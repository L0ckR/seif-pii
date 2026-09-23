"""Independent API process scaling with shared Redis and two NER groups.

Each group is one NER process plus N API workers sharing one listening socket.
All groups share a Redis DB and encryption key: any API can restore the
ciphertext record created by any other API. The driver round-robins groups.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import importlib
import json
import math
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, suppress
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
scaling = importlib.import_module("benchmarks.ner-models.rubert-throughput.http_scaling")
open_loop = importlib.import_module("benchmarks.ner-models.rubert-throughput.open_loop")
common, smoke, require = scaling.common, scaling.smoke, scaling.require
MODULE = "benchmarks.ner-models.rubert-throughput.cluster_scaling"


def create_api_app():
    """Add a private benchmark PID probe; the service implementation is intact."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    from seif.app import create_app

    app = create_app()

    async def benchmark_worker(request):
        supplied = request.headers.get("x-api-key", "")
        expected = os.environ["SEIF_RUBERT_HTTP_KEY"]
        if request.headers.get("x-system-id") != "benchmark" or not hmac.compare_digest(supplied, expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return {"pid": os.getpid()}

    benchmark_worker.__annotations__["request"] = Request
    app.get("/benchmark-worker")(benchmark_worker)
    # create_app mounts the demo at '/'; put this benchmark-only probe first.
    app.router.routes.insert(0, app.router.routes.pop())
    return app


def provenance(args):
    result = scaling.provenance(args)
    for path in (Path(__file__), Path(open_loop.__file__)):
        result["source_sha256"][str(path.relative_to(ROOT))] = scaling.legacy.file_digest(path)
    return result


def start_redis(args, temporary, children, log):
    from redis import Redis

    binary = str(args.redis_server) if args.redis_server else shutil.which("redis-server")
    require(binary is not None and os.path.isfile(binary) and os.access(binary, os.X_OK),
            "local_redis_server_exists")
    password = secrets.token_urlsafe(32)
    socket_path = temporary / "redis.sock"
    config = temporary / "redis.conf"
    config.write_text(f"port 0\ndir {temporary}\nunixsocket {socket_path}\nunixsocketperm 700\n"
                      f"requirepass {password}\nsave \"\"\nappendonly no\nmaxmemory {args.redis_maxmemory_mb}mb\n"
                      "maxmemory-policy noeviction\ndaemonize no\nloglevel warning\n")
    config.chmod(0o600)
    # Validated local executable and private generated config; no shell.
    process = subprocess.Popen([binary, str(config)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,  # noqa: S603
                               start_new_session=True)
    children.append(("redis", process))
    client = Redis(unix_socket_path=str(socket_path), password=password, socket_timeout=1)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            require(process.poll() is None, "isolated_redis_survives_startup")
            try:
                if client.ping():
                    info = client.info("server")
                    return socket_path, password, {"version": info["redis_version"], "transport": "private Unix socket",
                                                   "authenticated": True, "maxmemory_mb": args.redis_maxmemory_mb,
                                                   "persistence": False,
                                                   "before_load": redis_usage(socket_path, password)}
            except Exception:
                time.sleep(.1)
        require(False, "isolated_redis_readiness_deadline")
    finally:
        client.close()


def redis_usage(socket_path, password):
    from redis import Redis

    client = Redis(unix_socket_path=str(socket_path), password=password, socket_timeout=2)
    try:
        info = client.info("memory")
        return {**{name: info.get(name) for name in (
            "used_memory", "used_memory_peak", "used_memory_dataset", "maxmemory", "mem_fragmentation_ratio")},
            "dbsize": client.dbsize(), "record_ttl_seconds": 900}
    finally:
        client.close()


def start_api_workers(args, env, log, children, label):
    require(args.api_python.is_file(), "api_interpreter_exists")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1024)
        port = listener.getsockname()[1]
        argv = [str(args.api_python), "-m", "uvicorn", MODULE + ":create_api_app", "--factory",
                "--fd", str(listener.fileno()), "--workers", str(args.api_processes_per_replica),
                "--no-access-log", "--log-level", "warning"]
        # Validated local interpreter and fixed module arguments; no shell.
        process = subprocess.Popen(argv, cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)},  # noqa: S603
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   pass_fds=(listener.fileno(),), start_new_session=True)
        children.append((label, process))
    return process, f"http://127.0.0.1:{port}"


def call_connection(connection, headers, path, payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    connection.request("GET" if body is None else "POST", path, body=body,
                       headers={"Content-Type": "application/json", **headers})
    response = connection.getresponse()
    content = response.read(256 * 1024 + 1)
    require(len(content) <= 256 * 1024, "bounded_preflight_response")
    return response.status, json.loads(content)


def probe_worker(parsed, headers, timeout, gate):
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        require(gate.wait(timeout=timeout), "worker_probe_dispatch_deadline")
        status, body = call_connection(connection, headers, "/benchmark-worker")
        require(status == 200 and isinstance(body.get("pid"), int), "authenticated_worker_probe")
        return body["pid"], connection
    except BaseException:
        connection.close()
        raise


def discover_workers(args, parsed, headers):
    """Concurrent fresh connections avoid the shared listener's accept bias."""
    connections = {}
    try:
        deadline = time.monotonic() + min(args.startup_timeout, 30)
        burst_size = args.api_processes_per_replica * 8
        with ThreadPoolExecutor(max_workers=burst_size) as pool:
            while time.monotonic() < deadline:
                gate = threading.Event()
                timeout = min(args.timeout, max(.01, deadline - time.monotonic()))
                futures = [pool.submit(probe_worker, parsed, headers, timeout, gate) for _ in range(burst_size)]
                gate.set()
                first_error = None
                for future in as_completed(futures):
                    try:
                        pid, connection = future.result()
                    except Exception as exc:
                        first_error = first_error or exc
                        continue
                    if pid in connections:
                        connections[pid].close()
                    connections[pid] = connection
                if first_error is not None:
                    raise first_error
                if len(connections) == args.api_processes_per_replica:
                    break
        require(len(connections) == args.api_processes_per_replica, "all_api_worker_processes_observed")
        return connections
    except BaseException:
        for connection in connections.values():
            connection.close()
        raise


def verify_cross_worker(args, api_url, headers, case):
    """Retain connections to distinct PIDs and restore across those workers."""
    connections = discover_workers(args, urlsplit(api_url), headers)
    try:
        pids = list(connections)
        source, destination = pids[0], pids[-1]
        payload_id = "cross-worker-" + secrets.token_hex(16)
        status, body = call_connection(connections[source], headers, "/v1/mask", {
            "payload": case["text"], "payload_id": payload_id, "mode": "mask"})
        require(status == 200 and all(common.compare_response(case, payload_id, body)[key]
                for key in ("mask_match", "entities_match", "types_match")), "cross_worker_mask_reference")
        status, restored = call_connection(connections[destination], headers, "/v1/unmask", {
            "payload": body["result"], "payload_id": payload_id})
        require(status == 200 and restored.get("result") == case["text"], "cross_worker_restore_exact")
        return {"worker_pids": pids, "source_pid": source, "restore_pid": destination,
                "different_workers": source != destination, "exact": True,
                "all_configured_workers_observed": True, "excluded_from_timing": True}
    finally:
        for connection in connections.values():
            connection.close()


def launch_group(args, replica, temporary, children, logs, redis_socket, redis_password, master_key, case):  # noqa: PLR0913
    """Keep owned process resources and per-run credentials explicit at launch."""
    env = smoke.clean_environment()
    env.update(OMP_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads),
               MKL_NUM_THREADS=str(args.cpu_threads))
    token, api_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    policy = temporary / f"policy-{replica}.json"
    policy.write_text(json.dumps({"systems": {"benchmark": {"api_key_env": "SEIF_RUBERT_HTTP_KEY",
                               "mode": "mask", "allow_unmask": True, "rps": 100000}}}))
    ner_env = {**env, "SEIF_NER_TOKEN": token, "SEIF_NER_DEMO": "0",
               "SEIF_NER_MAX_MODEL_JOBS": str(args.ner_max_jobs),
               "SEIF_SCALING_MAX_HTTP": str(args.ner_max_http),
               "SEIF_SCALING_CPU_THREADS": str(args.cpu_threads),
               "SEIF_NER_BATCH_SIZE": str(args.model_batch_size), "SEIF_NER_BATCH_WAIT_MS": str(args.batch_wait_ms),
               "SEIF_RUBERT_HTTP_MODEL": str(args.model_path)}
    process, ner_url = smoke.start_service((args.ner_python, scaling.MODULE + ":create_ner_app", f"ner-{replica}"),
                                           ROOT, ner_env, logs[0], children)
    ner_health = smoke.await_ready(process, ner_url, args.startup_timeout)
    require(ner_health.get("model") == scaling.legacy.MODEL_ID, "actual_rubert_model")
    ner_process = process
    metric_directory = temporary / f"metrics-{replica}"
    metric_directory.mkdir(mode=0o700)
    api_env = {**env, "SEIF_DEMO": "0", "SEIF_CONFIG": str(policy), "SEIF_RUBERT_HTTP_KEY": api_key,
               "SEIF_MASTER_KEY": master_key,
               "SEIF_REDIS_URL": f"unix://{redis_socket}?db=0", "SEIF_REDIS_PASSWORD": redis_password,
               "PROMETHEUS_MULTIPROC_DIR": str(metric_directory),
               "SEIF_NER_URL": ner_url, "SEIF_NER_TOKEN": token, "SEIF_NER_TIMEOUT_SECONDS": "20",
               "SEIF_NER_MAX_CONCURRENCY": str(args.api_ner_concurrency), "SEIF_CPU_WORKERS": str(args.api_cpu_workers),
               "SEIF_MAX_INFLIGHT": str(args.api_max_inflight),
               "SEIF_NER_HTTP_BACKEND": args.ner_http_backend,
               "SEIF_REQUIRE_FREE_THREADING": "1", "SEIF_LOG_LEVEL": args.api_log_level}
    process, api_url = start_api_workers(args, api_env, logs[1], children, f"api-group-{replica}")
    api_health = smoke.await_ready(process, api_url, args.startup_timeout)
    require(api_health.get("storage") == "redis" and api_health.get("detector_profile") == "hybrid"
            and api_health.get("mode") == "restricted" and api_health.get("gil_enabled") is False,
            "shared_redis_free_threaded_api")
    headers = {"X-System-ID": "benchmark", "X-API-Key": api_key}
    worker_check = verify_cross_worker(args, api_url, headers, case)
    return {"api": common.LocalClient(api_url, headers, args.timeout),
            "ner": common.LocalClient(ner_url, {"Authorization": "Bearer " + token}, args.timeout),
            "ner_process": ner_process, "health": {"api": api_health, "ner": ner_health},
            "cross_worker_restore": worker_check}


def cross_group_preflight(pairs, case):
    """Create in each group, then restore through another independently served group."""
    checked = []
    for source, pair in enumerate(pairs):
        destination = (source + 1) % len(pairs)
        payload_id = "cross-group-" + secrets.token_hex(16)
        status, body = pair["api"].call("/v1/mask", {
            "payload": case["text"], "payload_id": payload_id, "mode": "mask"})
        require(status == 200 and all(common.compare_response(case, payload_id, body)[key]
                for key in ("mask_match", "entities_match", "types_match")), "cross_group_mask_reference")
        before = scaling.snapshot_pairs(pairs)
        status, restored = pairs[destination]["api"].call("/v1/unmask", {
            "payload": body["result"], "payload_id": payload_id})
        require(status == 200 and restored.get("result") == case["text"], "cross_group_restore_exact")
        require(scaling.snapshot_pairs(pairs) == before, "cross_group_restore_did_not_invoke_ner")
        checked.append({"source_group": source, "restore_group": destination, "exact": True,
                        "different_group": source != destination, "excluded_from_timing": True})
    return checked


def restore_after_open_loop(pairs, cases, outcomes, after):
    """Restore the retained load samples through a different API group."""
    by_id = {case["case_id"]: case for case in cases}
    samples = []
    for row in outcomes:
        if "masked" not in row or "error" in row:
            continue
        destination = (row["replica"] + 1) % len(pairs)
        status, body = pairs[destination]["api"].call("/v1/unmask", {
            "payload": row["masked"], "payload_id": row["payload_id"]})
        require(status == 200 and body.get("result") == by_id[row["case_id"]]["text"],
                "post_load_cross_group_restore_exact")
        samples.append({"source_group": row["replica"], "restore_group": destination,
                        "different_group": row["replica"] != destination, "exact": True})
    require(scaling.snapshot_pairs(pairs) == after, "post_load_restore_did_not_invoke_ner")
    return {"checked": len(samples), "exact": len(samples), "samples": samples,
            "excluded_from_mask_timing": True, "ner_counters_unchanged": True}


def latency_budget_report(phase, outcomes, budget_ms=500):
    """Performance acceptance is separate from correctness/accounting validity."""
    offered, dispatched = phase["offered_mask_requests"], phase["dispatched_mask_requests"]
    late = sum(row["scheduled_mask_latency_ms"] > budget_ms for row in outcomes)
    timely_correct = sum("error" not in row and row["mask_match"] and row["entities_match"]
                         and row["types_match"] and row["scheduled_mask_latency_ms"] <= budget_ms for row in outcomes)
    all_arrivals_sent = (phase["unsent_total"] == phase["callback_failures"] == 0 and dispatched == offered)
    return {"latency_budget_ms": budget_ms, "latency_origin": "scheduled arrival, including client delay",
            "offered_arrivals": offered, "all_offered_arrivals_dispatched": all_arrivals_sent,
            "served_attempts_over_budget": late,
            "fraction_dispatched_over_budget": late / dispatched if dispatched else None,
            "fraction_offered_served_over_budget": late / offered,
            "unsent_arrivals": phase["unsent_total"],
            "correct_successes_within_budget": timely_correct,
            "fraction_offered_correct_within_budget": timely_correct / offered,
            "at_least_95pct_offered_correct_within_budget": timely_correct / offered >= .95,
            "all_offered_correct_within_budget": timely_correct == offered,
            "note": "Diagnostic 500 ms budget; drops and failures cannot receive latency-SLA credit. No official SLA is inferred."}


def terminate_groups(children):
    """Stop only process groups created with start_new_session by this run."""
    results = []
    for label, process in reversed(children):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        forced = False
        try:
            code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            forced = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=10)
        # If a supervisor exited before its workers, release those owned workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
            forced = True
        except ProcessLookupError:
            pass
        results.append({"service": label, "return_code": code, "exited": process.poll() is not None,
                        "forced_group_kill": forced})
    return results


def warmup(pair):
    for _ in range(5):
        status, _body = pair["api"].call("/v1/mask", {
            "payload": "Иван Иванов приехал в Москву.", "payload_id": "warmup-" + secrets.token_hex(12)})
        require(status == 200, "warmup_success")


def verify_fail_closed(pairs):
    smoke.terminate_children([("ner-0", pairs[0]["ner_process"])])
    status, _body = pairs[0]["api"].call("/v1/mask", {
        "payload": "Иван Иванов", "payload_id": "failure-" + secrets.token_hex(16)})
    require(status == 503, "missing_ner_fails_closed")
    return True


def run(args, cases):
    children, pairs = [], []
    report = {"status": "FAIL", "phases": []}
    with tempfile.TemporaryDirectory(prefix="seif-rubert-cluster-") as folder, ExitStack() as stack:
        temporary = Path(folder)
        try:
            redis_log = stack.enter_context((temporary / "redis.log").open("wb"))
            redis_socket, redis_password, report["redis"] = start_redis(args, temporary, children, redis_log)
            master_key = base64.b64encode(secrets.token_bytes(32)).decode()
            case = next(case for case in cases if case["expected"] != case["text"])
            for replica in range(args.replicas):
                logs = [stack.enter_context((temporary / f"{label}-{replica}.log").open("wb"))
                        for label in ("ner", "api")]
                pair = launch_group(args, replica, temporary, children, logs, redis_socket, redis_password, master_key, case)
                pairs.append(pair)
                warmup(pair)
            report["health"] = [pair["health"] for pair in pairs]
            report["cross_worker_restore"] = [pair["cross_worker_restore"] for pair in pairs]
            report["cross_group_restore"] = cross_group_preflight(pairs, case)
            report["models"] = [common.ner_snapshot(pair["ner"]) for pair in pairs]
            for concurrency in args.concurrency if not args.offered_rps else []:
                frozen = provenance(args)
                print(json.dumps({"phase": "started", "api_processes": args.replicas * args.api_processes_per_replica,
                                  "ner_processes": args.replicas, "concurrency": concurrency}), flush=True)
                phase = scaling.run_phase(args, pairs, cases, "mask", concurrency)
                phase["immutable_sources_assets_models"] = frozen == provenance(args)
                phase["valid_successful_measurement"] &= phase["immutable_sources_assets_models"]
                report["phases"].append(phase)
                print(json.dumps({"phase": "completed", "mask_rps": phase["successful_mask_rps"],
                                  "p95_ms": phase["latency_ms_all_attempts"]["p95"],
                                  "valid": phase["valid_successful_measurement"]}), flush=True)
            if args.offered_rps:
                before = scaling.snapshot_pairs(pairs)
                print(json.dumps({"phase": "open-loop-started", "offered_rps": args.target_rps,
                                  "duration": args.duration}), flush=True)
                outcomes, traffic = open_loop.fixed_arrivals(args, pairs, cases)
                after = open_loop.drain_ner(args, pairs)
                phase = open_loop.report_phase(args, outcomes, traffic, before, after)
                phase["latency_budget"] = latency_budget_report(phase, outcomes)
                report["phases"].append(phase)
                phase["restore_samples"] = restore_after_open_loop(pairs, cases, outcomes, after)
                if phase["valid_successful_measurement"]:
                    require(phase["restore_samples"]["checked"] == min(8, phase["offered_mask_requests"]),
                            "all_retained_post_load_restore_samples_checked")
                print(json.dumps({"phase": "open-loop-completed", "mask_rps": phase["completed_mask_rps_with_drain"],
                                  "unsent": phase["unsent_total"], "valid": phase["valid_successful_measurement"]}), flush=True)
            report["redis"]["after_load"] = redis_usage(redis_socket, redis_password)
            report["missing_ner_returns_503"] = verify_fail_closed(pairs)
            report["status"] = "PASS" if all(phase["valid_successful_measurement"] for phase in report["phases"]) else "OVERLOAD"
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            for pair in pairs:
                pair["api"].close()
                pair["ner"].close()
            report["cleanup"] = terminate_groups(children)
            if report["status"] != "PASS":
                report["failure_logs"] = scaling.preserve_logs(args, args.replicas, temporary)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--ner-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--api-python", type=Path, default=ROOT.parent / "seif-pii/.venv/bin/python")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    parser.add_argument("--redis-server", type=Path, help="Executable path; default searches PATH")
    parser.add_argument("--redis-maxmemory-mb", type=int, choices=(512, 1024, 4096), default=512)
    parser.add_argument("--replicas", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--api-processes-per-replica", type=int, choices=(1, 2, 3, 4, 6), default=3)
    parser.add_argument("--concurrency", nargs="+", type=int, choices=(8, 16, 32, 64, 128, 256), default=[32, 64])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--api-cpu-workers", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--api-max-inflight", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--ner-http-backend", choices=("httpx", "aiohttp"), default="httpx")
    parser.add_argument("--api-log-level", choices=("INFO", "WARNING"), default="WARNING")
    parser.add_argument("--api-ner-concurrency", type=int, default=16)
    parser.add_argument("--model-batch-size", "--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=2)
    parser.add_argument("--ner-max-jobs", type=int, default=128)
    parser.add_argument("--ner-max-http", type=int, default=256)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument("--offered-rps", type=float, default=0, help="Optional fixed-rate mask load instead of closed-loop phases")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--max-inflight", type=int, choices=(32, 64, 128, 256, 512), default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency != sorted(set(args.concurrency)) or not 1 <= args.repeats <= 100:
        parser.error("Unique increasing concurrency and repeats1..100 are required")
    if not args.model_batch_size <= args.ner_max_jobs <= args.ner_max_http <= 512:
        parser.error("Require batch-size <= ner-max-jobs <= ner-max-http <= 512")
    if not 20 < args.timeout <= 120 or not 1 <= args.api_ner_concurrency <= 128:
        parser.error("Require timeout20..120 and API NER concurrency1..128")
    for name, bounds in {"batch_wait_ms": (0, 20), "startup_timeout": (1, 1800),
                         "offered_rps": (0, 10000), "duration": (1, 600)}.items():
        if not math.isfinite(getattr(args, name)) or not bounds[0] <= getattr(args, name) <= bounds[1]:
            parser.error(f"{name} must be finite and within {bounds}")
    if args.offered_rps and math.floor(args.offered_rps * args.duration) < 1:
        parser.error("At least one arrival must be scheduled")
    for name in ("model_path", "reference_cache", "ner_python", "api_python", "dataset", "output"):
        path = getattr(args, name).expanduser()
        setattr(args, name, path.absolute() if name in {"ner_python", "api_python"} else path.resolve())
    if args.redis_server:
        # Redis packages can use a multicall executable selected by argv[0].
        # Resolving redis-server -> redis-check-rdb would run the wrong command.
        args.redis_server = args.redis_server.expanduser().absolute()
    args.logs_dir = ROOT / "local-data/rubert-throughput" / (args.output.stem + "-" + secrets.token_hex(8))
    args.target_rps, args.mode, args.arrival_deadline_ms, args.drain_timeout = args.offered_rps, "mask", 250, 60
    return args


def main():
    args = parse_args()
    config_keys = ("replicas", "api_processes_per_replica", "concurrency", "repeats", "cpu_threads", "api_cpu_workers",
                   "api_max_inflight",
                   "ner_http_backend", "api_log_level", "redis_maxmemory_mb",
                   "api_ner_concurrency", "model_batch_size", "batch_wait_ms", "ner_max_jobs", "ner_max_http",
                   "offered_rps", "duration", "max_inflight")
    report = {"schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": {key: getattr(args, key) for key in config_keys},
              "limitations": ["Single physical GPU; process count does not imply multiple GPUs.",
                              "Round-robin API groups; connection acceptance by Uvicorn workers inside each group.",
                              "Authenticated isolated Redis and one encryption key shared by every API group.",
                              "Cross-worker and cross-group exact restoration are explicitly checked outside timing.",
                              "Redis uses a private Unix socket and no persistence, not production TCP/replication.",
                              "No prediction cache, capture, proxy or public tunnel; mask-only throughput."]}
    report["configuration"].update(total_api_processes=args.replicas * args.api_processes_per_replica,
                                    total_ner_processes=args.replicas, gpu_count=1,
                                    storage="shared authenticated Redis DB0 and common encryption key",
                                    api_metrics="multiprocess aggregate per API group",
                                    redis_server_executable=str(args.redis_server) if args.redis_server else shutil.which("redis-server"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        try:
            report["provenance"] = provenance(args)
            cases = scaling.prepare_cases(args)
            report["driver_environment"] = scaling.legacy.driver_environment()
            report["service"] = run(args, cases)
            require(report["provenance"] == provenance(args), "immutable_sources_assets_models")
            report["status"] = report["service"]["status"]
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
