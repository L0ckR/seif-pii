"""Bounded, real-inference scaling of independent API + RuBERT pipeline pairs.

The driver round-robins requests over 1/2/4 local pipeline pairs on one GPU.
Each API has an independent memory vault: restore checks use the same replica.
This is an experiment, not a shared-state production cluster or an SLA test.
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import os
import secrets
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
upgrade = importlib.import_module("benchmarks.ner-models.rubert-upgrade.http_benchmark")
legacy, common = upgrade.legacy, upgrade.common
smoke, require = legacy.smoke, legacy.require
MODULE = "benchmarks.ner-models.rubert-throughput.http_scaling"


class CountedAnalyzer(common.CountedAnalyzer):
    """Count documents, not just calls, when a real GPU batch is executed."""

    def analyze_batch(self, **kwargs):
        count = len(kwargs["texts"])
        self.counts.add(model_started=count, model_active=count, model_batches=1,
                        **{f"model_batch_size_{count}": 1})
        try:
            result = self.analyzer.analyze_batch(**kwargs)
        except BaseException:
            self.counts.add(model_failed=count)
            raise
        else:
            self.counts.add(model_completed=count)
            return result
        finally:
            self.counts.add(model_active=-count)


def create_ner_app():
    """Real word/native RuBERT, with bounded queue settings and instrumentation."""
    import torch

    from scripts.ner_service import NerSettings, create_app
    from seif.rubert_ner import RubertAnalyzer

    torch.set_num_threads(int(os.environ["SEIF_SCALING_CPU_THREADS"]))
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    counts, state = common.Counts(), {}

    def factory():
        analyzer = RubertAnalyzer.from_local(Path(os.environ["SEIF_RUBERT_HTTP_MODEL"]),
                                             decoder="word", profile="native",
                                             batch_size=int(os.environ["SEIF_NER_BATCH_SIZE"]))
        state["metadata"] = analyzer.metadata()
        state["server_environment"] = legacy.server_environment()
        wrapped = CountedAnalyzer(analyzer, counts)
        wrapped.supported_entities = analyzer.supported_entities
        return wrapped

    settings = NerSettings(token=os.environ["SEIF_NER_TOKEN"], demo=False,
                           max_http_inflight=int(os.environ["SEIF_SCALING_MAX_HTTP"]),
                           max_model_jobs=int(os.environ["SEIF_NER_MAX_MODEL_JOBS"]),
                           batch_size=int(os.environ["SEIF_NER_BATCH_SIZE"]),
                           batch_wait_ms=float(os.environ["SEIF_NER_BATCH_WAIT_MS"]))
    app = create_app(settings=settings, analyzer_factory=factory)
    app.add_middleware(common.CountedBoundary, counts=counts)

    @app.get("/benchmark-stats")
    async def benchmark_stats():
        return {"counts": counts.snapshot(), "model": state["metadata"],
                "server_environment": state["server_environment"]}

    return app


def provenance(args):
    result = upgrade.provenance(args)
    for path in (Path(__file__), ROOT / "pyproject.toml", ROOT / "requirements.lock"):
        result["source_sha256"][str(path.relative_to(ROOT))] = legacy.file_digest(path)
    return result


def prepare_cases(args):
    from scripts.evaluate_golden import load_cases, load_ner_cache

    prepared = legacy.prepare_cases(args)
    cache = load_ner_cache(args.reference_cache, load_cases(args.dataset))
    for case in prepared:
        case["ner_entities"] = [(row["start"], row["end"], row["entity_type"])
                                for row in cache[case["case_id"]]["entities"]]
    return prepared


def launch_pair(args, replica, temporary, children, logs):
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
               "SEIF_NER_BATCH_SIZE": str(args.model_batch_size),
               "SEIF_NER_BATCH_WAIT_MS": str(args.batch_wait_ms),
               "SEIF_RUBERT_HTTP_MODEL": str(args.model_path)}
    process, ner_url = smoke.start_service((args.ner_python, MODULE + ":create_ner_app", f"ner-{replica}"),
                                           ROOT, ner_env, logs[0], children)
    ner_health = smoke.await_ready(process, ner_url, args.startup_timeout)
    require(ner_health.get("model") == legacy.MODEL_ID, "real_rubert_readiness")
    ner_process = process
    api_env = {**env, "SEIF_DEMO": "0", "SEIF_CONFIG": str(policy), "SEIF_RUBERT_HTTP_KEY": api_key,
               "SEIF_MASTER_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
               "SEIF_NER_URL": ner_url, "SEIF_NER_TOKEN": token, "SEIF_NER_TIMEOUT_SECONDS": "20",
               "SEIF_NER_MAX_CONCURRENCY": str(args.api_ner_concurrency),
               "SEIF_NER_HTTP_BACKEND": args.ner_http_backend,
               "SEIF_CPU_WORKERS": str(args.api_cpu_workers),
               "SEIF_REQUIRE_FREE_THREADING": "1", "SEIF_LOG_LEVEL": args.api_log_level}
    process, api_url = smoke.start_service((args.api_python, "seif.app:create_app", f"api-{replica}"),
                                           ROOT, api_env, logs[1], children)
    api_health = smoke.await_ready(process, api_url, args.startup_timeout)
    require(api_health.get("storage") == "memory" and api_health.get("detector_profile") == "hybrid"
            and api_health.get("mode") == "restricted" and api_health.get("gil_enabled") is False,
            "isolated_restricted_free_threaded_hybrid_api")
    return {"api": common.LocalClient(api_url, {"X-System-ID": "benchmark", "X-API-Key": api_key}, args.timeout),
            "ner": common.LocalClient(ner_url, {"Authorization": "Bearer " + token}, args.timeout),
            "ner_process": ner_process, "health": {"api": api_health, "ner": ner_health}}


def ner_request(client, case):
    from seif.ner_contract import validate_entity

    started = time.perf_counter()
    outcome = {"case_id": case["case_id"], "status": None}
    try:
        status, body = client.call("/analyze", {"text": case["text"]})
        outcome["latency_ms"] = (time.perf_counter() - started) * 1000
        outcome["status"] = status
        if status != 200:
            outcome["error"] = f"http_{status}"
        else:
            require(isinstance(body, dict) and set(body) == {"entities"}
                    and isinstance(body["entities"], list), "ner_response_schema")
            for entity in body["entities"]:
                validate_entity(entity, len(case["text"]))
            entities = [(row["start"], row["end"], row["entity_type"]) for row in body["entities"]]
            outcome["entities_match"] = entities == case["ner_entities"]
    except Exception as exc:
        outcome.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        outcome["error"] = "timeout" if isinstance(exc, TimeoutError) else "invalid_response_or_transport"
    return outcome


def request(pairs, cases, index, stage, run_id):
    replica, case = index % len(pairs), cases[index % len(cases)]
    if stage == "mask":
        outcome = common.mask_request(pairs[replica]["api"], case, f"scaling-{run_id}-{index}")
    else:
        outcome = ner_request(pairs[replica]["ner"], case)
    outcome["replica"] = replica
    return outcome


def snapshot_pairs(pairs):
    return [{"ner": common.ner_snapshot(pair["ner"])["counts"],
             "api_ner_stages": common.ner_stage_count(pair["api"])} for pair in pairs]


def summarize(outcomes, elapsed, stage, concurrency, before, after):  # noqa: PLR0913
    """Keep measurement dimensions and independent counter snapshots explicit."""
    errors = Counter(row["error"] for row in outcomes if "error" in row)
    successes = [row for row in outcomes if "error" not in row]
    match_keys = ("mask_match", "entities_match", "types_match") if stage == "mask" else ("entities_match",)
    matches = sum(all(row[key] for key in match_keys) for row in successes)
    per_replica = []
    for replica, (old, new) in enumerate(zip(before, after, strict=True)):
        counts = common.delta_counts(old["ner"], new["ner"])
        attempts = sum(row["replica"] == replica for row in outcomes)
        inferred = attempts == counts.get("model_started") == counts.get("model_completed") == counts.get("http_200")
        stage_delta = new["api_ner_stages"] - old["api_ner_stages"]
        if stage == "mask":
            inferred &= stage_delta == attempts
        else:
            inferred &= stage_delta == 0
        drained = new["ner"]["http_active"] == new["ner"]["model_active"] == 0
        per_replica.append({"replica": replica, "attempts": attempts, "ner_counts": counts,
                            "api_ner_stage_attempts": stage_delta, "drained": drained,
                            "every_request_completed_real_ner": inferred and counts.get("model_failed", 0) == 0})
    latencies = sorted(row["latency_ms"] for row in outcomes)
    real_inference = all(row["every_request_completed_real_ner"] for row in per_replica)
    drained = all(row["drained"] for row in per_replica)
    return {"stage": stage, "replicas": len(before), "concurrency": concurrency, "requests": len(outcomes),
            "elapsed_seconds": elapsed, "attempted_rps": len(outcomes) / elapsed,
            "successful_rps": len(successes) / elapsed, "reference_matching_rps": matches / elapsed,
            "successful_mask_rps": len(successes) / elapsed if stage == "mask" else None,
            "errors": dict(errors), "http_statuses": dict(Counter(str(row["status"]) for row in outcomes)),
            "success_ratio": len(successes) / len(outcomes), "reference_matches": matches,
            "reference_match_ratio": matches / len(outcomes),
            "mismatches": {key: sum(not row[key] for row in successes) for key in match_keys},
            "every_request_completed_real_ner": real_inference, "ner_drained": drained,
            "valid_successful_measurement": matches == len(outcomes) and real_inference and drained,
            "latency_ms_all_attempts": {"p50": statistics.median(latencies),
                                       "p95": latencies[math.ceil(len(latencies) * .95) - 1],
                                       "p99": latencies[math.ceil(len(latencies) * .99) - 1],
                                       "max": max(latencies), "over_500ms": sum(value > 500 for value in latencies)},
            "per_replica": per_replica}


def run_phase(args, pairs, cases, stage, concurrency):
    before = snapshot_pairs(pairs)
    run_id = secrets.token_hex(16)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        started = time.perf_counter()
        futures = [pool.submit(request, pairs, cases, index, stage, run_id)
                   for index in range(len(cases) * args.repeats)]
        outcomes = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
    deadline = time.monotonic() + args.timeout
    after = snapshot_pairs(pairs)
    while any(row["ner"]["http_active"] or row["ner"]["model_active"] for row in after):
        if time.monotonic() >= deadline:
            break
        time.sleep(.1)
        after = snapshot_pairs(pairs)
    report = summarize(outcomes, elapsed, stage, concurrency, before, after)
    if stage == "mask" and report["ner_drained"]:
        restored = 0
        for index in range(0, len(cases), max(1, len(cases) // 8)):
            row = outcomes[index]
            if "error" in row:
                continue
            status, body = pairs[row["replica"]]["api"].call("/v1/unmask", {
                "payload": row["masked"], "payload_id": row["payload_id"]})
            require(status == 200 and body.get("result") == cases[index]["text"], "sticky_restore_exact")
            restored += 1
            if restored == 8:
                break
        report["restore"] = {"checked": restored, "exact": restored, "sticky_to_source_replica": True,
                             "excluded_from_mask_timing": True}
        require(snapshot_pairs(pairs) == after, "restore_did_not_invoke_ner")
    if args.include_traces:
        report["per_case"] = [{key: value for key, value in row.items() if key not in {"masked", "payload_id"}}
                              for row in outcomes]
    return report


def preserve_logs(args, replicas, temporary):
    directory = args.logs_dir / f"replicas-{replicas}"
    directory.mkdir(parents=True, mode=0o700)
    paths = []
    for source in temporary.glob("*.log"):
        destination = directory / source.name
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as stream:
            shutil.copyfileobj(stream, output)
        paths.append(str(destination))
    return paths


def run_configuration(args, replicas, cases):
    children, pairs = [], []
    report = {"replicas": replicas, "status": "FAIL", "phases": []}
    with tempfile.TemporaryDirectory(prefix="seif-rubert-scaling-") as folder, ExitStack() as stack:
        temporary = Path(folder)
        try:
            for replica in range(replicas):
                logs = [stack.enter_context((temporary / f"{label}-{replica}.log").open("wb"))
                        for label in ("ner", "api")]
                pair = launch_pair(args, replica, temporary, children, logs)
                pairs.append(pair)
                for _ in range(5):
                    status, _body = pair["api"].call("/v1/mask", {"payload": "Иван Иванов приехал в Москву.",
                                                                  "payload_id": "warmup-" + secrets.token_hex(12)})
                    require(status == 200, "warmup_success")
            report["health"] = [pair["health"] for pair in pairs]
            report["models"] = [common.ner_snapshot(pair["ner"]) for pair in pairs]
            for stage in args.stages:
                for concurrency in args.concurrency:
                    source_before = provenance(args)
                    print(json.dumps({"phase": "started", "replicas": replicas, "stage": stage,
                                      "concurrency": concurrency, "requests": len(cases) * args.repeats}), flush=True)
                    phase = run_phase(args, pairs, cases, stage, concurrency)
                    phase["immutable_sources_assets_models"] = source_before == provenance(args)
                    phase["valid_successful_measurement"] &= phase["immutable_sources_assets_models"]
                    report["phases"].append(phase)
                    print(json.dumps({"phase": "completed", "replicas": replicas, "stage": stage,
                                      "concurrency": concurrency, "rps": phase["successful_rps"],
                                      "p95_ms": phase["latency_ms_all_attempts"]["p95"], "errors": phase["errors"],
                                      "valid": phase["valid_successful_measurement"]}), flush=True)
                    if not phase["ner_drained"] or not phase["immutable_sources_assets_models"]:
                        return report
            smoke.terminate_children([("ner-0", pairs[0]["ner_process"])])
            status, _body = pairs[0]["api"].call("/v1/mask", {
                "payload": "Иван Иванов", "payload_id": "failure-" + secrets.token_hex(16)})
            require(status == 503, "missing_ner_fails_closed")
            report["missing_ner_returns_503"] = True
            report["status"] = "PASS" if all(row["valid_successful_measurement"] for row in report["phases"]) else "OVERLOAD"
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            for pair in pairs:
                pair["api"].close()
                pair["ner"].close()
            report["cleanup"] = smoke.terminate_children(children)
            if report["status"] != "PASS":
                report["failure_logs"] = preserve_logs(args, replicas, temporary)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--ner-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--api-python", type=Path, default=ROOT.parent / "seif-pii/.venv/bin/python")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    parser.add_argument("--replicas", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument("--concurrency", type=int, nargs="+", choices=(1, 4, 8, 16, 32, 64, 128), default=[8, 16, 32, 64])
    parser.add_argument("--stages", nargs="+", choices=("ner", "mask"), default=["ner", "mask"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--api-cpu-workers", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--ner-http-backend", choices=("httpx", "aiohttp"), default="httpx")
    parser.add_argument("--api-log-level", choices=("INFO", "WARNING"), default="WARNING")
    parser.add_argument("--api-ner-concurrency", type=int, default=4)
    parser.add_argument("--model-batch-size", "--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=1)
    parser.add_argument("--batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--ner-max-jobs", type=int, default=4)
    parser.add_argument("--ner-max-http", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--include-traces", action="store_true", help="Optional per-request metadata; no texts or IDs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate provenance and workload without starting servers or GPU.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for field in ("replicas", "concurrency"):
        if getattr(args, field) != sorted(set(getattr(args, field))):
            parser.error(f"{field} must be unique and increasing")
    if len(args.stages) != len(set(args.stages)) or not 1 <= args.repeats <= 100:
        parser.error("stages must be unique; repeats must be between 1 and 100")
    if not 1 <= args.ner_max_jobs <= args.ner_max_http <= 512:
        parser.error("Require 1 <= ner-max-jobs <= ner-max-http <= 512")
    if not 1 <= args.api_ner_concurrency <= 128 or args.model_batch_size > args.ner_max_jobs:
        parser.error("API NER concurrency must be 1..128; batch size must not exceed model capacity")
    if not math.isfinite(args.batch_wait_ms) or not 0 <= args.batch_wait_ms <= 20:
        parser.error("Batch delay must be finite and between 0 and 20 ms")
    if not 20 < args.timeout <= 120 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("Timeout must be 20..120 seconds and startup timeout finite and positive")
    for name in ("model_path", "reference_cache", "ner_python", "api_python", "dataset", "output"):
        path = getattr(args, name).expanduser()
        setattr(args, name, path.absolute() if name in {"ner_python", "api_python"} else path.resolve())
    args.logs_dir = ROOT / "local-data/rubert-throughput" / (args.output.stem + "-" + secrets.token_hex(8))
    return args


def main():
    args = parse_args()
    report = {"schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": {"decoder": "word", "profile": "native", "gpu_count": 1,
                                "pipeline_replicas": args.replicas, "concurrency": args.concurrency,
                                "stages": args.stages, "repeats": args.repeats, "unique_texts": 446,
                                "cpu_threads_per_ner": args.cpu_threads, "ner_max_jobs": args.ner_max_jobs,
                                "ner_max_http": args.ner_max_http, "api_cpu_workers_per_replica": args.api_cpu_workers,
                                "ner_http_backend": args.ner_http_backend, "api_log_level": args.api_log_level,
                                "api_ner_concurrency": args.api_ner_concurrency,
                                "model_batch_size": args.model_batch_size, "batch_wait_ms": args.batch_wait_ms,
                                "api_processes_per_replica": 1, "ner_processes_per_replica": 1,
                                "storage": "independent memory vaults; sticky restore", "capture": False,
                                "text_cache": False, "request_retries": False},
              "limitations": ["Closed-loop local HTTP throughput, not open-loop 2000 RPS endurance or a production SLA.",
                              "Round-robin traffic goes to independent API+NER pipelines on ONE GPU; no shared Redis or load balancer.",
                              "Every timed successful request runs real NER; frozen references are expectations only.",
                              "Mask RPS excludes warmup, restore, direct NER and admin traffic.",
                              "Parity is exact masks/types/offsets against frozen predictions, not perfect ground truth.",
                              "No original texts, model values, secrets or payload IDs are retained in report."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        try:
            report["provenance"] = provenance(args)
            cases = prepare_cases(args)
            report["driver_environment"] = legacy.driver_environment()
            if args.dry_run:
                report["status"] = "DRY_RUN_PASS"
            else:
                report["configurations"] = [run_configuration(args, replicas, cases) for replicas in args.replicas]
                require(report["provenance"] == provenance(args), "immutable_sources_assets_models")
                statuses = {row["status"] for row in report["configurations"]}
                report["status"] = "FAIL" if "FAIL" in statuses else "OVERLOAD" if "OVERLOAD" in statuses else "PASS"
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0 if report["status"] in {"PASS", "DRY_RUN_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
