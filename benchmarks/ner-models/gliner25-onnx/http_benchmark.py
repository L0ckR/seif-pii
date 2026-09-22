"""Isolated mask-only HTTP comparison with actual Torch/ONNX inference.

Each backend receives 446 fresh documents at concurrency 1, then the same 446
at concurrency 4 with fresh correlation IDs. This is a bounded closed-loop
measurement, not an open-loop capacity or production SLA claim. Frozen NER
predictions supply expectations only; they are never served to the API.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import secrets
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MODULE = "benchmarks.ner-models.gliner25-onnx.http_benchmark"
REFERENCE = ROOT / "benchmarks/ner-models/gliner25-multi-v1/selected-service.jsonl"
SMOKE_PATH = REFERENCE.with_name("http_smoke.py")
spec = importlib.util.spec_from_file_location("_seif_http_smoke", SMOKE_PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
require = smoke.require
THREADS = 4
SEED = 20260922
CASE_COUNT = 446


class Counts:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = Counter(model_started=0, model_completed=0, model_failed=0, model_active=0,
                              http_started=0, http_completed=0, http_active=0)

    def add(self, **changes):
        with self.lock:
            self.values.update(changes)

    def snapshot(self):
        with self.lock:
            return dict(self.values)


class CountedAnalyzer:
    def __init__(self, analyzer, counts):
        self.analyzer, self.counts = analyzer, counts
        self.model_name = analyzer.model_name

    def analyze(self, **kwargs):
        self.counts.add(model_started=1, model_active=1)
        try:
            result = self.analyzer.analyze(**kwargs)
        except BaseException:
            self.counts.add(model_failed=1)
            raise
        else:
            self.counts.add(model_completed=1)
            return result
        finally:
            self.counts.add(model_active=-1)


class CountedBoundary:
    def __init__(self, app, counts):
        self.app, self.counts = app, counts

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/analyze":
            return await self.app(scope, receive, send)
        self.counts.add(http_started=1, http_active=1)
        status = 500

        async def counted_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, counted_send)
        finally:
            self.counts.add(http_completed=1, http_active=-1, **{f"http_{status}": 1})


def create_ner_app():
    """Private subprocess factory; instrumentation wraps the real analyzer."""
    import torch

    from scripts.ner_service import create_app

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    counts, state = Counts(), {}

    def factory():
        backend = os.environ["SEIF_HTTP_BENCH_BACKEND"]
        native = Path(os.environ["SEIF_HTTP_BENCH_NATIVE"])
        device = os.environ["SEIF_HTTP_BENCH_DEVICE"]
        if backend == "torch":
            from seif.gliner_ner import GlinerAnalyzer

            analyzer = GlinerAnalyzer.from_local(native, device=device, schema="described-names", threshold=.8)
            metadata = {"backend": "torch", "device": device, "dtype": "torch.float32", "cuda_tf32": False}
        else:
            from seif.gliner_onnx import GlinerOnnxAnalyzer

            analyzer = GlinerOnnxAnalyzer.from_local(
                Path(os.environ["SEIF_HTTP_BENCH_ONNX"]), native, device=device, cpu_threads=THREADS,
            )
            metadata = analyzer.metadata()
        state["metadata"] = metadata
        return CountedAnalyzer(analyzer, counts)

    app = create_app(analyzer_factory=factory)
    app.add_middleware(CountedBoundary, counts=counts)

    @app.get("/benchmark-stats")
    async def benchmark_stats():
        # The existing NER authentication boundary also guards this endpoint.
        return {"counts": counts.snapshot(), "model": state["metadata"]}

    return app


class LocalClient:
    """One reusable loopback HTTP connection per worker; bounded responses."""
    def __init__(self, url, headers, timeout):
        parsed = urlsplit(url)
        require(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port
                and not parsed.username and not parsed.password, "private_loopback_client")
        self.host, self.port, self.headers, self.timeout = parsed.hostname, parsed.port, headers, timeout
        self.local, self.connections, self.lock = threading.local(), [], threading.Lock()

    def call(self, path, body=None, *, raw=False):
        if getattr(self.local, "connection", None) is None:
            self.local.connection = HTTPConnection(self.host, self.port, timeout=self.timeout)
            with self.lock:
                self.connections.append(self.local.connection)
        connection = self.local.connection
        try:
            encoded = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            connection.request("GET" if body is None else "POST", path, body=encoded,
                               headers={"Content-Type": "application/json", **self.headers})
            response = connection.getresponse()
            content = response.read(256 * 1024 + 1)
            require(len(content) <= 256 * 1024, "bounded_http_response")
            return response.status, content.decode("utf-8") if raw else json.loads(content)
        except Exception:
            connection.close()
            self.local.connection = None
            raise

    def close(self):
        for connection in self.connections:
            connection.close()


def canonical_entities(entities, text_length):
    require(isinstance(entities, list), "entity_list")
    rows = []
    for entity in entities:
        require(isinstance(entity, dict) and set(entity) == {"start", "end", "type", "confidence", "reason"},
                "entity_schema")
        start, end, score = entity["start"], entity["end"], entity["confidence"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= text_length
                and type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
                and isinstance(entity["type"], str) and isinstance(entity["reason"], str), "entity_values")
        rows.append((start, end, entity["type"]))
    return rows


def prepare_cases(args):
    from dataclasses import asdict

    from scripts.evaluate_golden import load_cases, load_ner_cache
    from seif.detector import Span, detect, merge_ner_candidates
    from seif.transform import mask

    cases = load_cases(args.dataset)
    require(len(cases) == CASE_COUNT and len({row["text"] for row in cases.values()}) == CASE_COUNT,
            "complete_unique_organizer_corpus")
    cache = load_ner_cache(args.reference_cache, cases)
    metadata = json.loads(args.reference_cache.with_suffix(".meta.json").read_text())
    require(metadata["configuration"]["schema"] == "described-names"
            and metadata["configuration"]["threshold"] == .8, "frozen_selected_reference")
    prepared = []
    for key, row in cases.items():
        text = row["text"]
        require(0 < len(text) <= 16000, "exactly_one_ner_chunk_per_document")
        candidates = [Span(e["start"], e["end"], e["entity_type"], e["score"], "presidio-ru-ner")
                      for e in cache[key]["entities"]]
        spans = merge_ner_candidates(text, detect(text), candidates)
        expected, _ = mask(text, spans, "mask")
        prepared.append({"case_id": key, "text": text, "expected": expected,
                         "entities": canonical_entities([asdict(span) for span in spans], len(text)),
                         "types": sorted({span.type for span in spans})})
    return prepared


def compare_response(case, payload_id, body):
    expected_keys = {"result", "payload_id", "entities", "types", "mode", "latency_ms", "masking_enabled"}
    require(isinstance(body, dict) and set(body) == expected_keys and body["payload_id"] == payload_id
            and isinstance(body["result"], str) and body["mode"] == "mask" and body["masking_enabled"] is True,
            "mask_response_contract")
    entities = canonical_entities(body["entities"], len(case["text"]))
    return {"mask_match": body["result"] == case["expected"], "entities_match": entities == case["entities"],
            "types_match": body["types"] == case["types"],
            "unexpected_unmasked": body["result"] == case["text"] and case["expected"] != case["text"]}


def mask_request(client, case, payload_id):
    started = time.perf_counter()
    outcome = {"case_id": case["case_id"], "payload_id": payload_id, "status": None}
    try:
        status, body = client.call("/v1/mask", {"payload": case["text"], "payload_id": payload_id, "mode": "mask"})
        outcome["latency_ms"] = (time.perf_counter() - started) * 1000
        outcome["status"] = status
        if status == 200:
            outcome.update(compare_response(case, payload_id, body))
            outcome["masked"] = body["result"]
        else:
            outcome["error"] = f"http_{status}"
    except Exception as exc:
        outcome.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        outcome["error"] = "timeout" if isinstance(exc, TimeoutError) else "invalid_response_or_transport"
    return outcome


def ner_stage_count(client):
    status, body = client.call("/metrics", raw=True)
    require(status == 200, "api_metrics_available")
    prefix = 'seif_stage_duration_seconds_count{stage="ner"} '
    values = [float(line[len(prefix):]) for line in body.splitlines() if line.startswith(prefix)]
    require(len(values) <= 1, "single_worker_ner_stage_counter")
    return values[0] if values else 0


def ner_snapshot(client):
    status, body = client.call("/benchmark-stats")
    require(status == 200, "ner_instrumentation_available")
    return body


def delta_counts(before, after):
    return {key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}


def phase_summary(outcomes, elapsed, concurrency, counts, stage_count):
    values = sorted(row["latency_ms"] for row in outcomes)
    errors = Counter(row["error"] for row in outcomes if "error" in row)
    successful = [row for row in outcomes if "error" not in row]
    matches = sum(all(row[key] for key in ("mask_match", "entities_match", "types_match")) for row in successful)
    all_inferred = (stage_count == len(outcomes) == counts.get("model_started")
                    == counts.get("model_completed") == counts.get("http_200")
                    and counts.get("model_failed", 0) == 0)
    return {
        "concurrency": concurrency, "mask_requests": len(outcomes), "elapsed_seconds": elapsed,
        "attempted_mask_rps": len(outcomes) / elapsed, "successful_mask_rps": len(successful) / elapsed,
        "reference_matching_mask_rps": matches / elapsed, "success_ratio": len(successful) / len(outcomes),
        "reference_match_ratio": matches / len(outcomes), "reference_matches": matches,
        "errors": dict(errors), "http_statuses": dict(Counter(str(row["status"]) for row in outcomes)),
        "mask_mismatches": sum(not row["mask_match"] for row in successful),
        "typed_entity_mismatches": sum(not row["entities_match"] for row in successful),
        "type_list_mismatches": sum(not row["types_match"] for row in successful),
        "unexpected_unmasked_responses": sum(row["unexpected_unmasked"] for row in successful),
        "ner_stage_attempts": stage_count, "requests_without_ner_stage": len(outcomes) - stage_count,
        "ner_counts": counts, "every_mask_completed_real_ner": all_inferred,
        "valid_successful_measurement": len(successful) == matches == len(outcomes) and all_inferred,
        "latency_ms_all_attempts": {"p50": statistics.median(values),
                                   "p95": values[math.ceil(len(values) * .95) - 1],
                                   "p99": values[math.ceil(len(values) * .99) - 1], "max": max(values),
                                   "over_500ms": sum(value > 500 for value in values)},
        "per_case": [{key: value for key, value in row.items() if key not in {"masked", "payload_id"}}
                     for row in outcomes],
    }


def restore_samples(client, cases, outcomes):
    selected = [index for index in range(0, len(cases), max(1, len(cases) // 8))
                if "error" not in outcomes[index]][:8]
    passed = 0
    for index in selected:
        row = outcomes[index]
        status, restored = client.call("/v1/unmask", {"payload": row["masked"], "payload_id": row["payload_id"]})
        require(status == 200 and restored.get("result") == cases[index]["text"], "sample_exact_restore")
        passed += 1
    return {"checked": passed, "exact": passed, "excluded_from_mask_timing": True}


def run_phase(api, ner, cases, concurrency):
    before, stage_before = ner_snapshot(ner)["counts"], ner_stage_count(api)
    run_id = secrets.token_hex(16)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        started = time.perf_counter()
        futures = [pool.submit(mask_request, api, case, f"http-bench-{run_id}-{index}")
                   for index, case in enumerate(cases)]
        outcomes = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
    deadline = time.monotonic() + api.timeout
    after = ner_snapshot(ner)["counts"]
    while (after["http_active"] or after["model_active"]) and time.monotonic() < deadline:
        time.sleep(.1)
        after = ner_snapshot(ner)["counts"]
    report = phase_summary(outcomes, elapsed, concurrency, delta_counts(before, after),
                           ner_stage_count(api) - stage_before)
    report["ner_drained"] = after["http_active"] == after["model_active"] == 0
    report["valid_successful_measurement"] &= report["ner_drained"]
    if not report["ner_drained"]:
        return report
    report["restore"] = restore_samples(api, cases, outcomes)
    require(ner_snapshot(ner)["counts"] == after, "restore_did_not_invoke_ner")
    return report


def launch(args, backend, temporary, children, logs):
    env = smoke.clean_environment()
    env.update(OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS))
    token, api_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    policy = temporary / "policy.json"
    policy.write_text(json.dumps({"systems": {"benchmark": {"api_key_env": "SEIF_HTTP_BENCH_KEY",
                           "mode": "mask", "allow_unmask": True, "rps": 100000}}}))
    ner_env = {**env, "SEIF_NER_TOKEN": token, "SEIF_NER_DEMO": "0", "SEIF_NER_MAX_MODEL_JOBS": "4",
               "SEIF_HTTP_BENCH_BACKEND": backend, "SEIF_HTTP_BENCH_NATIVE": str(args.model_path),
               "SEIF_HTTP_BENCH_ONNX": str(args.onnx_path or ""), "SEIF_HTTP_BENCH_DEVICE": args.device}
    process, ner_url = smoke.start_service((args.ner_python, MODULE + ":create_ner_app", "ner"),
                                           ROOT, ner_env, logs[0], children)
    ner_health = smoke.await_ready(process, ner_url, args.startup_timeout)
    expected_model = smoke.MODEL_NAME
    if backend == "onnx":
        from seif.gliner_onnx import MODEL_NAME

        expected_model = MODEL_NAME
    require(ner_health.get("model") == expected_model, "actual_gliner_health")
    api_env = {**env, "SEIF_DEMO": "0", "SEIF_CONFIG": str(policy), "SEIF_HTTP_BENCH_KEY": api_key,
               "SEIF_MASTER_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
               "SEIF_NER_URL": ner_url, "SEIF_NER_TOKEN": token, "SEIF_NER_TIMEOUT_SECONDS": "20",
               "SEIF_CPU_WORKERS": "1", "SEIF_REQUIRE_FREE_THREADING": "1", "SEIF_LOG_LEVEL": "WARNING"}
    process, api_url = smoke.start_service((args.api_python, "seif.app:create_app", "api"),
                                           ROOT, api_env, logs[1], children)
    api_health = smoke.await_ready(process, api_url, args.startup_timeout)
    require(api_health.get("storage") == "memory" and api_health.get("detector_profile") == "hybrid"
            and api_health.get("mode") == "restricted" and api_health.get("gil_enabled") is False,
            "isolated_restricted_free_threaded_hybrid_api")
    return (LocalClient(api_url, {"X-System-ID": "benchmark", "X-API-Key": api_key}, args.timeout),
            LocalClient(ner_url, {"Authorization": "Bearer " + token}, args.timeout),
            {"api": api_health, "ner": ner_health})


def run_backend(args, backend, cases):
    children, clients = [], []
    report = {"backend": backend, "status": "FAIL"}
    with tempfile.TemporaryDirectory(prefix="seif-gliner-http-benchmark-") as folder:
        temporary = Path(folder)
        with (temporary / "ner.log").open("wb") as ner_log, (temporary / "api.log").open("wb") as api_log:
            try:
                api, ner, report["health"] = launch(args, backend, temporary, children, (ner_log, api_log))
                clients.extend((api, ner))
                for _ in range(5):
                    status, _body = api.call("/v1/mask", {"payload": "Иван Иванов приехал в Москву.",
                                                           "payload_id": "warmup-" + secrets.token_hex(12)})
                    require(status == 200, "warmup_success")
                report["model"] = ner_snapshot(ner)["model"]
                report["phases"] = []
                for concurrency in (1, 4):
                    print(json.dumps({"backend": backend, "concurrency": concurrency,
                                      "phase": "started", "mask_requests": len(cases)}), flush=True)
                    report["phases"].append(run_phase(api, ner, cases, concurrency))
                    phase = report["phases"][-1]
                    print(json.dumps({"backend": backend, "concurrency": concurrency, "phase": "completed",
                                      "successful_mask_rps": phase["successful_mask_rps"],
                                      "reference_match_ratio": phase["reference_match_ratio"],
                                      "every_mask_completed_real_ner": phase["every_mask_completed_real_ner"]}), flush=True)
                    if not report["phases"][-1]["ner_drained"]:
                        return report
                # A fresh correlation ID must fail closed after our own NER child stops.
                report["ner_stop"] = smoke.terminate_children([children[0]])
                status, _body = api.call("/v1/mask", {"payload": "Иван Иванов", "payload_id": secrets.token_hex(16)})
                require(status == 503 and api.call("/health")[0] == 503, "missing_ner_fails_closed")
                report["missing_ner_returns_503"] = True
                report["status"] = "PASS" if all(p["valid_successful_measurement"] for p in report["phases"]) else "FAIL"
            except Exception as exc:
                report["error_type"] = type(exc).__name__
                if isinstance(exc, smoke.SmokeFailure):
                    report["failed_check"] = str(exc)
            finally:
                for client in clients:
                    client.close()
                report["cleanup"] = smoke.terminate_children(children)
                if report["status"] != "PASS":
                    report["failure_logs"] = preserve_failure_logs(args.logs_dir / backend, temporary)
    return report


def preserve_failure_logs(directory, temporary):
    directory.mkdir(parents=True, mode=0o700)
    paths = []
    for name in ("api.log", "ner.log"):
        path = directory / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output, (temporary / name).open("rb") as source:
            shutil.copyfileobj(source, output)
        paths.append(str(path))
    return paths


def provenance(args):
    from seif.gliner_onnx import NATIVE_FILES, ONNX_FILES

    sources = sorted((ROOT / "seif").glob("*.py")) + [Path(__file__), SMOKE_PATH,
               ROOT / "scripts/ner_service.py", ROOT / "scripts/evaluate_golden.py"]
    assets = [args.dataset, args.dataset.with_name("manifest.json"), args.reference_cache,
              args.reference_cache.with_suffix(".meta.json")]
    models = {args.model_path / name: digest for name, digest in NATIVE_FILES.items()}
    if args.onnx_path:
        models.update({args.onnx_path / name: digest for name, digest in ONNX_FILES.items()})

    def hashes(paths):
        result = {}
        for path in paths:
            with path.open("rb") as stream:
                result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result

    model_hashes = hashes(models)
    require(model_hashes == {str(path): digest for path, digest in models.items()}, "pinned_model_artifact_hashes")
    return {"source_sha256": hashes(sources), "asset_sha256": hashes(assets), "model_file_sha256": model_hashes}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "onnx", "both"), default="both")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ner-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--api-python", type=Path, default=ROOT.parent / "seif-pii/.venv/bin/python")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    parser.add_argument("--reference-cache", type=Path, default=REFERENCE)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend != "torch" and args.onnx_path is None:
        parser.error("ONNX backend requires --onnx-path")
    if not 20 < args.timeout <= 120 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("Client timeout must exceed the fixed 20-second NER deadline and be at most 120 seconds")
    for name in ("model_path", "onnx_path", "ner_python", "api_python", "dataset", "reference_cache", "output"):
        value = getattr(args, name)
        if value is not None:
            # Keep venv interpreter symlinks: resolving them bypasses pyvenv.cfg.
            value = value.expanduser()
            setattr(args, name, value.absolute() if name in {"ner_python", "api_python"} else value.resolve())
    if args.onnx_path is not None and not args.onnx_path.is_dir():
        parser.error("--onnx-path must be the installed local bundle directory")
    args.logs_dir = ROOT / "local-data/gliner25-onnx-http" / (args.output.stem + "-" + secrets.token_hex(8))
    return args


def main():
    args = parse_args()
    report = {"schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": "Closed-loop mask-only /v1/mask; 446 unique texts once per concurrency 1 and 4; fresh IDs.",
              "configuration": {"device": args.device, "cpu_threads": THREADS, "seed": SEED,
                                "schema": "described-names", "threshold": .8, "warmup": 5, "cuda_tf32": False,
                                "api_workers": 1, "ner_workers": 1, "ner_max_model_jobs": 4,
                                "ner_timeout_seconds": 20, "client_timeout_seconds": args.timeout,
                                "capture_enabled": False, "text_cache": False, "storage": "memory"},
              "limitations": ["Not an open-loop capacity, Redis, capture-enabled or production deployment measurement.",
                              "Restore checks and warmup are excluded; each timed request invokes actual NER.",
                              "Reference comparisons cover masks, ordered typed offsets and type lists, not float score equality.",
                              "Matching frozen predictions is inference parity, not perfect organizer ground truth."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            report["provenance"] = provenance(args)
            cases = prepare_cases(args)
            backends = ("torch", "onnx") if args.backend == "both" else (args.backend,)
            report["backends"] = [run_backend(args, backend, cases) for backend in backends]
            require(report["provenance"] == provenance(args), "immutable_sources_assets_models")
            report["status"] = "PASS" if all(item["status"] == "PASS" for item in report["backends"]) else "FAIL"
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
