"""Measure actual RuBERT TensorRT masking through an isolated local service.

Every phase submits all 446 organizer texts once, with fresh correlation IDs.
Frozen predictions are correctness expectations only, never an inference cache
served to the API. Reuses the established HTTP transport, counters and phase
checks, including idle socket renewal without retrying timed POST requests.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import math
import os
import secrets
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
common = importlib.import_module("benchmarks.ner-models.gliner25-onnx.http_benchmark")
smoke, require = common.smoke, common.require
MODULE = "benchmarks.ner-models.rubert-tensorrt.http_benchmark"
MODEL_ID = "lockR/rubert-base-pii-ner-tensorrt"
MODEL_REVISION = "73be581047bf123dac6505e7b3900ec292942296"
BACKEND = "trt-graph"
THREADS = 4
SEED = 20260922


def create_ner_app():
    """Private native subprocess factory; the published engine is required."""
    import torch

    from scripts.ner_service import create_app
    from seif.rubert_ner import RubertAnalyzer

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    counts, state = common.Counts(), {}

    def factory():
        analyzer = RubertAnalyzer.from_local(Path(os.environ["SEIF_RUBERT_HTTP_MODEL"]), device="cuda", warmup=True)
        state["metadata"] = analyzer.metadata()
        return common.CountedAnalyzer(analyzer, counts)

    app = create_app(analyzer_factory=factory)
    app.add_middleware(common.CountedBoundary, counts=counts)

    @app.get("/benchmark-stats")
    async def benchmark_stats():
        # Guarded by the existing private NER authentication boundary.
        return {"counts": counts.snapshot(), "model": state["metadata"]}

    return app


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_cases(args):
    from scripts.evaluate_golden import load_cases, load_ner_cache
    from seif.detector import detect, merge_ner_candidates
    from seif.ner import NerClient
    from seif.transform import mask

    cases = load_cases(args.dataset)
    require(len(cases) == 446 and len({row["text"] for row in cases.values()}) == 446,
            "complete_unique_organizer_corpus")
    metadata = json.loads(args.reference_cache.with_suffix(".meta.json").read_text())
    require(metadata.get("model") == MODEL_ID and metadata.get("model_revision") == MODEL_REVISION
            and metadata.get("dataset_sha256") == file_digest(args.dataset), "frozen_rubert_reference_identity")
    cache = load_ner_cache(args.reference_cache, cases)
    prepared = []
    for key, row in cases.items():
        text = row["text"]
        require(0 < len(text) <= 16000, "exactly_one_private_ner_request_per_document")
        entities = cache[key]["entities"]
        # Use the exact HTTP contract validator without constructing a client.
        candidates = NerClient._parse_entities(None, {"entities": entities}, text, 0, len(text))
        spans = merge_ner_candidates(text, detect(text), candidates)
        expected, _ = mask(text, spans, "mask")
        prepared.append({"case_id": key, "text": text, "expected": expected,
                         "entities": common.canonical_entities([asdict(span) for span in spans], len(text)),
                         "types": sorted({span.type for span in spans})})
    return prepared


def launch(args, temporary, children, logs):
    env = smoke.clean_environment()
    env.update(OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS))
    token, api_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    policy = temporary / "policy.json"
    policy.write_text(json.dumps({"systems": {"benchmark": {"api_key_env": "SEIF_RUBERT_HTTP_KEY",
                                       "mode": "mask", "allow_unmask": True, "rps": 100000}}}))
    ner_env = {**env, "SEIF_NER_TOKEN": token, "SEIF_NER_DEMO": "0", "SEIF_NER_MAX_MODEL_JOBS": "4",
               "SEIF_RUBERT_HTTP_MODEL": str(args.model_path)}
    process, ner_url = smoke.start_service((args.ner_python, MODULE + ":create_ner_app", "ner"),
                                           ROOT, ner_env, logs[0], children)
    ner_health = smoke.await_ready(process, ner_url, args.startup_timeout)
    require(ner_health.get("model") == MODEL_ID, "actual_rubert_model_health")
    api_env = {**env, "SEIF_DEMO": "0", "SEIF_CONFIG": str(policy), "SEIF_RUBERT_HTTP_KEY": api_key,
               "SEIF_MASTER_KEY": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
               "SEIF_NER_URL": ner_url, "SEIF_NER_TOKEN": token, "SEIF_NER_TIMEOUT_SECONDS": "20",
               "SEIF_CPU_WORKERS": "1", "SEIF_REQUIRE_FREE_THREADING": "1", "SEIF_LOG_LEVEL": "WARNING"}
    process, api_url = smoke.start_service((args.api_python, "seif.app:create_app", "api"),
                                           ROOT, api_env, logs[1], children)
    api_health = smoke.await_ready(process, api_url, args.startup_timeout)
    require(api_health.get("storage") == "memory" and api_health.get("detector_profile") == "hybrid"
            and api_health.get("mode") == "restricted" and api_health.get("gil_enabled") is False,
            "isolated_restricted_free_threaded_hybrid_api")
    return (common.LocalClient(api_url, {"X-System-ID": "benchmark", "X-API-Key": api_key}, args.timeout),
            common.LocalClient(ner_url, {"Authorization": "Bearer " + token}, args.timeout),
            {"api": api_health, "ner": ner_health})


def run_phases(args, api, ner, cases, report):
    for _ in range(5):
        status, _body = api.call("/v1/mask", {"payload": "Иван Иванов приехал в Москву.",
                                             "payload_id": "warmup-" + secrets.token_hex(12)})
        require(status == 200, "synthetic_api_warmup_success")
    report["model"] = common.ner_snapshot(ner)["model"]
    report["phases"] = []
    for concurrency in args.concurrency:
        print(json.dumps({"backend": BACKEND, "concurrency": concurrency,
                          "phase": "started", "mask_requests": len(cases)}), flush=True)
        phase = common.run_phase(api, ner, cases, concurrency, phase_reports=report["phases"])
        print(json.dumps({"backend": BACKEND, "concurrency": concurrency, "phase": "completed",
                          "successful_mask_rps": phase["successful_mask_rps"],
                          "reference_match_ratio": phase["reference_match_ratio"],
                          "every_mask_completed_real_ner": phase["every_mask_completed_real_ner"]}), flush=True)
        if not phase["ner_drained"]:
            return False
    return True


def run_service(args, cases):
    children, clients = [], []
    report = {"backend": BACKEND, "status": "FAIL"}
    with tempfile.TemporaryDirectory(prefix="seif-rubert-http-benchmark-") as folder:
        temporary = Path(folder)
        with (temporary / "ner.log").open("wb") as ner_log, (temporary / "api.log").open("wb") as api_log:
            try:
                api, ner, report["health"] = launch(args, temporary, children, (ner_log, api_log))
                clients.extend((api, ner))
                if not run_phases(args, api, ner, cases, report):
                    return report
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
                    report["failure_logs"] = common.preserve_failure_logs(args.logs_dir, temporary)
    return report


def provenance(args):
    from seif.rubert_ner import fingerprint_checkpoint

    sources = sorted((ROOT / "seif").glob("*.py")) + [Path(__file__), Path(common.__file__), common.SMOKE_PATH,
                ROOT / "scripts/ner_service.py", ROOT / "scripts/evaluate_golden.py",
                ROOT / "scripts/evaluate_annotations.py"]
    assets = [args.dataset, args.dataset.with_name("manifest.json"), args.reference_cache,
              args.reference_cache.with_suffix(".meta.json")]
    return {"source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in sources},
            "asset_sha256": {str(path): file_digest(path) for path in assets},
            "model_file_sha256": fingerprint_checkpoint(args.model_path)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True,
                        help="RuBERT organizer-only gateway predictions plus matching .meta.json.")
    parser.add_argument("--ner-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--api-python", type=Path, default=ROOT.parent / "seif-pii/.venv/bin/python")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    parser.add_argument("--concurrency", type=int, nargs="+", choices=(1, 4, 8, 16), default=[1, 4, 8, 16])
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency != sorted(set(args.concurrency)):
        parser.error("Concurrency phases must be unique and increasing")
    if not 20 < args.timeout <= 120 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("Client timeout must exceed the fixed 20-second NER deadline and be at most 120 seconds")
    for name in ("model_path", "ner_python", "api_python", "dataset", "reference_cache", "output"):
        path = getattr(args, name).expanduser()
        # Preserve venv paths instead of resolving symlinks to the base Python.
        setattr(args, name, path.absolute() if name in {"ner_python", "api_python"} else path.resolve())
    args.logs_dir = ROOT / "local-data/rubert-http" / (args.output.stem + "-" + secrets.token_hex(8))
    return args


def main():
    args = parse_args()
    report = {
        "schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Closed-loop mask-only /v1/mask; all 446 organizer texts per phase with fresh IDs.",
        "configuration": {"model": MODEL_ID, "model_revision": MODEL_REVISION, "backend": BACKEND,
                          "device": "cuda", "dtype": "engine FP16 with FP32 normalization accumulation",
                          "cpu_threads": THREADS, "seed": SEED, "min_confidence": .3, "batch_size": 1,
                          "max_tokens_per_window": 512, "window_overlap_tokens": 128,
                          "startup_graph_buckets_warmed": [32, 64, 128, 256, 512], "synthetic_api_warmup": 5,
                          "concurrency": args.concurrency, "api_workers": 1, "ner_workers": 1,
                          "ner_max_model_jobs": 4, "ner_timeout_seconds": 20,
                          "client_timeout_seconds": args.timeout, "capture_enabled": False,
                          "text_cache": False, "storage": "memory"},
        "limitations": [
            "Bounded closed-loop throughput; not an open-loop capacity, Redis or production SLA measurement.",
            "A single model worker serializes GPU calls; client concurrency measures queueing and HTTP overlap.",
            "All graph warmup, five synthetic API requests and sampled exact restores are excluded from mask RPS.",
            "Native labels map to gateway PERSON/LOCATION using the fixed adapter; other labels do not enter this NER path.",
            "Reference parity checks actual masks and typed offsets, not perfect organizer ground truth or float equality.",
            "No original text, predicted value, secret or correlation ID is retained in the JSON report.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            report["provenance"] = provenance(args)
            cases = prepare_cases(args)
            report["service"] = run_service(args, cases)
            require(report["provenance"] == provenance(args), "immutable_sources_assets_models")
            report["status"] = report["service"]["status"]
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
