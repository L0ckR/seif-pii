"""Actual HTTP throughput of word-decoded RuBERT with the complete PII interface.

Reuse the established traffic driver and correctness checks; each request runs
the real model. References only verify answers and are never served as a cache.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
legacy = importlib.import_module("benchmarks.ner-models.rubert-tensorrt.http_benchmark")
common = legacy.common
MODULE = "benchmarks.ner-models.rubert-upgrade.http_benchmark"


def create_ner_app():
    import torch

    from scripts.ner_service import create_app
    from seif.rubert_ner import RubertAnalyzer

    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    counts, state = common.Counts(), {}

    def factory():
        analyzer = RubertAnalyzer.from_local(Path(os.environ["SEIF_RUBERT_HTTP_MODEL"]),
                                             decoder="word", profile="native")
        state["metadata"] = analyzer.metadata()
        state["server_environment"] = legacy.server_environment()
        wrapped = common.CountedAnalyzer(analyzer, counts)
        wrapped.supported_entities = analyzer.supported_entities
        return wrapped

    app = create_app(analyzer_factory=factory)
    app.add_middleware(common.CountedBoundary, counts=counts)

    @app.get("/benchmark-stats")
    async def benchmark_stats():
        return {"counts": counts.snapshot(), "model": state["metadata"],
                "server_environment": state["server_environment"]}

    return app


def provenance(args):
    result = legacy.provenance(args)
    result["source_sha256"][str(Path(__file__).relative_to(ROOT))] = legacy.file_digest(Path(__file__))
    return result


def main():
    args = legacy.parse_args()
    # The shared launcher resolves its subprocess factory through this module.
    legacy.MODULE = MODULE
    report = {"schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": "Repeated complete446 organizer cycles, real inference, fresh correlation IDs, no text cache",
              "configuration": {"decoder": "word", "gateway_profile": "native", "confidence_threshold": None,
                                "concurrency": args.concurrency, "repeats": args.repeats, "batch_size": 1,
                                "cpu_threads": 4, "api_workers": 1, "ner_workers": 1,
                                "ner_max_model_jobs": 4, "capture_enabled": False, "storage": "memory"},
              "limitations": ["Local closed-loop throughput, not an open-loop load ceiling or production SLA.",
                              "Only masking requests contribute to RPS; warmup/restore checks are excluded.",
                              "All21 native types mapped into14 service types; consumer policy applies afterward."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            report["provenance"] = provenance(args)
            cases = legacy.prepare_workload(args)
            report["driver_environment"] = legacy.driver_environment()
            report["service"] = legacy.run_service(args, cases)
            legacy.require(report["provenance"] == provenance(args), "immutable_sources_assets_models")
            report["status"] = report["service"]["status"]
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, legacy.smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
