"""Refresh and time the existing spaCy NER on the same organizer inputs."""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_golden import DEFAULT_DATA, load_cases, save_json, sha256  # noqa: E402
from scripts.ner_service import build_analyzer, infer  # noqa: E402


def main():
    output = Path(__file__).with_name("spacy-fresh.jsonl")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise FileExistsError("Reference measurement is immutable")
    os.environ["SEIF_NER_BACKEND"] = "presidio"
    cases = load_cases(DEFAULT_DATA)
    analyzer = build_analyzer()
    for _ in range(5):
        infer(analyzer, "Иван Иванов приехал в Москву.")
    rows, timings = [], []
    started = time.perf_counter()
    for case_id, case in cases.items():
        text = case["text"]
        before = time.perf_counter()
        result = infer(analyzer, text)
        timings.append((time.perf_counter() - before) * 1000)
        rows.append({"case_id": case_id, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), **result})
    elapsed = time.perf_counter() - started
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    ordered = sorted(timings)
    metadata = {
        "model": "ru_core_news_sm", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_sha256": sha256(output), "dataset_sha256": sha256(DEFAULT_DATA), "cases": len(cases),
        "versions": {name: importlib.metadata.version(name) for name in ("spacy", "ru-core-news-sm", "presidio-analyzer")},
        "configuration": {"entities": ["PERSON", "LOCATION"], "device": "cpu", "batch_size": 1,
                          "blas_threads": 1, "warmup": 5, "python": platform.python_version()},
        "performance": {"elapsed_seconds": elapsed, "sequential_documents_per_second": len(cases) / elapsed,
                        "latency_ms": {"p50": statistics.median(timings), "p95": ordered[math.ceil(len(cases) * .95) - 1],
                                       "p99": ordered[math.ceil(len(cases) * .99) - 1], "max": max(timings)},
                        "scope": "Sequential warm local model calls only; not HTTP service RPS.",
                        "per_case_latency_ms": timings},
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                          "scripts/ner_service.py": sha256(ROOT / "scripts/ner_service.py")},
    }
    save_json(output.with_suffix(".meta.json"), metadata)
    print(json.dumps({k: v for k, v in metadata["performance"].items() if k != "per_case_latency_ms"}, indent=2))


if __name__ == "__main__":
    main()
