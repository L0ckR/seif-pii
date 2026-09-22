"""Run pinned local GLiNER predictions on the immutable organizer golden set."""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_golden import DEFAULT_DATA, load_cases, save_json, sha256  # noqa: E402
from scripts.ner_service import infer  # noqa: E402
from seif.gliner_ner import SCHEMAS, GlinerAnalyzer, _parse_output  # noqa: E402

MODEL_ID = "fastino/gliner2.5-multi-v1"
MODEL_REVISION = "a221b77a8baf4a613b8f8652661d41fa10a5641e"
MODEL_FILES = ("config.json", "encoder_config/config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


def cache_entities(text: str, result: dict, schema="person-location") -> list[dict]:
    """Use exactly the same validated conversion as the private NER service."""
    return [{"start": span.start, "end": span.end, "entity_type": span.entity_type, "score": span.score}
            for span in _parse_output(result, text, schema)]


def fingerprint_model(path: Path) -> dict:
    result = {}
    for name in MODEL_FILES:
        with (path / name).open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def infer_corpus(model, cases: dict, args, torch) -> tuple[list[dict], dict]:
    rows, timings = [], []
    adapter = GlinerAnalyzer(model, schema=args.schema, threshold=args.threshold) if args.service_adapter else None
    kwargs = {"threshold": args.threshold, "overlap_policy": "flat",
              "include_spans": True, "include_confidence": True, "max_len": None}
    for _ in range(5):
        model.extract_entities("Иван Иванов приехал в Москву.", SCHEMAS[args.schema], **kwargs)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index, (case_id, row) in enumerate(cases.items(), 1):
        text = row["text"]
        before = time.perf_counter()
        prediction = infer(adapter, text) if adapter else model.extract_entities(text, SCHEMAS[args.schema], **kwargs)
        if args.device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - before) * 1000)
        entities = prediction["entities"] if adapter else cache_entities(text, prediction, args.schema)
        rows.append({"case_id": case_id, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "entities": entities})
        if index % 50 == 0:
            print(json.dumps({"completed": index, "total": len(cases)}), flush=True)
    elapsed = time.perf_counter() - started
    return rows, {
        "cases": len(cases), "elapsed_seconds": elapsed, "sequential_documents_per_second": len(cases) / elapsed,
        "latency_ms": {"p50": statistics.median(timings), "p95": percentile(timings, .95),
                       "p99": percentile(timings, .99), "max": max(timings), "mean": statistics.mean(timings)},
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "scope": "Sequential warm local model calls only, no HTTP, Redis, masking or concurrency; not service RPS.",
        "per_case_latency_ms": timings,
    }


def generate(args) -> dict:
    sources = [Path(__file__), ROOT / "seif/gliner_ner.py", ROOT / "scripts/ner_service.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    if not args.model_path.is_dir():
        raise ValueError("Prepare the pinned model snapshot before running inference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(".meta.json").exists():
        raise FileExistsError("Inference outputs are immutable; choose a new output path")
    # Block accidental hub/API use while organizer texts are loaded.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from gliner2 import AutoExtractor

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20260922)
    model = AutoExtractor.from_pretrained(str(args.model_path), map_location=args.device,
                                         quantize=args.fp16, compile=False, local_files_only=True)
    tokenizer = model.processor.tokenizer
    markers = [item for item in json.loads((args.model_path / "tokenizer.json").read_text())["added_tokens"]
               if item["id"] >= 250102]
    if len(tokenizer) != 250112 or any(tokenizer.convert_tokens_to_ids(item["content"]) != item["id"] for item in markers):
        raise ValueError("Loaded tokenizer differs from the pinned vocabulary")
    cases = load_cases(args.dataset)
    rows, performance = infer_corpus(model, cases, args, torch)
    if source_hashes != {str(path.relative_to(ROOT)): sha256(path) for path in sources}:
        raise RuntimeError("Inference source changed during the measurement")
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    metadata = {
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_sha256": sha256(args.output), "dataset_sha256": sha256(args.dataset),
        "model_file_sha256": fingerprint_model(args.model_path), "cases": len(cases),
        "candidates": sum(len(row["entities"]) for row in rows),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("gliner2", "torch", "transformers", "tokenizers", "protobuf", "sentencepiece")},
        "configuration": {"schema": args.schema, "labels": SCHEMAS[args.schema],
                          "threshold": args.threshold, "overlap_policy": "flat",
                          "max_len": None, "batch_size": 1, "warmup": 5, "device": args.device,
                          "dtype": str(next(model.parameters()).dtype), "compile": False,
                          "cpu_threads": args.cpu_threads, "seed": 20260922, "offline": True,
                          "inference_path": "service-adapter" if args.service_adapter else "direct-model",
                          "service_word_window": 384 if args.service_adapter else None},
        "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                     "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None},
        "performance": performance,
        "source_sha256": source_hashes,
        "contract": {"candidates_over_200_characters": sum(e["end"] - e["start"] > 200 for row in rows for e in row["entities"])},
        "purpose": "Frozen predictions for a separate model comparison; never annotations; local inference only.",
    }
    save_json(args.output.with_suffix(".meta.json"), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--schema", choices=tuple(SCHEMAS), default="person-location")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--service-adapter", action="store_true")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.cpu_threads < 1 or (args.fp16 and args.device != "cuda"):
        parser.error("Invalid threshold, CPU threads or precision/device combination")
    metadata = generate(args)
    print(json.dumps({"output": str(args.output), "cases": metadata["cases"], "candidates": metadata["candidates"],
                      "performance": {k: v for k, v in metadata["performance"].items() if k != "per_case_latency_ms"}}, indent=2))


if __name__ == "__main__":
    main()
