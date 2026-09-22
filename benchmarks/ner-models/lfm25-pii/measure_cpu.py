"""Measure warm, sequential LFM CPU inference on the 446 organizer documents.

Both native decoders share one neural call. The cache matches the GPU runner's
record schema and contains only input digests, offsets and native entity types.
Run after the GPU experiment, with no competing inference or load test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

THREADS = 4
SEED = 20260922
WARMUP = 5
EXPECTED_CASES = 446
DECODERS = ("raw", "hybrid")
SOURCE_FILES = (
    Path(__file__).relative_to(ROOT),
    Path("scripts/evaluate_golden.py"),
    Path("scripts/evaluate_annotations.py"),
)


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_hashes():
    files = sorted((ROOT / "seif").glob("*.py")) + [ROOT / name for name in SOURCE_FILES]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in files}


def dataset_hashes(path):
    return {"cases": file_digest(path), "manifest": file_digest(path.with_name("manifest.json"))}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Installed, pinned local checkpoint.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    parser.add_argument("--cache", type=Path, required=True, help="New JSONL file inside this worktree's local-data/.")
    parser.add_argument("--report", type=Path, required=True, help="New JSON report, safe to track in Git.")
    args = parser.parse_args()
    for name in ("model_path", "dataset", "cache", "report"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.cache.is_relative_to(ROOT / "local-data") or args.cache.suffix != ".jsonl":
        parser.error("--cache must be a JSONL file inside this worktree's ignored local-data/ directory")
    if args.report.suffix != ".json" or args.report == args.cache:
        parser.error("--report must be a separate JSON file")
    for path in (args.cache, args.report):
        if path.exists():
            parser.error("Output files are immutable; choose new cache and report paths")
    return args


def configure_cpu():
    # Set these before importing Torch/tokenizers; loading remains strictly local.
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS), "OPENBLAS_NUM_THREADS": str(THREADS),
    })
    import torch

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    return torch


def validate_output(output, text, native_types):
    if not isinstance(output, dict) or set(output) != set(DECODERS):
        raise ValueError("LFM output must contain exactly raw and hybrid native decoders")
    for spans in output.values():
        if not isinstance(spans, list):
            raise ValueError("LFM decoder spans must be lists")
        for span in spans:
            if (not isinstance(span, dict) or set(span) != {"start", "end", "type"}
                    or type(span["start"]) is not int or type(span["end"]) is not int
                    or not 0 <= span["start"] < span["end"] <= len(text)
                    or not isinstance(span["type"], str) or span["type"] not in native_types):
                raise ValueError("Invalid native span; cache output rejected")


def measure(analyzer, cases, stream, native_types):
    timings, decoder_rows = [], {name: [] for name in DECODERS}
    started = time.perf_counter()
    for index, (case_id, case) in enumerate(cases.items(), 1):
        before = time.perf_counter()
        output = analyzer.predict_both(case["text"])
        latency_ms = (time.perf_counter() - before) * 1000
        validate_output(output, case["text"], native_types)
        key = "organizer/" + case_id
        record = {
            "case_id": key, "text_sha256": hashlib.sha256(case["text"].encode("utf-8")).hexdigest(),
            "decoders": output,
        }
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        timings.append({"case_id": key, "latency_ms": latency_ms})
        for name in DECODERS:
            decoder_rows[name].append({"case_id": key, "spans": output[name]})
        if index % 100 == 0:
            stream.flush()
            print(json.dumps({"completed": index, "total": len(cases)}), flush=True)
    stream.flush()
    return timings, time.perf_counter() - started, {name: json_digest(rows) for name, rows in decoder_rows.items()}


def summarize(timings, elapsed):
    values = [item["latency_ms"] for item in timings]
    ordered = sorted(values)
    call_seconds = sum(values) / 1000
    return {
        "cases": len(values), "summed_call_seconds": call_seconds,
        "sequential_documents_per_second": len(values) / call_seconds,
        "elapsed_seconds_including_cache": elapsed,
        "documents_per_second_including_cache": len(values) / elapsed,
        "p50_ms": statistics.median(values), "p95_ms": ordered[math.ceil(len(values) * .95) - 1],
        "p99_ms": ordered[math.ceil(len(values) * .99) - 1], "max_ms": max(values),
        "mean_ms": statistics.mean(values), "per_case_latency_ms": timings,
        "scope": "Sequential warm predict_both CPU calls, including both native decoders; "
                 "excludes model load, warmup and cache serialization. This is not HTTP service RPS.",
    }


def run(args):
    from scripts.evaluate_golden import load_cases
    from seif.lfm_pii import MAX_TOKENS, NATIVE_TYPES, LfmPiiAnalyzer, fingerprint_checkpoint

    sources, assets = source_hashes(), dataset_hashes(args.dataset)
    cases = load_cases(args.dataset)
    if len(cases) != EXPECTED_CASES:
        raise ValueError("CPU measurement requires all 446 organizer cases in original order")
    torch = configure_cpu()
    for path in (args.cache, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents overwriting prior runs, even if inference fails.
    with ExitStack() as stack:
        stream = stack.enter_context(args.cache.open("x", encoding="utf-8"))
        report_stream = stack.enter_context(args.report.open("x", encoding="utf-8"))
        analyzer = LfmPiiAnalyzer.from_local(args.model_path, device="cpu")
        model = analyzer.metadata()
        if model["device"] != "cpu" or model["dtype"] != "torch.float32":
            raise ValueError("CPU measurement requires the unchanged FP32 model")
        lengths = [analyzer.token_count(case["text"]) for case in cases.values()]
        if max(lengths) > MAX_TOKENS:
            raise ValueError("Organizer inputs exceed the no-truncation token budget")
        for _ in range(WARMUP):
            analyzer.predict_both("Иван Иванов приехал в Москву.")
        measured_at = datetime.now(timezone.utc).isoformat()
        timings, elapsed, predictions = measure(analyzer, cases, stream, NATIVE_TYPES)
        if (sources != source_hashes() or assets != dataset_hashes(args.dataset)
                or model["file_sha256"] != fingerprint_checkpoint(args.model_path)
                or model != analyzer.metadata()):
            raise ValueError("Sources, dataset or model changed during measurement; report rejected")
        report = {
            "schema_version": 1, "measured_at_utc": measured_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sources, "dataset_sha256": assets,
            "ordered_membership_sha256": json_digest(["organizer/" + key for key in cases]),
            "model": model, "cache_sha256": file_digest(args.cache),
            "decoder_prediction_sha256": predictions,
            "configuration": {
                "device": "cpu", "batch_size": 1, "cpu_threads": torch.get_num_threads(),
                "cpu_interop_threads": torch.get_num_interop_threads(), "seed": SEED,
                "warmup": WARMUP, "offline": True, "threshold_tuning": False,
                "maximum_tokens": MAX_TOKENS, "maximum_observed_tokens": max(lengths),
                "tokenizers_parallelism": False,
            },
            "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                         "logical_cpu_count": os.cpu_count(), "python": platform.python_version()},
            "timing": summarize(timings, elapsed),
            "limitations": [
                "One sequential pass; concurrent hardware load affects timing.",
                "Compare native raw and hybrid spans with the GPU cache before claiming quality parity.",
                "No HTTP, masking, Redis, batching or service throughput is measured.",
                "Reserved empty report or incomplete cache indicates a failed run, not a valid measurement.",
            ],
        }
        json.dump(report, report_stream, ensure_ascii=False, indent=2, allow_nan=False)
        report_stream.write("\n")
    return report


def main():
    report = run(parse_args())
    print(json.dumps({key: value for key, value in report["timing"].items() if key != "per_case_latency_ms"}, indent=2))


if __name__ == "__main__":
    main()
