"""Check real batched RuBERT against the complete frozen 5095-document reference.

The reference is an expectation, never a model-output cache served at inference.
Reports retain only counts, corpus case IDs, confidence differences and hashes.
This checks preservation of existing predictions, not a new held-out F1 estimate.
Run after throughput experiments finish; this command performs real GPU inference.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
evaluation = importlib.import_module("benchmarks.ner-models.rubert-upgrade.evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--model-path", type=Path, default=ROOT / "local-data/rubert-tensorrt/model")
    parser.add_argument("--reference", type=Path, default=ROOT / "local-data/rubert-upgrade/run-v2/native.jsonl")
    parser.add_argument("--public-run-dir", type=Path,
                        default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("model_path", "reference", "public_run_dir", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def provenance(args):
    from seif.rubert_ner import fingerprint_checkpoint

    source_paths = sorted((ROOT / "seif").glob("*.py")) + [
        Path(__file__), Path(evaluation.__file__), ROOT / "scripts/compare_ner_public.py",
        ROOT / "scripts/evaluate_golden.py", ROOT / "scripts/evaluate_annotations.py",
    ]
    assets = [args.reference, args.reference.with_suffix(".meta.json"),
              args.public_run_dir / "prepared-inputs.jsonl", args.public_run_dir / "protocol.json",
              ROOT / "datasets/golden/organizer-v1/cases.jsonl",
              ROOT / "datasets/golden/organizer-v1/manifest.json",
              ROOT / "local-data/rubert-tensorrt/run-v1/protocol.json"]
    return {"source_sha256": {str(path.relative_to(ROOT)): evaluation.sha(path) for path in source_paths},
            "asset_sha256": {str(path): evaluation.sha(path) for path in assets},
            "model_sha256": fingerprint_checkpoint(args.model_path)}


def native_signature(spans):
    return [(span["start"], span["end"], span["label"]) for span in spans]


def service_output(text, native):
    from seif.detector import detect, merge_ner_candidates
    from seif.ner import NerClient
    from seif.rubert_ner import gateway_entities
    from seif.transform import mask

    # Frozen cache intentionally omits matched text; recover it only in memory
    # for the same native-output validator used by the serving adapter.
    complete = [{**span, "text": text[span["start"]:span["end"]]} for span in native]
    gateway = gateway_entities(text, complete, profile="native")
    candidates = NerClient._parse_entities(None, {"entities": gateway}, text, 0, len(text))
    spans = merge_ner_candidates(text, detect(text), candidates)
    return {"masked": mask(text, spans, "mask")[0],
            "typed_spans": [(span.start, span.end, span.type) for span in spans]}


def empty_summary():
    return {"documents": 0, "native_span_matches": 0, "service_span_matches": 0, "mask_matches": 0,
            "mismatches": {"native_spans": [], "service_spans": [], "mask": []},
            "confidence_on_aligned_native_spans": {"pairs": 0, "different": 0,
                                                    "max_absolute_delta": 0.0, "sum_absolute_delta": 0.0}}


def compare_scores(actual, reference, summary):
    expected = {(span["start"], span["end"], span["label"]): span["score"] for span in reference}
    stats = summary["confidence_on_aligned_native_spans"]
    for span in actual:
        key = span["start"], span["end"], span["label"]
        if key not in expected:
            continue
        delta = abs(span["score"] - expected[key])
        stats["pairs"] += 1
        stats["different"] += delta != 0
        stats["max_absolute_delta"] = max(stats["max_absolute_delta"], delta)
        stats["sum_absolute_delta"] += delta


def compare_document(row, actual, reference, summary):
    text, case_id = row["text"], row["key"]
    previous = reference["native"]
    observed = service_output(text, actual)
    expected = service_output(text, previous)
    checks = {"native_spans": native_signature(actual) == native_signature(previous),
              "service_spans": observed["typed_spans"] == expected["typed_spans"],
              "mask": observed["masked"] == expected["masked"]}
    fields = {"native_spans": "native_span_matches", "service_spans": "service_span_matches", "mask": "mask_matches"}
    summary["documents"] += 1
    for category, matched in checks.items():
        summary[fields[category]] += matched
        if not matched:
            summary["mismatches"][category].append(case_id)
    compare_scores(actual, previous, summary)


def finish_summary(summary):
    count = summary["documents"]
    for name in ("native_span_matches", "service_span_matches", "mask_matches"):
        summary[name + "_ratio"] = summary[name] / count if count else None
    scores = summary["confidence_on_aligned_native_spans"]
    total = scores.pop("sum_absolute_delta")
    scores["mean_absolute_delta"] = total / scores["pairs"] if scores["pairs"] else None


def run_inference(args, rows, references, report):
    os.environ["SEIF_RUBERT_CPU_THREADS"] = str(args.cpu_threads)
    import torch

    from seif.rubert_ner import RubertAnalyzer

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["runtime"] = {"python": sys.version, "python_executable": sys.executable,
                         "packages": {name: importlib.metadata.version(name) for name in
                                      ("torch", "numpy", "transformers", "tokenizers", "tensorrt-cu12")}}
    model = RubertAnalyzer.from_local(args.model_path, decoder="word", profile="native", batch_size=args.batch_size)
    report["runtime"]["torch_num_threads"] = torch.get_num_threads()
    report["corpora"] = {name: empty_summary() for name in evaluation.COUNTS}
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        outputs = model.predict_native_batch([row["text"] for row in chunk])
        if not isinstance(outputs, list) or len(outputs) != len(chunk):
            raise ValueError("Incomplete quality-parity batch.")
        for row, actual in zip(chunk, outputs, strict=True):
            compare_document(row, actual, references[row["key"]], report["corpora"][row["dataset"]])
        if start // 1000 != (start + len(chunk)) // 1000:
            print(json.dumps({"completed": start + len(chunk), "total": len(rows)}), flush=True)
    report["elapsed_quality_seconds_not_rps"] = time.perf_counter() - started
    report["analyzer"] = model.metadata()
    for summary in report["corpora"].values():
        finish_summary(summary)
    report["all_documents_compared"] = (
        {name: summary["documents"] for name, summary in report["corpora"].items()} == evaluation.COUNTS
    )
    report["exact_native_span_parity"] = all(
        summary["native_span_matches"] == summary["documents"] for summary in report["corpora"].values()
    )
    report["exact_service_span_parity"] = all(
        summary["service_span_matches"] == summary["documents"] for summary in report["corpora"].values()
    )
    report["exact_mask_parity"] = all(
        summary["mask_matches"] == summary["documents"] for summary in report["corpora"].values()
    )


def main():
    args = parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    report = {"schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": {"batch_size": args.batch_size, "decoder": "word", "profile": "native",
                                "max_documents_per_model_batch": args.batch_size, "cpu_threads": args.cpu_threads,
                                "reference": str(args.reference), "fresh_inference": True, "text_cache": False},
              "limitations": ["Exact preservation check against frozen model predictions, not perfect ground truth.",
                              "All 5095 previously inspected development texts; not independent held-out evidence.",
                              "Native type/offset, service type/offset, and mask equality are separate checks.",
                              "Different float confidences are reported; no numerical-equality requirement.",
                              "Elapsed time includes CPU parity checks and lazy graph capture; it is not HTTP RPS.",
                              "No input text, matched values, original text hashes by case or credentials are retained."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        try:
            before = provenance(args)
            report["provenance_before"] = before
            modules = evaluation.load_modules(ROOT)
            rows, _cases, membership = evaluation.load_inputs(args, modules[0], modules[1])
            references, _metadata = evaluation.load_cache(args.reference, rows)
            report["corpus_identity"] = membership
            report["input_counts"] = dict(Counter(row["dataset"] for row in rows))
            run_inference(args, rows, references, report)
            after = provenance(args)
            report["provenance_after"] = after
            report["immutable_sources_assets_models"] = before == after
            complete = report["all_documents_compared"] and report["immutable_sources_assets_models"]
            parity = all(report[name] for name in ("exact_native_span_parity", "exact_service_span_parity",
                                                  "exact_mask_parity"))
            report["status"] = ("PASS" if parity else "MISMATCH") if complete else "FAIL"
        except Exception as error:
            # A third-party exception can contain original text. Retain only its
            # class and never print its message or traceback in a shareable report.
            report["error_type"] = type(error).__name__
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
