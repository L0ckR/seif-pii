"""Frozen RuBERT TensorRT experiment: native21 and unchanged SEIF merge profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = "protocol.json"
RUBERT_CACHE_FILE = "rubert.jsonl"
META_SUFFIX = ".meta.json"
ORGANIZER_PREFIX = "organizer/"
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts import compare_onnx as reference  # noqa: E402
from scripts import evaluate_annotations as annotations  # noqa: E402
from scripts import evaluate_golden as golden  # noqa: E402
from scripts.compare_ner_models import compact_evaluation, person_only  # noqa: E402
from seif.detector import Span  # noqa: E402
from seif.ner import _entity_values  # noqa: E402
from seif.transform import mask, restore_exact  # noqa: E402

MODEL = "lockR/rubert-base-pii-ner-tensorrt"
REVISION = "73be581047bf123dac6505e7b3900ec292942296"
PACKAGES = ("torch", "tensorrt-cu12", "transformers", "tokenizers", "numpy", "pyarrow")
SETTINGS = {
    "backend": "trt-graph",
    "device": "cuda",
    "batch_size": 1,
    "threshold": 0.3,
    "cpu_threads": 4,
    "max_length": 512,
    "stride": 128,
    "seed": 20260922,
    "precision": "FP16 with FP32 normalization accumulation",
    "tf32": False,
    "warmup": "All CUDA graph buckets 32/64/128/256/512, then 5 synthetic documents",
    "profiles": ["native21", "service_hybrid_PERSON_LOCATION", "service_person_only"],
    "name_labels": ["FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"],
    "location_labels": ["CITY", "COUNTRY", "DISTRICT", "REGION", "STREET", "HOUSE"],
    "coarse_merge": "Whitespace-adjacent equal mapped labels; max constituent confidence; no punctuation expansion",
    "policy": "No corpus tuning, no label changes; all original gold types retained in full masking",
}


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def input_hashes(args):
    files = reference.input_files(args)
    files["rubert_provenance"] = ROOT / "benchmarks/ner-models/rubert-tensorrt/provenance.json"
    return {k: golden.sha256(p) for k, p in files.items()}


def model_hashes(path):
    provenance = json.loads((ROOT / "benchmarks/ner-models/rubert-tensorrt/provenance.json").read_text())
    if provenance["repo"] != MODEL or provenance["revision"] != REVISION:
        raise ValueError("Unexpected model provenance")
    actual = {name: golden.sha256(path / name) for name in provenance["sha256"]}
    if actual != provenance["sha256"]:
        raise ValueError("Model differs from verified published revision")
    return actual


def prepare(args):
    rows = reference.load_inputs(args)
    if max(len(r["text"]) for r in rows) > 16000:
        raise ValueError("This single-call comparison requires texts within gateway character window")
    protocol = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "model_revision": REVISION,
        "settings": SETTINGS,
        "versions": versions(),
        "source_sha256": reference.source_hashes(),
        "model_sha256": model_hashes(args.model_path),
        "input_sha256": input_hashes(args),
        "cases": len(rows),
        "corpus_counts": dict(Counter(r["dataset"] for r in rows)),
        "ordered_membership_sha256": public.digest_json([r["key"] for r in rows]),
        "text_sha256": public.digest_json({r["key"]: text_hash(r["text"]) for r in rows}),
        "gold_sha256": public.digest_json({r["key"]: sorted(r["gold"]) for r in rows}),
        "maximum_characters": max(len(r["text"]) for r in rows),
        "baseline_control": reference.baseline_control(args, rows),
    }
    reference.save(args.run_dir / PROTOCOL_FILE, protocol)
    return {"prepared": True, "cases": len(rows), "baseline_control": protocol["baseline_control"]}


def verify(args, protocol):
    if protocol["source_sha256"] != reference.source_hashes() or protocol["versions"] != versions():
        raise ValueError("Experiment sources or package versions changed")
    if protocol["settings"] != SETTINGS or protocol["model_revision"] != REVISION:
        raise ValueError("Experiment settings changed")
    if protocol["input_sha256"] != input_hashes(args):
        raise ValueError("Corpus or reference files changed")
    rows = reference.load_inputs(args)
    if (
        protocol["ordered_membership_sha256"] != public.digest_json([r["key"] for r in rows])
        or protocol["text_sha256"] != public.digest_json({r["key"]: text_hash(r["text"]) for r in rows})
        or protocol["gold_sha256"] != public.digest_json({r["key"]: sorted(r["gold"]) for r in rows})
    ):
        raise ValueError("Corpus order, texts or annotations changed")
    return rows


def measured_call(analyzer, text):
    import torch

    started = time.perf_counter()
    try:
        output = analyzer.predict_both(text)
        if output.get("gateway_error"):
            raise ValueError("Gateway contract failure")
        for item in output["gateway"]:
            _entity_values(item, len(text))
        error = None
    except Exception as exc:
        output, error = {"native": [], "gateway": []}, type(exc).__name__
    torch.cuda.synchronize()
    return output, error, (time.perf_counter() - started) * 1000


def cache(args, rows, protocol):
    import torch

    from seif.rubert_ner import RubertAnalyzer

    target = args.run_dir / RUBERT_CACHE_FILE
    if target.exists() or target.with_suffix(META_SUFFIX).exists():
        raise FileExistsError("Prediction caches are immutable")
    if model_hashes(args.model_path) != protocol["model_sha256"]:
        raise ValueError("Model changed after protocol freeze")
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    analyzer = RubertAnalyzer.from_local(args.model_path)
    for _ in range(5):
        _, error, _ = measured_call(analyzer, "Иван Иванов приехал в Москву.")
        if error:
            raise RuntimeError("Synthetic warmup failed")
    by_corpus, failures, latencies = defaultdict(list), defaultdict(list), []
    started = time.perf_counter()
    with target.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            output, error, latency = measured_call(analyzer, row["text"])
            by_corpus[row["dataset"]].append(latency)
            latencies.append({"case_id": row["key"], "latency_ms": latency, "error_type": error})
            # Native text fields are redundant; keep raw text only in the frozen input corpus.
            native = [{k: v for k, v in e.items() if k != "text"} for e in output["native"]]
            record = {
                "case_id": row["key"],
                "text_sha256": text_hash(row["text"]),
                "entities": output["gateway"],
                "native": native,
            }
            if error:
                record["inference_error"] = error
                failures[row["dataset"]].append(row["key"])
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            if index % 500 == 0:
                stream.flush()
                print(json.dumps({"completed": index, "total": len(rows)}), flush=True)
    elapsed = time.perf_counter() - started
    verify(args, protocol)
    if model_hashes(args.model_path) != protocol["model_sha256"]:
        raise ValueError("Model changed during inference")
    metadata = {
        "model": MODEL,
        "model_revision": REVISION,
        "cases": len(rows),
        "settings": SETTINGS,
        "versions": versions(),
        "adapter": analyzer.metadata(),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_sha256": golden.sha256(target),
        "protocol_sha256": golden.sha256(args.run_dir / PROTOCOL_FILE),
        "hardware": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(),
            "logical_cpus": os.cpu_count(),
        },
        "timing_by_corpus": {
            k: {
                **reference.timing(v),
                "failed_cases": len(failures[k]),
                "valid_complete_corpus_measurement": not failures[k],
                "successful_documents_per_second": (len(v) - len(failures[k])) / (sum(v) / 1000),
            }
            for k, v in by_corpus.items()
        },
        "failures_by_corpus": dict(failures),
        "per_case_latency_ms": latencies,
        "elapsed_seconds_including_cache": elapsed,
        "documents_per_second_including_cache": len(rows) / elapsed,
        "timing_scope": "Warm synchronized sequential extraction + native validation + gateway mapping; batch1; not HTTP RPS",
    }
    reference.save(target.with_suffix(META_SUFFIX), metadata)
    return {"cases": len(rows), "timing": metadata["timing_by_corpus"], "failures": dict(failures)}


def load_cache(args, rows):
    target = args.run_dir / RUBERT_CACHE_FILE
    candidates, metadata = public.load_cache(target, rows, golden.sha256(args.run_dir / PROTOCOL_FILE))
    records = [json.loads(line) for line in target.read_text().splitlines()]
    if [r["case_id"] for r in records] != [r["key"] for r in rows]:
        raise ValueError("Cache order or coverage differs")
    actual = {r["case_id"] for r in records if "inference_error" in r}
    expected = {key for values in metadata["failures_by_corpus"].values() for key in values}
    if actual != expected:
        raise ValueError("Inference failure provenance differs")
    from seif.rubert_ner import gateway_entities

    native = {}
    for row, record in zip(rows, records, strict=True):
        if row["key"] in actual:
            continue
        entities = [{**e, "text": row["text"][e["start"] : e["end"]]} for e in record["native"]]
        if gateway_entities(row["text"], entities) != candidates[row["key"]]:
            raise ValueError("Native and gateway cache disagree")
        native[row["key"]] = record["native"]
    return candidates, native, metadata, actual


def organizer_scores(cases, predictions):
    inputs = {key: {"text": row["text"]} for key, row in cases.items()}
    weights = {key: row["traffic_weight"] for key, row in cases.items()}
    formatted = {}
    for key, row in cases.items():
        spans = predictions[ORGANIZER_PREFIX + key]
        masked, replacements = mask(row["text"], spans, "mask")
        if restore_exact({"masked": masked, "replacements": replacements}) != row["text"]:
            raise ValueError("Restoration failed")
        formatted[key] = {
            "case_id": key,
            "masked": masked,
            "entities": [{"type": s.type, "start": s.start, "end": s.end} for s in spans],
        }
    return compact_evaluation(annotations.evaluate(inputs, cases, formatted, weights))


def native_exact_redmad(rows, native):
    gold = {r["key"]: set(map(tuple, r["gold"])) for r in rows}
    predicted = {r["key"]: {(e["label"], e["start"], e["end"]) for e in native[r["key"]]} for r in rows}
    return public.scope_metrics(gold, {"rubert_native21": predicted})


def export_organizer(args, rows, candidates):
    target = args.run_dir / "rubert-organizer.jsonl"
    with target.open("x", encoding="utf-8") as stream:
        for row in rows:
            if row["dataset"] == "organizer":
                stream.write(
                    json.dumps(
                        {
                            "case_id": row["id"],
                            "text_sha256": text_hash(row["text"]),
                            "entities": candidates[row["key"]],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    reference.save(
        target.with_suffix(META_SUFFIX),
        {
            "model": MODEL,
            "model_revision": REVISION,
            "dataset_sha256": golden.sha256(reference.DATA),
            "cache_sha256": golden.sha256(target),
            "settings": SETTINGS,
            "parent_cache_sha256": golden.sha256(args.run_dir / RUBERT_CACHE_FILE),
            "protocol_sha256": golden.sha256(args.run_dir / PROTOCOL_FILE),
        },
    )


def evaluate(args, rows, protocol):
    candidates, native, metadata, failures = load_cache(args, rows)
    caches = reference.reference_caches(args, rows)
    caches["rubert"] = candidates
    predictions = defaultdict(dict)
    for row in rows:
        for name, spans in public.predictions_for(row["text"], {k: v[row["key"]] for k, v in caches.items()}).items():
            predictions[name][row["key"]] = spans
        if row["key"] not in failures:
            predictions["rubert_native21"][row["key"]] = [
                Span(e["start"], e["end"], e["label"], e["score"], "native21") for e in native[row["key"]]
            ]
    corpora, cases = {}, golden.load_cases(reference.DATA)
    for dataset in ("organizer", "pii", "redmadrobot"):
        part = [r for r in rows if r["dataset"] == dataset]
        failed = sorted(r["key"] for r in part if r["key"] in failures)
        if failed:
            corpora[dataset] = {"cases": len(part), "quality_available": False, "failed_case_ids": failed}
            continue
        full_mask = public.protection_metrics(part, predictions)
        result = {"cases": len(part), "full_masking_all_gold_types": full_mask}
        if dataset == "organizer":
            result["service_profiles"] = {
                name: organizer_scores(cases, spans) for name, spans in predictions.items() if name != "rubert_native21"
            }
            result["raw_person"] = person_only(cases, {k: {"entities": candidates[ORGANIZER_PREFIX + k]} for k in cases})
        else:
            service = {k: v for k, v in predictions.items() if k != "rubert_native21"}
            result["service_typed_all_types_unfiltered"] = reference.score_public(part, service, caches)[
                "typed_all_types_unfiltered"
            ]
            result["raw_model_person"] = public.raw_person_metrics(part, caches)
        if dataset == "redmadrobot":
            result["native_original21_exact_boundaries"] = native_exact_redmad(part, native)
        corpora[dataset] = result
    report = {
        "protocol_sha256": golden.sha256(args.run_dir / PROTOCOL_FILE),
        "protocol": protocol,
        "candidate_metadata": metadata,
        "corpora": corpora,
        "limitations": [
            "Organizer labels are provisional AI silver, not organizer-authoritative ground truth.",
            "Frozen public corpora were previously used for SEIF rules, so this is not a blind pipeline holdout.",
            "RuBERT is trained on redmadrobot pii_train; benchmark is same source family; training overlap not audited.",
            "Our RMR protocol preserves 2839 aligned rows and two historical exclusions; model card uses 2841 rows.",
            "Native21 is standalone masking; structured model predictions are NOT silently integrated into SEIF.",
            "No threshold tuning, no main service changes, no deployment; model doc/s is not HTTP RPS.",
        ],
    }
    verify(args, protocol)
    reference.save(args.output, report)
    if not any(key.startswith(ORGANIZER_PREFIX) for key in failures):
        export_organizer(args, rows, candidates)
    return {"output": str(args.output), "cases": len(rows), "failed": len(failures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cache", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_dir.resolve().is_relative_to(ROOT / "local-data"):
        parser.error("Raw caches belong in ignored local-data")
    if args.mode == "evaluate" and args.output is None:
        parser.error("Evaluation requires --output")
    public.disable_network()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        result = prepare(args)
    else:
        protocol = json.loads((args.run_dir / PROTOCOL_FILE).read_text())
        rows = verify(args, protocol)
        result = cache(args, rows, protocol) if args.mode == "cache" else evaluate(args, rows, protocol)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
