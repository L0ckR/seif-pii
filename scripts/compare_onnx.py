"""Measure a frozen GLiNER ONNX export against the existing native predictions.

Text copies and prediction caches stay in ignored local-data. Preparation fixes
all 5095 cases before inference; no model, schema, threshold or label tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts import evaluate_golden as golden  # noqa: E402
from scripts.cache_gliner import fingerprint_model, percentile  # noqa: E402
from scripts.compare_ner_models import compact_evaluation, person_only  # noqa: E402
from seif.gliner_ner import SCHEMAS, GlinerAnalyzer  # noqa: E402

ONNX_REPO = "DanKau/gliner2.5-multi-v1-onnx"
ONNX_REVISION = "481ad683a5420349c20f7ccc992efd25f9f3b809"
ONNX_FILES = (
    ".export_meta.json",
    "config.json",
    "gliner2_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/encoder.onnx",
    "onnx/boundary.onnx",
    "onnx/classifier.onnx",
)
BASE = ROOT / "benchmarks/ner-models/gliner25-multi-v1"
DATA = ROOT / "datasets/golden/organizer-v1/cases.jsonl"
VERSIONS = ("gliner2", "torch", "onnxruntime-gpu", "onnx", "transformers", "tokenizers", "numpy")
SETTINGS = {
    "schema": "described-names",
    "labels": SCHEMAS["described-names"],
    "threshold": 0.8,
    "precision": "FP32",
    "batch_size": 1,
    "cpu_threads": 4,
    "seed": 20260922,
    "warmup": 5,
    "word_window": 384,
    "word_overlap": 64,
    "overlap_policy": "native flat weighted interval scheduling",
    "tf32": False,
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint_onnx(path):
    return {name: golden.sha256(path / name) for name in ONNX_FILES}


def source_hashes():
    paths = [*sorted((ROOT / "seif").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]
    return {str(p.relative_to(ROOT)): golden.sha256(p) for p in paths}


def versions():
    return {name: importlib.metadata.version(name) for name in VERSIONS}


def input_files(args):
    names = {
        "organizer": DATA,
        "organizer_manifest": DATA.with_name("manifest.json"),
        "organizer_native": BASE / "selected-service.jsonl",
        "organizer_spacy": BASE / "spacy-fresh.jsonl",
        "organizer_native_metadata": BASE / "selected-service.meta.json",
        "organizer_spacy_metadata": BASE / "spacy-fresh.meta.json",
        "public_report": BASE / "public-transfer.json",
        "organizer_report": BASE / "comparison-selected-service.json",
    }
    names.update(
        {
            "public_" + name: args.public_run_dir / name
            for name in (
                "prepared-inputs.jsonl",
                "protocol.json",
                "spacy.jsonl",
                "spacy.meta.json",
                "gliner.jsonl",
                "gliner.meta.json",
            )
        }
    )
    return names


def load_inputs(args):
    cases = golden.load_cases(DATA)
    if len(cases) != 446:
        raise ValueError("Expected the unchanged 446 organizer cases")
    rows = [
        {
            "key": "organizer/" + key,
            "id": key,
            "dataset": "organizer",
            "split": "all",
            "text": case["text"],
            "gold": [(e["type"], e["start"], e["end"]) for e in case["entities"]],
        }
        for key, case in cases.items()
    ]
    protocol = json.loads((args.public_run_dir / "protocol.json").read_text())
    if (
        golden.sha256(args.public_run_dir / "protocol.json")
        != "6058374163fc3978ec4a882d19328e96fad1032d61bbac168ad8a3c78df68a27"
    ):
        raise ValueError("Public reference protocol differs")
    for name, checksum in protocol["source_sha256"].items():
        if golden.sha256(ROOT / name) != checksum:
            raise ValueError("Historical rules/evaluation source differs: " + name)
    rows.extend(public.load_prepared_inputs(args.public_run_dir / "prepared-inputs.jsonl", protocol))
    if len(rows) != 5095:
        raise ValueError("Expected the unchanged complete 5095-case experiment")
    return rows


def prepare(args):
    rows = load_inputs(args)
    selected = json.loads((BASE / "selected-service.meta.json").read_text())
    if fingerprint_model(args.native_path) != selected["model_file_sha256"]:
        raise ValueError("Native checkpoint differs from the selected reference")
    # Includes new adapter/runner while requiring every historical source above.
    protocol = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes(),
        "input_sha256": {k: golden.sha256(p) for k, p in input_files(args).items()},
        "model": ONNX_REPO,
        "revision": ONNX_REVISION,
        "onnx_file_sha256": fingerprint_onnx(args.model_path),
        "native_file_sha256": selected["model_file_sha256"],
        "settings": SETTINGS,
        "versions": versions(),
        "cases": len(rows),
        "corpus_counts": dict(Counter(r["dataset"] for r in rows)),
        "ordered_membership_sha256": public.digest_json([r["key"] for r in rows]),
        "text_sha256": public.digest_json({r["key"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in rows}),
        "gold_sha256": public.digest_json({r["key"]: sorted(r["gold"]) for r in rows}),
        "policy": "Full fixed corpora, selected native schema/threshold, no post-evaluation tuning. "
        "ONNX CPU/GPU and native CPU/GPU timed separately. Local inference only; no HTTP RPS claim.",
    }
    protocol["baseline_control"] = baseline_control(args, rows)
    save(args.run_dir / "protocol.json", protocol)
    return protocol


def verify(args, protocol):
    if protocol["source_sha256"] != source_hashes() or protocol["versions"] != versions():
        raise ValueError("Sources or dependencies changed after protocol freeze")
    if protocol["input_sha256"] != {k: golden.sha256(p) for k, p in input_files(args).items()}:
        raise ValueError("Corpus or frozen reference cache changed")
    rows = load_inputs(args)
    if (
        protocol["ordered_membership_sha256"] != public.digest_json([r["key"] for r in rows])
        or protocol["gold_sha256"] != public.digest_json({r["key"]: sorted(r["gold"]) for r in rows})
        or protocol["text_sha256"]
        != public.digest_json({r["key"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in rows})
    ):
        raise ValueError("Corpus order or annotations differ")
    return rows


def timing(values):
    total = sum(values) / 1000
    return {
        "cases": len(values),
        "summed_call_seconds": total,
        "documents_per_second": len(values) / total,
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
        "mean_ms": statistics.mean(values),
    }


def create_analyzer(args):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.backend == "native":
        return GlinerAnalyzer.from_local(args.native_path, device=args.device, schema="described-names", threshold=0.8)
    from seif.gliner_onnx import GlinerOnnxAnalyzer

    return GlinerOnnxAnalyzer.from_local(args.model_path, args.native_path, device=args.device, cpu_threads=4)


def measured_call(analyzer, text, device):
    import torch

    started = time.perf_counter()
    error = None
    try:
        entities = public.infer_document(analyzer, text)
    except Exception as exc:
        # Never score an unavailable corpus as a successful empty prediction.
        entities, error = [], type(exc).__name__
    if device == "cuda":
        torch.cuda.synchronize()
    return entities, error, (time.perf_counter() - started) * 1000


def cache(args, rows, protocol):
    import torch

    if (
        fingerprint_onnx(args.model_path) != protocol["onnx_file_sha256"]
        or fingerprint_model(args.native_path) != protocol["native_file_sha256"]
    ):
        raise ValueError("Checkpoint changed after preparation")
    target = args.run_dir / f"{args.backend}-{args.device}-{args.scope}.jsonl"
    if target.exists() or target.with_suffix(".meta.json").exists():
        raise FileExistsError("Refusing to overwrite measured output")
    selected = [r for r in rows if args.scope == "all" or r["dataset"] == "organizer"]
    analyzer = create_analyzer(args)
    for _ in range(5):
        public.infer_document(analyzer, "Иван Иванов приехал в Москву.")
    if args.device == "cuda":
        torch.cuda.synchronize()
    by_corpus, per_case, failures = defaultdict(list), [], defaultdict(list)
    started = time.perf_counter()
    with target.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(selected, 1):
            entities, error, ms = measured_call(analyzer, row["text"], args.device)
            by_corpus[row["dataset"]].append(ms)
            per_case.append({"case_id": row["key"], "latency_ms": ms, "error_type": error})
            record = {
                "case_id": row["key"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "entities": entities,
            }
            if error:
                record["inference_error"] = error
                failures[row["dataset"]].append({"case_id": row["key"], "error_type": error})
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            if index % 100 == 0:
                stream.flush()
                print(
                    json.dumps(
                        {"completed": index, "total": len(selected), "backend": args.backend, "device": args.device}
                    ),
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    verify(args, protocol)
    if (
        fingerprint_onnx(args.model_path) != protocol["onnx_file_sha256"]
        or fingerprint_model(args.native_path) != protocol["native_file_sha256"]
    ):
        raise ValueError("Weights changed during inference")
    metadata = {
        "schema_version": 1,
        "backend": args.backend,
        "device": args.device,
        "scope": args.scope,
        "cases": len(selected),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_sha256": golden.sha256(target),
        "protocol_sha256": golden.sha256(args.run_dir / "protocol.json"),
        "settings": SETTINGS,
        "versions": versions(),
        "source_sha256": protocol["source_sha256"],
        "hardware": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "timing_by_corpus": {
            k: {
                **timing(v),
                "failed_cases": len(failures[k]),
                "valid_complete_corpus_measurement": not failures[k],
                "successful_documents_per_second": (len(v) - len(failures[k])) / (sum(v) / 1000),
            }
            for k, v in by_corpus.items()
        },
        "failures_by_corpus": dict(failures),
        "elapsed_seconds_including_cache": elapsed,
        "documents_per_second_including_cache": len(selected) / elapsed,
        "per_case_latency_ms": per_case,
        "timing_scope": "Warm sequential full analyzer calls, batch1, no HTTP/Redis/masking/concurrency; not service RPS.",
    }
    if hasattr(analyzer, "metadata"):
        metadata["adapter"] = analyzer.metadata()
    save(target.with_suffix(".meta.json"), metadata)
    return {k: metadata[k] for k in ("backend", "device", "cases", "timing_by_corpus")}


def reference_caches(args, rows):
    cases = golden.load_cases(DATA)
    native = {
        "organizer/" + k: v["entities"]
        for k, v in golden.load_ner_cache(BASE / "selected-service.jsonl", cases).items()
    }
    spacy = {
        "organizer/" + k: v["entities"] for k, v in golden.load_ner_cache(BASE / "spacy-fresh.jsonl", cases).items()
    }
    external = [r for r in rows if r["dataset"] != "organizer"]
    digest = golden.sha256(args.public_run_dir / "protocol.json")
    for name, result in (("gliner", native), ("spacy", spacy)):
        predictions, _ = public.load_cache(args.public_run_dir / f"{name}.jsonl", external, digest)
        result.update(predictions)
    return {"native": native, "spacy": spacy}


def parity(rows, reference, candidate):
    differences, common, scores = [], 0, []
    for row in rows:
        key = row["key"]
        before = {(x["start"], x["end"], x["entity_type"]): x["score"] for x in reference[key]}
        after = {(x["start"], x["end"], x["entity_type"]): x["score"] for x in candidate[key]}
        if before.keys() != after.keys():
            differences.append(
                {
                    "case_id": key,
                    "missing": len(before.keys() - after.keys()),
                    "extra": len(after.keys() - before.keys()),
                }
            )
        common += len(before.keys() & after.keys())
        scores.extend(abs(before[k] - after[k]) for k in before.keys() & after.keys())
    return {
        "cases": len(rows),
        "identical_span_cases": len(rows) - len(differences),
        "changed_cases": differences,
        "matched_spans": common,
        "maximum_score_difference_on_matched_spans": max(scores, default=0),
        "mean_score_difference_on_matched_spans": statistics.mean(scores) if scores else 0,
    }


def score_public(rows, predictions, caches):
    redmad = rows[0]["dataset"] == "redmadrobot"
    mapping = {**public.redmad.SEIF_MAP, "LOCATION": "LOCATION"} if redmad else public.pii.SEIF_MAP
    truth, mapped = {}, {name: {} for name in predictions}
    for row in rows:
        key, text = row["key"], row["text"]
        gold = public.redmad.coarsen(row["gold"], public.redmad.GOLD_MAP, keep_unknown=True) if redmad else row["gold"]
        truth[key] = public.redmad.merge_adjacent(text, gold) if redmad else gold
        for name, values in predictions.items():
            spans = {(mapping.get(s.type, "UNMAPPED_PRED:" + s.type), s.start, s.end) for s in values[key]}
            mapped[name][key] = public.redmad.merge_adjacent(text, spans) if redmad else spans
    return {
        "cases": len(rows),
        "full_masking_all_gold_types": public.protection_metrics(rows, predictions),
        "typed_all_types_unfiltered": public.scope_metrics(truth, mapped),
        "raw_model_person": public.raw_person_metrics(rows, caches),
    }


def score_organizer(cases, cache):
    from scripts.evaluate_annotations import evaluate

    formatted = {key: {"entities": cache["organizer/" + key]} for key in cases}
    predicted = golden.predict(cases, profile="hybrid", cache=formatted)
    inputs = {key: {"text": row["text"]} for key, row in cases.items()}
    weights = {key: row["traffic_weight"] for key, row in cases.items()}
    return compact_evaluation(evaluate(inputs, cases, predicted, weights))


def baseline_control(args, rows):
    caches, cases = reference_caches(args, rows), golden.load_cases(DATA)
    expected = json.loads((BASE / "comparison-selected-service.json").read_text())
    for name, slot in (("native", "candidate"), ("spacy", "reference")):
        if score_organizer(cases, caches[name]) != expected["hybrid"][slot]:
            raise ValueError("Organizer baseline aggregates do not reproduce: " + name)
    expected_public = json.loads((BASE / "public-transfer.json").read_text())
    selected = [r for r in rows if r["dataset"] != "organizer"]
    predictions = defaultdict(dict)
    for row in selected:
        for name, spans in public.predictions_for(row["text"], {k: v[row["key"]] for k, v in caches.items()}).items():
            predictions[name][row["key"]] = spans
    for ds, split in (("pii", "pii/all"), ("redmadrobot", "redmadrobot/test")):
        measured = score_public([r for r in selected if r["dataset"] == ds], predictions, caches)
        metrics = measured["full_masking_all_gold_types"]["systems"]
        reference = expected_public["splits"][split]["full_masking_all_gold_types"]["systems"]
        typed = expected_public["splits"][split]["all_types_unfiltered"]["systems"]
        for name, slot in (("native", "gliner"), ("spacy", "spacy")):
            for profile in ("hybrid", "person_only"):
                if metrics[name + "_" + profile] != reference[slot + "_" + profile]:
                    raise ValueError("Public baseline aggregates do not reproduce: " + ds + "/" + name)
                if (
                    measured["typed_all_types_unfiltered"]["systems"][name + "_" + profile]
                    != typed[slot + "_" + profile]
                ):
                    raise ValueError("Public typed baseline aggregates differ: " + ds + "/" + name)
    return {
        "organizer_all_aggregates_exact": True,
        "public_full_masking_all_gold_types_exact": True,
        "public_typed_all_types_exact": True,
        "native_and_spacy": True,
        "cases": len(rows),
    }


def evaluate(args, rows, protocol):
    selected = [r for r in rows if args.scope == "all" or r["dataset"] == "organizer"]
    path = args.run_dir / f"{args.backend}-{args.device}-{args.scope}.jsonl"
    candidate, metadata = public.load_cache(path, selected, golden.sha256(args.run_dir / "protocol.json"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    failed_ids = {r["case_id"] for r in records if "inference_error" in r}
    expected_failed = {r["case_id"] for group in metadata["failures_by_corpus"].values() for r in group}
    if failed_ids != expected_failed:
        raise ValueError("Failure provenance differs between cache and metadata")
    caches = reference_caches(args, rows)
    caches["candidate"] = candidate
    predictions = defaultdict(dict)
    for row in selected:
        for name, spans in public.predictions_for(row["text"], {k: v[row["key"]] for k, v in caches.items()}).items():
            predictions[name][row["key"]] = spans
    corpora, cases = {}, golden.load_cases(DATA)
    for dataset in dict.fromkeys(r["dataset"] for r in selected):
        part = [r for r in selected if r["dataset"] == dataset]
        failures = sorted(r["key"] for r in part if r["key"] in failed_ids)
        if failures:
            corpora[dataset] = {
                "cases": len(part),
                "quality_available": False,
                "failed_case_ids": failures,
                "reason": "No complete-corpus quality claim when actual inference failed; no cases silently excluded.",
            }
            continue
        if dataset != "organizer":
            corpora[dataset] = score_public(part, predictions, caches)
            continue
        organizer_cache = {key: {"entities": candidate["organizer/" + key]} for key in cases}
        scored = {name: score_organizer(cases, values) for name, values in caches.items()}
        corpora[dataset] = {
            "cases": len(part),
            "service_hybrid": scored,
            "raw_person_candidate": person_only(cases, organizer_cache),
        }
    result = {
        "schema_version": 1,
        "protocol_sha256": golden.sha256(args.run_dir / "protocol.json"),
        "protocol": protocol,
        "candidate_metadata": metadata,
        "corpora": corpora,
        "parity_vs_native": {
            ds: (
                None
                if metadata["failures_by_corpus"].get(ds)
                else parity([r for r in selected if r["dataset"] == ds], caches["native"], candidate)
            )
            for ds in dict.fromkeys(r["dataset"] for r in selected)
        },
        "limitations": [
            "Organizer labels are provisional AI silver, not organizer ground truth.",
            "Corpora already used for rule development; selected GLiNER schema tuned on organizer before this experiment.",
            "No thresholds or annotations changed for the ONNX export.",
            "Full masking counts protected alphanumeric positions; typed metrics retain wrong classes.",
            "No inference speed claim is an HTTP RPS measurement.",
        ],
    }
    verify(args, protocol)
    save(args.output, result)
    return {"output": str(args.output), "cases": len(selected), "parity": result["parity_vs_native"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cache", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-path", type=Path, required=True)
    parser.add_argument("--backend", choices=("onnx", "native"), default="onnx")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--scope", choices=("all", "organizer"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_dir.resolve().is_relative_to(ROOT / "local-data"):
        parser.error("Use the ignored local-data directory for run outputs")
    if args.mode == "evaluate" and args.output is None:
        parser.error("Evaluation requires --output")
    public.disable_network()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        value = prepare(args)
        print(json.dumps({"cases": value["cases"], "prepared": True}))
        return
    protocol = json.loads((args.run_dir / "protocol.json").read_text())
    rows = verify(args, protocol)
    print(json.dumps(cache(args, rows, protocol) if args.mode == "cache" else evaluate(args, rows, protocol), indent=2))


if __name__ == "__main__":
    main()
