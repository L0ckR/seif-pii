"""Reconcile PII Guard paper metrics against frozen local RuBERT predictions.

No inference, network access, normalization, annotation edits, or raw-text output.
Run with the established SEIF analysis Python, optionally adding --word-cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts import compare_onnx as reference  # noqa: E402
from seif.detector import Span  # noqa: E402

COMMIT = "24230abb72949a9f85499244dd4f15a0ad0cdd9e"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_paper(upstream):
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required for pinned source verification")
    # Fixed read-only Git commands, argument list without a shell; the path is local.
    actual = subprocess.check_output([git, "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()  # noqa: S603
    if actual != COMMIT:
        raise ValueError("Upstream revision differs")
    source = upstream / "tests/quality/paper_score.py"
    committed = subprocess.check_output([git, "-C", str(upstream), "show", "HEAD:tests/quality/paper_score.py"])  # noqa: S603
    if source.read_bytes() != committed:
        raise ValueError("Upstream scorer has uncommitted changes")
    spec = importlib.util.spec_from_file_location("review_paper_score", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_native(path, rows):
    meta_path = path.with_suffix(".meta.json")
    metadata = json.loads(meta_path.read_text())
    if metadata["cache_sha256"] != sha(path):
        raise ValueError("Prediction cache digest differs")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if [r["case_id"] for r in records] != [r["key"] for r in rows]:
        raise ValueError("Prediction coverage/order differs from all 5095 frozen cases")
    for row, record in zip(rows, records, strict=True):
        if record["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("Prediction input text differs")
        for span in record["native"]:
            if not 0 <= span["start"] < span["end"] <= len(row["text"]):
                raise ValueError("Prediction offsets invalid")
    return {r["case_id"]: r for r in records}, metadata


def paper_result(paper, records, *, pipeline=False, strict=True):
    result = paper.score(records, pipeline=pipeline, strict=strict)
    return {"precision": result.micro[0], "recall": result.micro[1], "f1": result.micro[2],
            "tp": result.tp, "fp": result.fp, "fn": result.fn, "macro_f1": result.macro_f1,
            "negatives": result.negatives_total, "negatives_with_fp": result.negatives_with_fp,
            "per_type_tp_fp_fn": result.per_type}


def records_for(paper, rows, cache, *, folded):
    records = []
    for row in rows:
        gold, pred = [], []
        for kind, start, end in row["gold"]:
            mapped = paper.GOLD_TO_FAMILY["rmr"].get(kind) if folded else kind
            if mapped is None:
                raise ValueError("Unexpected unmapped original RMR gold label")
            gold.append({"type": mapped, "start": start, "end": end})
        for span in cache[row["key"]]["native"]:
            kind = span["label"]
            mapped = paper.NER_LABEL_TO_FAMILY.get(kind) if folded else kind
            if mapped is None:
                raise ValueError("Unexpected unmapped native model label")
            pred.append({"type": mapped, "start": span["start"], "end": span["end"]})
        # No scope exclusions: this frozen RMR set contains exactly the model's 21 types.
        records.append(paper.Record(id=row["key"], text=row["text"], gold=gold, pred=pred))
    return records


def score_cache(paper, rows, cache):
    scores = {}
    for dataset in ("organizer", "pii", "redmadrobot"):
        part = [r for r in rows if r["dataset"] == dataset]
        failures = [r["key"] for r in part if cache[r["key"]].get("inference_error")]
        if failures:
            scores[dataset] = {"cases": len(part), "quality_available": False, "failed_cases": len(failures)}
            continue
        predictions = {r["key"]: [Span(e["start"], e["end"], e["label"], e.get("score", .7), "native")
                                      for e in cache[r["key"]]["native"]] for r in part}
        scores[dataset] = {"cases": len(part), "quality_available": True,
                           "full_masking_all_gold_types": public.protection_metrics(part, {"native": predictions})["systems"]["native"]}
        if dataset == "redmadrobot":
            native = records_for(paper, part, cache, folded=False)
            folded = records_for(paper, part, cache, folded=True)
            scores[dataset].update({
                "native21_strict": paper_result(paper, native),
                "folded14_raw_strict": paper_result(paper, folded),
                "folded14_prediction_merge_strict": paper_result(paper, folded, pipeline=True),
                "folded14_raw_typed_overlap_one_to_one": paper_result(paper, folded, strict=False),
                "folded14_prediction_merge_typed_overlap_one_to_one": paper_result(paper, folded, pipeline=True, strict=False),
            })
    return scores


def validate_scorer_semantics(paper):
    gold = [{"type": "PERSON", "start": 0, "end": 1}, {"type": "PERSON", "start": 2, "end": 3}]
    broad = [{"type": "PERSON", "start": 0, "end": 3}]
    if paper.match(gold, broad, False)[0] != 1 or paper.match(gold, broad, True)[0] != 0:
        raise ValueError("Scorer no longer has documented one-to-one strict/soft semantics")
    record = paper.Record(id="synthetic", text="A B", gold=gold, pred=gold)
    if paper.score([record], strict=True, pipeline=False).tp != 2:
        raise ValueError("Raw scorer changed")
    merged = paper.score([record], strict=True, pipeline=True)
    if (merged.tp, merged.fp, merged.fn) != (0, 1, 2):
        raise ValueError("Prediction-only merge changed or now rewrites gold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "local-data/pii-guard-review/upstream")
    parser.add_argument("--public-run-dir", type=Path, default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public")
    parser.add_argument("--native-cache", type=Path, default=ROOT / "local-data/rubert-tensorrt/run-v1/rubert.jsonl")
    parser.add_argument("--word-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("metric-reconciliation.json"))
    args = parser.parse_args()
    public.disable_network()
    paper = load_paper(args.upstream)
    validate_scorer_semantics(paper)
    rows = reference.load_inputs(SimpleNamespace(public_run_dir=args.public_run_dir))
    previous_path = ROOT / "benchmarks/ner-models/rubert-tensorrt/comparison.json"
    previous = json.loads(previous_path.read_text())
    files = {"script": Path(__file__), "upstream_paper_score": args.upstream / "tests/quality/paper_score.py",
             "previous_comparison": previous_path, "organizer_gold": reference.DATA,
             "public_prepared": args.public_run_dir / "prepared-inputs.jsonl",
             "public_protocol": args.public_run_dir / "protocol.json"}
    systems = {}
    paths = {"published_trt_subword_decoder": args.native_cache}
    if args.word_cache is not None:
        paths["upstream_word_decoder_trt_backend"] = args.word_cache
    for name, path in paths.items():
        cache, metadata = load_native(path, rows)
        systems[name] = {"scores": score_cache(paper, rows, cache), "metadata": {
            key: metadata[key] for key in ("model", "model_revision", "settings", "protocol_sha256") if key in metadata}}
        files[name + "_cache"] = path
        files[name + "_metadata"] = path.with_suffix(".meta.json")
    # Recompute, then require parity with the already-published full-mask baseline.
    for dataset, result in systems["published_trt_subword_decoder"]["scores"].items():
        if result["full_masking_all_gold_types"] != previous["corpora"][dataset]["full_masking_all_gold_types"]["systems"]["rubert_native21"]:
            raise ValueError("Recomputed native full masking differs from frozen previous report")
    report = {
        "schema_version": 1, "upstream_commit": COMMIT,
        "cases": dict(Counter(r["dataset"] for r in rows)),
        "inputs_sha256": {name: sha(path) for name, path in files.items()},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          (ROOT / "scripts/compare_ner_public.py", ROOT / "scripts/compare_onnx.py", ROOT / "seif/transform.py")},
        "scorer_synthetic_checks": "PASS: exact vs overlap and prediction-only merge preserve original gold",
        "folded14_mapping": paper.NER_LABEL_TO_FAMILY,
        "results": systems,
        "historical_service_profiles_separate_protocol": {
            name: {"full_masking_all_gold_types": part["full_masking_all_gold_types"],
                   "service_typed_all_types_unfiltered": part.get("service_typed_all_types_unfiltered"),
                   "organizer_service_profiles": part.get("service_profiles")}
            for name, part in previous["corpora"].items()},
        "interpretation": [
            "All recomputed metrics retain frozen 2839 RMR rows; two historical alignment exclusions remain. Published card uses 2841.",
            "Native21 exact and folded14 raw exact are different taxonomies; neither merges or normalizes gold boundaries.",
            "Paper pipeline merges only PERSON/ADDRESS predictions across <=12-character allowed gaps; original gold remains split.",
            "Paper typed overlap is greedy one-to-one; the older quality gate can match one prediction to multiple gold entities.",
            "Full masking counts protected alphanumeric characters, all gold classes, ignoring types; it is not entity strict F1.",
            "Historical SEIF service uses only mapped PERSON/LOCATION from model plus existing rules; it is not native21 or PII Guard.",
            "Historical RMR service typed metric merges coarse gold and predictions alike, unlike paper prediction-only merge.",
            "83.6/88.9 published card claims are not assumed reproduced by different runtimes or these historical corpus exclusions.",
            "Pinned upstream docs/quality.md separately reports guard87.1 strict on14 families; no original prediction dumps or manifest are tracked.",
            "No new model tuning or inference is performed by this script; no raw texts are written to report."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "systems": list(systems), "cases": report["cases"]}))


if __name__ == "__main__":
    main()
