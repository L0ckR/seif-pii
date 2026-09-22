"""pii-guard rule/merge ablation on raw texts with frozen TensorRT predictions.

This is not the complete pii-guard pipeline: normalization, transliteration,
English-number/base64 preprocessing and fresh neural inference are excluded.
Only pinned upstream rules, label mapping, email gate and conflict resolution
run locally. The official word-decoder cache can be compared separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import sysconfig
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts import compare_onnx as reference  # noqa: E402
from scripts import compare_rubert as rubert  # noqa: E402
from seif.detector import Span  # noqa: E402
from seif.rubert_ner import NATIVE_TYPES  # noqa: E402

UPSTREAM = "https://github.com/redmadrobot-rnd/pii-guard"
REVISION = "24230abb72949a9f85499244dd4f15a0ad0cdd9e"
ALIASES = {"EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "CREDIT_CARD": "CARD"}
EXPECTED_COUNTS = {"organizer": 446, "pii": 1810, "redmadrobot": 2839}
BASELINE_REPORT = ROOT / "benchmarks/ner-models/rubert-tensorrt/comparison.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hashes(path, suffix=None):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and (suffix is None or p.suffix == suffix)}


def upstream_fingerprint(path):
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to verify upstream provenance")
    actual = subprocess.check_output(  # noqa: S603
        [git, "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(  # noqa: S603
        [git, "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if actual != REVISION or dirty:
        raise ValueError("Pinned upstream source differs")
    return {"revision": actual, "source_sha256": tree_hashes(path / "src", ".py")}


def provenance(args):
    files = {"raw_cache": args.raw_cache, "raw_metadata": args.raw_cache.with_suffix(".meta.json"),
             "organizer": reference.DATA, "public_inputs": args.public_run_dir / "prepared-inputs.jsonl",
             "public_protocol": args.public_run_dir / "protocol.json", "runner": Path(__file__).resolve(),
             "previous_rubert_comparison": BASELINE_REPORT}
    if args.word_cache:
        files.update(word_cache=args.word_cache, word_metadata=args.word_cache.with_suffix(".meta.json"))
    return {"upstream": upstream_fingerprint(args.upstream),
            "input_and_runner_sha256": {key: digest(path) for key, path in files.items()},
            "evaluation_source_sha256": reference.source_hashes(),
            "spacy_model_sha256": tree_hashes(args.spacy_model),
            "environment": {"python": sys.version, "platform": platform.platform(),
                            # The upstream src path is added only after pin verification.
                            # Its generated egg-info is not an installed venv dependency.
                            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions(
                                path=[sysconfig.get_paths()["purelib"]])}}}


def load_native(path, rows):
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if metadata["cache_sha256"] != digest(path) or metadata["model_revision"] != rubert.REVISION:
        raise ValueError("Native cache provenance differs")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if [r["case_id"] for r in records] != [r["key"] for r in rows]:
        raise ValueError("Native cache membership or order differs")
    failed = {r["case_id"] for r in records if r.get("inference_error")}
    if failed:
        raise ValueError("Native cache contains inference failures")
    for row, record in zip(rows, records, strict=True):
        if record["text_sha256"] != rubert.text_hash(row["text"]):
            raise ValueError("Native cache text fingerprint differs")
        if not isinstance(record["native"], list):
            raise ValueError("Native entities must be a list")
        for entity in record["native"]:
            start, end, score = entity["start"], entity["end"], entity["score"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row["text"])
                    or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                    or not isinstance(entity["label"], str) or entity["label"] not in NATIVE_TYPES):
                raise ValueError("Native entity contract differs")
    return {r["case_id"]: r for r in records}, {
        key: metadata[key] for key in ("model", "model_revision", "settings", "protocol_sha256",
                                     "cache_sha256", "source_sha256", "decoder", "backend", "max_tokens",
                                     "stride", "score", "failures") if key in metadata
    }


def converted_ner(text, native, mapping, score, result_factory, email_gate):  # noqa: PLR0913
    """Use the upstream mapping/priority score, preserving cached boundaries."""
    counts = Counter(native_unmapped_total=0)
    results = []
    for entity in native:
        label = entity["label"]
        counts["native:" + label] += 1
        mapped = mapping.get(label)
        if mapped is None:
            counts["native_unmapped_total"] += 1
            counts["unmapped_native:" + label] += 1
            continue
        counts["mapped:" + mapped] += 1
        results.append(result_factory(entity_type=mapped, start=entity["start"], end=entity["end"], score=score))
    gated = email_gate(results, text)
    counts["email_shape_discarded"] = len(results) - len(gated)
    return gated, counts


def safe_spans(results, text):
    spans = []
    end = 0
    for item in sorted(results, key=lambda r: (r.start, r.end, r.entity_type)):
        if not end <= item.start < item.end <= len(text):
            raise ValueError("Final rule merge overlaps or has invalid offsets")
        end = item.end
        spans.append(Span(item.start, item.end, ALIASES.get(item.entity_type, item.entity_type), item.score, "pii-guard"))
    return spans


def typed_metrics(rows, predictions):
    dataset = rows[0]["dataset"]
    truth = {}
    mapped = {name: {} for name in predictions}
    known_gold = {kind for row in rows for kind, *_ in row["gold"]}
    if dataset == "redmadrobot":
        known_gold = {public.redmad.GOLD_MAP.get(kind, kind) for kind in known_gold}
    for row in rows:
        key, text = row["key"], row["text"]
        gold = row["gold"]
        if dataset == "redmadrobot":
            gold = public.redmad.coarsen(gold, public.redmad.GOLD_MAP, keep_unknown=True)
            gold = public.redmad.merge_adjacent(text, gold)
        truth[key] = gold
        mapping = public.pii.SEIF_MAP if dataset == "pii" else (
            {**public.redmad.SEIF_MAP, "LOCATION": "LOCATION"} if dataset == "redmadrobot" else {})
        for name, values in predictions.items():
            # Existing aliases plus literal identical taxonomy names; unsupported
            # predictions remain false positives rather than disappearing.
            spans = set()
            for span in values[key]:
                kind = public.redmad.GOLD_MAP.get(span.type, span.type) if name == "rubert_native21" else span.type
                spans.add((mapping.get(kind, kind if kind in known_gold else "UNMAPPED_PRED:" + kind), span.start, span.end))
            mapped[name][key] = public.redmad.merge_adjacent(text, spans) if dataset == "redmadrobot" else spans
    return public.scope_metrics(truth, mapped)


def evaluate_corpus(rows, predictions):
    result = {"cases": len(rows), "full_masking_all_gold_types": public.protection_metrics(rows, predictions),
              "typed_all_types_unfiltered": typed_metrics(rows, predictions)}
    result["full_masking_all_gold_types"]["prediction_scope"] = (
        "Alphanumeric mask projection of each complete profile's final spans; not an HTTP/full-pipeline claim."
    )
    return result


def baseline_control(corpora):
    previous = json.loads(BASELINE_REPORT.read_text())["corpora"]
    checks = {}
    for dataset, result in corpora.items():
        if result.get("quality_available") is False:
            checks[dataset] = {"verified": False}
            continue
        names = {"seif_rules": "rules", "seif_rubert_hybrid": "rubert_hybrid", "rubert_native21": "rubert_native21"}
        checks[dataset] = {
            name: result["full_masking_all_gold_types"]["systems"][name] == previous[dataset]["full_masking_all_gold_types"]["systems"][old]
            for name, old in names.items()
        }
        if not all(checks[dataset].values()):
            raise ValueError("Historical baseline full-mask metrics differ")
    return checks


def run(args, report):  # noqa: PLR0915
    before = provenance(args)
    report["provenance"] = before
    rows = reference.load_inputs(args)
    if dict(Counter(r["dataset"] for r in rows)) != EXPECTED_COUNTS:
        raise ValueError("Frozen corpus counts differ")
    cache, metadata = load_native(args.raw_cache, rows)
    caches = {"guard_raw_trt_cache": cache}
    report["cache_metadata"] = {"guard_raw_trt_cache": metadata}
    if args.word_cache:
        caches["guard_word_decoder_cache"], report["cache_metadata"]["guard_word_decoder_cache"] = load_native(args.word_cache, rows)
    sys.path.insert(0, str(args.upstream / "src"))
    from pii_guard.config import NER_ONLY_ENTITIES, Config
    from pii_guard.detect import resolve_conflicts, resolve_ml_vs_rules_conflicts
    from pii_guard.engine import NER_ENTITY_MAPPING, Engine
    from pii_guard.entities._email_shape import filter_ner_email_spans
    from presidio_analyzer import RecognizerResult

    configuration = Config(spacy_model=str(args.spacy_model), enable_translit=False,
                           enable_en_numbers=False, enable_base64=False)
    engine = Engine(configuration, ner=False)
    report["upstream_policy"] = {"native_label_mapping": NER_ENTITY_MAPPING, "fixed_ner_priority_score": configuration.ner_score,
                                 "ner_only_entities_availability_warning_not_allowlist": sorted(NER_ONLY_ENTITIES),
                                 "output_allowlist": None, "type_aliases_for_comparison": ALIASES,
                                 "rules_nlp_pipeline": engine._rules_analyzer.nlp_engine.nlp["ru"].pipe_names}
    predictions = defaultdict(dict)
    counters = defaultdict(lambda: defaultdict(Counter))
    failures = defaultdict(list)
    for index, row in enumerate(rows, 1):
        key, text, dataset = row["key"], row["text"], row["dataset"]
        try:
            baseline = public.predictions_for(text, {"rubert": cache[key]["entities"]})
            predictions["seif_rules"][key] = baseline["rules"]
            predictions["seif_rubert_hybrid"][key] = baseline["rubert_hybrid"]
            predictions["rubert_native21"][key] = [Span(e["start"], e["end"], e["label"], e["score"], "native21")
                                                   for e in cache[key]["native"]]
            rules = engine._rules_analyzer.analyze(text=text, language="ru")
            registry = engine._numeric_recognizer.registry
            predictions["guard_rules_raw"][key] = safe_spans(resolve_conflicts(rules, text, registry=registry), text)
            for profile, records in caches.items():
                ner, counts = converted_ner(text, records[key]["native"], NER_ENTITY_MAPPING,
                                            configuration.ner_score, RecognizerResult, filter_ner_email_spans)
                merged = resolve_ml_vs_rules_conflicts(rules, ner)
                counts["ner_discarded_by_rule_overlap"] = len(rules) + len(ner) - len(merged)
                kept_ids = {id(entity) for entity in merged}
                counts.update("overlap_discarded_type:" + entity.entity_type for entity in ner if id(entity) not in kept_ids)
                final = resolve_conflicts(merged, text, registry=registry)
                counts["combined_discarded_by_final_resolution"] = len(merged) - len(final)
                counts["final_entities"] = len(final)
                counters[dataset][profile].update(counts)
                predictions[profile][key] = safe_spans(final, text)
        except Exception as exc:
            failures[dataset].append({"case_id": key, "error_type": type(exc).__name__})
        if index % 500 == 0:
            print(json.dumps({"completed": index, "total": len(rows), "failed": sum(map(len, failures.values()))}), flush=True)
    report["counters"] = counters
    report["failures"] = failures
    report["corpora"] = {}
    for dataset in EXPECTED_COUNTS:
        part = [r for r in rows if r["dataset"] == dataset]
        report["corpora"][dataset] = ({"cases": len(part), "quality_available": False, "failures": failures[dataset]}
                                      if failures[dataset] else evaluate_corpus(part, predictions))
    report["historical_fullmask_baseline_control"] = baseline_control(report["corpora"])
    if before != provenance(args):
        raise ValueError("Pinned source, cache, corpus, model data or environment changed during evaluation")
    report["status"] = "FAIL" if any(failures.values()) else "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "local-data/pii-guard-review/upstream")
    parser.add_argument("--spacy-model", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, default=ROOT / "local-data/rubert-tensorrt/run-v1/rubert.jsonl")
    parser.add_argument("--word-cache", type=Path)
    parser.add_argument("--public-run-dir", type=Path, default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public.disable_network()
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    logging.getLogger("pii_guard").setLevel(logging.ERROR)
    report = {"status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "study": "pii-guard rule/merge ablation on raw original texts with cached TensorRT predictions",
              "upstream_url": UPSTREAM, "upstream_revision": REVISION,
              "limitations": [
                  "Not the complete upstream pipeline: no text normalization, English numeral conversion, transliteration or base64 analysis.",
                  "Original raw TRT cache uses the published custom decoder and threshold0.3; upstream argmax decoder differs.",
                  "Upstream fixed NER score0.70 is a conflict priority, not confidence or a new threshold.",
                  "All21 labels map through the actual upstream policy; NER_ONLY_ENTITIES is not an allowlist.",
                  "Rule numeric normalize_safe remains active because it is internal and offset preserving.",
                  "Common full masking counts alphanumeric positions only; punctuation masking and roundtrip behavior are outside this study.",
                  "Organizer annotations are AI silver; public corpora previously informed SEIF rules; not a blind holdout.",
                  "RuBERT and RMR share source family; no new neural inference, no tuning, no HTTP RPS claim.",
              ]}
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            run(args, report)
        except Exception as exc:
            report["fatal_error_type"] = type(exc).__name__
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "fatal_error_type": report.get("fatal_error_type")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
