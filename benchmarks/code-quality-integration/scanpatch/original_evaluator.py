"""Offline regression on an already-used pinned Scanpatch test corpus.

No inference, downloads, API calls or training. Missing NER scores are recovered
only after validating the original fixed spaCy score path and reproducing the
first stored hybrid predictions on every case. The old reports are immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_redmadrobot import COMMON, PRESIDIO_MAP, coarsen, merge_adjacent, score_scope
from scripts.evaluate_scanpatch import FILES, GOLD_MAP, LOCAL_MAP, OTHER_LABELS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rows_and_cache(folder, original_report):
    import pyarrow.parquet as parquet

    for _, (filename, checksum) in FILES.items():
        if sha(folder / filename) != checksum:
            raise RuntimeError("Pinned local corpus hash mismatch; downloads are disabled")
    rows = parquet.read_table(folder / "scanpatch-test.parquet").to_pylist()
    if len(rows) != 532:
        raise RuntimeError("Unexpected local corpus size")
    cache_path = folder / "scanpatch-predictions-first.jsonl"
    if sha(cache_path) != original_report["raw_prediction_sha256"]:
        raise RuntimeError("Original prediction cache hash mismatch")
    cached = [json.loads(line) for line in cache_path.read_text().splitlines()]
    if len(cached) != 532 or len({row["id"] for row in cached}) != 532:
        raise RuntimeError("Prediction cache must cover all 532 unique cases")
    cache = {row["id"]: row for row in cached}
    for index, row in enumerate(rows):
        row["id"] = f"test_{index:04d}"
        values = [row[key] for key in ("entity_starts", "entity_ends", "entity_labels", "entity_texts")]
        if len({len(value) for value in values}) != 1:
            raise RuntimeError("Original annotation length mismatch")
        gold = set()
        for start, end, kind, value in zip(*values, strict=True):
            if (kind not in set(GOLD_MAP) | OTHER_LABELS or not 0 <= start < end <= len(row["text"])
                    or row["text"][start:end] != value):
                raise RuntimeError("Original annotation text or offset mismatch")
            gold.add((kind, start, end))
        row["fine_gold"] = gold
        row["merged_gold"] = merge_adjacent(row["text"], coarsen(gold, GOLD_MAP, keep_unknown=True))
        original = cache[row["id"]]
        if gold != {tuple(value) for value in original["fine_gold"]}:
            raise RuntimeError("Cache annotation identity mismatch")
        if row["merged_gold"] != {tuple(value) for value in original["merged_gold"]}:
            raise RuntimeError("Frozen annotation mapping mismatch")
    return rows, cache


def score_evidence(original_report):
    """Check the score path from installed source, without loading an NLP model."""
    package = Path(importlib.util.find_spec("presidio_analyzer").origin).parent
    files = {
        "ner_configuration": package / "nlp_engine/ner_model_configuration.py",
        "spacy_engine": package / "nlp_engine/spacy_nlp_engine.py",
        "spacy_recognizer": package / "predefined_recognizers/nlp_engine_recognizers/spacy_recognizer.py",
        "recognizer": package / "entity_recognizer.py",
        "context_enhancer": package / "context_aware_enhancers/lemma_context_aware_enhancer.py",
    }
    source = {name: path.read_text() for name, path in files.items()}
    version = importlib.metadata.version("presidio-analyzer")
    if version != "2.2.364":
        raise RuntimeError("Installed Presidio version differs from frozen run")
    checks = {
        "fixed_score": "default=0.85, ge=0.0, le=1.0" in source["ner_configuration"],
        "no_low_score_types": "LOW_SCORE_ENTITY_NAMES = set()" in source["ner_configuration"],
        "every_spacy_entity_gets_default":
            "scores = [self.ner_model_configuration.default_score] * len(entities)" in source["spacy_engine"],
        "recognizer_preserves_score": "score=ner_score" in source["spacy_recognizer"],
        "empty_default_context": "self.context = context if context else []" in source["recognizer"],
        "no_context_boost_for_empty_context": "if not recognizer.context:" in source["context_enhancer"],
    }
    configuration = original_report["presidio_configuration"]
    nlp_config = configuration["nlp_engine"]["ner_model_configuration"]
    checks["no_score_override_in_original_configuration"] = not {
        "default_score", "low_score_entity_names", "low_confidence_score_multiplier"
    } & set(nlp_config)
    providers = [r for r in configuration["recognizers"]
                 if {"PERSON", "LOCATION"} & set(r["supported_entities"])]
    checks["only_spacy_provides_person_location_without_context"] = (
        len(providers) == 1 and providers[0]["name"] == "SpacyRecognizer" and not providers[0]["context"]
    )
    if not all(checks.values()):
        raise RuntimeError("Missing-score reconstruction is not justified by installed source/configuration")
    return {"constant_person_location_score": 0.85, "presidio_version": version, "checks": checks,
            "evidence_source_sha256": {name: sha(path) for name, path in files.items()},
            "limitation": "The JSONL omitted scores. Recover fixed 0.85 from unchanged configuration/source, "
                          "then require exact reproduction of all original rule and hybrid spans."}


def infer_from_cache(detector, rows, cache):
    output = {"rules": {}, "hybrid": {}}
    for row in rows:
        key, text = row["id"], row["text"]
        base = detector.detect(text)
        candidates = [detector.Span(start, end, kind, 0.85, "ner")
                      for kind, start, end in cache[key]["original_predictions"]["presidio_ru"]
                      if kind in {"PERSON", "LOCATION"}]
        hybrid = detector.merge_ner_candidates(text, base, candidates)
        for name, values in (("rules", base), ("hybrid", hybrid)):
            output[name][key] = {(value.type, value.start, value.end) for value in values}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cache-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "output/golden-quality/scanpatch-regression.json")
    args = parser.parse_args()
    folder = ROOT / "output/external-bench-next"
    original_report = json.loads((ROOT / "docs/scanpatch-comparison.json").read_text())
    rows, cache = rows_and_cache(folder, original_report)
    evidence = score_evidence(original_report)
    original_path = ROOT / "output/golden-quality/detector-original-submitted-zip.py"
    original_detector = load_module("scanpatch_original_detector", original_path)
    reconstructed = infer_from_cache(original_detector, rows, cache)
    differences = {profile: [key for key, spans in values.items()
                            if spans != {tuple(item) for item in cache[key]["original_predictions"][old_name]}]
                   for profile, old_name in (("rules", "seif_fast"), ("hybrid", "seif_hybrid"))
                   for values in (reconstructed[profile],)}
    if any(differences.values()):
        raise RuntimeError("Cached NER did not exactly reproduce every original prediction: "
                           + json.dumps({key: len(value) for key, value in differences.items()}))
    evidence["original_reproduction"] = {"cases": len(rows), "rules_different_cases": 0,
                                         "hybrid_different_cases": 0, "detector_sha256": sha(original_path)}
    if args.verify_cache_only:
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return
    baseline_path = ROOT / "output/golden-quality/detector-75b52e8.py"
    baseline = load_module("scanpatch_baseline_75b52e8", baseline_path)
    from seif import detector as current

    tracked_paths = sorted((ROOT / "seif").glob("*.py"))
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in tracked_paths}
    raw = {}
    for version, detector in (("baseline_75b52e8", baseline), ("current", current)):
        for profile, predictions in infer_from_cache(detector, rows, cache).items():
            raw[f"{version}_{profile}"] = predictions
    raw["presidio_ru_unchanged"] = {
        key: {tuple(item) for item in row["original_predictions"]["presidio_ru"]} for key, row in cache.items()
    }
    predictions = {name: {} for name in raw}
    truth = {row["id"]: row["merged_gold"] for row in rows}
    for row in rows:
        key, text = row["id"], row["text"]
        for name, values in raw.items():
            mapping = PRESIDIO_MAP if name == "presidio_ru_unchanged" else LOCAL_MAP
            predictions[name][key] = merge_adjacent(text, coarsen(values[key], mapping))
    primary = score_scope(truth, predictions, COMMON)
    if len(primary["case_ids"]) != 164:
        raise RuntimeError("Frozen whole-case common5 subset changed")
    if hashes != {str(path.relative_to(ROOT)): sha(path) for path in tracked_paths}:
        raise RuntimeError("Detector sources changed during offline regression")
    report = {
        "evaluation_status": "Post-development regression on a previously used external test, not a fresh holdout",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(), "offered_cases": len(rows),
        "primary_common_case_count": len(primary["case_ids"]), "inference_calls": 0,
        "network_calls": 0, "source_sha256": hashes, "evaluator_sha256": sha(Path(__file__)),
        "baseline_detector_sha256": sha(baseline_path), "score_reconstruction_evidence": evidence,
        "corpus_sha256": sha(folder / "scanpatch-test.parquet"),
        "original_prediction_cache_sha256": sha(folder / "scanpatch-predictions-first.jsonl"),
        "primary_common_whole_cases": primary,
        "common_types_all_532_diagnostic": score_scope(truth, predictions, COMMON, whole_cases=False),
        "person_all_532_diagnostic": score_scope(truth, predictions, {"PERSON"}, whole_cases=False),
        "location_all_532_diagnostic": score_scope(truth, predictions, {"LOCATION"}, whole_cases=False),
        "document_tax_all_532_diagnostic": score_scope(truth, predictions, {"DOCUMENT", "INN"}, whole_cases=False),
        "limitations": ["Mixed synthetic Russian/Ukrainian corpus with no per-row language tags.",
                        "Common5 has no CARD positives and does not test all hackathon banking categories.",
                        "Repeated external regression; neither new holdout nor an estimate of the organizer score.",
                        "Models and raw Presidio predictions were reused unchanged; no new model inference.",
                        "Baseline and current rules use identical text, annotation mapping and case selection."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    prediction_path = args.output.with_name(args.output.stem + "-predictions.jsonl")
    with prediction_path.open("x") as file:
        for row in rows:
            key = row["id"]
            file.write(json.dumps({"id": key, "predictions": {name: sorted(values[key])
                                                            for name, values in raw.items()}}) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(rows), "common_cases": len(primary["case_ids"]),
                      "primary": {name: {key: value for key, value in scores["typed_character_primary"].items()
                                         if key in {"precision", "recall", "f1", "tp", "fp", "fn"}}
                                  for name, scores in primary["systems"].items()}}, indent=2))


if __name__ == "__main__":
    main()
