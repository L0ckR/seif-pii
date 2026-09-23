#!/usr/bin/env python3
"""Post-result PERSON+LOCATION ablation from saved RedMadRobot predictions.

No model inference and no edits to the first benchmark report. This corpus
motivated the change, so this output is explicitly development evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_redmadrobot import (  # noqa: E402
    COMMON,
    SEIF_MAP,
    coarsen,
    merge_adjacent,
    read_rows,
    score_scope,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "output/external-bench-next")
    parser.add_argument("--baseline", type=Path, default=ROOT / "docs/redmadrobot-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/redmadrobot-location-ablation.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Ablation output already exists; use a distinct explicit output path to retain history.")
    from seif.detector import Span, detect, merge_ner_candidates

    original_bytes = args.baseline.read_bytes()
    baseline = json.loads(original_bytes)
    csv_bytes = (args.data_dir / "redmadrobot-test.csv").read_bytes()
    if sha(csv_bytes) != baseline["dataset"]["file_sha256"]["test.csv"]:
        raise RuntimeError("Original corpus changed.")
    raw_bytes = (args.data_dir / "redmadrobot-predictions-first.jsonl").read_bytes()
    if sha(raw_bytes) != baseline["raw_prediction_sha256"]:
        raise RuntimeError("Original prediction cache changed.")
    cached = {row["id"]: row for row in map(json.loads, raw_bytes.decode().splitlines())}
    rows, excluded = read_rows(args.data_dir)
    if excluded != baseline["dataset"]["excluded_before_inference"]:
        raise RuntimeError("Exclusion policy changed.")
    source_hash = sha((ROOT / "seif/detector.py").read_bytes())
    truth, new_predictions = {}, {"seif_person_location_hybrid": {}}
    map_with_location = {**SEIF_MAP, "LOCATION": "LOCATION"}
    base_changes = 0
    for row in rows:
        key, text = row["id"], row["text"]
        original = cached[key]
        base = detect(text)
        base_changes += {(item.type, item.start, item.end) for item in base} != {
            tuple(item) for item in original["original_predictions"]["seif_fast"]
        }
        # The fixed spaCy engine gives PERSON/LOCATION score0.85; the old cache
        # retained spans, not scores. No context enhancer changes these labels.
        candidates = [
            Span(start, end, kind, 0.85, "cached-ner")
            for kind, start, end in original["original_predictions"]["presidio_ru"]
            if kind in {"PERSON", "LOCATION"}
        ]
        result = merge_ner_candidates(text, base, candidates)
        truth[key] = {tuple(item) for item in original["merged_gold"]}
        coarse = coarsen({(item.type, item.start, item.end) for item in result}, map_with_location)
        new_predictions["seif_person_location_hybrid"][key] = merge_adjacent(text, coarse)
    if source_hash != sha((ROOT / "seif/detector.py").read_bytes()):
        raise RuntimeError("Detector changed during ablation.")
    report = {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Post-result development ablation, NOT an independent holdout result",
        "baseline_report_sha256": sha(original_bytes),
        "baseline_detector_sha256": baseline["source_hashes"]["detector"],
        "current_detector_sha256": source_hash,
        "evaluator_sha256": sha(Path(__file__).read_bytes()),
        "recomputed_fast_cases_changed": base_changes,
        "prediction_source": "Unchanged original cached Presidio PERSON/LOCATION offsets; fixed spaCy default score0.85 reconstructed because the original cache stored type/offsets only",
        "common5": score_scope(truth, new_predictions, COMMON),
        "person_all_cases": score_scope(truth, new_predictions, {"PERSON"}, whole_cases=False),
        "location_all_cases": score_scope(truth, new_predictions, {"LOCATION"}, whole_cases=False),
        "limitations": [
            "Architecture selected after reviewing this corpus's aggregate results; no independent superiority claim.",
            "No NLP or HTTP inference; this measures the changed merge policy only.",
            "Original gold, rows, common-case selection and baseline report are unchanged.",
            "The separate untouched Scanpatch test is the final held-out mixed-language evaluation.",
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["common5"]["systems"]["seif_person_location_hybrid"]["typed_character_primary"], indent=2))


if __name__ == "__main__":
    main()
