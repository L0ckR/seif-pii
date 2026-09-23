#!/usr/bin/env python3
"""Post-result development ablation using frozen Presidio PERSON predictions.

The original comparison report is read only. This measures span quality of the
merge policy, not NLP/HTTP latency; no NLP package or external service is needed.
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

from scripts.compare_presidio import COMMON, metric, remap  # noqa: E402
from seif.detector import Span, detect, merge_person_candidates  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "docs/presidio-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/hybrid-ablation.json")
    args = parser.parse_args()
    original_bytes = args.baseline.read_bytes()
    baseline = json.loads(original_bytes)
    detector_hash = hashlib.sha256((ROOT / "seif/detector.py").read_bytes()).hexdigest()
    identity = {kind: kind for kind in COMMON.values()}
    corpora = {}
    for name, corpus in baseline["corpora"].items():
        truth, current_fast, hybrid, rows = {}, {}, {}, []
        for row in corpus["raw_cases"]:
            case_id, text = row["case_id"], row["text"]
            base = detect(text)
            candidates = [
                Span(item["start"], item["end"], "PERSON", item["score"], "cached-presidio")
                for item in row["presidio"]
                if item["type"] == "PERSON"
            ]
            merged = merge_person_candidates(text, base, candidates)
            truth[case_id] = {tuple(item) for item in row["expected"]}
            current_fast[case_id] = {(item.type, item.start, item.end) for item in base}
            hybrid[case_id] = {(item.type, item.start, item.end) for item in merged}
            rows.append(
                {
                    "case_id": case_id,
                    "expected": sorted(truth[case_id]),
                    "current_fast": sorted(current_fast[case_id]),
                    "hybrid": sorted(hybrid[case_id]),
                    "original_fast": row["seif"],
                    "cached_ner_candidates": len(candidates),
                }
            )
        common_ids = corpus["common_strict4"]["case_ids"]
        common_truth = {key: truth[key] for key in common_ids}
        common_results = {}
        for system, values in (("current_fast", current_fast), ("hybrid", hybrid)):
            mapped = remap(values, identity)
            common_results[system] = metric(common_truth, {key: mapped[key] for key in common_ids})
        corpora[name] = {
            "case_count": corpus["case_count"],
            "canonical_corpus_sha256": corpus["canonical_corpus_sha256"],
            "common_case_ids": common_ids,
            "common_strict4": common_results,
            "fine_taxonomy_requirement_fit": {
                "current_fast": metric(truth, current_fast),
                "hybrid": metric(truth, hybrid),
            },
            "original_presidio_common_strict4": corpus["common_strict4"]["presidio"],
            "raw_cases": rows,
        }
    if detector_hash != hashlib.sha256((ROOT / "seif/detector.py").read_bytes()).hexdigest():
        raise RuntimeError("Detector changed during measurement.")
    report = {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Post-result development ablation, not independent held-out evidence",
        "baseline_report_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "baseline_detector_sha256": baseline["source_hashes"]["seif_detector_sha256"],
        "current_detector_sha256": detector_hash,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_source": "Unchanged cached PERSON spans/scores from the original Russian Presidio comparison; no label or threshold edits",
        "method": "Exact typed Unicode character spans; same whole-case common4 selection as the original report",
        "corpora": corpora,
        "limitations": [
            "Hybrid architecture was selected after seeing the original results; fresh28 is now development evidence.",
            "Original dev48 was already used for detector development.",
            "This is offline merge quality using cached model outputs, not end-to-end API, availability, throughput or latency validation.",
            "Original report is preserved; current fast results are recomputed and source hashes distinguish revisions.",
            "Small authored synthetic examples cannot establish general superiority or replace external corpus evaluation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    system: {key: values[key] for key in ("precision", "recall", "f1")}
                    for system, values in corpus["common_strict4"].items()
                }
                for name, corpus in corpora.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
