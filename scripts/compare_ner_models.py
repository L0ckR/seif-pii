"""Compare frozen NER models on the same immutable organizer development corpus.

This is a model experiment, not the detector regression gate: changing the cache
is intentional here. The existing same-cache quality gate remains unchanged.
No inference, network requests, label edits, or raw-text report output occur.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_annotations import positions, scores  # noqa: E402
from scripts.evaluate_golden import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DATA,
    load_cases,
    load_ner_cache,
    run,
    save_json,
    sha256,
)

PERSON_GOLD_TYPES = frozenset({"PERSON", "CARDHOLDER"})
METADATA_FIELDS = (
    "model", "model_revision", "revision", "versions", "configuration", "settings",
    "inference_settings", "threshold", "labels", "device", "batch_size", "purpose",
)


def compact_evaluation(result: dict) -> dict:
    """Retain every aggregate from the established evaluator, without 446 rows."""
    return {key: value for key, value in result.items() if key != "case_diagnostics"}


def classify_errors(reference: dict, candidate: dict) -> str:
    """Classify character-error Pareto changes without hiding precision tradeoffs."""
    changes = tuple(candidate[key] - reference[key] for key in ("fp", "fn"))
    if changes == (0, 0):
        return "equal_error_counts"
    if all(change <= 0 for change in changes):
        return "improved"
    if all(change >= 0 for change in changes):
        return "regressed"
    return "mixed_tradeoff"


def compare_cases(reference: dict, candidate: dict, predictions: dict) -> dict:
    """Expose changed masks, typed entities, error counts, and exact-case flips."""
    expected = {row["case_id"]: row for row in reference["case_diagnostics"]}
    actual = {row["case_id"]: row for row in candidate["case_diagnostics"]}
    if set(expected) != set(actual) or any(set(rows) != set(expected) for rows in predictions.values()):
        raise ValueError("Comparison predictions must cover the same complete case set")
    groups = {key: [] for key in (
        "prediction_changed", "mask_changed", "entity_only_changed", "improved", "regressed",
        "mixed_tradeoff", "equal_error_counts", "became_exact", "lost_exact",
    )}
    details = []
    for case_id in sorted(expected):
        before, after = predictions["reference"][case_id], predictions["candidate"][case_id]
        if before == after:
            continue
        mask_changed = before["masked"] != after["masked"]
        classification = classify_errors(expected[case_id], actual[case_id])
        groups["prediction_changed"].append(case_id)
        groups["mask_changed" if mask_changed else "entity_only_changed"].append(case_id)
        groups[classification].append(case_id)
        if expected[case_id]["exact_cases"] != actual[case_id]["exact_cases"]:
            groups["became_exact" if actual[case_id]["exact_cases"] else "lost_exact"].append(case_id)
        details.append({
            "case_id": case_id, "classification": classification, "mask_changed": mask_changed,
            "reference": {key: expected[case_id][key] for key in ("tp", "fp", "fn", "exact_cases")},
            "candidate": {key: actual[case_id][key] for key in ("tp", "fp", "fn", "exact_cases")},
        })
    return {
        "definition": "Improved/regressed means nonincreasing/nondecreasing FP and FN with at least one strict change.",
        "counts": {key: len(value) for key, value in groups.items()},
        **{f"{key}_case_ids": value for key, value in groups.items()},
        "changed_cases": details,
    }


def person_only(cases: dict, cache: dict) -> dict:
    """Score raw PERSON candidates; do not equate addresses to LOCATION labels."""
    groups = {"unique_case_primary": Counter(), "certain_cases_sensitivity": Counter()}
    candidate_counts: Counter = Counter()
    for case_id, case in cases.items():
        text = case["text"]
        expected = positions(text, [row for row in case["entities"] if row["type"] in PERSON_GOLD_TYPES])
        entities = cache[case_id]["entities"]
        actual = positions(text, [row for row in entities if row["entity_type"] == "PERSON"])
        candidate_counts.update(row["entity_type"] for row in entities)
        counts = {
            "tp": len(expected & actual), "fp": len(actual - expected), "fn": len(expected - actual),
            "cases": 1, "exact_cases": int(expected == actual),
            "negative_cases": int(not expected),
            "false_positive_negative_cases": int(not expected and bool(actual)),
        }
        groups["unique_case_primary"].update(counts)
        if not case["uncertain"]:
            groups["certain_cases_sensitivity"].update(counts)
    return {
        "gold_types_mapped_to_PERSON": sorted(PERSON_GOLD_TYPES),
        "candidate_entities_by_type": dict(sorted(candidate_counts.items())),
        **{key: {
            **{name: value for name, value in counts.items() if name not in {"tp", "fp", "fn"}},
            "character_metrics": scores(counts["tp"], counts["fp"], counts["fn"]),
        } for key, counts in groups.items()},
        "location_metrics": None,
        "interpretation": (
            "Raw model PERSON candidates before rules and public-entity filtering; PERSON and CARDHOLDER labels "
            "are scored together. LOCATION is unscored because the corpus has no matching LOCATION annotation "
            "class; structured ADDRESS, BIRTH_PLACE, CITY, and CITIZENSHIP spans are not interchangeable with it."
        ),
    }


def cache_description(path: Path) -> dict:
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "cache_sha256": sha256(path), "metadata_sha256": sha256(metadata_path),
        "metadata": {key: metadata[key] for key in METADATA_FIELDS if key in metadata},
    }


def metric_deltas(reference: dict, candidate: dict) -> dict:
    result = {}
    for group in ("unique_case_primary", "certain_cases_sensitivity"):
        before, after = reference[group], candidate[group]
        result[group] = {
            **{name: round(after["character_metrics"][name] - before["character_metrics"][name], 6)
               for name in ("precision", "recall", "f1")},
            **{name: after.get(name, 0) - before.get(name, 0)
               for name in ("exact_cases", "false_positive_negative_cases")},
        }
    return result


def compare(dataset: Path, reference_cache: Path, candidate_cache: Path) -> dict:
    cases = load_cases(dataset)
    reference, baseline_predictions = run(dataset, reference_cache, ("rules", "hybrid"))
    candidate, candidate_predictions = run(dataset, candidate_cache, ("hybrid",))
    before, after = reference["profiles"]["hybrid"], candidate["profiles"]["hybrid"]
    if reference["dataset_sha256"] != candidate["dataset_sha256"]:
        raise ValueError("Dataset changed during the model comparison")
    if reference["source_sha256"] != candidate["source_sha256"]:
        raise ValueError("Evaluation source changed during the model comparison")
    return {
        "schema_version": 1, "measured_at_utc": candidate["measured_at_utc"],
        "method": "Frozen model caches compared through the same detector, NER merge, mask, and character evaluator.",
        "evaluation_runtime": {"python": sys.version, "implementation": sys.implementation.name},
        "dataset_sha256": candidate["dataset_sha256"],
        "dataset_manifest_sha256": sha256(dataset.with_name("manifest.json")),
        "source_sha256": {**candidate["source_sha256"], "scripts/compare_ner_models.py": sha256(Path(__file__))},
        "reference": cache_description(reference_cache), "candidate": cache_description(candidate_cache),
        "hybrid": {"reference": compact_evaluation(before), "candidate": compact_evaluation(after)},
        "rules_control": compact_evaluation(reference["profiles"]["rules"]),
        "candidate_minus_reference": metric_deltas(before, after),
        "hybrid_changes": compare_cases(before, after, {
            "reference": baseline_predictions["hybrid"], "candidate": candidate_predictions["hybrid"],
        }),
        "model_only_person_secondary": {
            "reference": person_only(cases, load_ner_cache(reference_cache, cases)),
            "candidate": person_only(cases, load_ner_cache(candidate_cache, cases)),
        },
        "limitations": [*candidate["limitations"],
            "The organizer texts have provisional AI silver labels, not organizer-provided ground truth.",
            "This offline model comparison does not certify latency, throughput, model availability, or memory usage.",
            "Threshold/model choices evaluated on this already inspected corpus require a separate held-out set.",
            "Per-type coverage measures protection regardless of predicted class; typed metrics use exact taxonomy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace an existing report: {args.output}")
    report = compare(args.dataset, args.reference_cache, args.candidate_cache)
    save_json(args.output, report)
    print(json.dumps({
        "hybrid": {key: value["unique_case_primary"] for key, value in report["hybrid"].items()},
        "candidate_minus_reference": report["candidate_minus_reference"],
        "changed_cases": report["hybrid_changes"]["counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
