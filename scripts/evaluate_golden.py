"""Reproducible offline golden regression; labels are never inferred or modified.

Hybrid measurements reuse pinned model predictions to isolate detector changes.
This is a development regression score, not the organizer's official metric.
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

from scripts.evaluate_annotations import evaluate, positions, read_rows, shape_mask, validate  # noqa: E402
from seif.detector import Span, detect, merge_ner_candidates  # noqa: E402
from seif.transform import mask, restore_exact  # noqa: E402

DEFAULT_DATA = ROOT / "datasets/golden/organizer-v1/cases.jsonl"
DEFAULT_CACHE = ROOT / "benchmarks/golden/ner-cache.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path) -> dict:
    metadata = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["cases_sha256"] != sha256(path):
        raise ValueError("Golden version or checksum mismatch")
    rows = read_rows(path)
    if len(rows) != metadata["case_count"] or not rows:
        raise ValueError("Golden coverage mismatch")
    for row in rows.values():
        validate(row["text"], row["entities"], annotation=True)
        if row["expected_masked"] != shape_mask(row["text"], positions(row["text"], row["entities"])):
            raise ValueError("Golden mask disagrees with its immutable spans")
        if type(row["traffic_weight"]) is not int or row["traffic_weight"] < 1:
            raise ValueError("Golden weights must be positive integers")
    return rows


def load_ner_cache(path: Path, cases: dict) -> dict:
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if metadata["cache_sha256"] != sha256(path):
        raise ValueError("NER cache checksum mismatch")
    rows = read_rows(path)
    if set(rows) != set(cases):
        raise ValueError("NER cache must cover the complete golden corpus")
    for case_id, row in rows.items():
        expected = hashlib.sha256(cases[case_id]["text"].encode("utf-8")).hexdigest()
        if row["text_sha256"] != expected:
            raise ValueError("NER cache belongs to different input text")
    return rows


def predict(cases: dict, *, profile: str, cache: dict | None = None) -> dict:
    predictions = {}
    for case_id, case in cases.items():
        text = case["text"]
        spans = detect(text)
        if profile == "hybrid":
            if cache is None:
                raise ValueError("Hybrid evaluation requires pinned NER predictions")
            candidates = [Span(e["start"], e["end"], e["entity_type"], e["score"], "frozen-ner")
                          for e in cache[case_id]["entities"]]
            spans = merge_ner_candidates(text, spans, candidates)
        masked, replacements = mask(text, spans, "mask")
        if restore_exact({"masked": masked, "replacements": replacements}) != text:
            raise ValueError("Exact restoration failed; quality result rejected")
        predictions[case_id] = {
            "case_id": case_id, "masked": masked,
            "entities": [{"type": s.type, "start": s.start, "end": s.end} for s in spans],
        }
    return predictions


def compare_quality(current: dict, baseline: dict) -> list[str]:
    """Require the same corpus and nondecreasing global/certain P/R/F1."""
    if current["dataset_sha256"] != baseline["dataset_sha256"]:
        raise ValueError("Baseline belongs to a different golden version")
    if current.get("ner_cache_sha256") != baseline.get("ner_cache_sha256"):
        raise ValueError("Baseline uses different NER predictions")
    failures = []
    for profile, reference in baseline["profiles"].items():
        if profile not in current["profiles"]:
            raise ValueError("Missing baseline profile")
        actual = current["profiles"][profile]
        for group in ("unique_case_primary", "certain_cases_sensitivity"):
            failures.extend(
                f"{profile}.{group}.{metric}" for metric in ("precision", "recall", "f1")
                if actual[group]["character_metrics"][metric] + 1e-6 < reference[group]["character_metrics"][metric]
            )
        if actual["unique_case_primary"]["false_positive_negative_cases"] > reference["unique_case_primary"]["false_positive_negative_cases"]:
            failures.append(f"{profile}.false_positive_negative_cases")
    return failures


def run(dataset: Path, cache_path: Path, profiles: tuple[str, ...]) -> tuple[dict, dict]:
    cases = load_cases(dataset)
    cache = load_ner_cache(cache_path, cases) if "hybrid" in profiles else None
    inputs = {key: {"text": row["text"]} for key, row in cases.items()}
    weights = {key: row["traffic_weight"] for key, row in cases.items()}
    predictions = {profile: predict(cases, profile=profile, cache=cache) for profile in profiles}
    sources = sorted((ROOT / "seif").glob("*.py")) + [Path(__file__), ROOT / "scripts/evaluate_annotations.py"]
    report = {
        "schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256(dataset), "ner_cache_sha256": sha256(cache_path) if cache else None,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "profiles": {profile: evaluate(inputs, cases, rows, weights) for profile, rows in predictions.items()},
        "limitations": [
            "Golden labels are provisional independent AI annotations, not organizer ground truth.",
            "After tuning, improvements on this corpus are in-sample development evidence, not independent validation.",
            "Character P/R/F1 is not the organizer's private span-Levenshtein or code-review score.",
            "Cached NER isolates rule/merge behavior; this does not measure HTTP availability or RPS.",
        ],
    }
    return report, predictions


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ner-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--profile", choices=("rules", "hybrid", "both"), default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    profiles = ("rules", "hybrid") if args.profile == "both" else (args.profile,)
    report, _ = run(args.dataset, args.ner_cache, profiles)
    failures = compare_quality(report, json.loads(args.baseline.read_text())) if args.baseline else []
    report["regressions_against_baseline"] = failures
    if args.output:
        save_json(args.output, report)
    print(json.dumps({"profiles": {key: value["unique_case_primary"] for key, value in report["profiles"].items()},
                      "regressions": failures}, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
