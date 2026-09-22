"""Evaluate frozen predictions against independently prepared local annotations.

No inference or annotation editing occurs here. Reports contain counts and offsets,
never input text. AI annotations remain provisional, not organizer gold labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate case IDs")
    return result


def validate(text, entities, *, annotation=False):
    previous = 0
    for entity in sorted(entities, key=lambda value: (value["start"], value["end"])):
        start, end = entity["start"], entity["end"]
        if type(start) is not int or type(end) is not int or not previous <= start < end <= len(text):
            raise ValueError("Invalid or overlapping entity offsets")
        if not isinstance(entity["type"], str) or not entity["type"]:
            raise ValueError("Invalid entity type")
        if annotation and entity.get("text") != text[start:end]:
            raise ValueError("Annotation substring mismatch")
        previous = end


def positions(text, entities, *, typed=False):
    return {(entity["type"], index) if typed else index
            for entity in entities for index in range(entity["start"], entity["end"])
            if text[index].isalnum()}


def scores(tp, fp, fn):
    if tp + fp:
        precision = tp / (tp + fp)
    elif not fn:
        precision = 1.0
    else:
        precision = 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
            "true_positive": tp, "false_positive": fp, "false_negative": fn}


def shape_mask(text, protected):
    return "".join("*" if index in protected else char for index, char in enumerate(text))


def summarize(counts):
    result = {key: value for key, value in counts.items() if key not in {"tp", "fp", "fn"}}
    result["character_metrics"] = scores(counts["tp"], counts["fp"], counts["fn"])
    result["exact_case_rate"] = round(counts["exact_cases"] / counts["cases"], 6) if counts["cases"] else None
    result["fully_protected_positive_rate"] = (round(counts["fully_protected_positive_cases"] /
                                                      counts["positive_cases"], 6)
                                                if counts["positive_cases"] else None)
    result["false_positive_negative_rate"] = (round(counts["false_positive_negative_cases"] /
                                                    counts["negative_cases"], 6)
                                              if counts["negative_cases"] else None)
    return result

def _update_type_coverage(text, gold, actual, type_counts):
    missing = []
    for kind in {s["type"] for s in gold}:
        relevant = positions(text, [s for s in gold if s["type"] == kind])
        type_counts[kind].update(gold_characters=len(relevant), protected_characters=len(relevant & actual),
                                 cases=1, fully_protected_cases=int(relevant <= actual))
        if relevant - actual:
            missing.append(kind)
    return missing


def evaluate(inputs, annotations, predictions, weights=None):
    if set(inputs) != set(annotations) or set(inputs) != set(predictions):
        raise ValueError("Inputs, annotations and predictions must cover the same complete case set")
    weights = weights or dict.fromkeys(inputs, 1)
    if set(weights) != set(inputs) or any(type(value) is not int or value < 1 for value in weights.values()):
        raise ValueError("Weights must be positive integers for every case")
    totals = Counter()
    weighted = Counter()
    certain = Counter()
    typed_totals = Counter()
    exact_span = Counter()
    type_counts = defaultdict(Counter)
    diagnostics = []
    for case_id, source in inputs.items():
        text = source["text"]
        label, prediction = annotations[case_id], predictions[case_id]
        gold, guessed = label["entities"], prediction["entities"]
        validate(text, gold, annotation=True)
        validate(text, guessed)
        if type(label.get("uncertain")) is not bool:
            raise ValueError("Every annotation needs an explicit uncertainty flag")
        if label.get("decision") != ("positive" if gold else "negative"):
            raise ValueError("Annotation decision conflicts with its entities")
        expected, actual = positions(text, gold), positions(text, guessed)
        if prediction["masked"] != shape_mask(text, actual):
            raise ValueError("Frozen spans do not reproduce the actually observed mask")
        counts = {"tp": len(expected & actual), "fp": len(actual - expected), "fn": len(expected - actual),
                      "cases": 1, "exact_cases": int(expected == actual),
                      "positive_cases": int(bool(expected)), "negative_cases": int(not expected),
                      "fully_protected_positive_cases": int(bool(expected) and expected <= actual),
                      "false_positive_negative_cases": int(not expected and bool(actual)),
                      "uncertain_cases": int(label["uncertain"])}
        totals.update(counts)
        weighted.update({key: value * weights[case_id] for key, value in counts.items()})
        if not label["uncertain"]:
            certain.update(counts)
        gold_typed, pred_typed = positions(text, gold, typed=True), positions(text, guessed, typed=True)
        typed_totals.update(tp=len(gold_typed & pred_typed), fp=len(pred_typed - gold_typed),
                            fn=len(gold_typed - pred_typed))
        gold_spans = {(s["type"], s["start"], s["end"]) for s in gold}
        pred_spans = {(s["type"], s["start"], s["end"]) for s in guessed}
        exact_span.update(tp=len(gold_spans & pred_spans), fp=len(pred_spans - gold_spans),
                          fn=len(gold_spans - pred_spans))
        missing = _update_type_coverage(text, gold, actual, type_counts)
        diagnostics.append({"case_id": case_id, **counts, "traffic_weight": weights[case_id],
                            "gold_types": sorted({s["type"] for s in gold}), "missed_types": sorted(missing)})


    return {
        "unique_case_primary": summarize(totals), "traffic_weighted_secondary": summarize(weighted),
        "certain_cases_sensitivity": summarize(certain),
        "typed_character_secondary": scores(typed_totals["tp"], typed_totals["fp"], typed_totals["fn"]),
        "exact_typed_span_secondary": scores(exact_span["tp"], exact_span["fp"], exact_span["fn"]),
        "gold_type_coverage": {kind: {**counts, "character_recall": round(
            counts["protected_characters"] / counts["gold_characters"], 6) if counts["gold_characters"] else None}
                               for kind, counts in sorted(type_counts.items())},
        "case_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(read_rows(args.inputs), read_rows(args.annotations), read_rows(args.predictions),
                      {key: row["weight"] for key, row in read_rows(args.corpus).items()})
    result.update({
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Alphanumeric character protection against frozen independent AI spans; unique-case micro precision/recall/F1",
        "source_sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                          for name in ("inputs", "annotations", "predictions", "corpus")},
        "limitations": [
            "Annotations were produced by an AI annotator, not by the organizer; annotation mistakes can change these scores.",
            "These metrics are not the organizer's normalized span-based Levenshtein score or proof of the hackathon 95% criterion.",
            "Corpus-wide unique texts are primary; traffic weights repeat correlated templates and are secondary.",
            "Typed span metrics are sensitive to address-component conventions and fragmented entity boundaries.",
            "Once inspected for development this corpus is no longer independent evidence for a tuned model.",
        ],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "unique_case_primary", "traffic_weighted_secondary", "certain_cases_sensitivity", "gold_type_coverage",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
