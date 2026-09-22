"""Reproduce the immutable organizer-v1 corpus from its locally retained sources.

This utility copies existing annotations. It never runs a detector or revises a
label. Input hashes are pinned so the same version cannot silently change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

SOURCE_SHA256 = {
    "input-only.jsonl": "893f5660d0c7170fab8640b467acba8fa49ef4a9b2fc9a6ac934b0854e38a53c",
    "annotations.jsonl": "056a4d3744c8428854c07df95d221d1fa99479e3ffb6156ef0e9d145d5e9972c",
    "annotations-adjudicated.jsonl": "1c5bba96f00a970c074221aa6be0ae18efa345b55608a6b3cf7e2dcbd6b510b9",
    "load-corpus.jsonl": "407f1090c19a1c7a7b763ba897fad6d5fd49da009502a9300d35ac715ec18b46",
    "review-annotations.jsonl": "1239a03d3eee519a070c15033320ec3ae0213ef02c532981ba87372e9b5f8872",
    "review-sample-ids.json": "2b845e50725110906239993bd398b249f7a3cfbb4418c1756f076da6666bfd9f",
    "adjudication-decisions.jsonl": "9049316fbd23f58e5559a33ea41d766539f1a0b1add6778c5b13fcec73d37fa5",
    "annotation-protocol.md": "ca257fa2f77ac868813ab180ea623ed77de0144aee341917d88003bbb9d9496b",
}
LABEL_FIELDS = {"case_id", "decision", "entities", "uncertain", "confidence", "notes"}
ENTITY_FIELDS = {"type", "start", "end", "text"}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    indexed = {row["case_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate case IDs in {path.name}")
    return indexed


def masked(text: str, entities: list[dict]) -> str:
    chars = list(text)
    previous = 0
    for entity in sorted(entities, key=lambda item: (item["start"], item["end"])):
        start, end = entity["start"], entity["end"]
        if set(entity) != ENTITY_FIELDS or not isinstance(entity["type"], str) or not entity["type"]:
            raise ValueError("Invalid annotation entity schema")
        if type(start) is not int or type(end) is not int or not previous <= start < end <= len(text):
            raise ValueError("Invalid or overlapping annotation offsets")
        if entity["text"] != text[start:end]:
            raise ValueError("Annotation substring mismatch")
        for index in range(start, end):
            if text[index].isalnum():
                chars[index] = "*"
        previous = end
    return "".join(chars)


def build(source: Path) -> tuple[bytes, dict]:
    for filename, expected in SOURCE_SHA256.items():
        if hashlib.sha256((source / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {filename}")
    inputs = read_rows(source / "input-only.jsonl")
    annotations = read_rows(source / "annotations-adjudicated.jsonl")
    weights = read_rows(source / "load-corpus.jsonl")
    if set(inputs) != set(annotations) or set(inputs) != set(weights) or len(inputs) != 446:
        raise ValueError("Source case sets differ from the frozen 446-case corpus")
    rows = []
    for case_id, source_row in inputs.items():
        text = source_row["text"]
        annotation, traffic = annotations[case_id], weights[case_id]
        if (set(source_row) != {"case_id", "text"}
                or set(annotation) not in (LABEL_FIELDS, LABEL_FIELDS | {"adjudication_reason"})):
            raise ValueError("Unexpected source fields; refusing to copy extra metadata")
        if not isinstance(text, str) or traffic["payload"] != text:
            raise ValueError("Original text differs between sources")
        if type(traffic["weight"]) is not int or traffic["weight"] < 1:
            raise ValueError("Traffic weight must be a positive integer")
        if type(annotation["uncertain"]) is not bool or annotation["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("Invalid uncertainty or confidence")
        if not isinstance(annotation["notes"], str):
            raise ValueError("Annotation notes must be preserved as a string")
        if "adjudication_reason" in annotation and not isinstance(annotation["adjudication_reason"], str):
            raise ValueError("Adjudication reason must be preserved as a string")
        if annotation["decision"] != ("positive" if annotation["entities"] else "negative"):
            raise ValueError("Annotation decision disagrees with entity spans")
        rows.append({**annotation, "text": text, "traffic_weight": traffic["weight"],
                     "expected_masked": masked(text, annotation["entities"])})
    cases = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    entity_counts = Counter(entity["type"] for row in rows for entity in row["entities"])
    manifest = {
        "dataset_id": "organizer-v1",
        "version": "organizer-v1",
        "release_version": "1.0.0",
        "schema_version": 1,
        "case_count": len(rows),
        "frozen_on": "2026-09-22",
        "cases_file": "cases.jsonl",
        "cases_sha256": hashlib.sha256(cases).hexdigest(),
        "serialization": "UTF-8 JSON Lines; sorted object keys; compact separators; trailing LF",
        "offset_unit": "Unicode code points; zero-based start inclusive, end exclusive",
        "masking_rule": "Replace each alphanumeric code point inside a gold span with *; preserve all other code points",
        "counts": {
            "cases": len(rows),
            "positive": sum(row["decision"] == "positive" for row in rows),
            "negative": sum(row["decision"] == "negative" for row in rows),
            "uncertain": sum(row["uncertain"] for row in rows),
            "entities": sum(entity_counts.values()),
            "entity_types": dict(sorted(entity_counts.items())),
            "confidence": dict(sorted(Counter(row["confidence"] for row in rows).items())),
            "traffic_weight_total": sum(row["traffic_weight"] for row in rows),
        },
        "provenance": {
            "source": "446 unique original texts captured from an organizer evaluation run",
            "source_date": "2026-09-22",
            "source_sha256": SOURCE_SHA256,
            "label_source": "Previously locked annotations-adjudicated.jsonl; all labels copied without revision",
            "primary_ai_annotator_cases": 446,
            "independent_review_sample_cases": 80,
            "independent_review_sample_seed": 20260922,
            "adjudicated_cases": 8,
            "blind_annotation": "Annotators saw source text and protocol; outputs were locked before scoring",
            "reviewer_limitation": "Secondary reviewer had prior familiarity with detector code from earlier work",
            "official_organizer_ground_truth": False,
            "annotation_quality": "Provisional AI annotations (silver quality), frozen as golden regression expectations",
        },
        "authorization": {
            "private_repository": "L0ckR/seif-pii",
            "date": "2026-09-22",
            "basis": "User explicitly requested committing and pushing the annotated organizer dataset to their private repo",
            "supersedes": "Earlier local-only/no-Git instruction for these selected texts and frozen annotations",
            "excluded": ["source IP metadata", "API keys", "live request identifiers", "raw request logs",
                         "observed service outputs", "model predictions"],
            "public_distribution_authorized": False,
        },
        "evaluation_limits": [
            "Golden means frozen regression expectations, not official organizer ground truth or human-validated labels.",
            "117 cases are uncertain; report both all-case and certain-case metrics, retaining every case.",
            "There is no held-out split; scores after tuning on this corpus are in-sample development metrics.",
            "Do not reinterpret character F1 as the organizer's span-based Levenshtein score or overall hackathon score.",
            "Traffic weights repeat correlated texts and are secondary to equally weighted unique-case metrics.",
            "Do not revise v1 labels to match predictions; document corrections in a separately versioned dataset.",
        ],
    }
    return cases, manifest


def write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite changed immutable artifact: {path.name}")
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    cases, manifest = build(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    write_immutable(args.output / "cases.jsonl", cases)
    write_immutable(args.output / "manifest.json",
                    (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"cases": manifest["counts"]["cases"], "cases_sha256": manifest["cases_sha256"]}))


if __name__ == "__main__":
    main()
