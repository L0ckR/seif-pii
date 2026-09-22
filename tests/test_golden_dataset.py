"""Verify the privately committed golden corpus, when present in this checkout.

The source submission ZIP intentionally omits datasets. Missing datasets in that
distribution are expected; partial/corrupt datasets in a checkout are errors.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "golden" / "organizer-v1"
FROZEN_SHA256 = "3bec44d5eafbffc799cd525bbe573f8b0adf84ec82e16271bd406fe92b521e46"


@pytest.fixture(scope="module")
def corpus():
    if not DATASET.exists():
        pytest.skip("Private golden dataset is intentionally excluded from the source submission ZIP")
    raw = (DATASET / "cases.jsonl").read_bytes()
    return raw, json.loads((DATASET / "manifest.json").read_text(encoding="utf-8")), [
        json.loads(line) for line in raw.decode("utf-8").splitlines()
    ]


def test_canonical_content_and_frozen_hash(corpus):
    raw, manifest, rows = corpus
    canonical = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                           for row in rows) + "\n").encode("utf-8")
    assert raw == canonical
    assert hashlib.sha256(raw).hexdigest() == manifest["cases_sha256"] == FROZEN_SHA256
    assert manifest["dataset_id"] == "organizer-v1"
    assert manifest["version"] == "organizer-v1"
    assert manifest["release_version"] == "1.0.0"
    assert manifest["case_count"] == 446
    assert manifest["schema_version"] == 1


def test_complete_unique_cases_and_only_authorized_fields(corpus):
    _, _, rows = corpus
    assert [row["case_id"] for row in rows] == [f"org_{index:04d}" for index in range(1, 447)]
    assert len({row["text"] for row in rows}) == 446
    allowed = {"case_id", "text", "decision", "entities", "uncertain", "confidence", "notes",
               "traffic_weight", "expected_masked"}
    for row in rows:
        assert set(row) in (allowed, allowed | {"adjudication_reason"})
        if "adjudication_reason" in row:
            assert isinstance(row["adjudication_reason"], str)
        assert isinstance(row["text"], str)
        assert isinstance(row["notes"], str)
        assert type(row["uncertain"]) is bool
        assert row["confidence"] in {"high", "medium", "low"}
        assert type(row["traffic_weight"]) is int
        assert row["traffic_weight"] > 0


def test_gold_masks_come_only_from_valid_annotation_spans(corpus):
    _, _, rows = corpus
    for row in rows:
        text, spans = row["text"], row["entities"]
        assert row["decision"] == ("positive" if spans else "negative")
        expected, previous = list(text), 0
        for entity in sorted(spans, key=lambda value: (value["start"], value["end"])):
            assert set(entity) == {"type", "start", "end", "text"}
            start, end = entity["start"], entity["end"]
            assert type(start) is int
            assert type(end) is int
            assert previous <= start < end <= len(text)
            assert entity["text"] == text[start:end]
            assert isinstance(entity["type"], str)
            assert entity["type"]
            for index in range(start, end):
                if text[index].isalnum():
                    expected[index] = "*"
            previous = end
        assert row["expected_masked"] == "".join(expected)


def test_counts_and_annotation_limitations_are_preserved(corpus):
    _, manifest, rows = corpus
    counts = manifest["counts"]
    assert len(rows) == counts["cases"] == 446
    assert sum(row["decision"] == "positive" for row in rows) == counts["positive"] == 369
    assert sum(row["decision"] == "negative" for row in rows) == counts["negative"] == 77
    assert sum(row["uncertain"] for row in rows) == counts["uncertain"] == 117
    assert sum(len(row["entities"]) for row in rows) == counts["entities"] == 469
    assert sum(row["traffic_weight"] for row in rows) == counts["traffic_weight_total"] == 35845
    assert dict(Counter(entity["type"] for row in rows for entity in row["entities"])) == counts["entity_types"]
    assert dict(Counter(row["confidence"] for row in rows)) == counts["confidence"]
    provenance = manifest["provenance"]
    assert provenance["primary_ai_annotator_cases"] == 446
    assert provenance["independent_review_sample_cases"] == 80
    assert provenance["adjudicated_cases"] == 8
    assert sum("adjudication_reason" in row for row in rows) == 8
    assert provenance["official_organizer_ground_truth"] is False
    assert manifest["authorization"]["public_distribution_authorized"] is False
