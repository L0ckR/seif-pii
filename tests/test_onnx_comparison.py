"""Parity and full-corpus metrics must not hide an ONNX regression."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from scripts import compare_onnx as comparison
from seif.detector import Span


def test_parity_separates_numeric_drift_from_missing_and_extra_spans():
    rows = [{"key": "drift-only"}, {"key": "changed-spans"}]
    reference = {
        "drift-only": [{"start": 0, "end": 5, "entity_type": "PERSON", "score": .9}],
        "changed-spans": [{"start": 0, "end": 5, "entity_type": "PERSON", "score": .95},
                          {"start": 6, "end": 10, "entity_type": "LOCATION", "score": .9}],
    }
    candidate = {
        "drift-only": [{"start": 0, "end": 5, "entity_type": "PERSON", "score": .90001}],
        # A type change at identical boundaries is a dropped and an extra span.
        "changed-spans": [{"start": 0, "end": 5, "entity_type": "LOCATION", "score": .95}],
    }

    result = comparison.parity(rows, reference, candidate)

    assert result["cases"] == 2
    assert result["identical_span_cases"] == 1
    assert result["changed_cases"] == [{"case_id": "changed-spans", "missing": 2, "extra": 1}]
    assert result["matched_spans"] == 1
    assert result["maximum_score_difference_on_matched_spans"] == pytest.approx(.00001)
    assert result["mean_score_difference_on_matched_spans"] == pytest.approx(.00001)


@pytest.mark.parametrize(
    ("dataset", "name_label", "unsupported_label", "typed_unsupported", "typed_location"),
    [("pii", "NAME", "TOKEN", "TOKEN", "UNMAPPED_PRED:LOCATION"),
     ("redmadrobot", "FIRST_NAME", "OMS", "UNMAPPED:OMS", "LOCATION")],
)
def test_public_scores_keep_wrong_types_unsupported_gold_and_negative_cases(
    dataset, name_label, unsupported_label, typed_unsupported, typed_location,
):
    rows = [
        {"key": "positive", "dataset": dataset, "text": "Alice XY12 ZZ34",
         "gold": {(name_label, 0, 5), (unsupported_label, 6, 10), ("SNILS", 11, 15)}},
        {"key": "negative", "dataset": dataset, "text": "word", "gold": set()},
    ]
    spans = {
        "positive": [Span(0, 5, "LOCATION", .9, "fixture"), Span(6, 10, "PERSON", .9, "fixture")],
        "negative": [Span(0, 4, "LOCATION", .9, "fixture")],
    }
    caches = {"candidate": {key: [{"start": s.start, "end": s.end, "entity_type": s.type, "score": s.confidence}
                                 for s in values] for key, values in spans.items()}}

    result = comparison.score_public(rows, {"candidate_hybrid": spans}, caches)
    protection = result["full_masking_all_gold_types"]["systems"]["candidate_hybrid"]
    typed = result["typed_all_types_unfiltered"]["systems"]["candidate_hybrid"]

    # Wrong-type protection counts for masking, but cannot become a typed TP.
    assert protection["cases"] == 2
    assert protection["true_positive"] == 9
    assert protection["false_negative"] == 4
    assert protection["false_positive"] == 4
    assert protection["negative_cases_with_fp"] == 1
    assert typed["exact_span"]["true_positive"] == 0
    assert typed["exact_span"]["false_negative"] == 3
    assert typed["exact_span"]["false_positive"] == 3
    assert typed["exact_span"]["by_type"][typed_unsupported]["false_negative"] == 1
    assert typed["exact_span"]["by_type"][typed_location]["false_positive"] == 2
    assert typed["typed_character"]["true_positive"] == 0
    assert typed["typed_character"]["false_negative"] == 13
    assert typed["typed_character"]["false_positive"] == 13


@pytest.mark.parametrize("change", ["source", "dependency", "input_file", "text", "annotation", "order"])
def test_frozen_protocol_rejects_changes_before_evaluation(monkeypatch, tmp_path, change):
    source = {"adapter.py": "frozen-source"}
    packages = {"onnxruntime-gpu": "frozen-version"}
    rows = [{"key": "a", "text": "Alice", "gold": {("NAME", 0, 5)}},
            {"key": "b", "text": "word", "gold": set()}]
    input_file = tmp_path / "fixture.jsonl"
    input_file.write_text("frozen input bytes", encoding="utf-8")
    protocol = {
        "source_sha256": source.copy(), "versions": packages.copy(),
        "input_sha256": {"fixture": hashlib.sha256(input_file.read_bytes()).hexdigest()},
        "ordered_membership_sha256": comparison.public.digest_json(["a", "b"]),
        "gold_sha256": comparison.public.digest_json({"a": [("NAME", 0, 5)], "b": []}),
        "text_sha256": comparison.public.digest_json({
            "a": hashlib.sha256(b"Alice").hexdigest(), "b": hashlib.sha256(b"word").hexdigest(),
        }),
    }
    monkeypatch.setattr(comparison, "source_hashes", lambda: source)
    monkeypatch.setattr(comparison, "versions", lambda: packages)
    monkeypatch.setattr(comparison, "input_files", lambda _args: {"fixture": input_file})
    monkeypatch.setattr(comparison, "load_inputs", lambda _args: rows)
    args = SimpleNamespace()
    assert comparison.verify(args, protocol) == rows

    if change == "source":
        source["adapter.py"] = "changed-source"
    elif change == "dependency":
        packages["onnxruntime-gpu"] = "changed-version"
    elif change == "input_file":
        input_file.write_text("changed input bytes", encoding="utf-8")
    elif change == "text":
        rows[0]["text"] = "Other"
    elif change == "annotation":
        rows[0]["gold"] = set()
    else:
        rows.reverse()

    with pytest.raises(ValueError):
        comparison.verify(args, protocol)
