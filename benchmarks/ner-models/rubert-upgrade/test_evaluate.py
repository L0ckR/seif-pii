"""Failure handling and metric semantics for the upgrade comparison."""
# ruff: noqa: S101
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("upgrade_evaluate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_overlapping_predictions_are_rejected():
    with pytest.raises(ValueError, match="overlapping"):
        MODULE.validate_spans([{"start": 0, "end": 3, "type": "PERSON"},
                               {"start": 2, "end": 4, "type": "PERSON"}], "Abcd", "type")


def test_nonfinite_confidence_is_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        MODULE.validate_spans([{"start": 0, "end": 1, "type": "PERSON", "score": float("nan")}], "A", "type")


def test_cache_failure_is_not_scored_as_empty_prediction(tmp_path):
    path = tmp_path / "cache.jsonl"
    record = {"case_id": "a", "text_sha256": "unused", "native": [], "inference_error": "RuntimeError"}
    path.write_text(json.dumps(record) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps({"cache_sha256": MODULE.sha(path), "model_revision": "73be581047bf123dac6505e7b3900ec292942296"}))
    with pytest.raises(ValueError, match="Incomplete"):
        MODULE.load_cache(path, [{"key": "a", "text": "abc"}])


def test_unknown_predictions_count_as_false_positives_and_gold_is_not_merged():
    upstream = MODULE.ROOT / "local-data/pii-guard-review/upstream"
    if not upstream.exists():
        pytest.skip("Pinned upstream scorer is a local experiment dependency")
    paper = MODULE.load_paper(SimpleNamespace(upstream=upstream))
    rows = [{"key": "synthetic", "text": "A B 1", "gold": [("FIRST_NAME", 0, 1), ("LAST_NAME", 2, 3)]}]
    span = SimpleNamespace
    predictions = {"system": {"synthetic": [span(type="CARDHOLDER", start=0, end=1),
                                              span(type="PERSON", start=2, end=3),
                                              span(type="UNSUPPORTED_KIND", start=4, end=5)]}}
    scores = MODULE.paper_metrics(paper, rows, predictions)["systems"]["system"]
    assert (scores["raw_strict"]["tp"], scores["raw_strict"]["fp"], scores["raw_strict"]["fn"]) == (2, 1, 0)
    assert (scores["prediction_merge_strict"]["tp"], scores["prediction_merge_strict"]["fp"],
            scores["prediction_merge_strict"]["fn"]) == (0, 2, 2)
    assert scores["raw_strict_common14_prediction_scope"]["excluded_predictions"] == 1
    assert scores["raw_strict_common14_prediction_scope"]["fp"] == 0
    assert rows[0]["gold"] == [("FIRST_NAME", 0, 1), ("LAST_NAME", 2, 3)]


def test_serialized_upstream_spans_are_ordered_for_exact_restore():
    span = SimpleNamespace
    values = [span(start=3, end=4, type="PHONE", confidence=.7, reason="test"),
              span(start=0, end=1, type="PERSON", confidence=.7, reason="test")]
    assert [item["start"] for item in MODULE.encode(values)] == [0, 3]


def test_strict_miss_decomposition_distinguishes_boundary_from_exposure():
    upstream = MODULE.ROOT / "local-data/pii-guard-review/upstream"
    if not upstream.exists():
        pytest.skip("Pinned upstream scorer is a local experiment dependency")
    paper = MODULE.load_paper(SimpleNamespace(upstream=upstream))
    record = paper.Record(id="synthetic", text="A B 12 34", gold=[
        {"type": "PERSON", "start": 0, "end": 1}, {"type": "PERSON", "start": 2, "end": 3},
        {"type": "PHONE", "start": 4, "end": 6}, {"type": "INN", "start": 7, "end": 9},
    ], pred=[{"type": "PERSON", "start": 0, "end": 3}, {"type": "PHONE", "start": 4, "end": 5}])
    counts = MODULE.strict_miss_decomposition(paper, [record])["counts"]
    assert counts == {"fully_masked_same_type_boundary_difference": 2, "partly_masked": 1, "unmasked": 1}


def test_symmetric_merge_preserves_name_granularity_and_out_of_scope_errors():
    upstream = MODULE.ROOT / "local-data/pii-guard-review/upstream"
    if not upstream.exists():
        pytest.skip("Pinned upstream scorer is a local experiment dependency")
    paper = MODULE.load_paper(SimpleNamespace(upstream=upstream))
    public = MODULE.load_modules(MODULE.ROOT)[0]
    span = SimpleNamespace
    rows = [{"key": "synthetic", "text": "A B 12", "gold": [("NAME", 0, 3), ("SNILS", 4, 6)]}]
    predictions = {
        "parts": {"synthetic": [span(type="FIRST_NAME", start=0, end=1), span(type="LAST_NAME", start=2, end=3),
                                 span(type="SNILS", start=4, end=6)]},
        "whole_wrong_type": {"synthetic": [span(type="PERSON", start=0, end=3), span(type="UNSUPPORTED", start=4, end=6)]},
    }
    result = MODULE.shared_typed_metrics(paper, rows, predictions, public)
    assert result["original_boundaries"]["systems"]["parts"]["exact_span"]["f1"] < 1
    assert result["symmetric_whitespace_merge"]["systems"]["parts"]["exact_span"]["f1"] == 1
    assert result["predictions_outside_gold_taxonomy_retained_as_fp"]["whole_wrong_type"] == 1
    assert result["symmetric_whitespace_merge"]["systems"]["whole_wrong_type"]["exact_span"]["false_positive"] == 1
