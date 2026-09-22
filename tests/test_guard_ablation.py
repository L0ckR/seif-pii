"""Guard-cache ablation invariants without importing upstream/ML dependencies."""
import importlib
import json
from types import SimpleNamespace

import pytest

bench = importlib.import_module("benchmarks.ner-models.pii-guard-review.guard_ablation")


def test_all_mapped_native_labels_retained_fixed_priority_not_threshold():
    rows = [{"start": 0, "end": 1, "label": label, "score": .01} for label in ("FIRST_NAME", "SNILS", "URL")]
    results, counts = bench.converted_ner("x", rows, {"FIRST_NAME": "PERSON", "SNILS": "SNILS", "URL": "URL"},
                                         .7, SimpleNamespace, lambda values, _: values)
    assert [r.entity_type for r in results] == ["PERSON", "SNILS", "URL"]
    assert all(r.score == .7 for r in results)
    assert counts["mapped:SNILS"] == 1


def test_unknown_native_label_counted_and_email_gate_called():
    seen = []

    def gate(results, text):
        seen.append((results, text))
        return []

    rows = [{"start": 0, "end": 1, "label": label} for label in ("EMAIL", "UNSUPPORTED")]
    results, counts = bench.converted_ner("я", rows, {"EMAIL": "EMAIL_ADDRESS"}, .7, SimpleNamespace, gate)
    assert not results and seen[0][1] == "я"
    assert counts["email_shape_discarded"] == counts["unmapped_native:UNSUPPORTED"] == 1


def test_sort_original_unicode_offsets_and_only_explicit_aliases():
    results = [SimpleNamespace(start=2, end=3, entity_type="DATE_TIME", score=.8),
               SimpleNamespace(start=0, end=1, entity_type="EMAIL_ADDRESS", score=.9)]
    spans = bench.safe_spans(results, "я —")
    assert [(s.start, s.end, s.type) for s in spans] == [(0, 1, "EMAIL"), (2, 3, "DATE_TIME")]
    results.append(SimpleNamespace(start=0, end=3, entity_type="PERSON", score=.7))
    with pytest.raises(ValueError):
        bench.safe_spans(results, "я —")


def test_missing_gold_type_not_dropped_from_fullmask_or_typed_scoring():
    rows = [{"key": "x", "dataset": "pii", "text": "abc 123", "gold": [("SNILS", 4, 7)]}]
    predictions = {"empty": {"x": []}, "literal": {"x": [bench.Span(4, 7, "SNILS", .7, "test")]}}
    score = bench.evaluate_corpus(rows, predictions)
    systems = score["full_masking_all_gold_types"]["systems"]
    assert systems["empty"]["false_negative"] == 3 and systems["literal"]["false_negative"] == 0
    typed = score["typed_all_types_unfiltered"]["systems"]
    assert typed["empty"]["exact_span"]["false_negative"] == 1
    assert typed["literal"]["exact_span"]["true_positive"] == 1


def test_cache_coverage_failure_and_nonfinite_score_fail_closed(tmp_path):
    path = tmp_path / "cache.jsonl"
    rows = [{"key": "x", "text": "abc"}]
    record = {"case_id": "x", "text_sha256": bench.rubert.text_hash("abc"),
              "native": [{"start": 0, "end": 1, "label": "FIRST_NAME", "score": .9}]}

    def save(value):
        path.write_text(json.dumps(value) + "\n")
        path.with_suffix(".meta.json").write_text(json.dumps({
            "cache_sha256": bench.digest(path), "model_revision": bench.rubert.REVISION,
        }))

    save(record)
    assert bench.load_native(path, rows)[0]["x"]["native"] == record["native"]
    save({**record, "case_id": "different"})
    with pytest.raises(ValueError):
        bench.load_native(path, rows)
    save({**record, "inference_error": "RuntimeError"})
    with pytest.raises(ValueError):
        bench.load_native(path, rows)
    record["native"][0]["label"] = "UNRECOGNIZED"
    save(record)
    with pytest.raises(ValueError):
        bench.load_native(path, rows)
    record["native"][0]["label"] = "FIRST_NAME"
    record["native"][0]["score"] = float("nan")
    save(record)
    with pytest.raises(ValueError):
        bench.load_native(path, rows)
