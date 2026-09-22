"""RuBERT comparisons retain fine labels, unsupported gold, and failed cases."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import compare_rubert as comparison


def test_native_redmad_exact_metrics_do_not_coarsen_name_or_location_labels():
    rows = [{"key": "r", "gold": {("FIRST_NAME", 0, 4), ("LAST_NAME", 5, 11), ("CITY", 12, 16)}}]
    native = {"r": [{"label": "FIRST_NAME", "start": 0, "end": 4},
                    {"label": "LAST_NAME", "start": 5, "end": 11},
                    {"label": "COUNTRY", "start": 12, "end": 16}]}

    result = comparison.native_exact_redmad(rows, native)["systems"]["rubert_native21"]

    assert result["exact_span"]["true_positive"] == 2
    assert result["exact_span"]["false_negative"] == 1
    assert result["exact_span"]["false_positive"] == 1
    assert result["exact_span"]["by_type"]["FIRST_NAME"]["true_positive"] == 1
    assert result["exact_span"]["by_type"]["LAST_NAME"]["true_positive"] == 1
    assert result["typed_character"]["by_type"]["CITY"]["false_negative"] == 4
    assert result["typed_character"]["by_type"]["COUNTRY"]["false_positive"] == 4


@pytest.fixture
def evaluation_fixture(monkeypatch, tmp_path):
    rows = [
        {"key": "organizer/a", "id": "a", "dataset": "organizer", "text": "Alice 1234",
         "gold": {("PERSON", 0, 5), ("CVV", 6, 10)}},
        {"key": "pii/a", "id": "a", "dataset": "pii", "text": "Alice 1234",
         "gold": {("NAME", 0, 5), ("TOKEN", 6, 10)}},
        {"key": "redmadrobot/a", "id": "a", "dataset": "redmadrobot", "text": "Alice 1234",
         "gold": {("FIRST_NAME", 0, 5), ("SNILS", 6, 10)}},
    ]
    candidates = {row["key"]: [{"start": 0, "end": 5, "entity_type": "PERSON", "score": .9}] for row in rows}
    native = {row["key"]: [{"start": 0, "end": 5, "label": "FIRST_NAME", "score": .9}] for row in rows}
    failures = set()
    metadata = {"failures_by_corpus": {}}
    monkeypatch.setattr(comparison, "load_cache", lambda *_args: (candidates, native, metadata, failures))
    monkeypatch.setattr(comparison.reference, "reference_caches", lambda *_args: {"native": candidates})
    cases = {"a": {"text": "Alice 1234", "traffic_weight": 1, "uncertain": False,
                   "entities": [{"type": "PERSON", "start": 0, "end": 5},
                                {"type": "CVV", "start": 6, "end": 10}]}}
    monkeypatch.setattr(comparison.golden, "load_cases", lambda *_args: cases)
    monkeypatch.setattr(comparison, "organizer_scores", lambda *_args: {})
    monkeypatch.setattr(comparison, "verify", lambda *_args: rows)
    exports = []
    monkeypatch.setattr(comparison, "export_organizer", lambda *_args: exports.append(True))
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(run_dir=tmp_path, output=tmp_path / "report.json")
    return SimpleNamespace(rows=rows, candidates=candidates, native=native, failures=failures,
                           metadata=metadata, args=args, exports=exports)


def test_native_and_service_full_masking_retain_all_unsupported_gold(evaluation_fixture):
    fixture = evaluation_fixture
    summary = comparison.evaluate(fixture.args, fixture.rows, {})
    report = json.loads(fixture.args.output.read_text())

    assert summary["cases"] == 3
    assert summary["failed"] == 0
    for dataset in ("organizer", "pii", "redmadrobot"):
        systems = report["corpora"][dataset]["full_masking_all_gold_types"]["systems"]
        for model in ("rubert_native21", "rubert_hybrid", "rubert_person_only"):
            assert systems[model]["cases"] == 1
            assert systems[model]["true_positive"] == 5
            assert systems[model]["false_negative"] == 4
            assert systems[model]["false_positive"] == 0
        assert systems["rubert_native21"] == systems["rubert_hybrid"]
    assert fixture.exports == [True]


def test_any_inference_failure_disables_entire_corpus_quality_and_organizer_export(
    monkeypatch, evaluation_fixture,
):
    fixture = evaluation_fixture
    # Keep a successful organizer case too: it must not become a reduced scored subset.
    second = {**fixture.rows[0], "key": "organizer/b", "id": "b"}
    fixture.rows.insert(1, second)
    fixture.candidates["organizer/b"] = []
    fixture.failures.add("organizer/b")
    fixture.metadata["failures_by_corpus"]["organizer"] = ["organizer/b"]

    def forbidden_score(*_args):
        pytest.fail("Organizer with a failed document must not publish quality or an HTTP reference cache")

    monkeypatch.setattr(comparison, "organizer_scores", forbidden_score)
    monkeypatch.setattr(comparison, "export_organizer", forbidden_score)
    summary = comparison.evaluate(fixture.args, fixture.rows, {})
    report = json.loads(fixture.args.output.read_text())

    assert summary["cases"] == 4
    assert summary["failed"] == 1
    assert report["corpora"]["organizer"] == {
        "cases": 2, "quality_available": False, "failed_case_ids": ["organizer/b"],
    }
    assert report["corpora"]["pii"]["cases"] == 1
    assert report["corpora"]["redmadrobot"]["cases"] == 1
    assert "full_masking_all_gold_types" in report["corpora"]["pii"]


def test_failure_metadata_cannot_hide_an_empty_prediction_from_a_failed_model(monkeypatch, tmp_path):
    records = [{"case_id": "organizer/a", "native": [], "entities": [], "inference_error": "RuntimeError"}]
    (tmp_path / "rubert.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(comparison.public, "load_cache", lambda *_args: (
        {"organizer/a": []}, {"failures_by_corpus": {"organizer": []}},
    ))
    with pytest.raises(ValueError, match="failure provenance"):
        comparison.load_cache(SimpleNamespace(run_dir=tmp_path), [{"key": "organizer/a"}])
