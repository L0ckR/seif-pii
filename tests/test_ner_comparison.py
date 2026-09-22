"""Hand-counted model changes, taxonomy boundaries, and immutable input checks."""
import hashlib
import json

import pytest

from scripts.compare_ner_models import compare, compare_cases, person_only
from scripts.evaluate_annotations import evaluate, positions, shape_mask


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def case(case_id, text, entities, *, uncertain=False):
    return {
        "case_id": case_id, "text": text,
        "entities": [{**row, "text": text[row["start"]:row["end"]]} for row in entities],
        "expected_masked": shape_mask(text, positions(text, entities)),
        "traffic_weight": 1, "uncertain": uncertain, "decision": "positive" if entities else "negative",
    }


def prediction(row, entities):
    return {
        "case_id": row["case_id"], "entities": entities,
        "masked": shape_mask(row["text"], positions(row["text"], entities)),
    }


def span(start, end, kind="PERSON"):
    return {"type": kind, "start": start, "end": end}


def write_cache(path, cases, predictions):
    rows = [{
        "case_id": key, "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
        "entities": [{"entity_type": entity["type"], "start": entity["start"],
                      "end": entity["end"], "score": .9} for entity in predictions[key]],
    } for key, row in cases.items()]
    write_rows(path, rows)
    write_json(path.with_suffix(".meta.json"), {"cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


@pytest.fixture
def corpus_files(tmp_path):
    row = case("fictional-name", "Обсудили всё с Зогваром.", [span(15, 23)])
    cases = {row["case_id"]: row}
    dataset = tmp_path / "cases.jsonl"
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_rows(dataset, [row])
    write_json(tmp_path / "manifest.json", {
        "schema_version": 1, "case_count": 1, "cases_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    })
    write_cache(reference, cases, {row["case_id"]: []})
    write_cache(candidate, cases, {row["case_id"]: [span(15, 23)]})
    return dataset, reference, candidate


def test_candidate_cache_changes_real_hybrid_predictions_without_editing_golden(corpus_files):
    dataset, reference, candidate = corpus_files
    before_bytes = {path: path.read_bytes() for path in corpus_files}
    report = compare(dataset, reference, candidate)
    baseline = report["hybrid"]["reference"]["unique_case_primary"]
    actual = report["hybrid"]["candidate"]["unique_case_primary"]
    assert baseline["character_metrics"]["false_negative"] == 8
    assert actual["character_metrics"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "true_positive": 8, "false_positive": 0, "false_negative": 0,
    }
    assert report["rules_control"]["unique_case_primary"] == baseline
    assert report["hybrid_changes"]["became_exact_case_ids"] == ["fictional-name"]
    assert report["hybrid_changes"]["improved_case_ids"] == ["fictional-name"]
    assert report["reference"]["cache_sha256"] != report["candidate"]["cache_sha256"]
    assert report["candidate_minus_reference"]["unique_case_primary"]["exact_cases"] == 1
    assert "Зогваром" not in json.dumps(report, ensure_ascii=False)
    assert all(path.read_bytes() == before_bytes[path] for path in corpus_files)


@pytest.mark.parametrize(("mutation", "message"), [
    ("missing", "complete golden corpus"),
    ("different_id", "complete golden corpus"),
    ("different_text", "different input text"),
    ("duplicate", "Duplicate case IDs"),
    ("corrupt_checksum", "checksum mismatch"),
    ("invalid_offset", "invalid span bounds"),
])
def test_model_comparison_rejects_unrelated_incomplete_or_invalid_cache(corpus_files, mutation, message):
    dataset, reference, candidate = corpus_files
    row = json.loads(candidate.read_text())
    rows = [row]
    if mutation == "missing":
        rows = []
    elif mutation == "different_id":
        row["case_id"] = "another-input"
    elif mutation == "different_text":
        row["text_sha256"] = hashlib.sha256(b"different text").hexdigest()
    elif mutation == "duplicate":
        rows.append(row)
    elif mutation == "invalid_offset":
        row["entities"][0]["end"] = 1000
    write_rows(candidate, rows)
    checksum = "incorrect" if mutation == "corrupt_checksum" else hashlib.sha256(candidate.read_bytes()).hexdigest()
    write_json(candidate.with_suffix(".meta.json"), {"cache_sha256": checksum})
    with pytest.raises(ValueError, match=message):
        compare(dataset, reference, candidate)


def test_case_diagnostics_keep_regressions_tradeoffs_and_type_only_changes_separate():
    rows = [
        case("improve", "АБВ", [span(0, 3)]),
        case("regress", "АБВ", [], uncertain=True),
        case("tradeoff", "АБВ ГДЕ", [span(0, 3)]),
        case("equal-errors", "АБВ", [span(0, 3)]),
        case("type-only", "АБВ", [span(0, 3)]),
    ]
    cases = {row["case_id"]: row for row in rows}
    before_spans = {"improve": [], "regress": [], "tradeoff": [],
                    "equal-errors": [span(0, 2)], "type-only": [span(0, 3)]}
    after_spans = {"improve": [span(0, 3)], "regress": [span(0, 3)], "tradeoff": [span(0, 7)],
                   "equal-errors": [span(1, 3)], "type-only": [span(0, 3, "LOCATION")]}
    before = {key: prediction(row, before_spans[key]) for key, row in cases.items()}
    after = {key: prediction(row, after_spans[key]) for key, row in cases.items()}
    baseline, actual = evaluate(cases, cases, before), evaluate(cases, cases, after)
    changes = compare_cases(baseline, actual, {"reference": before, "candidate": after})
    assert changes["improved_case_ids"] == ["improve"]
    assert changes["regressed_case_ids"] == ["regress"]
    assert changes["mixed_tradeoff_case_ids"] == ["tradeoff"]
    assert changes["equal_error_counts_case_ids"] == ["equal-errors", "type-only"]
    assert changes["entity_only_changed_case_ids"] == ["type-only"]
    assert changes["lost_exact_case_ids"] == ["regress"]
    assert changes["became_exact_case_ids"] == ["improve"]
    assert changes["counts"]["mask_changed"] == 4
    assert actual["unique_case_primary"]["false_positive_negative_cases"] == 1
    assert actual["certain_cases_sensitivity"]["false_positive_negative_cases"] == 0
    assert actual["typed_character_secondary"]["false_positive"] == 9


def test_raw_person_secondary_maps_cardholders_but_does_not_invent_location_labels():
    rows = [case("holder", "АБВ ГДЕ", [span(0, 3, "CARDHOLDER")]),
            case("address", "АБВ", [span(0, 3, "ADDRESS")], uncertain=True)]
    cases = {row["case_id"]: row for row in rows}
    cache = {
        "holder": {"entities": [{"entity_type": "PERSON", "start": 0, "end": 3},
                                {"entity_type": "LOCATION", "start": 4, "end": 7}]},
        "address": {"entities": [{"entity_type": "PERSON", "start": 0, "end": 3}]},
    }
    result = person_only(cases, cache)
    assert result["unique_case_primary"]["character_metrics"] == {
        "precision": .5, "recall": 1.0, "f1": .666667,
        "true_positive": 3, "false_positive": 3, "false_negative": 0,
    }
    assert result["certain_cases_sensitivity"]["character_metrics"]["f1"] == 1.0
    assert result["gold_types_mapped_to_PERSON"] == ["CARDHOLDER", "PERSON"]
    assert result["candidate_entities_by_type"] == {"LOCATION": 1, "PERSON": 2}
    assert result["location_metrics"] is None


def test_same_model_control_has_no_prediction_or_metric_changes(corpus_files):
    dataset, reference, _ = corpus_files
    report = compare(dataset, reference, reference)
    assert report["hybrid"]["reference"] == report["hybrid"]["candidate"]
    assert report["hybrid_changes"]["prediction_changed_case_ids"] == []
    assert all(value == 0 for group in report["candidate_minus_reference"].values() for value in group.values())
