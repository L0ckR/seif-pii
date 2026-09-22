"""Hand-counted checks for immutable-corpus evaluation and regression gates."""
import copy
import hashlib
import json

import pytest

from scripts.evaluate_golden import compare_quality, load_cases, load_ner_cache, predict, save_json


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_golden_loader_rejects_changed_text_or_label_mask(tmp_path):
    row = {"case_id": "one", "text": "АБ-12", "entities": [
        {"type": "ID", "start": 0, "end": 5, "text": "АБ-12"}],
        "expected_masked": "**-**", "traffic_weight": 1}
    data = tmp_path / "cases.jsonl"
    dump(data, row)
    manifest = {"schema_version": 1, "case_count": 1,
                "cases_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    dump(tmp_path / "manifest.json", manifest)
    assert load_cases(data)["one"]["text"] == "АБ-12"
    row["expected_masked"] = "АБ-12"
    dump(data, row)
    with pytest.raises(ValueError, match="checksum"):
        load_cases(data)
    manifest["cases_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    dump(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="mask"):
        load_cases(data)


def test_model_cache_cannot_be_applied_to_different_text(tmp_path):
    data = tmp_path / "ner-cache.jsonl"
    row = {"case_id": "one", "text_sha256": hashlib.sha256(b"old").hexdigest(), "entities": []}
    dump(data, row)
    dump(data.with_suffix(".meta.json"), {"cache_sha256": hashlib.sha256(data.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="different input"):
        load_ner_cache(data, {"one": {"text": "new"}})


def test_inference_never_uses_expected_labels():
    case = {"text": "hello@example.invalid", "entities": [], "expected_masked": "wrong label"}
    prediction = predict({"one": case}, profile="rules")["one"]
    assert prediction["entities"][0]["type"] == "EMAIL"
    assert prediction["masked"] != case["text"]


def baseline():
    group = {"character_metrics": {"precision": .8, "recall": .7, "f1": .75},
             "false_positive_negative_cases": 2}
    return {"dataset_sha256": "fixed", "profiles": {"hybrid": {
        "unique_case_primary": copy.deepcopy(group), "certain_cases_sensitivity": copy.deepcopy(group)}}}


def test_quality_gate_exposes_tradeoffs_and_rejects_different_labels():
    reference = baseline()
    current = copy.deepcopy(reference)
    current["profiles"]["hybrid"]["unique_case_primary"]["character_metrics"]["recall"] = .69
    assert compare_quality(current, reference) == ["hybrid.unique_case_primary.recall"]
    current["dataset_sha256"] = "edited labels"
    with pytest.raises(ValueError, match="different golden"):
        compare_quality(current, reference)


def test_reports_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "quality.json"
    save_json(path, {"f1": .7})
    with pytest.raises(FileExistsError):
        save_json(path, {"f1": 1})
    assert json.loads(path.read_text()) == {"f1": .7}


def test_quality_gate_rejects_different_model_predictions():
    reference = baseline()
    current = copy.deepcopy(reference)
    current["ner_cache_sha256"] = "different model output"
    with pytest.raises(ValueError, match="different NER"):
        compare_quality(current, reference)
