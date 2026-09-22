"""Hand-counted fixtures verify independent annotation scoring semantics."""
import pytest

from scripts.evaluate_annotations import evaluate, validate


def test_character_scoring_preserves_negatives_and_exposes_partial_leakage():
    inputs = {"a": {"text": "АБ-12"}, "b": {"text": "Нет"}}
    labels = {
        "a": {"decision": "positive", "uncertain": False, "entities": [
            {"type": "PASSPORT", "start": 0, "end": 5, "text": "АБ-12"}]},
        "b": {"decision": "negative", "uncertain": True, "entities": []},
    }
    predictions = {
        "a": {"masked": "**-12", "entities": [{"type": "PASSPORT", "start": 0, "end": 2}]},
        "b": {"masked": "*ет", "entities": [{"type": "PERSON", "start": 0, "end": 1}]},
    }
    result = evaluate(inputs, labels, predictions, {"a": 2, "b": 3})
    metrics = result["unique_case_primary"]["character_metrics"]
    assert (metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]) == (2, 1, 2)
    assert result["unique_case_primary"]["fully_protected_positive_rate"] == 0
    assert result["unique_case_primary"]["false_positive_negative_rate"] == 1
    assert result["gold_type_coverage"]["PASSPORT"]["character_recall"] == .5
    assert result["traffic_weighted_secondary"]["character_metrics"]["false_positive"] == 3
    assert result["certain_cases_sensitivity"]["cases"] == 1


def test_missing_cases_and_wrong_frozen_masks_are_rejected():
    inputs = {"a": {"text": "АБ"}}
    labels = {"a": {"decision": "negative", "uncertain": False, "entities": []}}
    with pytest.raises(ValueError, match="complete case set"):
        evaluate(inputs, labels, {})
    with pytest.raises(ValueError, match="actually observed mask"):
        evaluate(inputs, labels, {"a": {"entities": [], "masked": "**"}})


def test_unicode_codepoints_and_nonoverlap_are_enforced():
    text = "😊 Имя"
    validate(text, [{"type": "PERSON", "start": 2, "end": 5, "text": "Имя"}], annotation=True)
    with pytest.raises(ValueError, match="substring"):
        validate(text, [{"type": "PERSON", "start": 1, "end": 4, "text": "Имя"}], annotation=True)
    with pytest.raises(ValueError, match="overlapping"):
        validate(text, [{"type": "PERSON", "start": 2, "end": 5},
                        {"type": "CITY", "start": 4, "end": 5}])
