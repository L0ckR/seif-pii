"""Check diagnostic attribution without model loading or corpus fixtures."""
# Ruff's project-wide test exemption covers tests/, while this replay test lives
# beside the benchmark artifact. Assertions here are intentional pytest checks.
# ruff: noqa: S101
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def analysis():
    path = Path(__file__).with_name("rule_analysis.py")
    spec = importlib.util.spec_from_file_location("rule_analysis_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native(text, value, label):
    start = text.index(value)
    return {"start": start, "end": start + len(value), "label": label, "text": value, "score": .9}


def test_union_geometry_handles_overlaps_without_duplicate_masking(analysis):
    spans = [analysis.detector.Span(0, 3, "A"), analysis.detector.Span(2, 5, "B")]
    assert analysis.geometry("A B C", spans) == {0, 2, 4}


def test_projection_loss_is_separate_from_context_filtering(analysis):
    text = "СНИЛС: 111-222-333 44"
    raw = [native(text, "111-222-333 44", "SNILS")]
    _, _, positions, _ = analysis.replay(text, raw, [])
    assert len(positions["native21"]) == 11
    assert positions["native_and_rules_union"] - positions["gateway_and_rules_union"] == positions["native21"]
    assert positions["gateway_and_rules_union"] == positions["service_hybrid"]


def test_priority_rejection_is_attributed_after_normalization(analysis):
    text = "пр-т Примерная"
    raw = [native(text, text, "STREET")]
    gateway = analysis.gateway_entities(text, raw)
    _, _, positions, reasons = analysis.replay(text, raw, gateway)
    assert positions["normalized_gateway_and_rules"] == positions["gateway_and_rules_union"]
    lost = positions["refined_gateway_and_rules"] - positions["service_hybrid"]
    assert lost == {0, 1, 3}
    assert any(key.startswith("priority:LOCATION:ner-location -> STREET:") and value == lost
               for key, value in reasons.items())


def test_personal_by_name_collision_is_visible_as_normalization_loss(analysis):
    text = "Жил человек по имени Алёна Никитична."
    raw = [native(text, "Алёна", "FIRST_NAME"), native(text, "Никитична", "MIDDLE_NAME")]
    gateway = analysis.gateway_entities(text, raw)
    _, _, positions, reasons = analysis.replay(text, raw, gateway)
    lost = positions["gateway_and_rules_union"] - positions["normalized_gateway_and_rules"]
    assert len(lost) == 14
    assert lost == reasons["public_name_context"]
    assert positions["normalized_gateway_and_rules"] == positions["service_hybrid"]


def test_report_retains_every_gold_type_and_counts_scope_loss(analysis):
    text = "СНИЛС: 111-222-333 44"
    item = native(text, "111-222-333 44", "SNILS")
    row = {"key": "test/one", "dataset": "test", "text": text,
           "gold": [("SNILS", item["start"], item["end"])]}
    report = analysis.analyze([row], {row["key"]: {"native": [item], "entities": []}})
    assert report["systems"]["native21"]["tp"] == 11
    assert report["systems"]["service_hybrid"]["fn"] == 11
    assert report["pipeline_removed_characters_by_stage"]["excluded_native_types"]["gold_characters"] == 11
    assert report["pipeline_removed_characters_by_stage"]["shared_context_refinement"]["gold_characters"] == 0
    assert report["rule_false_negative_characters_by_gold_type"]["SNILS"]["characters"] == 11
