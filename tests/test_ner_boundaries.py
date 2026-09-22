"""Coverage-preserving model boundaries, independent of benchmark examples."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seif.app import create_app
from seif.config import Policy, Settings
from seif.detector import Span
from seif.ner_boundaries import preserve_model_boundaries
from seif.transform import mask, restore_exact, restore_tokens


def span(text, value, kind="PERSON", reason="rule", confidence=.97):
    start = text.index(value)
    return Span(start, start + len(value), kind, confidence, reason)


def test_full_name_components_preserve_unicode_mask_and_rule_semantics():
    text = "🔒 Дина\tМарковна\nШтольц; конец."
    original = span(text, "Дина\tМарковна\nШтольц")
    parts = [span(text, word, confidence=.8, reason="ner-person") for word in ("Дина", "Марковна", "Штольц")]
    result = preserve_model_boundaries(text, [original], list(reversed(parts)))
    assert [text[s.start:s.end] for s in result] == ["Дина", "Марковна", "Штольц"]
    assert all((s.type, s.confidence, s.reason) == (original.type, original.confidence, original.reason) for s in result)
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]
    assert preserve_model_boundaries(text, result, parts) == result


@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_refined_spans_restore_exact_text_and_token_reply(mode):
    text = "🔒 José, е\u0301; конец"
    original = span(text, "José, е\u0301")
    parts = [span(text, "José"), span(text, "е\u0301")]
    result = preserve_model_boundaries(text, [original], parts)
    masked, replacements = mask(text, result, mode)
    record = {"masked": masked, "replacements": replacements}
    assert restore_exact(record) == text
    if mode == "token":
        assert restore_tokens("Ответ: " + masked, record) == "Ответ: " + text


@pytest.mark.parametrize("protected", ["Дина Марковна", "Марковна Штольц", "Дина Штольц"])
def test_incomplete_model_name_cannot_drop_rule_protected_letters(protected):
    text = "Дина Марковна Штольц"
    original = span(text, text)
    parts = [span(text, word) for word in protected.split()]
    result = preserve_model_boundaries(text, [original], parts)
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]
    assert [text[s.start:s.end] for s in result] == text.split()


def test_address_markers_and_numeric_suffixes_cannot_disappear():
    text = "г. Макетный, ул. Полевая, д. 17Б"
    original = span(text, text, "ADDRESS")
    incomplete = [span(text, "Макетный", "LOCATION"), span(text, "Полевая", "LOCATION"), span(text, "17", "LOCATION")]
    assert preserve_model_boundaries(text, [original], incomplete) == [original]
    complete = [span(text, value, "LOCATION") for value in ("г. Макетный", "ул. Полевая", "д. 17Б")]
    result = preserve_model_boundaries(text, [original], complete)
    assert len(result) == 3 and {s.type for s in result} == {"ADDRESS"}
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]


def test_unmodeled_address_designators_and_house_suffix_remain_protected():
    text = "г. Макетный, ул. Полевая, д. 17Б"
    original = span(text, text, "ADDRESS")
    parts = [span(text, value, "LOCATION") for value in ("Макетный", "Полевая", "17Б")]
    result = preserve_model_boundaries(text, [original], parts)
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]
    assert [text[s.start:s.end] for s in result] == ["г.", "Макетный", ", ул.", "Полевая", ", д.", "17Б"]
    assert {s.type for s in result} == {"ADDRESS"}


@pytest.mark.parametrize("text,part", [("Марковна", "Мар"), ("Штольц", "тольц"), ("е\u0301", "е")])
def test_model_cannot_partition_inside_letters_or_unicode_combining_marks(text, part):
    original = span(text, text)
    assert preserve_model_boundaries(text, [original], [span(text, part)]) == [original]


@pytest.mark.parametrize("parts", [
    [Span(0, 4, "PERSON"), Span(3, 11, "PERSON")],
    [Span(0, 11, "PERSON"), Span(0, 4, "PERSON")],
    [Span(0, 4, "PERSON"), Span(5, 11, "LOCATION")],
    [Span(0, 4, "PERSON"), Span(5, 12, "PERSON")],
])
def test_overlapping_crossing_or_wrong_family_candidates_keep_original(parts):
    text = "Дина Штольц!"
    original = Span(0, 11, "PERSON")
    assert preserve_model_boundaries(text, [original], parts) == [original]


def test_custom_rule_boundaries_are_authoritative():
    text = "Дина Штольц"
    original = span(text, text, reason="custom-rule")
    parts = [span(text, value) for value in text.split()]
    assert preserve_model_boundaries(text, [original], parts) == [original]


def test_model_cannot_extend_into_neighboring_protected_span():
    text = "Дина 1234"
    originals = [span(text, "Дина"), span(text, "1234", "PIN")]
    assert preserve_model_boundaries(text, originals, [span(text, text)]) == originals


def test_empty_models_and_matching_boundaries_keep_original_values():
    text = "Дина"
    original = span(text, text)
    assert preserve_model_boundaries(text, [original], []) == [original]
    assert preserve_model_boundaries(text, [], [original]) == []
    assert preserve_model_boundaries(text, [original], [original]) == [original]


@pytest.mark.parametrize("mode", ["mask", "token"])
def test_api_name_components_mask_and_restore_full_payload(monkeypatch, mode):
    text = "🔒 ФИО: Дина Марковна Штольц."
    parts = [span(text, word, reason="model") for word in ("Дина", "Марковна", "Штольц")]

    class FakeNer:
        def __init__(self, *_args):
            pass

        async def health(self):
            pass

        async def close(self):
            pass

        async def detect(self, value):
            assert value == text
            return parts

    monkeypatch.setattr("seif.app.NerClient", FakeNer)
    settings = Settings(demo=True, ner_url="http://ner.internal", ner_token="test",
                        policies={"demo": Policy(mode=mode)})
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/mask", json={"payload": text, "payload_id": "name-components"})
        assert response.status_code == 200
        masked = response.json()["result"]
        assert all(value not in masked for value in ("Дина", "Марковна", "Штольц"))
        if mode == "mask":
            assert masked == "🔒 ФИО: **** ******** ******."
        else:
            assert masked.count("⟦PD:PERSON:") == 3
        restored = client.post("/v1/unmask", json={"payload": masked, "payload_id": "name-components"})
        assert restored.status_code == 200 and restored.json()["result"] == text
