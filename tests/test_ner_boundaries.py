"""Coverage-preserving model boundaries, independent of benchmark examples."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seif.app import create_app
from seif.config import Policy, Settings
from seif.detector import Span, detect, merge_ner_candidates
from seif.ner_boundaries import preserve_model_boundaries
from seif.transform import mask, restore_exact, restore_tokens


def span(text, value, kind="PERSON", reason="rule", confidence=.97):
    start = text.index(value)
    return Span(start, start + len(value), kind, confidence, reason)


def test_full_name_stays_one_entity_despite_model_components():
    text = "🔒 Дина\tМарковна\nШтольц; конец."
    original = span(text, "Дина\tМарковна\nШтольц")
    parts = [span(text, word, confidence=.8, reason="ner-person") for word in ("Дина", "Марковна", "Штольц")]
    result = preserve_model_boundaries(text, [original], list(reversed(parts)))
    assert result == [original]
    assert all((s.type, s.confidence, s.reason) == (original.type, original.confidence, original.reason) for s in result)
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]
    assert preserve_model_boundaries(text, result, parts) == result


@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_refined_spans_restore_exact_text_and_token_reply(mode):
    text = "🔒 José, е\u0301; конец"
    original = span(text, "José, е\u0301", "LOCATION")
    parts = [span(text, "José", "LOCATION"), span(text, "е\u0301", "LOCATION")]
    result = preserve_model_boundaries(text, [original], parts)
    assert result == parts
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
    assert result == [original]


def test_address_markers_and_numeric_suffixes_cannot_disappear():
    text = "г. Макетный, ул. Полевая, д. 17Б"
    original = span(text, text, "ADDRESS")
    incomplete = [span(text, "Макетный", "LOCATION"), span(text, "Полевая", "LOCATION"), span(text, "17", "LOCATION")]
    assert preserve_model_boundaries(text, [original], incomplete) == [original]
    complete = [span(text, value, "LOCATION") for value in ("г. Макетный", "ул. Полевая", "д. 17Б")]
    result = preserve_model_boundaries(text, [original], complete)
    assert result == [original]
    assert {s.type for s in result} == {"ADDRESS"}
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]


def test_unmodeled_address_designators_and_house_suffix_remain_protected():
    text = "г. Макетный, ул. Полевая, д. 17Б"
    original = span(text, text, "ADDRESS")
    parts = [span(text, value, "LOCATION") for value in ("Макетный", "Полевая", "17Б")]
    result = preserve_model_boundaries(text, [original], parts)
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]
    assert result == [original]
    assert {s.type for s in result} == {"ADDRESS"}


@pytest.mark.parametrize("mode", ["token", "synthetic"])
def test_complete_personal_address_is_replaced_once(mode):
    value = "г. Москва, ул. Лесная, д. 12, кв. 34"
    text = "Адрес проживания: " + value
    parts = [span(text, word, "LOCATION", "model") for word in ("Москва", "ул. Лесная", "д. 12, кв. 34")]
    result = merge_ner_candidates(text, detect(text), parts)
    assert [(s.type, text[s.start:s.end]) for s in result] == [("ADDRESS", value)]
    masked, replacements = mask(text, result, mode)
    assert len(replacements) == 1
    assert restore_exact({"masked": masked, "replacements": replacements}) == text
    if mode == "synthetic":
        assert masked == "Адрес проживания: г. Макетный, ул. Тестовая, д. 1"


@pytest.mark.parametrize("text,part", [("Марковна", "Мар"), ("Штольц", "тольц"), ("е\u0301", "е")])
def test_model_cannot_partition_inside_letters_or_unicode_combining_marks(text, part):
    original = span(text, text, "LOCATION")
    assert preserve_model_boundaries(text, [original], [span(text, part, "LOCATION")]) == [original]


@pytest.mark.parametrize("parts", [
    [Span(0, 4, "LOCATION"), Span(3, 11, "LOCATION")],
    [Span(0, 11, "LOCATION"), Span(0, 4, "LOCATION")],
    [Span(0, 4, "LOCATION"), Span(5, 11, "PERSON")],
    [Span(0, 4, "LOCATION"), Span(5, 12, "LOCATION")],
])
def test_overlapping_crossing_or_wrong_family_candidates_keep_original(parts):
    text = "Дина Штольц!"
    original = Span(0, 11, "LOCATION")
    assert preserve_model_boundaries(text, [original], parts) == [original]


def test_custom_rule_boundaries_are_authoritative():
    text = "Дина Штольц"
    original = span(text, text, "LOCATION", reason="custom-rule")
    parts = [span(text, value, "LOCATION") for value in text.split()]
    assert preserve_model_boundaries(text, [original], parts) == [original]


def test_model_cannot_extend_into_neighboring_protected_span():
    text = "Дина 1234"
    originals = [span(text, "Дина", "LOCATION"), span(text, "1234", "PIN")]
    assert preserve_model_boundaries(text, originals, [span(text, text, "LOCATION")]) == originals


@pytest.mark.parametrize("value,pieces", [
    ("192.0.2.17", ["192", "17"]),
    ("192 . 0 . 2 . 17", ["192", "17"]),
    ("2001:db8:abcd::17", ["2001", "db8", "abcd", "17"]),
    ("2001 : db8 : abcd : : 17", ["2001", "db8", "abcd", "17"]),
    ("::ffff:192.0.2.17", ["ffff", "192", "17"]),
])
def test_complete_valid_ip_stays_one_address_despite_model_fragments(value, pieces):
    text = f"🔒 IP: {value}; конец"
    original = span(text, value, "IP_ADDRESS", "ip-address-format")
    model = [span(text, piece, "IP_ADDRESS", "ner-structured") for piece in pieces]
    result = preserve_model_boundaries(text, [original], model)
    assert result == [original]
    masked, replacements = mask(text, result, "token")
    assert masked.count("⟦PD:IP_ADDRESS:") == 1
    assert restore_exact({"masked": masked, "replacements": replacements}) == text


def test_atomic_ip_exception_does_not_conflate_separate_addresses():
    text = "192.0.2.17 198.51.100.9"
    original = span(text, text, "IP_ADDRESS")
    model = [span(text, value, "IP_ADDRESS") for value in text.split()]
    result = preserve_model_boundaries(text, [original], model)
    assert [text[item.start:item.end] for item in result] == text.split()


def test_atomic_ip_exception_does_not_prevent_document_series_components():
    text = "Паспорт: 4500 123456"
    original = span(text, "4500 123456", "PASSPORT")
    model = [span(text, value, "PASSPORT") for value in ("4500", "123456")]
    result = preserve_model_boundaries(text, [original], model)
    assert [text[item.start:item.end] for item in result] == ["4500", "123456"]
    assert mask(text, result, "mask")[0] == mask(text, [original], "mask")[0]


def test_empty_models_and_matching_boundaries_keep_original_values():
    text = "Дина"
    original = span(text, text)
    assert preserve_model_boundaries(text, [original], []) == [original]
    assert preserve_model_boundaries(text, [], [original]) == []
    assert preserve_model_boundaries(text, [original], [original]) == [original]


@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_api_name_components_mask_and_restore_full_payload(monkeypatch, mode):
    text = "🔒 ФИО: Дина Марковна Штольц."
    parts = [span(text, word, reason="model") for word in ("Дина", "Марковна", "Штольц")]

    class FakeNer:
        def __init__(self, *_args, max_concurrency=4, backend="httpx"):
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
        body = response.json()
        masked = body["result"]
        assert [(entity["type"], text[entity["start"]:entity["end"]]) for entity in body["entities"]] == [
            ("PERSON", "Дина Марковна Штольц"),
        ]
        assert all(value not in masked for value in ("Дина", "Марковна", "Штольц"))
        if mode == "mask":
            assert masked == "🔒 ФИО: **** ******** ******."
        elif mode == "token":
            assert masked.count("⟦PD:PERSON:") == 1
        else:
            assert masked == "🔒 ФИО: Тестов1 Макет Макетович."
        restored = client.post("/v1/unmask", json={"payload": masked, "payload_id": "name-components"})
        assert restored.status_code == 200
        assert restored.json()["result"] == text


@pytest.mark.parametrize("separator", ["; Клиент: ", "\nКлиент: ", ", "])
def test_complete_names_stay_separate_and_tokens_restore_reordered_reply(separator):
    names = ["Иванов Иван Иванович", "Петрова Анна Сергеевна"]
    text = "Клиент: " + separator.join(names)
    parts = []
    cursor = 0
    for name in names:
        for word in name.split():
            start = text.index(word, cursor)
            cursor = start + len(word)
            parts.append(Span(start, cursor, "PERSON", .9, "model"))
    result = merge_ner_candidates(text, detect(text), parts)
    assert [(s.type, text[s.start:s.end]) for s in result] == [("PERSON", name) for name in names]
    masked, replacements = mask(text, result, "token")
    record = {"masked": masked, "replacements": replacements}
    assert len(replacements) == 2
    assert restore_exact(record) == text
    reply = ", ".join(replacements[index]["replacement"] for index in (1, 0, 1))
    assert restore_tokens(reply, record) == ", ".join(names[index] for index in (1, 0, 1))
