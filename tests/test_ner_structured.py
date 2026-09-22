"""Synthetic regressions for structured NER candidates and compatible overlaps."""

import pytest

from seif.detector import Span, detect, merge_ner_candidates
from seif.transform import mask, restore_exact


def candidate(text, value, kind, reason="external"):
    start = text.index(value)
    return Span(start, start + len(value), kind, 0.7, reason)


def values(text, spans):
    return [(span.type, text[span.start:span.end]) for span in spans]


@pytest.mark.parametrize(
    "kind,value,prefix",
    [
        ("EMAIL", "a . serov@example . net", "Почта: "),
        ("PHONE", "+44 (20) 7946 0958", "Телефон: "),
        ("CARD", "1234 - 5678 - 9012 - 3456", "Номер карты Visa: "),
        ("INN", "123456789012", "ИНН клиента: "),
        ("PASSPORT", "12 34 567890", "Паспорт: "),
        ("DRIVER_LICENSE", "12АБ345678", "Водительское удостоверение: "),
        ("SNILS", "123-456-789 01", "СНИЛС: "),
        ("OMS", "1234 5678 9012 3456", "Полис ОМС: "),
        ("IP_ADDRESS", "192 . 0 . 2 . 1", "Адрес соединения: "),
        ("IP_ADDRESS", "2001:db8::1", "Адрес соединения: "),
        ("URL", "https://example.net/account?id=42", "Ссылка: "),
        ("MILITARY_ID", "АБ 1234567", "Военный билет: "),
        ("BIRTH_CERTIFICATE", "IV-АБ 123456", "Свидетельство о рождении: "),
    ],
)
def test_all_structured_types_keep_original_offsets_and_restore(kind, value, prefix):
    text = "🔐 " + prefix + value + "."
    spans = merge_ner_candidates(text, [], [candidate(text, value, kind)])
    assert values(text, spans) == [(kind, value)]
    assert spans[0].reason == "ner-structured"
    masked, replacements = mask(text, spans, "mask")
    assert restore_exact({"masked": masked, "replacements": replacements}) == text


@pytest.mark.parametrize(
    "kind,value,prefix",
    [
        ("INN", "123456789012", "Значение: "),
        ("CARD", "1234567890123456", "Значение: "),
        ("PHONE", "1234567890", "Паспорт: "),
        ("PHONE", "79991112233", "Номер заказа: "),
        ("PASSPORT", "1234567890", "Номер заказа: "),
        ("INN", "1234567890", "ИНН организации: "),
        ("IP_ADDRESS", "999.0.0.1", "Адрес соединения: "),
        ("EMAIL", "a@example", "Почта: "),
        ("SNILS", "12345", "СНИЛС: "),
        ("OMS", "12345", "Полис ОМС: "),
        ("URL", "обычное слово", "Ссылка: "),
    ],
)
def test_structured_model_label_cannot_bypass_format_or_field_policy(kind, value, prefix):
    text = prefix + value
    assert merge_ner_candidates(text, [], [candidate(text, value, kind)]) == []


def test_structured_model_priority_never_overrides_explicit_rule_even_with_high_score():
    text = "Код: 1234567890"
    base = [Span(5, 15, "FOREIGN_DOCUMENT", 0.8, "custom-rule")]
    model = [Span(5, 15, "PASSPORT", 1.0, "external")]
    assert merge_ner_candidates(text, base, model) == base


def test_corporate_inn_filter_also_applies_to_spaced_model_values():
    text = 'Организация ООО «Макет»: ИНН: 12 34 56 78 90'
    assert merge_ner_candidates(text, [], [candidate(text, "12 34 56 78 90", "INN")]) == []


@pytest.mark.parametrize("prefix", ["Человек по имени ", "Здесь жила женщина по имени: "])
def test_by_the_name_of_is_personal_without_changing_public_institution_policy(prefix):
    text = prefix + "Дина Марковна"
    model = candidate(text, "Дина Марковна", "PERSON")
    assert values(text, merge_ner_candidates(text, [], [model])) == [("PERSON", "Дина Марковна")]
    public = "Университет имени Дины Марковны"
    assert merge_ner_candidates(public, [], [candidate(public, "Дины Марковны", "PERSON")]) == []


@pytest.mark.parametrize("value", ["79991112233.5", "+79991112233.5", "+447911123456.25", "0.79991112233"])
def test_phone_rules_reject_fractional_numbers(value):
    assert not [span for span in detect("Значение: " + value) if span.type == "PHONE"]


@pytest.mark.parametrize("value", ["79991112233", "+7.999.111.22.33", "+44 (20) 7946 0958"])
def test_phone_rules_keep_sentence_ending_dot(value):
    text = "Телефон: " + value + ". Следующая запись."
    assert values(text, [s for s in detect(text) if s.type == "PHONE"]) == [("PHONE", value)]


def test_phone_ner_cannot_restore_decimal_fragment_removed_from_rules():
    text = "Значение: 79991112233.5"
    assert merge_ner_candidates(text, [], [candidate(text, "79991112233", "PHONE")]) == []


@pytest.mark.parametrize("join", [",", "."])
def test_adjacent_complete_phone_numbers_remain_protected(join):
    first, second = "79991112233", "79994445566"
    text = first + join + second
    assert values(text, [s for s in detect(text) if s.type == "PHONE"]) == [("PHONE", first), ("PHONE", second)]


@pytest.mark.parametrize(
    "value,core,kind,expected",
    [
        ("пр-т Примерный", "Примерный", "STREET", [("LOCATION", "пр-т"), ("STREET", "Примерный")]),
        ("Примерная улица", "Примерная", "STREET", [("STREET", "Примерная"), ("LOCATION", "улица")]),
        ("ул. Примерная, д. 18Б", "Примерная", "STREET", [("LOCATION", "ул."), ("STREET", "Примерная"), ("LOCATION", "д. 18Б")]),
        ("г. Неведомск", "Неведомск", "CITY", [("LOCATION", "г."), ("CITY", "Неведомск")]),
    ],
)
def test_short_address_rule_preserves_compatible_model_continuations(value, core, kind, expected):
    text = "Адрес клиента: " + value
    base = [candidate(text, core, kind, "core-field")]
    model = [candidate(text, value, "LOCATION")]
    spans = merge_ner_candidates(text, base, model)
    assert values(text, spans) == expected
    assert next(s for s in spans if s.type == kind).reason == "core-field"
    assert all(first.end <= second.start for first, second in zip(spans, spans[1:], strict=False))
    assert merge_ner_candidates(text, spans, model) == spans


def test_long_location_cannot_extend_custom_rules_or_incompatible_field_types():
    text = "ул. Примерная, д. 18Б"
    for kind, reason in [("PERSON", "core-field"), ("STREET", "custom-rule")]:
        base = [candidate(text, "Примерная", kind, reason)]
        assert merge_ner_candidates(text, base, [candidate(text, text, "LOCATION")]) == base


def test_wider_location_does_not_mask_arbitrary_prose_around_address():
    text = "перед значением после"
    base = [candidate(text, "значением", "STREET", "core-field")]
    assert merge_ner_candidates(text, base, [candidate(text, text, "LOCATION")]) == base
