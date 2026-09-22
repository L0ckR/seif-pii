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
        ("INN", "123456789012", ""),
        ("CARD", "1234567890123456", ""),
        ("PHONE", "1234567890", "Паспорт: "),
        ("PHONE", "79991112233", "Номер заказа: "),
        ("PASSPORT", "1234567890", "Номер заказа: "),
        ("INN", "1234567890", "ИНН организации: "),
        ("IP_ADDRESS", "999.0.0", "Адрес соединения: "),
        ("EMAIL", "a@example", "Почта: "),
        ("SNILS", "обычное слово", "СНИЛС: "),
        ("OMS", "обычное слово", "Полис ОМС: "),
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


@pytest.mark.parametrize("suffix", [".0", ".00", ",000"])
def test_integral_phone_float_exports_preserve_only_integer_digits(suffix):
    number = "79992223344"
    text = number + suffix
    rules = detect(text)
    assert values(text, [span for span in rules if span.type == "PHONE"]) == [("PHONE", number)]
    neural = merge_ner_candidates(text, [], [candidate(text, text, "PHONE")])
    assert values(text, neural) == [("PHONE", number)]


@pytest.mark.parametrize("value", ["1234-567890", "1234-5678-9012", "123456789012.0"])
def test_model_cannot_reinterpret_ambiguous_bare_document_shapes_as_phone(value):
    assert merge_ner_candidates(value, [], [candidate(value, value, "PHONE")]) == []


def test_split_client_role_is_removed_but_same_word_as_explicit_name_is_kept():
    text = "Клиент Дина ждёт"
    neural = [candidate(text, "Клиент", "PERSON"), candidate(text, "Дина", "PERSON")]
    assert values(text, merge_ner_candidates(text, [], neural)) == [("PERSON", "Дина")]
    name = "ФИО: Клиент Дина"
    neural = [candidate(name, "Клиент", "PERSON"), candidate(name, "Дина", "PERSON")]
    assert values(name, merge_ner_candidates(name, [], neural)) == [("PERSON", "Клиент"), ("PERSON", "Дина")]


@pytest.mark.parametrize(
    "text,value,kind",
    [("Оформил зарплатную карту", "зарплатную", "LOCATION"),
     ("Аренда сейфовой ячейки", "сейфовой ячейки", "LOCATION"),
     ("Отдел закупок", "закупок", "PERSON")],
)
def test_nonpersonal_financial_products_and_department_names_are_contextual(text, value, kind):
    assert merge_ner_candidates(text, [], [candidate(text, value, kind)]) == []


def test_product_or_department_word_can_still_be_a_name_in_explicit_personal_field():
    text = "ФИО: Закупок; адрес: Сейфовая"
    result = merge_ner_candidates(text, [], [candidate(text, "Закупок", "PERSON"), candidate(text, "Сейфовая", "LOCATION")])
    assert values(text, result) == [("PERSON", "Закупок"), ("LOCATION", "Сейфовая")]


@pytest.mark.parametrize(
    "kind,value",
    [("PASSPORT", "12 34"), ("DRIVER_LICENSE", "34 56"),
     ("BIRTH_CERTIFICATE", "IV – АБ"), ("MILITARY_ID", "АБ"),
     ("SNILS", "42"), ("OMS", "1234|5678|9012|3456")],
)
def test_model_document_components_need_not_be_complete_validated_documents(kind, value):
    text = "Значение: " + value
    assert values(text, merge_ner_candidates(text, [], [candidate(text, value, kind)])) == [(kind, value)]


@pytest.mark.parametrize("value", ["192.0.2.1", "192 . 0 . 2 . 1", "2001:db8::1", "::1", "::ffff", "2001 : db8 : 1 : 2 : 3 : 4 : 5 : 6"])
def test_ip_rules_keep_complete_original_value_and_offset(value):
    text = "🔐 IP: " + value + ". Запись."
    result = [span for span in detect(text) if span.type == "IP_ADDRESS"]
    assert values(text, result) == [("IP_ADDRESS", value)]


@pytest.mark.parametrize("value", ["999.1.2.3", "12.03.2026", "12:34:56", "1.2.3.4.5", "12345678", "::", ":::1"])
def test_ip_rules_require_address_syntax_and_validity(value):
    assert not [span for span in detect("Значение: " + value) if span.type == "IP_ADDRESS"]


def test_ipv6_model_fragment_is_validated_in_containing_address():
    text = "IP: 2001 : db8 : abcd : 2 : 3 : 4 : 5 : 6"
    result = merge_ner_candidates(text, [], [candidate(text, "abcd", "IP_ADDRESS")])
    assert values(text, result) == [("IP_ADDRESS", "abcd")]


def test_mistyped_ip_remains_sensitive_when_the_model_recognizes_it():
    text = "IP: 999.0.2.1"
    assert values(text, merge_ner_candidates(text, [], [candidate(text, "999.0.2.1", "IP_ADDRESS")])) == [("IP_ADDRESS", "999.0.2.1")]


@pytest.mark.parametrize("value", ["123-456-789 42", "123:456:789:42", "123_456_789_42"])
def test_explicit_snils_is_protected_even_with_mistyped_checksum(value):
    text = "СНИЛС: " + value
    assert values(text, [span for span in detect(text) if span.type == "SNILS"]) == [("SNILS", value)]


def test_snils_field_never_extracts_eleven_digit_prefix_of_longer_identifier():
    assert not [span for span in detect("СНИЛС: 123-456-789-421") if span.type == "SNILS"]


def test_electronic_registration_does_not_borrow_an_address_hint_from_later_sentence():
    text = "Клиент зарегистрирован в приложении. Его заказ 123456 подтверждён. IP: 192.0.2.1."
    assert not [span for span in detect(text) if span.type == "ADDRESS"]
    assert values(text, [span for span in detect(text) if span.type == "IP_ADDRESS"]) == [("IP_ADDRESS", "192.0.2.1")]


def test_address_sentence_boundary_retains_abbreviations_and_named_street_initial():
    text = "Адрес: г. Псков, ул. С. Разина, д. 14. Следующая запись: 123456."
    address = [span for span in detect(text) if span.type == "ADDRESS"]
    assert values(text, address) == [("ADDRESS", "г. Псков, ул. С. Разина, д. 14")]


def test_explicit_inn_fragment_with_typo_is_sensitive_but_bare_number_stays_ambiguous():
    value = "123456789"
    text = "ИНН клиента: " + value
    assert values(text, merge_ner_candidates(text, [], [candidate(text, value, "INN")])) == [("INN", value)]
    assert merge_ner_candidates(value, [], [candidate(value, value, "INN")]) == []


def test_birth_certificate_keeps_precise_class_with_model_components():
    text = "Свидетельство о рождении: IV-АБ 123456"
    spans = merge_ner_candidates(text, detect(text), [candidate(text, "IV-АБ", "BIRTH_CERTIFICATE"), candidate(text, "123456", "BIRTH_CERTIFICATE")])
    assert values(text, spans) == [("BIRTH_CERTIFICATE", "IV-АБ"), ("BIRTH_CERTIFICATE", "123456")]
