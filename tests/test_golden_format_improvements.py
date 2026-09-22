"""Independent format and ownership regressions for the second golden iteration."""

import pytest

from seif.detector import Span, detect, merge_ner_candidates
from seif.transform import mask, restore_exact


def values(text, kind):
    return [text[s.start:s.end] for s in detect(text) if s.type == kind]


@pytest.mark.parametrize("label", ["ИНН поручителя:", "ИНН клиента", "ИНН:"])
def test_personal_inn_accepts_readable_digit_groups(label):
    text = f"{label} 5623 1098 7654, заявление принято."
    assert values(text, "INN") == ["5623 1098 7654"]


@pytest.mark.parametrize("label", ["Последние 4 цифры карты:", "последние четыре цифры номера банковской карты"])
def test_explicit_card_suffix_is_sensitive_without_claiming_a_complete_pan(label):
    assert values(label + " 6821.", "CARD") == ["6821"]


@pytest.mark.parametrize("text", ["Последние 4 цифры заказа: 6821.", "Последние 4 цифры карты: 68210."])
def test_card_suffix_needs_exact_length_and_card_ownership(text):
    assert values(text, "CARD") == []


def test_driver_licence_does_not_acquire_passport_types():
    text = "ВУ серия 67 23, номер 918263. Дата выдачи 16.08.2019."
    assert values(text, "DRIVER_LICENSE") == ["67 23", "918263"]
    assert values(text, "PASSPORT") == []
    assert values(text, "PASSPORT_DATE") == []


def test_explicit_passport_date_overrides_previous_licence_owner():
    text = "Водительское удостоверение проверено. Дата выдачи паспорта: 16.08.2019."
    assert values(text, "PASSPORT_DATE") == ["16.08.2019"]


def test_inflected_issue_verb_retains_authority_and_separate_date():
    authority = "УМВД России по г. Твери"
    text = f"Предъявлен паспорт, выданный {authority} 16.08.2019."
    assert values(text, "PASSPORT_ISSUER") == [authority]
    assert values(text, "PASSPORT_DATE") == ["16.08.2019"]


def test_repeated_issuer_reference_keeps_explanatory_words_open():
    authority = "УФМС России по Тестовой области"
    text = f"Орган выдачи {authority} совпадает с указанным в анкете: {authority}."
    assert values(text, "PASSPORT_ISSUER") == [authority, authority]


@pytest.mark.parametrize("relative", ["прошлом", "позапрошлом", "этом", "текущем"])
def test_relative_year_stays_a_separate_explicit_passport_date(relative):
    authority = "ОВД «Тестовое» города Твери"
    text = f"Паспорт выдан {authority} в {relative} году."
    assert values(text, "PASSPORT_ISSUER") == [authority]
    assert values(text, "PASSPORT_DATE") == [f"{relative} году"]


@pytest.mark.parametrize("text", [
    "Паспорт оборудования выдан отделом в прошлом году.",
    "Паспорт выдан УФМС. Офис отремонтирован в прошлом году.",
    "В прошлом году я прочитал паспорт оборудования.",
])
def test_relative_issue_date_does_not_inherit_unrelated_ownership(text):
    assert values(text, "PASSPORT_DATE") == []


@pytest.mark.parametrize("address", [
    "с. Тестовое, ул. Полевая, д. 7",
    "Тестовая обл., Примерный г.о., д. Речная, ул. Полевая, д. 7",
    "пгт им. Лескова, мкр. Речной, д. 7, к. 3, кв. 8",
    "Республика Условная, г. Примерово, пр-кт Дальний, д. 7, кв. 8",
    "г. Примерово, Речной пр-кт, д. 7, кв. 8",
])
def test_residential_address_components_keep_whole_record(address):
    assert values(address, "ADDRESS") == [address]
    masked, replacements = mask(address, detect(address), "mask")
    assert restore_exact({"masked": masked, "replacements": replacements}) == address


def test_city_owner_and_postal_inflection_are_not_masked_as_values():
    text = "Город проживания — Тверь. Отправить по индексу 170001."
    assert values(text, "CITY") == ["Тверь"]
    assert values(text, "POSTAL_CODE") == ["170001"]


def test_season_does_not_start_a_birthplace_by_matching_a_letter_v():
    text = "Я родился весной. В Твери распускались листья."
    assert values(text, "BIRTH_PLACE") == []


def test_ner_country_qualifier_does_not_hide_passport_field_word():
    text = "Паспорт РФ: серия 6723, номер 918263"
    start = text.index("РФ")
    merged = merge_ner_candidates(text, detect(text), [Span(start, start + 2, "LOCATION", 0.85)])
    assert all(text[span.start:span.end] != "РФ" for span in merged)


def test_a_named_country_inside_citizenship_field_remains_protected():
    text = "Гражданство — Российская Федерация."
    assert values(text, "CITIZENSHIP") == ["Российская Федерация"]
