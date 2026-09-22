"""Independent adversarial controls for generalized context recognition.

These are authored examples, not organizer inputs or copied golden cases.
They exercise context boundaries, full names, and numeric-list ambiguity.
"""

import pytest

from seif.detector import detect


def values(text, kind):
    return [text[span.start:span.end] for span in detect(text) if span.type == kind]


@pytest.mark.parametrize("text", [
    "Талон выдан вчера. ОВД района принимает посетителей.",
    "Чек выдан покупателю, УМВД области объявило конкурс.",
    "Сертификат выдан компании. МО МВД области опубликовало расписание.",
])
def test_business_issuance_does_not_create_a_passport_authority(text):
    assert values(text, "PASSPORT_ISSUER") == []


def test_passport_authority_ownership_does_not_cross_a_completed_sentence():
    authority = "УФМС города Твери"
    text = f"Паспорт выдан {authority}. ОВД района принимает посетителей."
    assert values(text, "PASSPORT_ISSUER") == [authority]


def test_previous_authority_within_one_passport_record_is_retained():
    first, second = "УМВД Псковской области", "ОВД Псковского района"
    text = f"Паспорт выдан {first} (ранее — {second})."
    result = [value.rstrip(")") for value in values(text, "PASSPORT_ISSUER")]
    assert result == [first, second]


@pytest.mark.parametrize("prefix,region", [
    ("Доставьте в ", "Тверскую область"),
    ("Проверьте ", "Тверской район"),
    ("Выбран ", "Тверской район"),
])
def test_region_does_not_consume_preceding_narrative(prefix, region):
    address = region + ", г. Тверь, ул. Мира, д. 7"
    text = prefix + address + "."
    spans = detect(text)
    assert all(span.start >= len(prefix) for span in spans)
    assert values(text, "ADDRESS") == [address]


def test_public_biography_does_not_gain_new_birthplace_candidate():
    text = "Поэт Василий Лебедев родился 24.05.1882 в г. Псков."
    assert values(text, "BIRTH_PLACE") == []


@pytest.mark.parametrize("locality", ["г. Нижний Новгород", "городе Нижний Новгород"])
def test_birthdate_before_multiword_birthplace_does_not_leak_second_word(locality):
    text = f"Клиент родился 14.08.1992 в {locality}."
    assert values(text, "BIRTH_PLACE") == [locality]
    assert values(text, "BIRTH_DATE") == ["14.08.1992"]


def test_hyphenated_birthplace_remains_complete():
    text = "Клиент родился 14.08.1992 в г. Ростов-на-Дону."
    assert values(text, "BIRTH_PLACE") == ["г. Ростов-на-Дону"]


@pytest.mark.parametrize("separator", [",", ", ", ";"])
def test_decimal_filter_retains_both_values_in_explicit_inn_list(separator):
    # Authored ten-digit values with valid control digits.
    first, second = "8123456782", "9123456784"
    text = f"Выписка: ИНН владельцев {first}{separator}{second}."
    assert values(text, "INN") == [first, second]


@pytest.mark.parametrize("value", ["8123456782.0", "8123456782,123", "0.8123456782"])
def test_unowned_decimal_measurement_does_not_become_an_inn(value):
    assert values("Значение показателя: " + value, "INN") == []
