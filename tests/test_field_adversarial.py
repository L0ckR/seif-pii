"""Independent synthetic controls for ownership boundaries and regex growth."""

import subprocess
import sys

import pytest

from seif.context_filters import refine_candidates
from seif.detector import Span
from seif.structured_fields import structured_candidates


def _span(text, value, kind):
    start = text.rindex(value)
    return Span(start, start + len(value), kind, 0.9, "adversarial-control")


@pytest.mark.parametrize("boundary", [". ", "; ", "\n\n"])
@pytest.mark.parametrize("value", ["1234567890", "123456789012"])
def test_corporate_owner_cannot_exempt_identifier_in_a_later_record(boundary, value):
    text = "Компания завершила работу" + boundary + "ИНН " + value
    span = _span(text, value, "INN")
    assert refine_candidates(text, [span]) == [span]


@pytest.mark.parametrize("boundary", [". ", "; ", "\n\n"])
def test_office_cannot_exempt_geography_in_a_later_record(boundary):
    text = "ОВД закрыто" + boundary + "г. Омск"
    span = _span(text, "Омск", "CITY")
    assert refine_candidates(text, [span]) == [span]


def test_explicit_address_remains_personal_beside_an_office_reference():
    text = "УВД находится рядом, адрес: г. Омск"
    span = _span(text, "Омск", "CITY")
    assert refine_candidates(text, [span]) == [span]


@pytest.mark.parametrize("prefix", ["ОВД г. ", "УФМС,\nг. ", r"УФМС,\nг. ", "Отдел внутренних дел, ул. Мира, г. "])
def test_office_name_abbreviations_do_not_start_a_private_record(prefix):
    text = prefix + "Омск"
    assert refine_candidates(text, [_span(text, "Омск", "CITY")]) == []


@pytest.mark.parametrize("prefix", ["АО «Макет» г. Омск, ", "Реквизиты АО. «Макет», ", "ООО, ул. Мира, "])
def test_corporate_name_and_address_abbreviations_preserve_ownership(prefix):
    text = prefix + "ИНН 1234567890"
    assert refine_candidates(text, [_span(text, "1234567890", "INN")]) == []


@pytest.mark.parametrize("text", [
    "Дата рождения ожидается — дата релиза: 23.04.1991.",
    "Дата рождения будет уточнена дата договора: 23.04.1991.",
    "Дата рождения не заполнена — дата договора: 23.04.1991.",
])
def test_birth_field_does_not_take_a_date_owned_by_another_field(text):
    assert list(structured_candidates(text)) == []


@pytest.mark.parametrize("text", [
    "Паспорт оборудования номер 567890.",
    "Паспорт компании номер 123456.",
    "Электронный паспорт транспортного средства: номер 123456.",
    "Паспорт получен, заказу присвоена серия 1234 номер 567890.",
    "Паспорт не указан, регистрационный номер 123456.",
])
def test_nonpersonal_or_missing_passport_does_not_own_later_numbers(text):
    assert list(structured_candidates(text)) == []


@pytest.mark.parametrize("text", [
    "Дата рождения заёмщика по договору: 23.04.1991.",
    "Дата рождения сотрудника банка: 23.04.1991.",
])
def test_contract_and_employer_qualifiers_do_not_make_a_persons_birth_public(text):
    assert [(kind, text[start:end]) for start, end, kind, *_ in structured_candidates(text)] == [
        ("BIRTH_DATE", "23.04.1991")
    ]


@pytest.mark.parametrize("script", [
    "from seif.context_filters import refine_candidates; from seif.detector import Span; "
    "text='1234'+' '*65536+'123456'; span=Span(0,len(text),'PASSPORT'); "
    "assert refine_candidates(text,[span]) == [span]",
    "from seif.structured_fields import structured_candidates; "
    "assert list(structured_candidates('PIN'+' '*65536+'1')) == []",
    "from seif.structured_fields import structured_candidates; "
    "assert list(structured_candidates('CVV'+' '*65536+'1')) == []",
])
def test_long_incomplete_field_does_not_lock_up_recognizer(script):
    # A subprocess deadline catches polynomial/cubic backtracking without a
    # fragile millisecond performance assertion or a hung test suite.
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=3)


def test_document_split_stays_idempotent_through_merge_and_round_trip():
    text = "🔐 Серия 12 34, номер 567890"
    spans = [_span(text, "12 34, номер 567890", "PASSPORT")]
    once = refine_candidates(text, spans)
    assert len(once) == 2
    assert refine_candidates(text, once) == once


def test_person_name_filter_leaves_unrelated_public_and_private_names_intact():
    for text in ("Поэт Антон Полевой", "Клиент Антон Полевой"):
        span = _span(text, "Антон Полевой", "PERSON")
        assert refine_candidates(text, [span]) == [span]
