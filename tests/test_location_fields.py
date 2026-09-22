"""Fictional variations for explicit issuers and residential-address grammar."""
from __future__ import annotations

import pytest

from seif.location_fields import location_candidates


def values(text):
    return [(text[start:end], kind) for start, end, kind, _score, _reason in location_candidates(text)]


@pytest.mark.parametrize("label", ["Орган выдачи:", "Орган выдачи паспорта —", "Орган, выдавший паспорт:",
                                   "Кем выдан паспорт:"])
def test_authority_field_includes_institution_qualifiers(label):
    authority = "Паспортно-визовое отделение УВД Тестового района"
    assert values(f"😊 {label} {authority}.") == [(authority, "PASSPORT_ISSUER")]


def test_repeat_passport_issuer_retains_internal_quotes_and_locality():
    second = "ОВД «Условное» г. Примерово"
    text = f"Первый паспорт был выдан УФМС Тестовой области, повторный — {second}."
    assert (second, "PASSPORT_ISSUER") in values(text)


@pytest.mark.parametrize("recipient", ["гражданину", "гражданке", "клиенту", "заявительнице"])
def test_issuing_authority_before_personal_passport_verb(recipient):
    authority = "УФМС России по Условной области"
    assert values(f"{authority} выдало паспорт {recipient}.") == [(authority, "PASSPORT_ISSUER")]


def test_issuer_ends_before_another_field_and_keeps_embedded_quotes():
    authority = "ОВД «Условное»"
    text = f"Орган выдачи: {authority}, код подразделения: 123-456; дата выдачи: 01.02.2024"
    assert values(text) == [(authority, "PASSPORT_ISSUER")]


def test_outer_field_quotes_are_excluded():
    authority = "УФМС России по Условной области"
    assert values(f"Орган выдачи: «{authority}».") == [(authority, "PASSPORT_ISSUER")]


def test_private_registration_retains_rural_and_region_components():
    address = "Тестовая обл., Условное с.п., дер. Примерная, ул. Лесная, д. 17"
    assert values(f"Место регистрации заёмщика — {address}.") == [(address, "ADDRESS")]


@pytest.mark.parametrize("address", [
    "123456, г. Примерово-на-Реке, просп. Условных Героев, д. 12а, кв. 4",
    "город Примерово, Тестовый бульвар, дом 14, строение 3, квартира 217",
    "г. Примерово, ул. Условная, д. 12/2, корп. 3а, кв. 17",
])
def test_complete_residential_line_composes_all_components(address):
    assert values(address + ".") == [(address, "ADDRESS")]


@pytest.mark.parametrize("text", [
    "УФМС России по Тестовой области принимает посетителей.",
    "Орган выдачи сертификата: отдел сертификации.",
    "Орган выдачи: отдел сертификации.",
    "Орган выдачи: не указан, обратитесь в УФМС Тестовой области.",
    "Орган выдачи: неизвестен; уточните в УФМС Тестовой области.",
    "Орган выдачи: для уточнения обратитесь в УФМС Тестовой области.",
    "Орган, выдавший паспорт оборудования: отдел снабжения.",
    "УВД Тестовой области выдало паспорт оборудования организации.",
    "Повторный — ОВД Тестового района.",
    "Город Примерово известен своим бульваром.",
    "Место регистрации компании — Тестовая обл., г. Примерово, ул. Лесная, д. 17.",
    "Реквизиты ООО «Пример»:\nгород Примерово, Тестовый бульвар, дом 14, строение 3, квартира 217.",
    "Офис банка:\n123456, г. Примерово, просп. Условных Героев, д. 12а, кв. 4.",
    "Реквизиты филиала:\n123456, г. Примерово, просп. Условных Героев, д. 12а, кв. 4.",
    "Юридическое лицо:\nг. Примерово, ул. Условная, д. 12/2, корп. 3а, кв. 17.",
])
def test_public_or_unowned_context_does_not_create_location_candidate(text):
    assert values(text) == []


def test_later_private_owner_can_follow_an_earlier_company():
    address = "г. Примерово, ул. Условная, д. 12, кв. 17"
    text = f"ООО «Пример». Домашний адрес клиента:\n{address}"
    assert values(text) == [(address, "ADDRESS")]


def test_a_new_paragraph_does_not_inherit_earlier_corporate_ownership():
    address = "г. Примерово, ул. Условная, д. 12, кв. 17"
    text = f"Реквизиты ООО «Пример» проверены.\n\n{address}"
    assert values(text) == [(address, "ADDRESS")]


def test_bounded_field_does_not_emit_a_truncated_value():
    text = "Орган выдачи: УФМС " + "Условной " * 100 + "области"
    assert values(text) == []


def test_repeated_fields_keep_distinct_unicode_offsets():
    authority = "УФМС России по Условной области"
    text = f"😀 Орган выдачи: {authority}; Орган выдачи: {authority}."
    spans = list(location_candidates(text))
    assert len(spans) == 2
    assert spans[0][0] == text.index(authority)
    assert spans[1][0] == text.rindex(authority)
    assert all(text[start:end] == authority for start, end, *_rest in spans)
