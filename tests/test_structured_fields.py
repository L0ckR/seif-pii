"""Authored variations and negative controls for bounded field recognizers."""

import pytest

from seif.structured_fields import structured_candidates


def values(text):
    return [(kind, text[start:end]) for start, end, kind, _, _ in structured_candidates(text)]


@pytest.mark.parametrize("value", [
    "23.04.91", "23-04-1991", "1991/04/23", "1991.23.04", "04.23.1991",
    "23 04 1991", "23 04 91", "23 04", "23/04", "23-апр-91",
    "23 АПР 1991", "23-Апреля-1991", "23 апреля", "29 фев", "29.02",
    "29 февраля 00", "23 апреля 1991 года",
])
def test_explicit_birth_dates_with_qualified_labels(value):
    text = f"Анкета: дата рождения доверенного лица (день и месяц): {value}."
    assert values(text) == [("BIRTH_DATE", value)]


@pytest.mark.parametrize("label", [
    "Дата выдачи паспорта представителя: ", "Дата выдачи паспорта: ", "Паспорт выдан ",
])
def test_explicit_passport_date(label):
    assert values(label + "24-сен-19.") == [("PASSPORT_DATE", "24-сен-19")]


@pytest.mark.parametrize("text", [
    "Дата рождения: 31.02.2004.", "Дата рождения: 29.02.2001.",
    "Дата рождения: 29-фев-01.", "Дата рождения: 00.04.91.",
    "Дата рождения: 23.00.91.", "Дата рождения: 23.04.991.",
    "Дата рождения: 23.04.19911.", "Дата рождения: 23.04.1991/04.",
    "Дата рождения: 23 Ммм 1991.", "Дата рождения: 23.04.1799.",
    "Дата рождения: 23.04.2101.", "Дата рождения не указана: 23.04.1991.",
    "Дата рождения отсутствует. Релиз: 23.04.1991.",
    "Дата рождения неизвестна; заявка: 23.04.1991.",
    "Дата рождения поэта: 23.04.1991.", "Дата рождения банка: 23.04.1991.",
    "Дата выпуска релиза: 23.04.1991.", "Встреча: 23 апреля 1991.",
    "Товар выдан 23.04.1991.", "Дата выдачи талона: 23.04.1991.",
    "Паспорт изготовлен. Заказ выдан 23.04.1991.",
    "23.04.1991", "23 04", "00-апр-91",
])
def test_dates_need_an_owned_valid_value(text):
    assert values(text) == []


def test_private_owner_is_not_exempted_by_unrelated_public_name():
    text = "Клиент работает у писателя; дата рождения клиента: 23-апр-91."
    assert values(text) == [("BIRTH_DATE", "23-апр-91")]


def test_bank_questionnaire_does_not_make_a_birth_date_public():
    text = "В справке банка указана дата рождения (день и месяц): 23/04."
    assert values(text) == [("BIRTH_DATE", "23/04")]


@pytest.mark.parametrize("value", [
    "двадцать третьего апреля тысяча девятьсот восемьдесят второго года",
    "первое мая две тысячи шестого года", "третье июня две тысячи года",
    "двадцать девятое февраля две тысячи четвертого года",
    "тридцать первого января тысяча девятьсот девятого года",
])
def test_written_dates_use_calendar_and_number_grammar(value):
    assert values("Дата рождения указана словами: " + value + ".") == [("BIRTH_DATE", value)]


@pytest.mark.parametrize("value", [
    "тридцать первого февраля тысяча девятьсот девятого года",
    "двадцать девятое февраля две тысячи первого года",
    "пятидесятого апреля две тысячи четвертого года",
    "двадцать девятнадцатого апреля две тысячи четвертого года",
])
def test_invalid_written_dates_are_rejected(value):
    assert values("Дата рождения указана словами: " + value + ".") == []


@pytest.mark.parametrize("text,kind,value", [
    ("pin-code: 2468", "PIN", "2468"),
    ("Пин-код(2468)", "PIN", "2468"),
    ("ПИН-код «2468»", "PIN", "2468"),
    ("ПИН дополнительной карты: 2468", "PIN", "2468"),
    ("В поле CVC указан код 246", "CVV", "246"),
    ("CVV кода 246", "CVV", "246"),
    ("ЦВВ 246", "CVV", "246"),
    ("Код 246 неверно введен CVC", "CVV", "246"),
    ("Проверьте код на обороте карты 246", "CVV", "246"),
    ("Код на обратной стороне карты: 246", "CVV", "246"),
    ("Платёж с банковской карты 5123 4567 8901 2345", "CARD", "5123 4567 8901 2345"),
    ("В/У № 65 43 987654", "DRIVER_LICENSE", "65 43 987654"),
    ("ВУ 6543 987654", "DRIVER_LICENSE", "6543 987654"),
    ("65 43 987654 — водительское удостоверение", "DRIVER_LICENSE", "65 43 987654"),
    ("Паспорт поручителя 6543 987654", "PASSPORT", "6543 987654"),
    ("Данные паспорта: серия 65 43", "PASSPORT", "65 43"),
])
def test_owned_payment_and_document_variants(text, kind, value):
    assert values(text) == [(kind, value)]


def test_repeated_card_fields_preserve_separate_values_and_offsets():
    text = "🔑 ПИН первой карты: 2468, ПИН второй карты: 9753."
    matches = list(structured_candidates(text))
    assert [(start, end) for start, end, *_ in matches] == [(20, 24), (44, 48)]
    assert values(text) == [("PIN", "2468"), ("PIN", "9753")]


def test_separate_document_parts_do_not_include_intervening_labels():
    text = "Водительское удостоверение, серия 65 43, выданное ГИБДД, номер 987654."
    assert values(text) == [("DRIVER_LICENSE", "65 43"), ("DRIVER_LICENSE", "987654")]


@pytest.mark.parametrize("text", [
    "Заказ с серией 6543 и номером 987654.",
    "Паспорт нужен; заказ, серия 6543, номер 987654.",
    "Паспорт нужен. Серия товара 6543, номер 987654.",
    "Паспорт нужен, накладная: серия 6543, номер 987654.",
    "6543 987654", "Номер 987654",
    "ПИН-код неизвестен. Заказ 2468.", "ПИН-код 246", "ПИН-код 2468975",
    "Система вернула код ошибки 246.", "Код на карте товара 246.",
    "CVV указан в заявке номер 246.", "CVV 24689", "CVV 246a",
    "Код 246 — ошибка ввода другого поля, проверьте CVV.",
    "Банковская карта 5123 4567 8901 2345 5678", "Банковская карта 512345",
])
def test_unowned_numbers_and_prose_are_not_values(text):
    assert values(text) == []


def test_adjacent_series_and_number_supply_document_structure():
    assert values("серия 6543 / номер 987654") == [("PASSPORT", "6543"), ("PASSPORT", "987654")]
