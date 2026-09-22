"""Authored format variants and business-number negative controls."""

import pytest

from seif.document_fields import document_candidates


def values(text):
    return [(kind, text[start:end]) for start, end, kind, _, _ in document_candidates(text)]


@pytest.mark.parametrize("value", ["6721 908172", "67 21 908172", "67-21-908172", "67/21/908172"])
@pytest.mark.parametrize("template", ["Паспорт: {}", "Данные документа: ({}).", "В документах: паспорт {} и второй паспорт."])
def test_grouped_document_format(value, template):
    assert values(template.format(value)) == [("PASSPORT", value)]


@pytest.mark.parametrize("template", [
    "Серия {series}, номер {number}.", "номер {number} серия {series}",
    "серия: {series}; номер: {number}", "сер. {series} № {number}",
    "Документ с серией {series} и номером {number} утрачен.",
    "Документ: серия {series}. Номер {number}. Выдан миграционным отделом.",
])
def test_paired_parts(template):
    text = template.format(series="6721", number="908172")
    result = values(text)
    assert sorted(result) == [("PASSPORT", "6721"), ("PASSPORT", "908172")]


@pytest.mark.parametrize("label", ["ВУ", "В/У", "Водительское удостоверение", "Удостоверение:"])
def test_license_parts_keep_ownership(label):
    assert values(label + " серия 67 21, номер 908172") == [
        ("DRIVER_LICENSE", "67 21"), ("DRIVER_LICENSE", "908172"),
    ]


@pytest.mark.parametrize("prefix", [
    "Заказ", "Номер договора", "Паспорт оборудования", "Артикул товара", "SKU", "ИНН", "Телефон", "Сумма",
])
@pytest.mark.parametrize("value", ["6721 908172", "серия 6721 / номер 908172"])
def test_business_owner_vetoes_document_format(prefix, value):
    assert values(f"{prefix}: {value}.") == []


@pytest.mark.parametrize("text", [
    "6721908172", "6721", "6721 908172", "67 21 908172", "67/21/908172", "Номер 908172", "серия 6721; дата рождения 908172",
    "серия 6721 и номер товара 908172", "6721 908172 3", "6721 908172a",
    "a6721 908172", "67/21/908172/4", "Серия 6721, товар 7, номер 908172",
    "Серия 6721, номер 908172 товара",
    "Удостоверение сотрудника: серия 6721, номер 908172",
    "Пенсионное удостоверение: серия 6721, номер 908172",
])
def test_incomplete_or_embedded_numbers(text):
    assert values(text) == []


def test_new_explicit_owner_overrides_business_context():
    assert values("Номер договора проверен; паспорт: 6721 908172.") == [("PASSPORT", "6721 908172")]


@pytest.mark.parametrize("verb", ["стоит", "указан", "записан", "равен", "составляет"])
def test_department_field_verb(verb):
    assert values(f"В графе код подразделения {verb} 123-567.") == [("DEPARTMENT_CODE", "123-567")]


def test_department_field_does_not_search_free_prose():
    assert values("Код подразделения уточните в офисе, заказ 123-567.") == []
