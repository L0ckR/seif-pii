"""Independent ownership, record-boundary and field-continuation controls."""

import pytest

from seif.document_fields import document_candidates, document_kind_at_value

NUMBER_PARTS = [("DRIVER_LICENSE", "34 56"), ("DRIVER_LICENSE", "876543")]


def values(text):
    return [(kind, text[start:end]) for start, end, kind, _, _ in document_candidates(text)]


@pytest.mark.parametrize("label", [
    "Карточка водителя", "Данные карточки водителя", "Предъявите карточку водителя",
    "С карточкой водителя", "В карточке водителя", "Водительская карточка",
    "Данные водительской карточки", "Предъявите водительскую карточку",
])
@pytest.mark.parametrize("transform", [str.lower, str.upper, lambda value: value])
def test_personal_driver_card_inflections_are_recognized(label, transform):
    text = transform(label + ", категория C, серия 34 56, номер 876543.")
    assert values(text) == NUMBER_PARTS


@pytest.mark.parametrize("bridge", [
    ". ", ".\n", "\n", "\r\n", "; ",
    ". В графе 3 указано: ", ". В пункте 2. 3 записано, что ",
    ". На обороте приведены: ", ". В ней указаны ",
    " проверено. ", " клиента предъявлено. В строке № 2 указаны: ",
])
@pytest.mark.parametrize("transform", [str.lower, str.upper, lambda value: value])
def test_explicit_owner_continues_only_into_a_field_clause(bridge, transform):
    text = transform("Водительское удостоверение" + bridge + "Серия 34 56, номер 876543.")
    assert values(text) == NUMBER_PARTS


@pytest.mark.parametrize("separator", ["\n", "\r\n"])
def test_separate_lines_in_one_document_preserve_both_parts(separator):
    text = "Водительское удостоверение" + separator + "серия 34 56" + separator + "номер 876543"
    assert values(text) == NUMBER_PARTS


@pytest.mark.parametrize("bridge", [
    ". Обсудили встречу. ", ". На следующий день ", ". Другие сведения: ",
    ". Новая запись: ", ". Следующая анкета: ", "\n\n", "\r\n\r\n", "! ", "? ",
])
def test_unrelated_record_does_not_inherit_driver_license(bridge):
    # The pre-existing unowned paired-series policy is PASSPORT. This control
    # asserts that an unrelated earlier licence does not alter that policy.
    text = "Водительское удостоверение" + bridge + "серия 34 56, номер 876543."
    assert values(text) == [("PASSPORT", "34 56"), ("PASSPORT", "876543")]


@pytest.mark.parametrize("business", ["Заказ", "Паспорт оборудования", "Накладная", "Номер договора", "Артикул товара"])
def test_new_business_owner_stops_personal_document_context(business):
    text = "Водительское удостоверение проверено. " + business + ": серия 34 56, номер 876543."
    assert values(text) == []


@pytest.mark.parametrize("business", ["Заказ", "Накладная", "Паспорт оборудования"])
def test_business_record_continuation_does_not_become_a_personal_document(business):
    text = business + " проверен. В графе 2 указано: серия 34 56, номер 876543."
    assert values(text) == []


def test_new_explicit_personal_owner_overrides_prior_business_record():
    text = "Накладная проверена. Водительское удостоверение: серия 34 56, номер 876543."
    assert values(text) == NUMBER_PARTS


def test_multiple_documents_preserve_their_individual_owners():
    text = ("Водительское удостоверение. Серия 34 56, номер 876543; "
            "паспорт: серия 78 12, номер 345678.")
    assert values(text) == NUMBER_PARTS + [("PASSPORT", "78 12"), ("PASSPORT", "345678")]


@pytest.mark.parametrize("label", [
    "Тахографическая карточка водителя", "Топливная карточка водителя",
    "Корпоративная карточка водителя", "Коммерческая карточка водителя",
    "Карточка водителя для тахографа", "Для цифрового тахографа карточка водителя",
    "Карта водителя", "Водительская карта", "Карта водителя для оплаты топлива",
])
def test_non_license_or_ambiguous_driver_cards_are_not_assumed_to_be_licenses(label):
    assert values(label + ": серия 34 56, номер 876543.") == []


def test_plain_grouped_number_needs_ownership_in_its_own_clause():
    assert values("Водительское удостоверение проверено. 3456 876543") == []


def test_blank_line_does_not_pair_document_parts_from_separate_records():
    assert values("Водительское удостоверение: серия 34 56\n\nномер 876543") == []


def test_owner_context_does_not_expand_without_limit():
    text = "Водительское удостоверение. " + " " * 170 + "серия 34 56, номер 876543."
    assert values(text) == [("PASSPORT", "34 56"), ("PASSPORT", "876543")]


@pytest.mark.parametrize("device", ["принтере", "сканере", "компьютере", "ноутбуке", "устройстве", "двигателе"])
def test_new_product_record_rejects_personal_document_inference(device):
    text = f"Паспорт заявителя проверен.\n\nОтчёт о {device}: серия 34 56, номер 876543."
    assert values(text) == []


@pytest.mark.parametrize("template,kind", [
    ("Карточка водителя: серия {value}, номер 876543.", "DRIVER_LICENSE"),
    ("Водительское удостоверение. Серия {value}, номер 876543.", "DRIVER_LICENSE"),
    ("Водительское удостоверение. В графе 3 указано: серия {value}, номер 876543.", "DRIVER_LICENSE"),
    ("Паспорт заявителя проверен.\n\nОтчёт о принтере: серия {value}, номер 876543.", None),
    ("Паспорт: серия {value}, номер 876543.", "PASSPORT"),
])
def test_legacy_numeric_match_uses_the_same_field_owner(template, kind):
    text = template.format(value="34 56")
    assert document_kind_at_value(text, text.index("34 56")) == kind
