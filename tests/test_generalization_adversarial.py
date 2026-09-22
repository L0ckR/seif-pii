"""Independent semantic controls for generalized personal-field extraction.

These authored examples are not copied from organizer or external benchmarks.
Assertions concern protected values and visible prose, not regex structure,
confidence constants, reason strings, or a prescribed internal decomposition.
"""

import pytest

from seif.cardholder_fields import cardholder_candidates
from seif.detector import detect
from seif.document_fields import document_candidates
from seif.transform import mask, restore_exact


def alphanumeric_positions(text, spans):
    return {
        offset
        for start, end, *_ in spans
        for offset in range(start, end)
        if text[offset].isalnum()
    }


def positions_of(text, values):
    positions = set()
    cursor = 0
    for value in values:
        start = text.index(value, cursor)
        positions.update(
            offset for offset in range(start, start + len(value)) if text[offset].isalnum()
        )
        cursor = start + len(value)
    return positions


def holder_spans(text):
    # Empty caller dictionary prevents a passing test from depending on these
    # independent names being present in the production Russian name list.
    return list(cardholder_candidates(text, given_names=frozenset()))


@pytest.mark.parametrize("name", [
    "Özlem Şentürk", "Nkosazana Dlamini", "Aïcha Benali", "Łukasz Żmuda",
    "Саят Нұрғали", "Meryem El-Karoui", "Amara O’Nwosu",
])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.swapcase])
def test_closed_cardholder_value_does_not_require_a_seen_name(name, change_case):
    value = change_case(name)
    text = change_case("🔒 Name on card: ") + value + change_case("; Review ticket: ZX-42.")
    assert alphanumeric_positions(text, holder_spans(text)) == positions_of(text, [value])


@pytest.mark.parametrize("label, separator, value_end", [
    ("Держатель карты", " — ", ". Квитанция сохранена."),
    ("Имя держателя карты", " – ", "; Следующий реквизит: 901."),
    ("CARDHOLDER", " - ", "! Processing complete."),
    ("name on card", "=", "\nAudit status: complete"),
    ("Эмбоссированное имя", " «", "». Следующая запись."),
    ("card holder", " “", "”. Next record."),
    ("Держатель карты", " ", ". Платёж рассмотрен."),
    ("Карта выпущена на имя", " ", "; договор сохранён."),
])
def test_typographic_field_boundaries_keep_labels_and_next_fields_visible(label, separator, value_end):
    value = "Sizwe Mbeki"
    text = label + separator + value + value_end
    assert alphanumeric_positions(text, holder_spans(text)) == positions_of(text, [value])


@pytest.mark.parametrize("role", ["Держатель", "Держателя", "держателю"])
def test_generic_object_holder_without_card_or_field_delimiter_is_not_a_person_field(role):
    text = role + " Sizwe Mbeki. Деталь отправлена на склад."
    assert holder_spans(text) == []


@pytest.mark.parametrize("owner", [
    "Карта 5555 4444 3333 1111", "Номер карты 5555444433331111", "Card number 5555444433331111",
])
def test_adjacent_explicit_card_can_own_a_generic_holder_field(owner):
    value = "Xolani Radebe"
    text = owner + " держатель " + value + "."
    assert alphanumeric_positions(text, holder_spans(text)) == positions_of(text, [value])


@pytest.mark.parametrize("text", [
    "Карта 5555444433331111.\n\nПроект «Держатель Xolani Radebe».",
    "Номер заказа 5555444433331111 держатель Xolani Radebe.",
    "Карта клиента проверена. Артикул товара 5555444433331111 держатель Xolani Radebe.",
])
def test_numeric_similarity_and_an_older_card_record_do_not_create_ownership(text):
    assert holder_spans(text) == []


@pytest.mark.parametrize("text", [
    "Name on card: Pending Approval.",
    "Cardholder: Account Locked.",
    "Держатель карты: Ожидает подтверждения.",
    "Имя держателя карты: отдел взыскания.",
    "Держатель карты: ООО Кристалл.",
    "Cardholder: Acme LLC.",
    "Name on card: Meryem El-Karoui@example.net",
    "Name on card: Meryem El-Karoui.example.net",
    "Cardholder: Sizwe Mbeki/profile",
    "Name on card: https://accounts.example.net/profile",
    "Имя держателя карты: имя@example.net",
    "Cardholder: Sizwe Mbeki approved the request.",
    "Держатель карты: Саят Нұрғали подписал заявление.",
])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.swapcase])
def test_status_business_identifiers_and_unclosed_prose_are_not_cardholder_values(text, change_case):
    assert holder_spans(change_case(text)) == []


def test_each_cardholder_record_is_local_and_does_not_propagate_by_identical_text():
    text = (
        "Имя держателя карты: Aïcha Benali.\n"
        "Следующая запись:\n"
        "Название тестового макета: Aïcha Benali.\n"
        "Name on card=Özlem Şentürk; transaction accepted."
    )
    assert alphanumeric_positions(text, holder_spans(text)) == positions_of(
        text, ["Aïcha Benali", "Özlem Şentürk"]
    )


@pytest.mark.parametrize("value", ["Özlem Şentürk", "Саят Нұрғали", "Amara O’Nwosu"])
def test_personal_values_survive_full_mask_restore_with_unrelated_prose_visible(value):
    email = "m.relay@example.org"
    text = f"🔐 Name on card — {value}; contact: {email}. Сумма операции 37 рублей."
    spans = detect(text)
    covered = {
        offset for span in spans for offset in range(span.start, span.end)
        if text[offset].isalnum()
    }
    assert covered == positions_of(text, [value, email])
    masked, replacements = mask(text, spans, "mask")
    assert masked.endswith(". Сумма операции 37 рублей.")
    assert masked.startswith("🔐 Name on card — ")
    assert restore_exact({"masked": masked, "replacements": replacements}) == text


@pytest.mark.parametrize("text", [
    "Паспорт заявителя проверен.\n\nОтчёт о принтере: серия 5814, номер 620731.",
    "Паспорт принят. Заказ оборудования: серия 5814; номер 620731.",
    "Данные паспорта архивированы. URL: https://example.net/5814/620731",
    "Документ человека проверен. Email: 5814.620731@example.net",
    "Водительское удостоверение проверено.\n\nАртикул изделия: 5814 620731.",
    "Документ: производственный сертификат, серия 5814; номер 620731.",
    "Электронный паспорт оборудования: серия 5814; номер 620731.",
])
def test_personal_document_owner_does_not_leak_into_business_records(text):
    assert list(document_candidates(text)) == []


@pytest.mark.parametrize("text", [
    "Паспорт клиента: серия 5814.\n\nНомер 620731 указан на коробке.",
    "Серия 5814 относится к паспорту. Следующая запись: номер 620731.",
    "Документ: серия 5814; сумма 37; номер 620731.",
])
def test_series_and_number_are_not_joined_across_another_record_or_field(text):
    spans = list(document_candidates(text))
    number_positions = positions_of(text, ["620731"])
    assert not alphanumeric_positions(text, spans) & number_positions


@pytest.mark.parametrize("text, expected_kind, values", [
    (
        "Заказ оборудования закрыт. Паспорт гражданина: 5814 620731.",
        "PASSPORT", ["5814 620731"],
    ),
    (
        "Артикул изделия: 902.\nВодительское удостоверение: серия 58 14; номер 620731.",
        "DRIVER_LICENSE", ["58 14", "620731"],
    ),
    (
        "Водительское удостоверение проверено. Серия 5814; номер 620731.",
        "DRIVER_LICENSE", ["5814", "620731"],
    ),
])
def test_new_explicit_owner_restores_personal_document_protection(text, expected_kind, values):
    spans = list(document_candidates(text))
    assert spans
    assert {span[2] for span in spans} == {expected_kind}
    assert alphanumeric_positions(text, spans) == positions_of(text, values)


@pytest.mark.parametrize("prefix", [
    "Карточка водителя: ",
    "Водительское удостоверение проверено. ",
])
@pytest.mark.parametrize("join", [" ", ", "])
def test_full_detector_preserves_license_ownership_without_a_conjunction(prefix, join):
    text = prefix + "серия 58 14" + join + "номер 620731. Реквизиты сверены."
    spans = detect(text)
    assert spans
    assert {span.type for span in spans} == {"DRIVER_LICENSE"}
    covered = {
        offset for span in spans for offset in range(span.start, span.end)
        if text[offset].isalnum()
    }
    assert covered == positions_of(text, ["58 14", "620731"])
    masked, replacements = mask(text, spans, "mask")
    assert masked == prefix + "серия ** **" + join + "номер ******. Реквизиты сверены."
    assert restore_exact({"masked": masked, "replacements": replacements}) == text


@pytest.mark.parametrize("prefix", ["", "Паспорт заявителя проверен.\n\n"])
def test_full_detector_keeps_printer_business_numbers_visible(prefix):
    text = prefix + "Отчёт о принтере: серия 5814 номер 620731."
    masked, replacements = mask(text, detect(text), "mask")
    assert masked == text
    assert replacements == []


@pytest.mark.parametrize("suffix", [
    "; номер партии 84.", "\nКод заказа 123-567.", " в инструкции указан пример 123-567.",
])
def test_document_field_labels_do_not_search_arbitrary_later_values(suffix):
    text = "Код подразделения не заполнен" + suffix
    assert list(document_candidates(text)) == []


def test_initial_punctuation_stays_visible_without_swallowing_the_following_sentence():
    text = "Плательщик: Зорин Т. Ф. Подтверждение получено; номер операции 57."
    spans = detect(text)
    expected = positions_of(text, ["Зорин Т. Ф."])
    actual = {
        offset for span in spans for offset in range(span.start, span.end)
        if text[offset].isalnum()
    }
    assert actual == expected
    masked, replacements = mask(text, spans, "mask")
    assert masked == "Плательщик: ***** *. *. Подтверждение получено; номер операции 57."
    assert restore_exact({"masked": masked, "replacements": replacements}) == text
