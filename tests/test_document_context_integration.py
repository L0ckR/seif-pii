"""End-to-end ownership checks including competing legacy recognizers."""

import pytest

from seif.detector import detect


def document_values(text):
    return [(span.type, text[span.start:span.end]) for span in detect(text)
            if span.type in {"PASSPORT", "DRIVER_LICENSE"}]


@pytest.mark.parametrize("prefix", [
    "Карточка водителя: ",
    "В водительской карточке указана категория C, ",
    "Водительское удостоверение. ",
    "Водительское удостоверение. В графе 3 указано: ",
    "Водительское удостоверение\n",
])
@pytest.mark.parametrize("transform", [str.lower, str.upper, str.swapcase])
def test_explicit_license_owner_beats_legacy_passport_pattern(prefix, transform):
    text = transform(prefix + "серия 34 56, номер 876543.")
    assert document_values(text) == [("DRIVER_LICENSE", "34 56"), ("DRIVER_LICENSE", "876543")]


@pytest.mark.parametrize("device", ["принтере", "сканере", "ноутбуке", "устройстве"])
def test_previous_passport_does_not_mask_a_new_equipment_record(device):
    text = f"Паспорт заявителя проверен.\n\nОтчёт о {device}: серия 34 56, номер 876543."
    assert detect(text) == []


def test_explicit_document_owner_changes_without_global_priority_changes():
    text = ("Карточка водителя: серия 34 56, номер 876543; "
            "паспорт: серия 78 12, номер 345678.")
    assert document_values(text) == [
        ("DRIVER_LICENSE", "34 56"), ("DRIVER_LICENSE", "876543"),
        ("PASSPORT", "78 12"), ("PASSPORT", "345678"),
    ]
