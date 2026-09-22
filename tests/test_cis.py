"""Synthetic regional development regressions, not a held-out quality benchmark.

Official format sources are recorded in docs/cis.md. Values are invented; these
tests verify protection of explicitly marked fields, not document authenticity.
No external benchmark examples or predictions were used to construct this suite.
"""

import pytest

from seif.detector import detect


def extracted(text):
    return [(span.type, text[span.start : span.end]) for span in detect(text)]


@pytest.mark.parametrize(
    "text,kind,value",
    [
        ("ИИН: 123456789012", "INN", "123456789012"),
        ("жсн = 123456789012", "INN", "123456789012"),
        ("ЖСН / ИИН: 123456789012", "INN", "123456789012"),
        ("ИИН Казахстана № 123456789012", "INN", "123456789012"),
        ("ИИН (KZ): 123456789012", "INN", "123456789012"),
        ("Казахстан; ИИН: 123456789012", "INN", "123456789012"),
        ("РНОКПП: 1234567890", "INN", "1234567890"),
        ("рнокпп України = 1234567890", "INN", "1234567890"),
        ("ІПН: 1234567890", "INN", "1234567890"),
        ("ІПН України: 1234567890", "INN", "1234567890"),
        ("ПИНФЛ: 12345678901234", "INN", "12345678901234"),
        ("JShShIR = 12345678901234", "INN", "12345678901234"),
        ("PINFL (UZ): 12345678901234", "INN", "12345678901234"),
        ("Узбекистан, ПИНФЛ: 12345678901234", "INN", "12345678901234"),
        ("Кыргызстан, ПИН: 12345678901234", "INN", "12345678901234"),
        ("ПИН Кыргызстана: 12345678901234", "INN", "12345678901234"),
        ("ПИН Кыргызской Республики: 12345678901234", "INN", "12345678901234"),
        ("Персональный идентификационный номер Кыргызстана: 12345678901234", "INN", "12345678901234"),
        ("Удостоверение личности Казахстана: 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("Номер удостоверения личности гражданина Республики Казахстан: 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("Казахстан\nудостоверение личности: 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("ID-карта (KZ): 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("Жеке куәлік (KZ): 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("ID-карта Украины № 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("ID-картка України: 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("Паспорт гражданина Украины: 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("Украина; паспорт: 123456789", "FOREIGN_DOCUMENT", "123456789"),
        ("Личный номер Беларуси: 1234567A123PB5", "FOREIGN_DOCUMENT", "1234567A123PB5"),
        ("Беларусь, идентификационный номер: 1234567A123PB5", "FOREIGN_DOCUMENT", "1234567A123PB5"),
        ("Персональный номер (BY): 1234567a123pb5", "FOREIGN_DOCUMENT", "1234567a123pb5"),
        ("IDNP Молдовы: 1234567890123", "INN", "1234567890123"),
        ("Молдова, IDNP: 1234567890123", "INN", "1234567890123"),
        ("IDNP (MD): 1234567890123", "INN", "1234567890123"),
        ("FIN Азербайджана: 1A2B3C4", "FOREIGN_DOCUMENT", "1A2B3C4"),
        ("Азербайджан, FİN: 1A2B3C4", "FOREIGN_DOCUMENT", "1A2B3C4"),
        ("FIN (AZ): 1a2b3c4", "FOREIGN_DOCUMENT", "1a2b3c4"),
        ("FIN (AZ): ABCDEFG", "FOREIGN_DOCUMENT", "ABCDEFG"),
    ],
)
def test_explicit_regional_identifier(text, kind, value):
    assert extracted(text) == [(kind, value)]


@pytest.mark.parametrize(
    "text",
    [
        "Заказ: 123456789", "Номер заявки: 1234567890", "Артикул: 123456789012",
        "Номер заказа: 12345678901234", "ID: 123456789", "ID-карта: 123456789",
        "Удостоверение личности: 123456789", "Казахстан: 123456789",
        "ПИН: 12345678901234", "Кыргызстан: 12345678901234",
        "Казахстан. Удостоверение личности: 123456789",
        "Украина\n\nID-карта: 123456789", "Кыргызстан. ПИН: 12345678901234",
        "Казахстан, заказ, удостоверение личности: 123456789",
        "ID-карта Германии: 123456789", "БИН: 123456789012",
        "my_jshshir: 12345678901234", "антиИИН: 123456789012",
        "ЖСН: 0123456789012", "ЖСН: 01234567890", "РНОКПП: 01234567890",
        "ПИНФЛ: 012345678901234", "ПИНФЛ: 0123456789012",
        "ПИН Кыргызстана: 012345678901234", "ИИН: 123456789012x",
        "ПИНФЛ: 12345678901234x", "ID-карта Украины: 123456789x",
        "Личный номер: 1234567A123PB5", "Беларусь. Личный номер: 1234567A123PB5",
        "Личный номер Беларуси: 1234567А123РВ5",  # Cyrillic lookalikes are not Latin.
        "Личный номер Беларуси: 1234567A123PB55", "Артикул: 1234567A123PB5",
        "IDNP: 1234567890123", "MD: 1234567890123", "Молдова. IDNP: 1234567890123",
        "IDNP (MD): 01234567890123", "FIN: 1A2B3C4", "ФИН: 1A2B3C4",
        "Азербайджан. FIN: 1A2B3C4", "FIN (AZ): 1A2B3C45", "FIN (AZ): 1A2B3C",
    ],
)
def test_no_generic_number_masking_or_country_leakage(text):
    assert extracted(text) == []


@pytest.mark.parametrize(
    "label,value",
    [("ЖСН", "000000000000"), ("РНОКПП", "0000000000"),
     ("ПИНФЛ", "00000000000000"), ("ПИН Кыргызстана", "00000000000000")],
)
def test_explicit_personal_fields_remain_protected_without_validity_claim(label, value):
    assert extracted(f"{label}: {value}") == [("INN", value)]


def test_personal_pin_and_bank_pin_have_different_classes():
    text = "Кыргызстан, ПИН: 12345678901234; ПИН-код карты: 4567"
    assert extracted(text) == [("INN", "12345678901234"), ("PIN", "4567")]


def test_explicit_personal_id_beats_coincidental_card_checksum():
    # This invented 14-digit sequence also satisfies Luhn; that does not change
    # the meaning of an explicitly named personal-identifier field.
    value = "30000000000004"
    assert extracted(f"ПИНФЛ: {value}") == [("INN", value)]
    assert extracted(f"Кыргызстан, ПИН: {value}") == [("INN", value)]
    assert extracted(f"Карта: {value}") == [("CARD", value)]


def test_unicode_offsets_and_adjacent_regional_fields():
    text = "🔐 ИИН: 123456789012; ПИНФЛ: 12345678901234."
    spans = detect(text)
    assert extracted(text) == [("INN", "123456789012"), ("INN", "12345678901234")]
    assert spans[0].start == text.index("123456789012")
    assert spans[0].end < spans[1].start


@pytest.mark.parametrize(
    "text,value",
    [
        ("ФИО: Нұрсұлтан Әбішұлы", "Нұрсұлтан Әбішұлы"),
        ("ФИО: Айгүл Әлиева.", "Айгүл Әлиева"),
        ("Фамилия и имя: Өмүрбек Жээнбеков;", "Өмүрбек Жээнбеков"),
        ("Аты-жөні: Әлия Мұратқызы Сәрсенова", "Әлия Мұратқызы Сәрсенова"),
        ("Имя и фамилия: Sean O'Neil", "Sean O'Neil"),
        ("ФИО: Shaun O’Neil", "Shaun O’Neil"),
        ("ФИО: О'Нил Анна", "О'Нил Анна"),
        ("ФИО: Нұрсұлтан Әбішұлы; ИИН: 123456789012", "Нұрсұлтан Әбішұлы"),
    ],
)
def test_strong_name_field_accepts_regional_letters_without_a_name_dictionary(text, value):
    assert ("PERSON", value) in extracted(text)


@pytest.mark.parametrize(
    "text",
    ["Клиент позвонил", "Клиент отправил письмо.", "Аты-жөні обсуждали на встрече.",
     "ФИО обсуждали на встрече.", "Номер заказа: QWERTY ABCDEF", "ФИО: test@example.net"],
)
def test_unicode_name_rule_requires_a_strong_name_field(text):
    assert not any(span.type == "PERSON" for span in detect(text))
