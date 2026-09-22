"""Post-benchmark development examples written before the address implementation.

These cases were authored from format requirements, without reading external
benchmark cases, gold spans, or failures. They are development regressions, not
an independent quality measurement. The implementation uses original code;
Natasha/Yargy's component-grammar design is inspiration, not copied source.
"""

import pytest

from seif.detector import detect

POSITIVE = [
    ("ул. Тихая, д. 14", "ул. Тихая, д. 14"),
    ("Доставить: ул. Тихая 14.", "ул. Тихая 14"),
    ("г. Тверь, ул. Дальняя, д. 12, кв. 4", "г. Тверь, ул. Дальняя, д. 12, кв. 4"),
    ("300001, г. Тула, улица Верхняя Полевая, дом 3", "300001, г. Тула, улица Верхняя Полевая, дом 3"),
    ("220001, Минск, ул. Новая, д. 7", "220001, Минск, ул. Новая, д. 7"),
    ("г. Алматы, пр-т Абая, д. 27, кв. 16", "г. Алматы, пр-т Абая, д. 27, кв. 16"),
    ("город Нижний Новгород, пер. Садовый, дом 6А", "город Нижний Новгород, пер. Садовый, дом 6А"),
    ("ул. 8 Марта, д. 5/2", "ул. 8 Марта, д. 5/2"),
    ("ул. 50 лет Победы, дом 17", "ул. 50 лет Победы, дом 17"),
    ("ул. 1-я Полевая, д. 21", "ул. 1-я Полевая, д. 21"),
    ("ул. Верхняя Полевая, д. 12, корп. 2, стр. 1, кв. 8", "ул. Верхняя Полевая, д. 12, корп. 2, стр. 1, кв. 8"),
    ("проспект Научный, 18к2", "проспект Научный, 18к2"),
    ("шоссе Северное, дом 8, литера Б", "шоссе Северное, дом 8, литера Б"),
    ("б-р Солнечный, д. 4, кв. 15", "б-р Солнечный, д. 4, кв. 15"),
    ("наб. Речная, д. 2/4, корпус 1", "наб. Речная, д. 2/4, корпус 1"),
    ("проезд Новый, дом 11", "проезд Новый, дом 11"),
    ("ул. Әуезова, дом 23", "ул. Әуезова, дом 23"),
    ("ул.Тихая,д.14,кв.3", "ул.Тихая,д.14,кв.3"),
    ("г. Тверь,\nул. Дальняя,\nд. 12, кв. 4", "г. Тверь,\nул. Дальняя,\nд. 12, кв. 4"),
    ("ул. Тихая, д. 14, г. Тверь, 170001", "ул. Тихая, д. 14, г. Тверь, 170001"),
    ("🔐 ул. Тихая, д. 14. Позвоните завтра.", "ул. Тихая, д. 14"),
    ("Адрес клиента: г. Тверь, ул. Тихая, д. 14.", "г. Тверь, ул. Тихая, д. 14"),
    ("Клиент проживает: ул. Тихая, д. 14, строение 2.", "ул. Тихая, д. 14, строение 2"),
    ("пр-т Солнечный, д. №18, кв. №4", "пр-т Солнечный, д. №18, кв. №4"),
]

NEGATIVE = [
    "Прочитайте историю улицы Тихой.",
    "Улица Верхняя Полевая переименована.",
    "На ул. Тихая состоится праздник.",
    "ул. Тихая; номер заказа: 14",
    "ул. Тихая. Дом 14 выставлен на продажу.",
    "ул. Тихая\n\nдом 14",
    "ул. Тихая 2024 год — время ремонта.",
    "ул. Тихая 15 километров от центра.",
    "ул. Тихая, заказ 14",
    "Улица закрыта на 12 часов.",
    "Пер. — это сокращение слова переулок.",
    "Паспорт 4510 123456, документ №14.",
    "Артикул UL-MIRA-14-K2",
    "Квартира 14 в доме у реки.",
    "Москва, Тверь, Нижний Новгород — города России.",
]


@pytest.mark.parametrize("text,value", POSITIVE)
def test_structural_address_keeps_exact_original_span(text, value):
    spans = [span for span in detect(text) if span.type == "ADDRESS"]
    assert [(span.start, span.end) for span in spans] == [(text.index(value), text.index(value) + len(value))]


@pytest.mark.parametrize("text", NEGATIVE)
def test_address_requires_linked_street_and_house_not_unrelated_prose(text):
    assert not any(span.type == "ADDRESS" for span in detect(text))


@pytest.mark.parametrize(
    "text",
    [
        "Адрес банка: г. Тверь, ул. Дальняя, д. 12.",
        "Отделение банка: 300001, г. Тула, ул. Дальняя, д. 12.",
        "Филиал банка расположен: пр-т Научный, д. 17.",
    ],
)
def test_public_bank_address_remains_unmasked(text):
    assert detect(text) == []


def test_public_bank_does_not_exempt_a_later_private_address():
    text = "Адрес банка: ул. Дальняя, д. 12; адрес клиента: ул. Тихая, д. 14."
    spans = detect(text)
    assert [(span.type, text[span.start : span.end]) for span in spans] == [("ADDRESS", "ул. Тихая, д. 14")]


def test_adjacent_addresses_and_other_personal_fields_are_not_joined():
    text = "ул. Тихая, д. 14; ул. Новая, д. 7; email: client@example.net"
    assert [(span.type, text[span.start : span.end]) for span in detect(text)] == [
        ("ADDRESS", "ул. Тихая, д. 14"),
        ("ADDRESS", "ул. Новая, д. 7"),
        ("EMAIL", "client@example.net"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Улица Сосновая появилась в 1987 году.",
        "На улице Сосновой зарегистрирован заказ №18.",
        "В конкурсе «Улица года» участвовало 12 городов.",
        "Встретимся у памятника на улице Академика.",
        "ул. Сосновая, домофон 18",
    ],
)
def test_independently_suggested_prose_negatives(text):
    assert not any(span.type == "ADDRESS" for span in detect(text))


def test_order_number_before_a_real_address_is_not_a_postcode():
    text = "Номер заказа: 300001, ул. Тихая, д. 14"
    assert [(span.type, text[span.start : span.end]) for span in detect(text)] == [("ADDRESS", "ул. Тихая, д. 14")]


def test_a_window_boundary_never_truncates_the_house_number():
    text = "ул. Тихая," + " " * 310 + "д. 1234"
    spans = detect(text)
    assert not any(span.type == "ADDRESS" for span in spans)
    assert any(span.type == "HOUSE" and text[span.start : span.end] == "1234" for span in spans)


def test_public_bank_exception_stays_local_before_a_private_record():
    text = "Офис банка: ул. Сосновая, д. 18. Адрес клиента: ул. Лесная, д. 20."
    assert [(span.type, text[span.start : span.end]) for span in detect(text)] == [("ADDRESS", "ул. Лесная, д. 20")]
