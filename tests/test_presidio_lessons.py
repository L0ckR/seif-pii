"""Development regressions inspired by Presidio, implemented independently.

Presidio's PatternRecognizer separates candidate validation from invalidation,
while ContextAwareEnhancer limits supporting context to a nearby word window:
https://microsoft.github.io/presidio/analyzer/
https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/pattern_recognizer.py

No upstream source code is copied. These handwritten minimal pairs test Russian
field ownership and sentence boundaries, not general benchmark superiority.
"""

from __future__ import annotations

import pytest

from seif.detector import detect

CASES = [
    ("Номер заказа: 79991234567.", []),
    ("Номер заказа: 7707083893.", []),
    ("Номер накладной: 4111111111111111.", []),
    ("Артикул товара: 4111-1111-1111-1111.", []),
    ("Тикет № 79991234567.", []),
    ("SKU: 4111111111111111", []),
    ("ORDER_ID=7707083893", []),
    ("Телефон для заказа: 79991234567.", [("PHONE", "79991234567")]),
    ("ИНН получателя заказа: 7707083893.", [("INN", "7707083893")]),
    ("Номер карты для оплаты заказа: 4111111111111111.", [("CARD", "4111111111111111")]),
    ("Номер заказа: 118; номер карты: 4111111111111111.", [("CARD", "4111111111111111")]),
    ("Номер заказа: 4111111111111111; карта: 4111111111111111.", [("CARD", "4111111111111111")]),
    ("Номер заявки: 79991234567; телефон: 79991234567.", [("PHONE", "79991234567")]),
    ("Паспорт проверен. Талон выдан 17.03.2024.", []),
    ("Паспорт проверен; товар выдан складом 17.03.2024.", []),
    ("Паспорт проверен. ГУ МВД России опубликовало отчёт.", []),
    ("Паспорт клиента осмотрен.\n\nДокумент выдан 17.03.2024.", []),
    ("Паспорт: 4512 654321; выдан 17.03.2024.", [("PASSPORT", "4512 654321"), ("PASSPORT_DATE", "17.03.2024")]),
    ("Паспорт: 4512 654321\nВыдан 17.03.2024.", [("PASSPORT", "4512 654321"), ("PASSPORT_DATE", "17.03.2024")]),
    (
        "Паспорт выдан ОВД района Арбат г. Москвы 17.03.2024.",
        [("PASSPORT_ISSUER", "ОВД района Арбат г. Москвы"), ("PASSPORT_DATE", "17.03.2024")],
    ),
    ("Паспорт: 4512 654321.\nДата выдачи: 17.03.2024.", [("PASSPORT", "4512 654321"), ("PASSPORT_DATE", "17.03.2024")]),
    ("Паспорт выдан 17.03.2024.\n\nТовар выдан 18.03.2024.", [("PASSPORT_DATE", "17.03.2024")]),
    ("Кем выдан: ОВД района Арбат.", [("PASSPORT_ISSUER", "ОВД района Арбат")]),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_local_field_ownership_and_document_context(text, expected):
    actual = [(span.type, text[span.start : span.end]) for span in detect(text)]
    assert actual == expected


def test_repeated_equal_numbers_are_scoped_to_their_own_field():
    text = "Номер заказа: 79991234567; Телефон: 79991234567; Номер заказа: 79991234567."
    spans = detect(text)
    assert [(span.type, span.start, span.end) for span in spans] == [
        (
            "PHONE",
            text.index("79991234567", text.index("Телефон:")),
            text.index("79991234567", text.index("Телефон:")) + 11,
        )
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Товар: дата выдачи: 17.03.2024.", []),
        ("Талон, дата выдачи: 17.03.2024.", []),
        ("Дата выдачи: 17.03.2024.", [("PASSPORT_DATE", "17.03.2024")]),
        ("Товар получен; дата выдачи паспорта: 17.03.2024.", [("PASSPORT_DATE", "17.03.2024")]),
        ("Паспорт Товарова выдан 17.03.2024.", [("PASSPORT_DATE", "17.03.2024")]),
        ("Телефон не указан, номер заказа: 79991234567.", []),
        ("Номер карты не указан, артикул товара: 4111111111111111.", []),
        ("Номер заказа не указан, телефон: 79991234567.", [("PHONE", "79991234567")]),
    ],
)
def test_explicit_owner_priority_and_context_word_boundaries(text, expected):
    assert [(span.type, text[span.start : span.end]) for span in detect(text)] == expected
