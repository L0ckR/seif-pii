"""Regression and boundary tests; these are development tests, not a bank benchmark."""

import time

import pytest

from seif.detector import TYPES, Span, detect, validate_extra_rule


def extracted(text):
    return [(span.type, text[span.start : span.end]) for span in detect(text)]


@pytest.mark.parametrize(
    "text,kind,value",
    [
        ("Клиент Иванов Иван Иванович.", "PERSON", "Иванов Иван Иванович"),
        ("клиент иван иванович петров", "PERSON", "иван иванович петров"),
        ("ИВАНОВ ИВАН ИВАНОВИЧ", "PERSON", "ИВАНОВ ИВАН ИВАНОВИЧ"),
        ("Перевод Ивану Петрову", "PERSON", "Ивану Петрову"),
        ("Заёмщик: Петров А. С.", "PERSON", "Петров А. С."),
        ("Клиент Иван Петров пришёл.", "PERSON", "Иван Петров"),
        ("дата рождения: 31.12.1990", "BIRTH_DATE", "31.12.1990"),
        ("дата рождения: 12.31.1990", "BIRTH_DATE", "12.31.1990"),
        ("дата рождения: 1990.31.12", "BIRTH_DATE", "1990.31.12"),
        ("дата рождения: 1990-12-31", "BIRTH_DATE", "1990-12-31"),
        ("Родилась 29 февраля 2000 года", "BIRTH_DATE", "29 февраля 2000 года"),
        ("д.р.: 2000 года, 29 февраля", "BIRTH_DATE", "2000 года, 29 февраля"),
        ("Дата рождения: февраля 29, 2000", "BIRTH_DATE", "февраля 29, 2000"),
        ("12 мая 1990 года рождения", "BIRTH_DATE", "12 мая 1990 года"),
        ("место рождения: г. Самара.", "BIRTH_PLACE", "г. Самара"),
        ("Родилась в Казани.", "BIRTH_PLACE", "Казани"),
        ("паспорт 4509 123456", "PASSPORT", "4509 123456"),
        ("ПАСПОРТ РФ: 4509123456", "PASSPORT", "4509123456"),
        ("гражданство: РФ", "CITIZENSHIP", "РФ"),
        ("Гражданин России", "CITIZENSHIP", "России"),
        ("Гражданство: Российская Федерация", "CITIZENSHIP", "Российская Федерация"),
        ("Кем выдан: ОУФМС России по г. Москве;", "PASSPORT_ISSUER", "ОУФМС России по г. Москве"),
        ("Код подразделения: 770-001", "DEPARTMENT_CODE", "770-001"),
        ("Дата выдачи: 20 января 2020 года", "PASSPORT_DATE", "20 января 2020 года"),
        ("Паспорт выдан 2020.20.01", "PASSPORT_DATE", "2020.20.01"),
        ("Водительское удостоверение: 77 12 123456", "DRIVER_LICENSE", "77 12 123456"),
        ("В/У: 77 АА 123456", "DRIVER_LICENSE", "77 АА 123456"),
        ("Адрес: г. Москва, ул. Мира, д. 3, кв. 4.", "ADDRESS", "г. Москва, ул. Мира, д. 3, кв. 4"),
        ("Страна проживания: Россия.", "COUNTRY", "Россия"),
        ("Индекс: 125009", "POSTAL_CODE", "125009"),
        ("Город: Ростов-на-Дону", "CITY", "Ростов-на-Дону"),
        ("Улица: Большая Никитская.", "STREET", "Большая Никитская"),
        ("Дом: 12А корпус 3", "HOUSE", "12А корпус 3"),
        ("Квартира: 35", "APARTMENT", "35"),
        ("Почта First.Last+tag@example.com.", "EMAIL", "First.Last+tag@example.com"),
        ("почта тест@пример.рф", "EMAIL", "тест@пример.рф"),
        ("Телефон +7 (999) 123-45-67", "PHONE", "+7 (999) 123-45-67"),
        ("Телефон: 999 123 45 67", "PHONE", "999 123 45 67"),
        ("Связаться +44 20 7946 0958", "PHONE", "+44 20 7946 0958"),
        ("ИНН: 500100732259", "INN", "500100732259"),
        ("INN 500100732259", "INN", "500100732259"),
        ("Карта: 4111 1111 1111 1111", "CARD", "4111 1111 1111 1111"),
        ("Перевод на 4111111111111111", "CARD", "4111111111111111"),
        ("CVV2: 123", "CVV", "123"),
        ("CVC-код: 123", "CVV", "123"),
        ("ПИН-код карты: 9876", "PIN", "9876"),
        ("Держатель карты: IVAN PETROV", "CARDHOLDER", "IVAN PETROV"),
        ("Загранпаспорт: 75 1234567", "FOREIGN_DOCUMENT", "75 1234567"),
        ("Свидетельство о рождении: IV-АБ 123456", "FOREIGN_DOCUMENT", "IV-АБ 123456"),
    ],
)
def test_exact_entities(text, kind, value):
    assert extracted(text) == [(kind, value)]


def test_document_field_label_is_not_personal_data():
    assert extracted("паспорт: серия 45 09 номер 123456") == [
        ("PASSPORT", "45 09"), ("PASSPORT", "123456"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Александр Пушкин написал стихотворение.",
        "Поэт Александр Сергеевич Пушкин родился в Москве.",
        "Роман Александра Пушкина находится в библиотеке.",
        "Адрес отделения Банка: г. Москва, ул. Ленина, д. 12.",
        "Филиал банка расположен по адресу: г. Казань, ул. Мира, д. 4.",
        "Встреча назначена на 12.01.2025 в 15:30.",
        "Товар выдан 12.03.2024.",
        "ПИН-код карты нужен для оплаты, но здесь нет его значения.",
        "Номер заявки 1234567890, сумма 1000 рублей, версия 3.12.",
        "Контрольная сумма 1111111111111111.",
        "Дата рождения: 31.02.2000",  # Both DMY and MDY invalid.
        "Дата рождения: 29 февраля 2001 года",
        "Email: not-an-address@invalid",
        "Пришло 100 сообщений; продолжительность 123 секунд.",
        "",
    ],
)
def test_negatives(text):
    assert detect(text) == []


def test_public_reference_does_not_disable_private_namesake():
    text = "Поэт Александр Пушкин известен; клиент Александр Пушкин, телефон +7 999 123-45-67."
    assert extracted(text) == [("PERSON", "Александр Пушкин"), ("PHONE", "+7 999 123-45-67")]
    assert detect(text)[0].start == text.rindex("Александр")


def test_bank_branch_does_not_disable_later_personal_address():
    text = "Адрес отделения банка: г. Москва, ул. Мира, д. 2; адрес клиента: г. Казань, ул. Ленина, д. 3."
    assert extracted(text) == [("ADDRESS", "г. Казань, ул. Ленина, д. 3")]


def test_address_does_not_swallow_following_prose():
    text = "Адрес: г. Москва, ул. Мира, д. 3. Это новый договор."
    assert extracted(text) == [("ADDRESS", "г. Москва, ул. Мира, д. 3")]


def test_issuer_overlap_city_and_late_passport_date():
    text = "Паспорт 4509 123456 выдан ОУФМС России по г. Москве, 12.03.2020, код подразделения 770-001."
    assert extracted(text) == [
        ("PASSPORT", "4509 123456"),
        ("PASSPORT_ISSUER", "ОУФМС России по г. Москве"),
        ("PASSPORT_DATE", "12.03.2020"),
        ("DEPARTMENT_CODE", "770-001"),
    ]


def test_unicode_offsets_and_deterministic_sorted_nonoverlap():
    text = "🔐 Иванов Иван Иванович — паспорт 4509 123456; email: x@example.com; CVV: 123."
    spans = detect(text)
    assert spans == detect(text)
    assert all(isinstance(s, Span) and 0 <= s.start < s.end <= len(text) for s in spans)
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:], strict=False))
    assert text[spans[0].start : spans[0].end] == "Иванов Иван Иванович"
    assert spans[0].start == 2


def test_custom_entity_extension():
    rule = {"type": "EMPLOYEE_ID", "pattern": r"\bEMP-\d{6}\b"}
    text = "Табельный EMP-000123; email a@example.com"
    spans = detect(text, extra_rules=[rule])
    assert [(s.type, text[s.start : s.end]) for s in spans] == [
        ("EMPLOYEE_ID", "EMP-000123"),
        ("EMAIL", "a@example.com"),
    ]


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "invalid type", "pattern": "test"},
        {"type": "CUSTOM", "pattern": "["},
        {"type": "CUSTOM", "pattern": ".*"},
        {"type": "CUSTOM", "pattern": "a" * 513},
        {"type": "CUSTOM"},
    ],
)
def test_custom_rule_validation(rule):
    with pytest.raises(ValueError):
        validate_extra_rule(rule)


def test_pathological_custom_regex_fails_closed():
    started = time.monotonic()
    with pytest.raises(ValueError, match="time budget"):
        detect("a" * 10000 + "!", extra_rules=[{"type": "CUSTOM", "pattern": r"(a+)+$"}])
    assert time.monotonic() - started < 2


def test_custom_rule_count_bound():
    with pytest.raises(ValueError, match="32"):
        detect("test", extra_rules=[{"type": "CUSTOM", "pattern": "test"}] * 33)


def test_large_text_has_bounded_runtime_and_preserves_end_offsets():
    # 100k whitespace tokens, >600k code points; protects against regex blowups.
    text = "нейтральный текст " * 50000 + "\nEmail: end@example.com"
    started = time.monotonic()
    spans = detect(text)
    assert time.monotonic() - started < 5
    assert [(s.type, text[s.start : s.end]) for s in spans] == [("EMAIL", "end@example.com")]


def test_every_documented_type_has_russian_label():
    # Core categories plus optional model-only PII classes.
    assert len(TYPES) == 31
    assert all(repr(label) and any("А" <= c <= "я" for c in label) for label in TYPES.values())


@pytest.mark.parametrize(
    "text,value",
    [
        ("Иванов Иван Иванович работает в банке.", "Иванов Иван Иванович"),
        ("Петров Петр Петрович отправил письмо.", "Петров Петр Петрович"),
        ("Клиент Иванов Иван Иванович заключил договор.", "Иванов Иван Иванович"),
        ("Анна Ивановна Соколова обратилась.", "Анна Ивановна Соколова"),
        ("Обратился Иван Иванович Ким.", "Иван Иванович Ким"),
        ("Письмо отправлено Марии Ивановне Петровой.", "Марии Ивановне Петровой"),
    ],
)
def test_full_name_boundaries_in_prose(text, value):
    assert extracted(text) == [("PERSON", value)]


def test_public_reference_and_bank_requisites_realistic_paragraph():
    # This is the complete unchanged context example from the demo. Narrow
    # ownership/apposition rules must handle prose, not just minimal fixtures.
    text = (
        "Подготовь справку для посетителей.\n\n"
        "Александр Сергеевич Пушкин — русский поэт, автор романа «Евгений Онегин».\n\n"
        "Публичные реквизиты банка:\nПАО Сбербанк\n"
        "Адрес банка: 117312, г. Москва, ул. Вавилова, д. 19.\n"
        "БИК банка: 044525225\nИНН банка: 7707083893\n\n"
        "Сравни описание банковского сервиса с литературной метафорой."
    )
    assert detect(text) == []


@pytest.mark.parametrize(
    "text,kind,value",
    [
        ("Клиент Александр Сергеевич Пушкин — русский поэт.", "PERSON", "Александр Сергеевич Пушкин"),
        ("Клиент Иван Петров — известный писатель.", "PERSON", "Иван Петров"),
        ("ИНН клиента банка: 7707083893", "INN", "7707083893"),
        ("ИНН клиента банка: 1234567890", "INN", "1234567890"),
        ("ИНН сотрудника банка: 500100732259", "INN", "500100732259"),
        ("ИНН банка: 500100732259", "INN", "500100732259"),
        ("Адрес клиента: 117312, г. Москва, ул. Вавилова, д. 19.", "ADDRESS", "117312, г. Москва, ул. Вавилова, д. 19"),
    ],
)
def test_public_context_never_exempts_explicit_personal_records(text, kind, value):
    assert extracted(text) == [(kind, value)]


@pytest.mark.parametrize(
    "text",
    [
        "Александр Сергеевич Пушкин — русский поэт.",
        "Иван Петров — известный писатель.",
        "Адрес банка: 117312, г. Москва, ул. Вавилова, д. 19.",
        "ИНН банка: 7707083893",
        "ИНН компании: 7707083893",
    ],
)
def test_explicit_public_ownership(text):
    assert detect(text) == []


def test_public_fields_do_not_exempt_later_private_fields():
    text = (
        "ИНН банка: 7707083893; ИНН клиента банка: 7707083893; "
        "Адрес банка: 117312, г. Москва, ул. Вавилова, д. 19; "
        "адрес клиента: г. Москва, ул. Вавилова, д. 19."
    )
    assert extracted(text) == [("INN", "7707083893"), ("ADDRESS", "г. Москва, ул. Вавилова, д. 19")]
