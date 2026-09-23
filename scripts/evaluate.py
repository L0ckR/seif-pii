#!/usr/bin/env python3
"""Independent, source-controlled synthetic detector check with exact span metrics.

This is a small development suite, not a representative bank dataset and not
the organizer's span-Levenshtein score. Expected spans are explicitly marked
in source before running the detector; results are never used as labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Labels include the value only: textual field names are safe context, not PII.
# All examples are fictional; example.net and example.invalid are reserved.
FIXTURES = [
    ("full_name", "Клиент {{PERSON|Иванов Иван Иванович}}, здравствуйте."),
    ("full_name_lowercase", "ФИО: {{PERSON|петров пётр петрович}}."),
    ("full_name_uppercase", "ФИО: {{PERSON|СИДОРОВ СЕРГЕЙ ИВАНОВИЧ}}."),
    ("birth_dotted", "Дата рождения: {{BIRTH_DATE|12.03.1990}}."),
    ("birth_iso", "Дата рождения: {{BIRTH_DATE|1990-03-12}}."),
    ("birth_words", "Родился {{BIRTH_DATE|12 марта 1990 года}}."),
    ("birth_uppercase", "ДАТА РОЖДЕНИЯ: {{BIRTH_DATE|12 МАРТА 1990 ГОДА}}."),
    ("birth_place", "Место рождения: {{BIRTH_PLACE|г. Казань}}."),
    ("passport_compact", "Паспорт: {{PASSPORT|4509 123456}}."),
    ("passport_split", "Паспорт: серия {{PASSPORT|4509}} номер {{PASSPORT|123456}}."),
    ("passport_uppercase", "ПАСПОРТ {{PASSPORT|45 09 123456}}."),
    ("citizenship", "Гражданство: {{CITIZENSHIP|Российская Федерация}}."),
    ("citizenship_lowercase", "гражданство: {{CITIZENSHIP|россия}}."),
    ("issuer", "Паспорт выдан {{PASSPORT_ISSUER|ОТДЕЛОМ УФМС РОССИИ ПО Г. МОСКВЕ}}; благодарю."),
    ("department", "Код подразделения: {{DEPARTMENT_CODE|770-001}}."),
    ("passport_date", "Дата выдачи паспорта: {{PASSPORT_DATE|20.04.2010}}."),
    ("passport_date_words", "Дата выдачи: {{PASSPORT_DATE|20 апреля 2010 года}}."),
    ("driver_license", "Водительское удостоверение: {{DRIVER_LICENSE|77 12 345678}}."),
    ("driver_license_short", "В/у: {{DRIVER_LICENSE|7712 345678}}."),
    ("address", "Адрес проживания: {{ADDRESS|г. Москва, ул. Тестовая, д. 17, кв. 8}}."),
    ("country", "Страна проживания: {{COUNTRY|Россия}}."),
    ("postal_code", "Индекс: {{POSTAL_CODE|123456}}."),
    ("city", "Город: {{CITY|Казань}}."),
    ("street", "Улица: {{STREET|Тестовая}}."),
    ("house", "Дом: {{HOUSE|17А}}."),
    ("apartment", "Квартира: {{APARTMENT|8}}."),
    ("email", "Напишите {{EMAIL|ivan.petrov+test@example.net}}, пожалуйста."),
    ("email_uppercase", "EMAIL: {{EMAIL|USER@EXAMPLE.NET}}!"),
    ("phone", "Телефон: {{PHONE|+7 (999) 123-45-67}}."),
    ("phone_8", "Телефон: {{PHONE|8 999 123 45 67}}."),
    ("inn", "ИНН: {{INN|7707083893}}."),
    ("inn_person", "ИНН физлица: {{INN|500100732259}}."),
    ("card", "Номер карты: {{CARD|4111 1111 1111 1111}}."),
    ("card_dashes", "Карта {{CARD|4111-1111-1111-1111}}."),
    ("cvv", "CVV: {{CVV|123}}."),
    ("pin", "ПИН-код карты: {{PIN|4321}}."),
    ("cardholder", "Имя держателя карты: {{CARDHOLDER|IVAN PETROV}}."),
    ("foreign_passport", "Загранпаспорт: {{FOREIGN_DOCUMENT|72 1234567}}."),
    (
        "complex",
        "Клиент {{PERSON|Иванов Иван Иванович}} (email: {{EMAIL|ivan@example.net}}), паспорт {{PASSPORT|4509 123456}}; телефон {{PHONE|+7 (999) 123-45-67}}.",
    ),
    ("unicode", "«Email: {{EMAIL|hello@example.net}}» — 😊;\nтелефон: {{PHONE|+7 (999) 123-45-67}}!"),
    ("repeated_email", "{{EMAIL|one@example.net}} и снова {{EMAIL|one@example.net}}."),
    ("public_poet", "Расскажи о поэте Александре Пушкине и его стихах."),
    ("bank_office", "Адрес отделения банка: г. Москва, ул. Тверская, д. 1."),
    ("weather_city", "Какая погода в Москве?"),
    ("release_date", "Дата релиза программы: 12.03.2024."),
    ("arithmetic", "Посчитай 123 + 456 и объясни значение числа 4321."),
    ("no_pii", "Поясни разницу между вкладом и накопительным счётом."),
    ("label_only", "Пин-код карты — это секрет, его нельзя сообщать другим."),
]

MARKER = re.compile(r"\{\{([A-Z_]+)\|([^{}]+)\}\}")


def parse_fixture(marked: str) -> tuple[str, set[tuple[str, int, int]]]:
    chunks: list[str] = []
    expected: set[tuple[str, int, int]] = set()
    cursor = 0
    position = 0
    for match in MARKER.finditer(marked):
        prefix = marked[cursor : match.start()]
        chunks.append(prefix)
        position += len(prefix)
        value = match.group(2)
        expected.add((match.group(1), position, position + len(value)))
        chunks.append(value)
        position += len(value)
        cursor = match.end()
    chunks.append(marked[cursor:])
    return "".join(chunks), expected


def scores(tp: int, fp: int, fn: int) -> dict:
    if tp + fp:
        precision = tp / (tp + fp)
    elif fn == 0:
        precision = 1.0
    else:
        precision = 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def evaluate() -> dict:
    from seif.detector import detect

    totals: Counter[str] = Counter()
    per_type: dict[str, Counter[str]] = {}
    failures = []
    passed = 0
    for case_id, marked in FIXTURES:
        text, expected = parse_fixture(marked)
        spans = detect(text)
        predicted = {(span.type, span.start, span.end) for span in spans}
        true_positive = expected & predicted
        false_positive = predicted - expected
        false_negative = expected - predicted
        if expected == predicted:
            passed += 1
        else:
            failures.append({"case": case_id, "expected": sorted(expected), "predicted": sorted(predicted)})
        for name, items in (("tp", true_positive), ("fp", false_positive), ("fn", false_negative)):
            totals[name] += len(items)
            for kind, _, _ in items:
                per_type.setdefault(kind, Counter())[name] += 1
    return {
        "schema_version": 1,
        "method": "exact (type, Unicode character start, end) span match; micro averaged",
        "fixture_count": len(FIXTURES),
        "cases_exact_match": passed,
        "metrics": scores(totals["tp"], totals["fp"], totals["fn"]),
        "by_type": {kind: scores(c["tp"], c["fp"], c["fn"]) for kind, c in sorted(per_type.items())},
        "failures": failures,
        "limitations": [
            "Fixtures are a small synthetic development suite, not a representative held-out bank corpus.",
            "These scores do not establish 95% accuracy on the organizer's private evaluation data.",
            "Exact-span F1 differs from the organizer's normalized span-based Levenshtein score.",
            "Reversibility and HTTP correctness are tested independently by tests/test_api.py.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
