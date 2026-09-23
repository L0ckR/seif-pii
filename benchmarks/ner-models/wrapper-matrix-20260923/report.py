"""Render the frozen benchmark JSON as reviewable Markdown tables."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = (("rubert", "RuBERT"), ("spacy", "ru_core_news_sm"))
WRAPPERS = (("seif", "СЕЙФ"), ("pii_guard", "PII Guard"), ("presidio", "Presidio"))
CORPORA = (("organizer", "Golden"), ("pii", "PII-bench"), ("redmadrobot", "Redmadrobot"))


def percent(value):
    return f"{value * 100:.2f}%".replace(".", ",")


def table(result, dataset):
    lines = [
        "| NER | Обёртка | Покрытие P | Покрытие R | Покрытие F1 | Точные сущности F1 | С объединением F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, model_label in MODELS:
        for wrapper, wrapper_label in WRAPPERS:
            row = result["corpora"][dataset][wrapper + "_" + model]
            character = row["character"]
            lines.append("| " + " | ".join((
                model_label, wrapper_label, percent(character["precision"]), percent(character["recall"]),
                percent(character["f1"]), percent(row["entity_exact"]["f1"]),
                percent(row["entity_joined"]["f1"]),
            )) + " |")
    return "\n".join(lines)


def main():
    result = json.loads((HERE / "result.json").read_text())
    sections = ["# Сравнение трёх обёрток и двух NER-моделей", ""]
    sections.append(
        "Пересчёт выполнен после исправлений целостности ФИО/адреса и публичного контекста "
        "в СЕЙФ. На всех 5095 текстах применены одни и те же сохранённые выходы моделей "
        "и одна неизменённая разметка. Заново прогнаны все шесть связок модель + обёртка; "
        "нейросетевой инференс взят из проверенных кэшей."
    )
    sections.extend(["", "## Результаты", ""])
    for key, label in CORPORA:
        sections.extend([f"### {label} — {result['cases'][key]} текстов", "", table(result, key), ""])
    sections.extend([
        "## Как считали", "",
        "**Покрытие P/R/F1** — микрооценка букв и цифр внутри всех исходных размеченных "
        "фрагментов. Каждая обёртка передаёт итоговые распознанные интервалы; берётся их "
        "объединение. Все gold-типы и все предсказанные типы учтены. Тип не влияет на эту метрику.",
        "",
        "**Точные сущности F1** — совпадение типа и обоих исходных смещений после одной общей "
        "карты семейств. **С объединением F1** — тот же критерий после симметричного "
        "объединения пересекающихся или разделённых пробелами соседних фрагментов одного "
        "типа как в разметке, так и в ответе. Например, три отдельные части ФИО в "
        "Redmadrobot могут считаться одним PERSON. Незнакомые типы не удаляются и дают FP.",
        "",
        "PII-bench состоит из 900 domain и 910 entity текстов. Redmadrobot содержит 2839 "
        "проверенных текстов: две строки опубликованного test набора исключены до инференса "
        "из-за несовпадения BIO-токенов с исходным текстом. Golden содержит 446 уникальных "
        "примеров с предварительной AI-разметкой, из них 329 уверенных. Эти выборки ранее "
        "использовались при разработке, поэтому результаты не являются слепым holdout.",
        "",
        "RuBERT — `lockR/rubert-base-pii-ner-tensorrt` revision "
        "`73be581047bf123dac6505e7b3900ec292942296`, декодер `word/native`, "
        "без порога уверенности. Хеши весов совпали с текущими закреплёнными файлами. "
        "spaCy — `ru_core_news_sm 3.8.0`, один общий сохранённый транспорт PERSON/LOCATION. "
        "RuBERT обучался на данных семейства Redmadrobot; возможное пересечение с тестом "
        "не проверено. Сопоставлять его лидерство там с оценкой независимого теста нельзя.",
        "",
        "СЕЙФ — текущие правила и merge. PII Guard — исходные правила, mapping, email-gate и "
        "разрешение пересечений commit `24230abb72949a9f85499244dd4f15a0ad0cdd9e`; "
        "его транслитерация, преобразование английских чисел и base64 выключены ради "
        "совпадения смещений на исходном тексте. Его фиксированный NER score 0,70 — "
        "приоритет конфликта, а не порог модели. Это контролируемый прогон слоя обнаружения, "
        "не полного upstream pipeline. Presidio `2.2.364` использует штатные русские "
        "распознаватели email, телефона, карты, даты, URL, IP, IBAN, криптоадреса; "
        "spaCy PERSON/LOCATION распознаватель заменён входом из общего кэша. "
        "Его остальные правила и механизм контекста сохранены.",
        "",
        "Результаты показывают качество обнаружения на исходных смещениях, без сетевого API "
        "и замера задержки. Для СЕЙФ точное обратное восстановление маски проверено на "
        "каждом из 5095 текстов. Результаты Golden не являются официальной метрикой организатора.",
        "",
        "## Дополнительные срезы", "",
        "Golden, только 329 уверенных примеров:", "", table(result, "organizer_certain"), "",
        "Детальные метрики PII-bench domain/entity и полные TP/FP/FN всех профилей "
        "доступны в `result.json`; плоская таблица — `table.csv`.",
        "",
        "## Повторение", "",
        "```bash",
        ".venv/bin/python benchmarks/ner-models/wrapper-matrix-20260923/prepare.py",
        ".venv/bin/python benchmarks/ner-models/wrapper-matrix-20260923/seif_run.py",
        "SEIF_BENCH_SPACY_MODEL=/path/to/ru_core_news_sm-3.8.0 \\",
        "  /home/lockr/projects/seif-pii-rubert/local-data/pii-guard-review/venv/bin/python \\",
        "  benchmarks/ner-models/wrapper-matrix-20260923/wrappers_run.py",
        ".venv/bin/python benchmarks/ner-models/wrapper-matrix-20260923/score.py",
        ".venv/bin/python benchmarks/ner-models/wrapper-matrix-20260923/report.py",
        "```",
        "",
        "Объединённая копия исходных текстов и покейсные предсказания хранятся в игнорируемом "
        "`local-data/wrapper-matrix-20260923`. Публичный отчёт содержит агрегаты и хеши.",
    ])
    (HERE / "README.md").write_text("\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
