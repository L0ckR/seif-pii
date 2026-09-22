#!/usr/bin/env python3
"""Reproducible synthetic RU detector comparison, never an HTTP SLA benchmark.

Run in an isolated ordinary CPython 3.13 environment with presidio-analyzer
2.2.364, spacy 3.8.16 and the official ru_core_news_sm 3.8.0 model installed.
Both detectors run in the same process. No API, Redis, remote model or service
is called. The small fresh challenge was frozen before either detector ran.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make the numerical-library thread count explicit before spaCy imports NumPy.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRESH_FIXTURES = [
    ("fresh_01", "Передайте договор {{PERSON|Дине Шварц}}, пожалуйста."),
    ("fresh_02", "Вчера я встретил {{PERSON|Александру Михайловну Коваленко}}."),
    ("fresh_03", "Меня зовут {{PERSON|Ноа Бен-Ами}}."),
    ("fresh_04", "Заёмщик {{PERSON|Ли Марк Юрьевич}} просит изменить срок платежа."),
    ("fresh_05", "Клиентка {{PERSON|Sofia Müller}} оставила заявку."),
    ("fresh_06", "Свяжитесь с {{PERSON|Карлосом да Силвой}} завтра."),
    ("fresh_07", "Номер клиента в Лондоне: {{PHONE|+44 20 7946 0123}}."),
    ("fresh_08", "Для связи: {{PHONE|+1 (202) 555-0198}}."),
    ("fresh_09", "Его телефон — {{PHONE|+49 30 123456}}."),
    ("fresh_10", "Контактный номер: {{PHONE|+33 1 99 00 12 34}}."),
    ("fresh_11", "Мобильный клиента: {{PHONE|+7.916.234.56.78}}."),
    ("fresh_12", "Почта для ответа: {{EMAIL|first_last+notice@example.org}}; тема — заявление."),
    ("fresh_13", "Отправитель <{{EMAIL|a.b-c@example.net}}> ожидает звонка."),
    ("fresh_14", "Оплата картой {{CARD|5555 5555 5555 4444}}."),
    ("fresh_15", "Реквизит клиента, карта: {{CARD|378282246310005}}."),
    ("fresh_16", "Рождена {{BIRTH_DATE|29 февраля 1992 года}}, место рождения: {{BIRTH_PLACE|г. Томск}}."),
    ("fresh_17", "Дата рождения: {{BIRTH_DATE|1985.31.12}}; дата выдачи: {{PASSPORT_DATE|2010-01-20}}."),
    ("fresh_18", "Паспорт серия {{PASSPORT|40 11 номер 654321}}."),
    ("fresh_19", "ИНН: {{INN|123456789012}}; ПИН: {{PIN|9087}}."),
    ("fresh_20", "Адрес регистрации: {{ADDRESS|г. Тула, ул. Северная, д. 14/2, кв. 91}}; прошу обновить запись."),
    ("fresh_21", "Памятник Михаилу Лермонтову находится в Москве."),
    ("fresh_22", "Перескажи биографию поэта Бориса Пастернака."),
    ("fresh_23", "Адрес филиала банка: г. Тула, ул. Центральная, д. 4."),
    ("fresh_24", "Номер заказа 123456789012, размер партии 4321, версия 3.12.1."),
    ("fresh_25", "Конференция состоится 20.11.2027, доклад длится 40 минут."),
    ("fresh_26", "Компания Северный Ветер открыла новый офис."),
    ("fresh_27", "Как безопасно хранить номер карты и CVV, если сами значения не указаны?"),
    ("fresh_28", "В статье о поэте Борисе Пастернаке клиент {{PERSON|Борис Пастернак}} указал {{EMAIL|reader@example.org}}."),
]
FROZEN_AT = "2026-09-22T07:41:45Z"
FROZEN_SOURCE_SHA256 = "31c36e36e19fe553858ba4f335030919ed3344fccdf638ee1c4880beff2ebdbb"

# Only these four types have a direct semantic counterpart without inference.
COMMON = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "CREDIT_CARD": "CARD"}
COARSE = {
    "PERSON": "PERSON", "CARDHOLDER": "PERSON", "EMAIL": "EMAIL_ADDRESS", "PHONE": "PHONE_NUMBER",
    "CARD": "CREDIT_CARD", "BIRTH_DATE": "DATE_TIME", "PASSPORT_DATE": "DATE_TIME",
    "BIRTH_PLACE": "LOCATION", "ADDRESS": "LOCATION", "COUNTRY": "LOCATION", "CITY": "LOCATION",
    "STREET": "LOCATION", "CITIZENSHIP": "NRP", "PASSPORT_ISSUER": "ORGANIZATION",
}
SOURCES = {
    "languages": "https://presidio.dataprivacystack.org/analyzer/languages/",
    "nlp_configuration": "https://presidio.dataprivacystack.org/analyzer/nlp_engines/spacy_stanza/",
    "entities": "https://presidio.dataprivacystack.org/supported_entities/",
    "russian_model": "https://spacy.io/models/ru",
    "model_release": "https://github.com/explosion/spacy-models/releases/tag/ru_core_news_sm-3.8.0",
    "project_transition": "https://presidio.dataprivacystack.org/project_transition/",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(fixtures) -> str:
    return hashlib.sha256(json.dumps(fixtures, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def build_presidio():
    import tldextract
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        DateRecognizer,
        EmailRecognizer,
        PhoneRecognizer,
        SpacyRecognizer,
    )

    # EmailRecognizer calls tldextract.extract. Use its bundled PSL snapshot,
    # preventing the library's optional network refresh during measurement.
    tldextract.extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ru", "model_name": "ru_core_news_sm"}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION"},
            "labels_to_ignore": [],
        },
    }
    engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    recognizers = [
        SpacyRecognizer(supported_language="ru"),
        EmailRecognizer(supported_language="ru", context=["email", "e-mail", "почта", "электронная"]),
        PhoneRecognizer(supported_language="ru", supported_regions=("RU",) + PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS,
                        context=["телефон", "тел", "мобильный", "номер", "связь", "контакт"]),
        CreditCardRecognizer(supported_language="ru", context=["карта", "карты", "платёжная", "кредитная", "банковская", "card"]),
        DateRecognizer(supported_language="ru", context=["дата", "рождение", "рождения", "выдан", "выдача", "выдачи", "родился", "родилась"]),
    ]
    registry = RecognizerRegistry(supported_languages=["ru"])
    for recognizer in recognizers:
        registry.add_recognizer(recognizer)
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=engine, supported_languages=["ru"], default_score_threshold=0.0)
    metadata = {
        "name": "stock Russian spaCy model + upstream generic recognizers adapted to ru",
        "upstream_english_default": False,
        "nlp_engine": configuration,
        "spacy_pipeline": engine.nlp["ru"].pipe_names,
        "spacy_ner_labels": list(engine.nlp["ru"].get_pipe("ner").labels),
        "score_threshold": 0.0,
        "threshold_selection": "Upstream analyzer default, fixed before measurement; no corpus-based tuning",
        "context_adaptation": "Russian context words and RU phone region; unchanged upstream patterns, checksums and NER model",
        "network_during_analysis": "none; tldextract uses bundled public suffix snapshot",
        "excluded_entities": "URL, IP, crypto, IBAN and country-specific non-Russian IDs are outside this assignment and are not enabled",
        "custom_russian_identifier_rules": False,
        "recognizers": [{"name": item.name, "supported_language": item.supported_language,
                         "supported_entities": item.supported_entities, "context": item.context,
                         "phone_regions": list(getattr(item, "supported_regions", []))} for item in recognizers],
    }
    assert all(item.supported_language == "ru" for item in recognizers)
    return analyzer, metadata


def metric(expected, predicted) -> dict:
    from scripts.evaluate import scores

    totals = Counter()
    by_type = {}
    exact_cases = 0
    negatives = negatives_with_fp = 0
    failures = []
    for case_id in expected:
        wanted, found = expected[case_id], predicted[case_id]
        exact_cases += wanted == found
        if not wanted:
            negatives += 1
            negatives_with_fp += bool(found)
        groups = {"tp": wanted & found, "fp": found - wanted, "fn": wanted - found}
        for name, spans in groups.items():
            totals[name] += len(spans)
            for kind, _, _ in spans:
                by_type.setdefault(kind, Counter())[name] += 1
        if wanted != found:
            failures.append({"case_id": case_id, "false_positive": sorted(groups["fp"]), "false_negative": sorted(groups["fn"])})
    return {**scores(totals["tp"], totals["fp"], totals["fn"]), "exact_cases": exact_cases,
            "evaluated_cases": len(expected), "negative_cases": negatives, "negative_cases_with_fp": negatives_with_fp,
            "by_type": {kind: scores(c["tp"], c["fp"], c["fn"]) for kind, c in sorted(by_type.items())},
            "failures": failures}


def remap(rows, mapping, keep_unknown=False):
    return {case_id: {(mapping.get(kind, "UNMAPPED:" + kind), start, end)
                      for kind, start, end in spans if keep_unknown or kind in mapping}
            for case_id, spans in rows.items()}


def evaluate_corpus(fixtures, detect, analyzer):
    from scripts.evaluate import parse_fixture

    truth, ours, theirs, texts, raw = {}, {}, {}, {}, []
    for case_id, marked in fixtures:
        text, expected = parse_fixture(marked)
        truth[case_id], texts[case_id] = expected, text
        ours[case_id] = {(span.type, span.start, span.end) for span in detect(text)}
        results = analyzer.analyze(text=text, language="ru", score_threshold=0.0)
        theirs[case_id] = {(span.entity_type, span.start, span.end) for span in results}
        raw.append({"case_id": case_id, "text": text, "expected": sorted(expected), "seif": sorted(ours[case_id]),
                    "presidio": [{"type": span.entity_type, "start": span.start, "end": span.end, "score": round(span.score, 5)} for span in results]})
    strict_identity = {kind: kind for kind in COMMON.values()}
    common_truth = remap(truth, strict_identity)
    coarse_truth = remap(truth, COARSE)
    coarse_theirs = remap(theirs, {kind: kind for kind in COARSE.values()})
    common_case_ids = [case_id for case_id, spans in truth.items() if all(kind in strict_identity for kind, _, _ in spans)]

    def selected(rows):
        return {case_id: rows[case_id] for case_id in common_case_ids}

    def untyped(rows):
        return {key: {("ANY_PII", start, end) for _, start, end in spans} for key, spans in rows.items()}

    report = {
        "case_count": len(fixtures), "annotated_spans": sum(map(len, truth.values())), "canonical_corpus_sha256": canonical_hash(fixtures),
        "common_strict4": {
            "selection_method": "Whole cases whose gold labels are a subset of PERSON/EMAIL/PHONE/CARD, including all gold-empty cases; no filtering based on predictions",
            "case_ids": common_case_ids,
            "seif": metric(selected(common_truth), selected(remap(ours, strict_identity))),
            "presidio": metric(selected(common_truth), selected(remap(theirs, COMMON))),
        },
        "strict4_allcases_diagnostic": {"seif": metric(common_truth, remap(ours, strict_identity)), "presidio": metric(common_truth, remap(theirs, COMMON))},
        "fine_taxonomy_requirement_fit": {"seif": metric(truth, ours), "presidio": metric(truth, remap(theirs, COMMON, keep_unknown=True))},
        "coarse_diagnostic": {"seif": metric(coarse_truth, remap(ours, COARSE)), "presidio": metric(coarse_truth, coarse_theirs)},
        "untyped_exact_span_coverage": {"seif": metric(untyped(truth), untyped(ours)), "presidio": metric(untyped(truth), untyped(theirs))},
        "raw_cases": raw,
    }
    return report, texts


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 6)


def benchmark(texts_by_corpus, detect, analyzer, repeats, warmup):
    calls = {"seif": lambda text: detect(text), "presidio": lambda text: analyzer.analyze(text=text, language="ru", score_threshold=0.0)}
    rows = [(corpus, case_id, text) for corpus, texts in texts_by_corpus.items() for case_id, text in texts.items()]
    for _ in range(warmup):
        for _, _, text in rows:
            for call in calls.values():
                call(text)
    rng = random.Random(20260922)
    samples = {corpus: {name: [] for name in calls} for corpus in texts_by_corpus}
    for _ in range(repeats):
        rng.shuffle(rows)
        for corpus, _, text in rows:
            order = list(calls)
            rng.shuffle(order)
            for name in order:
                start = time.perf_counter_ns()
                calls[name](text)
                samples[corpus][name].append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "scope": "detector-only, same process/interpreter, one caller, full analyzer including NLP versus full local detector; no mask/vault/HTTP",
        "warmup_full_corpus_passes": warmup, "measured_passes": repeats, "order": "seeded interleaved random order, seed=20260922",
        "corpora": {corpus: {name: {"samples": len(values), "mean_ms": round(sum(values) / len(values), 6),
                                   "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95),
                                   "p99_ms": percentile(values, 99), "max_ms": round(max(values), 6)}
                              for name, values in data.items()} for corpus, data in samples.items()},
        "raw_latency_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/presidio-comparison.json")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    logging.basicConfig(level=logging.ERROR)
    from scripts.evaluate import FIXTURES
    from seif.detector import TYPES, detect

    detector_sha = sha256(ROOT / "seif/detector.py")
    init_started = time.perf_counter()
    analyzer, configuration = build_presidio()
    init_seconds = time.perf_counter() - init_started
    corpora, texts = {}, {}
    for name, fixtures in (("development48", FIXTURES), ("fresh28", FRESH_FIXTURES)):
        corpora[name], texts[name] = evaluate_corpus(fixtures, detect, analyzer)
    load_before = os.getloadavg()
    latency = benchmark(texts, detect, analyzer, args.repeats, args.warmup)
    if sha256(ROOT / "seif/detector.py") != detector_sha:
        raise RuntimeError("Detector changed during comparison; discard this run and use a fixed revision")
    reverse_common = {value: key for key, value in COMMON.items()}
    report = {
        "schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Exact (type, Unicode character start, end) micro precision/recall/F1; untyped spans and coarse taxonomy explicitly separate",
        "environment": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
                        "load_average_before": load_before, "load_average_after": os.getloadavg(),
                        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "same_interpreter_for_both": True,
                        "numpy_thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                        "installed_packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))},
        "source_hashes": {"seif_detector_sha256": detector_sha, "development_evaluator_sha256": sha256(ROOT / "scripts/evaluate.py"),
                          "comparison_script_sha256": sha256(Path(__file__))},
        "fresh_challenge_freeze": {"frozen_at_utc": FROZEN_AT, "original_file_sha256": FROZEN_SOURCE_SHA256,
                                   "canonical_data_sha256": canonical_hash(FRESH_FIXTURES),
                                   "before_first_measurement": True, "text_withheld_from_detector_author_until_changes_finished": True},
        "presidio_configuration": configuration, "presidio_cold_initialization_seconds": round(init_seconds, 4),
        "type_mapping": {kind: {"label": label, "strict_presidio_counterpart": reverse_common.get(kind),
                                "coarse_counterpart_only": None if kind in reverse_common else COARSE.get(kind),
                                "custom_ru_rule_or_context_classifier_required": kind not in reverse_common} for kind, label in TYPES.items()},
        "corpora": corpora, "latency": latency, "official_sources": SOURCES,
        "limitations": [
            "Development48 was used to improve SEIF and is biased in its favor; do not present its score as independent validation.",
            "Fresh28 is a small author-written synthetic challenge frozen before this comparison, not a representative independent corpus.",
            "The main like-for-like comparison is common_strict4: only whole cases whose gold types are a subset of PERSON/EMAIL/PHONE/CARD plus all gold-empty cases; selection never uses predictions.",
            "strict4_allcases_diagnostic filters only span types across every case; this can count native PERSON on a CARDHOLDER gold span as FP, so it is not the headline comparison.",
            "Fine taxonomy requirement-fit penalizes DATE_TIME/LOCATION/ORGANIZATION labels lacking the requested Russian field meaning; it is not a universal NER quality ranking.",
            "Coarse diagnostics deliberately collapse meanings (e.g. date purpose, cardholder vs person) and retain strict boundaries; partial address extraction still fails exact matching.",
            "Public authors, bank offices and event dates are annotated by the assignment's PII policy; recognizing them can be correct NER yet a false positive under this policy.",
            "Presidio uses its pretrained small Russian model and upstream generic rules with Russian context, not default English; custom Russian IDs, stronger models or policy recognizers could improve it.",
            "No thresholds, labels, patterns or models were tuned on this frozen challenge after observing results.",
            "Latency is sequential detector-only in the same ordinary CPython process; it does not measure HTTP RPS, storage, masking, multi-worker scaling or production 3.14t performance.",
            "No external service or model endpoint receives the examples; dependencies and the model were downloaded before measurement.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"versions": {name: report["environment"]["installed_packages"].get(name) for name in ("presidio_analyzer", "presidio-analyzer", "spacy", "ru_core_news_sm")},
               "common_strict4": {name: {system: {key: results["common_strict4"][system][key] for key in ("precision", "recall", "f1")} for system in ("seif", "presidio")} for name, results in corpora.items()},
               "latency": latency["corpora"], "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
