"""Replay fixed NER spans through Presidio and PII Guard rules/resolvers.

Run with the pinned local 3.12 environment. PII Guard's optional translit,
base64 and number preprocessing is disabled to retain raw Unicode offsets.
The same rule configuration is used with both cached NER models.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "local-data/wrapper-matrix-20260923"
UPSTREAM = ROOT.parent / "seif-pii-rubert/local-data/pii-guard-review/upstream"
SPACY_MODEL = Path(os.environ["SEIF_BENCH_SPACY_MODEL"])
RUBERT = ROOT.parent / "seif-pii-rubert/local-data/rubert-upgrade/run-v2/native.jsonl"
GUARD_REVISION = "24230abb72949a9f85499244dd4f15a0ad0cdd9e"
sys.path[:0] = [str(ROOT), str(UPSTREAM / "src")]

logging.disable(logging.WARNING)


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def verify_runtime():
    if not SPACY_MODEL.is_dir() or json.loads((SPACY_MODEL / "meta.json").read_text())["version"] != "3.8.0":
        raise ValueError("Expected ru_core_news_sm 3.8.0 model directory")
    versions = {name: importlib.metadata.version(name) for name in ("presidio-analyzer", "spacy")}
    if versions != {"presidio-analyzer": "2.2.364", "spacy": "3.8.16"}:
        raise ValueError("Wrapper dependency versions differ")
    git = shutil.which("git")
    if git is None:
        raise ValueError("Git is required to verify PII Guard revision")
    revision = subprocess.check_output([git, "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()  # noqa: S603
    dirty = subprocess.check_output(  # noqa: S603
        [git, "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if revision != GUARD_REVISION or dirty:
        raise ValueError("PII Guard source revision differs")
    return versions


def encode(results):
    return [{"start": result.start, "end": result.end, "type": result.entity_type,
             "score": round(float(result.score), 6)}
            for result in sorted(results, key=lambda item: (item.start, item.end, item.entity_type))]


def build_presidio():
    import tldextract
    from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        CryptoRecognizer,
        DateRecognizer,
        EmailRecognizer,
        IbanRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        UrlRecognizer,
    )

    tldextract.extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    configuration = {
        "nlp_engine_name": "spacy", "models": [{"lang_code": "ru", "model_name": str(SPACY_MODEL)}],
        "ner_model_configuration": {"model_to_presidio_entity_mapping": {
            "PER": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION",
        }, "labels_to_ignore": []},
    }
    nlp = NlpEngineProvider(nlp_configuration=configuration).create_engine()

    class CachedNerRecognizer(EntityRecognizer):
        def __init__(self):
            kinds = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                     "PASSPORT", "DRIVER_LICENSE", "INN", "SNILS", "OMS", "MILITARY_ID",
                     "BIRTH_CERTIFICATE", "IP_ADDRESS", "URL"]
            super().__init__(supported_entities=kinds, supported_language="ru", name="FixedCachedNER")
            self.current = []

        def load(self):
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            return [RecognizerResult(entity_type=entity["type"], start=entity["start"],
                                     end=entity["end"], score=entity["score"])
                    for entity in self.current if entity["type"] in entities]

    recognizers = [
        EmailRecognizer(supported_language="ru", context=["email", "e-mail", "почта", "электронная"]),
        PhoneRecognizer(supported_language="ru", supported_regions=("RU",) + PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS,
                        context=["телефон", "тел", "мобильный", "номер", "связь", "контакт"]),
        CreditCardRecognizer(supported_language="ru",
                             context=["карта", "карты", "платёжная", "кредитная", "банковская", "card"]),
        DateRecognizer(supported_language="ru",
                       context=["дата", "рождение", "рождения", "выдан", "выдача", "выдачи", "родился", "родилась"]),
        UrlRecognizer(supported_language="ru"), IpRecognizer(supported_language="ru"),
        IbanRecognizer(supported_language="ru"), CryptoRecognizer(supported_language="ru"),
    ]
    cache = CachedNerRecognizer()
    registry = RecognizerRegistry(supported_languages=["ru"])
    for recognizer in (*recognizers, cache):
        registry.add_recognizer(recognizer)
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp,
                              supported_languages=["ru"], default_score_threshold=0.0)
    return analyzer, cache


def neural_rubert(record, mapping, text, score):
    spans = []
    for entity in record["native"]:
        kind = mapping[entity["label"]]
        start, end = entity["start"], entity["end"]
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid RuBERT span")
        spans.append({"type": kind, "start": start, "end": end, "score": score(entity)})
    return spans


def main():
    versions = verify_runtime()
    from pii_guard.config import Config
    from pii_guard.detect import resolve_conflicts, resolve_ml_vs_rules_conflicts
    from pii_guard.engine import NER_ENTITY_MAPPING, Engine
    from pii_guard.entities._email_shape import filter_ner_email_spans
    from presidio_analyzer import RecognizerResult

    guard_config = Config(spacy_model=str(SPACY_MODEL), enable_translit=False,
                          enable_en_numbers=False, enable_base64=False)
    guard = Engine(guard_config, ner=False)
    presidio, cached = build_presidio()
    output = LOCAL / "wrappers.jsonl"
    with output.open("w") as stream:
        for index, (row, rubert, spacy) in enumerate(zip(
            records(LOCAL / "inputs.jsonl"), records(RUBERT), records(LOCAL / "spacy.jsonl"), strict=True,
        ), 1):
            text = row["text"]
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            if not (row["key"] == rubert["case_id"] == spacy["case_id"]
                    and fingerprint == rubert["text_sha256"] == spacy["text_sha256"]):
                raise ValueError("Cache/input mismatch")
            rules = guard._rules_analyzer.analyze(text=text, language="ru")
            models = {
                "rubert": neural_rubert(rubert, NER_ENTITY_MAPPING, text, lambda e: e["score"]),
                "spacy": [{"type": e["entity_type"], "start": e["start"], "end": e["end"],
                           "score": e["score"]} for e in spacy["entities"]],
            }
            profiles = {}
            for model, candidates in models.items():
                fixed = [RecognizerResult(entity_type=e["type"], start=e["start"],
                                          end=e["end"], score=guard_config.ner_score) for e in candidates]
                fixed = filter_ner_email_spans(fixed, text)
                merged = resolve_ml_vs_rules_conflicts(rules, fixed)
                final = resolve_conflicts(merged, text, registry=guard._numeric_recognizer.registry)
                profiles["pii_guard_" + model] = encode(final)
                cached.current = candidates
                found = presidio.analyze(text=text, language="ru", score_threshold=0.0)
                profiles["presidio_" + model] = encode(found)
            stream.write(json.dumps({"case_id": row["key"], "text_sha256": fingerprint,
                                     "profiles": profiles}, ensure_ascii=False) + "\n")
            if index % 250 == 0:
                print(f"Wrappers replay: {index}", flush=True)
    print(f"Wrappers replay complete: {index} cases", flush=True)
    with output.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    (LOCAL / "wrappers.meta.json").write_text(json.dumps({
        "cache_sha256": fingerprint, "cases": index, "versions": versions,
        "pii_guard_revision": GUARD_REVISION,
        "spacy_model_sha256": hashlib.sha256((SPACY_MODEL / "meta.json").read_bytes()).hexdigest(),
        "pii_guard_preprocessors_enabled": {"translit": False, "english_numbers": False, "base64": False},
        "presidio_recognizers": ["cached-NER", "email", "phone", "credit-card", "date",
                                  "URL", "IP", "IBAN", "crypto"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
