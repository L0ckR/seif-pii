"""Experimental, local-only GLiNER adapter for the existing private NER API.

The default experiment uses PERSON/LOCATION labels and a 0.5 threshold.
Named schema presets and a configurable threshold support recorded ablations.
Model installation belongs to the operator; importing this module does not
import Torch or load/download any model.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from threading import Lock

MODEL_NAME = "fastino/gliner2.5-multi-v1"
LABELS = {"person": "PERSON", "location": "LOCATION"}
SCHEMAS = {
    "person-location": list(LABELS),
    "presidio-labels": ["person", "name", "organization", "location"],
    "described-names": {
        "person": "Proper names of individual people: given names, surnames, patronymics or initials. "
                  "Exclude generic roles, job titles, kinship terms, pronouns and numeric identifiers.",
        "location": "Geographic proper names of cities, settlements, regions and countries. "
                    "Exclude organizations, institution names, generic facility types and field labels.",
        "organization": "Names or references to companies, banks, government agencies and other institutions.",
    },
}
ENTITY_TYPES = {**LABELS, "name": "PERSON", "organization": "ORGANIZATION"}
CHUNK_WORDS = 384
CHUNK_OVERLAP = 64


@dataclass(frozen=True)
class GlinerSpan:
    start: int
    end: int
    entity_type: str
    score: float


def _validate_span(item, entity_type, text):
    if not isinstance(item, dict):
        raise ValueError("Invalid GLiNER model output.")
    start, end, score = item.get("start"), item.get("end"), item.get("confidence")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(score) or not 0 <= score <= 1
            or not isinstance(item.get("text"), str) or item["text"] != text[start:end]):
        raise ValueError("Invalid GLiNER model output.")
    return GlinerSpan(start=start, end=end, entity_type=entity_type, score=float(score))


def _parse_output(output, text, schema="person-location"):
    if not isinstance(output, dict) or set(output) != {"entities"}:
        raise ValueError("Invalid GLiNER model output.")
    entities = output["entities"]
    if not isinstance(entities, dict) or set(entities) != set(SCHEMAS[schema]):
        raise ValueError("Invalid GLiNER model output.")
    unique = {}
    for label, items in entities.items():
        if not isinstance(items, list):
            raise ValueError("Invalid GLiNER model output.")
        for item in items:
            span = _validate_span(item, ENTITY_TYPES[label], text)
            # Organization is a competing label, not a gateway PII target. Its
            # output must still satisfy the same offset/confidence contract.
            if span.entity_type == "ORGANIZATION":
                continue
            identity = (span.start, span.end, span.entity_type)
            if identity not in unique or unique[identity].score < span.score:
                unique[identity] = span
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.entity_type))


def _validate_configuration(schema, threshold):
    if not isinstance(schema, str) or schema not in SCHEMAS:
        raise ValueError("SEIF_GLINER_SCHEMA must name a supported schema preset.")
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("SEIF_GLINER_THRESHOLD must be finite and strictly between 0 and 1.")


def _local_checkpoint(model_path):
    if not model_path:
        raise RuntimeError("SEIF_GLINER_MODEL_PATH must name an installed local checkpoint directory.")
    path = Path(model_path).expanduser().resolve()
    required = ("config.json", "encoder_config/config.json", "tokenizer_config.json", "tokenizer.json",
                "model.safetensors")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise RuntimeError("Required local GLiNER checkpoint files are missing.")
    return path


class GlinerAnalyzer:
    """Presidio-compatible analyzer using a preloaded GLiNER2 extractor.

    A per-instance lock serializes model calls even outside the service's
    bounded worker. Short input uses the unmodified original string; long
    input uses GLiNER's word windows and document-offset merge API.
    """

    model_name = MODEL_NAME

    def __init__(self, extractor, *, schema="person-location", threshold=0.5):
        _validate_configuration(schema, threshold)
        self.extractor = extractor
        self.schema = schema
        self.threshold = float(threshold)
        self.entity_schema = SCHEMAS[schema].copy()
        self._model_lock = Lock()

    @classmethod
    def from_local(cls, model_path, *, device="cpu", schema="person-location", threshold=0.5):
        _validate_configuration(schema, threshold)
        path = _local_checkpoint(model_path)
        if device not in {"cpu", "cuda"}:
            raise ValueError("SEIF_GLINER_DEVICE must be cpu or cuda.")
        from gliner2 import AutoExtractor

        extractor = AutoExtractor.from_pretrained(
            str(path), map_location=device, quantize=False, compile=False, local_files_only=True,
        )
        extractor.eval()
        return cls(extractor, schema=schema, threshold=threshold)

    @classmethod
    def from_env(cls):
        return cls.from_local(os.getenv("SEIF_GLINER_MODEL_PATH", ""),
                              device=os.getenv("SEIF_GLINER_DEVICE", "cpu"),
                              schema=os.getenv("SEIF_GLINER_SCHEMA", "person-location"),
                              threshold=float(os.getenv("SEIF_GLINER_THRESHOLD", "0.5")))

    def _extract(self, text):
        # Use the exact model splitter for the same word limit as its collator.
        # If an injected extractor has no splitter, the long API is always safe.
        processor = getattr(self.extractor, "processor", None)
        splitter = getattr(processor, "word_splitter", None)
        short = callable(splitter) and len(list(islice(splitter(text, lower=False), CHUNK_WORDS + 1))) <= CHUNK_WORDS
        options = {"threshold": self.threshold, "include_spans": True, "include_confidence": True,
                   "overlap_policy": "flat"}
        if short:
            # Upstream may append a final period; reserve its extra word slot
            # without changing the original text or truncating any input word.
            return self.extractor.extract_entities(text, self.entity_schema, max_len=CHUNK_WORDS + 1, **options)
        return self.extractor.extract_entities_long(
            text, self.entity_schema, chunk_size=CHUNK_WORDS, chunk_overlap=CHUNK_OVERLAP,
            batch_size=1, num_workers=0, **options,
        )

    def analyze(self, *, text, language, entities, score_threshold):
        if not isinstance(text, str) or language != "ru" or entities != ["PERSON", "LOCATION"]:
            raise ValueError("Unsupported GLiNER analyzer request.")
        if score_threshold != 0.0:
            raise ValueError("The gateway score threshold must be zero; configure the GLiNER threshold on the analyzer.")
        if not text:
            return []
        with self._model_lock:
            return _parse_output(self._extract(text), text, self.schema)
