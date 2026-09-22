"""Bounded private NER vocabulary, independent of a particular model runtime.

The model advertises its capabilities; a consumer policy still decides which
types to mask. Native name/address components keep their original boundaries.
"""
from __future__ import annotations

import math

LEGACY_NER_TYPES = ("PERSON", "LOCATION")
NER_ENTITY_TYPES = (*LEGACY_NER_TYPES, "EMAIL", "PHONE", "CARD", "PASSPORT", "DRIVER_LICENSE", "INN",
                    "SNILS", "OMS", "IP_ADDRESS", "URL", "MILITARY_ID", "BIRTH_CERTIFICATE")
NER_TYPES = frozenset(NER_ENTITY_TYPES)
MAX_ENTITIES = 2048
MAX_NER_SPAN = 2048


def max_entity_chars(kind: str) -> int:
    """Transport limits; chunk overlap covers the largest accepted entity."""
    if kind == "URL":
        return MAX_NER_SPAN
    return 320 if kind == "EMAIL" else 200


def validate_entity(item: dict, text_length: int) -> tuple[int, int, float, str]:
    """Shared fail-closed wire contract, with no model or HTTP dependencies."""
    if not isinstance(item, dict) or set(item) != {"start", "end", "score", "entity_type"}:
        raise ValueError("Invalid NER entity.")
    start, end, score, kind = item["start"], item["end"], item["score"], item["entity_type"]
    if (not isinstance(kind, str) or kind not in NER_TYPES
            or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= text_length or end - start > max_entity_chars(kind)
            or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("Invalid NER entity.")
    return start, end, score, kind


def analyzer_entities(analyzer) -> list[str]:
    """Legacy analyzers retain PERSON/LOCATION; extended models opt in explicitly."""
    kinds = getattr(analyzer, "supported_entities", LEGACY_NER_TYPES)
    if (not isinstance(kinds, (list, tuple)) or not kinds
            or any(not isinstance(kind, str) or kind not in NER_TYPES for kind in kinds)
            or len(set(kinds)) != len(kinds)):
        raise ValueError("Invalid NER model capabilities.")
    return list(kinds)
