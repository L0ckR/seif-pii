"""Keep model entity boundaries when rule coverage and meaning are identical.

Rules often recognize a complete name/address while a model identifies its
components. Keeping those components improves typed extraction without changing
which alphanumeric characters are masked. No labels, corpus IDs, dictionaries,
or learned thresholds participate in this decision.
"""
from __future__ import annotations

import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import replace
from ipaddress import ip_address
from itertools import accumulate
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seif.detector import Span

_ADDRESSES = frozenset({"ADDRESS", "LOCATION", "COUNTRY", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE",
                        "APARTMENT", "POSTAL_CODE"})


def _family(kind: str) -> str:
    return "ADDRESS" if kind in _ADDRESSES else kind


def _word_boundary(text: str, offset: int) -> bool:
    if offset in {0, len(text)}:
        return True
    return not all(character.isalnum() or unicodedata.category(character).startswith("M")
                   for character in text[offset - 1:offset + 1])


def _compatible_parts(text: str, original: Span, parts: Sequence[Span]) -> bool:
    """Only complete non-overlapping words/components with matching semantics."""
    cursor = original.start
    for part in parts:
        if (part.start < cursor or part.end > original.end or part.end <= part.start
                or _family(part.type) != _family(original.type)
                or not _word_boundary(text, part.start) or not _word_boundary(text, part.end)):
            return False
        cursor = part.end
    return True


def _remainder(text: str, original: Span, start: int, end: int) -> list[Span]:
    if not any(character.isalnum() for character in text[start:end]):
        return []
    while start < end and text[start].isspace():
        start += 1
    while start < end and text[end - 1].isspace():
        end -= 1
    return [replace(original, start=start, end=end)]


def _partition(text: str, original: Span, parts: Sequence[Span]) -> list[Span]:
    result = []
    cursor = original.start
    for part in parts:
        result.extend(_remainder(text, original, cursor, part.start))
        result.append(replace(original, start=part.start, end=part.end))
        cursor = part.end
    result.extend(_remainder(text, original, cursor, original.end))
    return result


def _atomic_ip(text: str, span: Span) -> bool:
    if span.type != "IP_ADDRESS":
        return False
    # A valid complete network address is one identifier. Native IPv6 groups
    # must not split it into several purported addresses; they are not IPs.
    value = text[span.start:span.end].replace(" ", "").replace("\t", "")
    try:
        ip_address(value)
    except ValueError:
        return False
    return True


def preserve_model_boundaries(text: str, resolved: Sequence[Span], model_candidates: Sequence[Span]) -> list[Span]:
    """Refine already validated spans; preserve coverage, rule type and provenance.

    Only wholly contained, unambiguous model components of the same semantic
    family may partition a resolved span. Unmodeled rule remainders retain every
    protected letter/digit. Explicit custom rules and valid complete IP addresses
    keep their boundaries; individual hex groups are not independent IPs.
    Inputs are validated by the detector; output retains its non-overlap invariant.
    """
    if not resolved or not model_candidates:
        return list(resolved)
    models = sorted(model_candidates, key=lambda span: (span.start, span.end, span.type))
    starts = [span.start for span in models]
    cover_ends = list(accumulate((span.end for span in models), max))
    output = []
    for original in resolved:
        first = bisect_right(cover_ends, original.start)
        last = bisect_left(starts, original.end)
        parts = [span for span in models[first:last] if span.end > original.start]
        unchanged = len(parts) == 1 and (parts[0].start, parts[0].end) == (original.start, original.end)
        if (unchanged or original.reason == "custom-rule" or not parts
                or _atomic_ip(text, original) or not _compatible_parts(text, original, parts)):
            output.append(original)
            continue
        # Keep the authoritative rule's type, confidence and reason. Only its
        # boundaries change; model LOCATION does not assert a particular CITY.
        output.extend(_partition(text, original, parts))
    return output
