"""Lossless reversible replacements, without saving the entire original text."""
from __future__ import annotations

import re
import secrets

_TOKEN = re.compile(r"⟦PD:[A-Z][A-Z0-9_]{1,39}:[0-9a-f]{16}⟧")


class RestorationTooLarge(ValueError):
    """Token restoration would exceed the configured output character limit."""


def shape_mask(value: str) -> str:
    return "".join("*" if char.isalnum() else char for char in value)


def synthetic_value(kind: str, value: str, ordinal: int) -> str:
    # Deliberately invalid identifiers: synthetic values must not become real cards.
    values = {
        "PERSON": f"Тестов{ordinal} Макет Макетович", "CARDHOLDER": f"DEMO USER {ordinal}",
        "EMAIL": f"person{ordinal}@example.invalid", "PHONE": "+0 (000) 000-00-00",
        "PASSPORT": "0000 000000", "DRIVER_LICENSE": "00 00 000000", "INN": "000000000000",
        "CARD": "0000 0000 0000 0000", "CVV": "***", "PIN": "****",
        "ADDRESS": f"г. Макетный, ул. Тестовая, д. {ordinal}", "CITY": "Макетный",
        "STREET": "Тестовая", "HOUSE": "0", "APARTMENT": "0", "POSTAL_CODE": "000000",
        "LOCATION": "Макетная область", "COUNTRY": "Тестовая страна", "CITIZENSHIP": "Тестовое гражданство",
        "BIRTH_PLACE": "г. Макетный", "PASSPORT_ISSUER": "ТЕСТОВЫЙ ОРГАН",
        "DEPARTMENT_CODE": "000-000", "BIRTH_DATE": "01.01.1900", "PASSPORT_DATE": "01.01.2000",
        "FOREIGN_DOCUMENT": "DEMO000000",
    }
    return values.get(kind, shape_mask(value))


def mask(text: str, spans, mode: str) -> tuple[str, list[dict]]:
    parts, replacements, offset, length = [], [], 0, 0
    seen = {}
    for span in spans:
        before = text[offset:span.start]
        parts.append(before)
        length += len(before)
        original = text[span.start:span.end]
        identity = (span.type, original)
        if mode == "mask":
            replacement = shape_mask(original)
        elif mode == "token":
            if identity not in seen:
                token = f"⟦PD:{span.type}:{secrets.token_hex(8)}⟧"
                while token in text:
                    token = f"⟦PD:{span.type}:{secrets.token_hex(8)}⟧"
                seen[identity] = token
            replacement = seen[identity]
        else:
            seen.setdefault(identity, synthetic_value(span.type, original, len(seen) + 1))
            replacement = seen[identity]
        replacements.append({"start": length, "end": length + len(replacement),
                             "original": original, "replacement": replacement})
        parts.append(replacement)
        length += len(replacement)
        offset = span.end
    parts.append(text[offset:])
    return "".join(parts), replacements


def restore_exact(record: dict) -> str:
    text = record["masked"]
    parts, offset = [], 0
    for item in record["replacements"]:
        start, end = item["start"], item["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not offset <= start < end <= len(text)
            or text[start:end] != item["replacement"]
        ):
            raise ValueError("Invalid replacement coordinates or masked value")
        parts.extend((text[offset:start], item["original"]))
        offset = end
    parts.append(text[offset:])
    return "".join(parts)


def restore_tokens(text: str, record: dict, *, max_output_chars: int | None = None) -> str:
    if max_output_chars is not None and (type(max_output_chars) is not int or max_output_chars < 0):
        raise ValueError("Output character limit must be a nonnegative integer")
    mapping = {item["replacement"]: item["original"] for item in record["replacements"]}
    found, output_chars = False, len(text)
    for match in _TOKEN.finditer(text):
        token = match.group()
        if token not in mapping:
            raise ValueError("Unknown or missing token")
        found = True
        output_chars += len(mapping[token]) - (match.end() - match.start())
    if not found:
        raise ValueError("Unknown or missing token")
    if max_output_chars is not None and output_chars > max_output_chars:
        # Count the full result before substitution. Repeated tokens must not
        # turn a bounded request into a much larger allocation and response.
        raise RestorationTooLarge("Restored text exceeds the output character limit")
    # One pass avoids interpreting original text as another replacement token.
    return _TOKEN.sub(lambda match: mapping[match.group()], text)
