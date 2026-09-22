"""Lossless reversible replacements, without saving the entire original text."""
from __future__ import annotations

import secrets


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
    for item in reversed(record["replacements"]):
        text = text[:item["start"]] + item["original"] + text[item["end"]:]
    return text


def restore_tokens(text: str, record: dict) -> str:
    import re
    mapping = {item["replacement"]: item["original"] for item in record["replacements"]}
    pattern = re.compile(r"⟦PD:[A-Z_]+:[0-9a-f]{16}⟧")
    matches = list(pattern.finditer(text))
    if not matches or any(match.group() not in mapping for match in matches):
        raise ValueError("Unknown or missing token")
    # One pass avoids interpreting original text as another replacement token.
    return pattern.sub(lambda match: mapping[match.group()], text)
