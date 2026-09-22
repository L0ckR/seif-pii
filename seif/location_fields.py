"""Bounded issuer and residential-address fields with explicit ownership cues.

This module returns unchanged Unicode offsets and has no dependency on detector
resolution. Bare geography and institutions without personal context are omitted.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

Candidate = tuple[int, int, str, float, str]
_FLAGS = re.IGNORECASE
_AUTHORITY_HEAD = (
    r"(?:паспортно-визов[а-яё]*[ \t]+(?:отдел[а-яё]*|служб[а-яё]*)|"
    r"отдел(?:ение|ением|ом)?|управлени[ея]м?|[гу]+[ \t]+мвд|[оугм]вд|[оу]{0,2}фмс|мвд)"
)
_AUTHORITY = re.compile(
    r"\b(?:[гу]+[ \t]+мвд|[оугм]вд|[оу]{0,2}фмс|мвд|"
    r"паспортно-визов[а-яё]*[ \t]+(?:отдел[а-яё]*|служб[а-яё]*)|"
    r"отдел[а-яё]*[ \t]+внутренних[ \t]+дел)\b", _FLAGS,
)
_ISSUER_LABEL = re.compile(
    r"\b(?:орган[ \t]+выдачи(?:[ \t]+паспорта)?|"
    r"орган[ \t]*,[ \t]*выдавший[ \t]+паспорт|кем[ \t]+выдан[ \t]+паспорт)"
    r"[ \t]*[:=—–-][ \t]*", _FLAGS,
)
_REISSUE_LABEL = re.compile(
    r"\bповторный(?:[ \t]+паспорт)?[ \t]+(?:(?:был[ \t]+)?выдан[ \t]*|[—–-][ \t]*)", _FLAGS,
)
_PERSONAL_PASSPORT = re.compile(r"\bпаспорт(?:[ \t]+был)?[ \t]+выдан\b", _FLAGS)
_ISSUER_POST = re.compile(
    rf"\b(?P<value>{_AUTHORITY_HEAD}\b[^;,\n!?]{{0,160}}?)[ \t]+"
    r"выдал[ао]?[ \t]+паспорт[ \t]+(?:гражданину|гражданке|клиенту|клиентке|заявителю|заявительнице)\b",
    _FLAGS,
)
_ISSUER_NEXT_FIELD = re.compile(
    r"[, \t]+(?=\b(?:код[ \t]+подразделения|дата[ \t]+выдачи|дата[ \t]+рождения|"
    r"повторный|телефон|адрес|фио|паспорт)[ \t:—–-])", _FLAGS,
)
_MISSING_AUTHORITY = re.compile(
    r"\b(?:не[ \t]+(?:указан[а-яё]*|извест[а-яё]*|определ[её]н[а-яё]*)|"
    r"неизвест[а-яё]*|отсутств[а-яё]*|нет[ \t]+данных|"
    r"обратит[а-яё]*|обратиться|уточнит[а-яё]*|уточнить)\b", _FLAGS,
)
_SENTENCE_END = re.compile(r"[;\n!?]|[.](?=[ \t]*(?:[А-ЯЁ]|$))")
_ABBREVIATION = re.compile(r"\b(?:г|гор|обл|р-н|им|д|ул|кв|корп|стр|пос|дер|просп|пр|пер|наб|бул|п|с)\.$", _FLAGS)
_PRIVATE_ADDRESS_LABEL = re.compile(
    r"\bместо[ \t]+регистрации[ \t]+(?:за[её]мщика|клиента|пациента|заявителя|поручителя)"
    r"[ \t]*[:=—–-][ \t]*", _FLAGS,
)
_HOUSE = re.compile(r"(?<!\w)(?:дом|д)[.]?[ \t]+\d{1,4}(?!\d)", _FLAGS)
_ADDRESS_COMPONENT = re.compile(
    r"(?<!\w)(?:ул[.]|улица\b|дер[.]|деревня\b|обл[.]|область\b|город\b|г[.])", _FLAGS,
)
_PUBLIC_OWNER = re.compile(
    r"\b(?:ооо|пао|оао|зао|ао|банк[а-яё]*|офис[а-яё]*|филиал[а-яё]*|компани[а-яё]*|"
    r"организаци[а-яё]*|юрлиц[а-яё]*|юридическ[а-яё]*[ \t]+лиц[а-яё]*)\b", _FLAGS,
)
_PRIVATE_OWNER = re.compile(
    r"\b(?:клиент[а-яё]*|за[её]мщик[а-яё]*|пациент[а-яё]*|поручител[а-яё]*|"
    r"проживани[а-яё]*|домашн[а-яё]*)\b", _FLAGS,
)
_WORD = r"[а-яё]{1,35}(?:-[а-яё]{1,35}){0,3}"
_NAME = rf"{_WORD}(?:[ \t]+{_WORD}){{0,3}}"
_NUMBER = r"[0-9]{1,4}[а-яё]?(?:/[0-9а-яё]{1,5})?"
_STREET_TYPE = r"(?:улица|ул[.]|проспект|просп[.]|пр[.]|бульвар|бул[.]|переулок|пер[.])"
_RESIDENTIAL_LINE = re.compile(
    rf"^[ \t]*(?P<value>(?:[0-9]{{6}},[ \t]*)?(?:город|г[.])[ \t]*{_NAME},[ \t]*"
    rf"(?:{_STREET_TYPE}[ \t]+{_NAME}|{_NAME}[ \t]+{_STREET_TYPE}),[ \t]*"
    rf"(?:дом|д[.])[ \t]*{_NUMBER}"
    rf"(?:,[ \t]*(?:строение|стр[.]|корпус|корп[.])[ \t]*{_NUMBER}){{0,2}},[ \t]*"
    rf"(?:квартира|кв[.])[ \t]*{_NUMBER})(?=[ \t]*(?:$|[.;\n]))", _FLAGS | re.MULTILINE,
)


def _field_end(text: str, start: int, limit: int = 220) -> int:
    end = min(len(text), start + limit)
    for match in _SENTENCE_END.finditer(text, start, end):
        if match.group() == "." and _ABBREVIATION.search(text[max(start, match.start() - 8):match.end()]):
            continue
        return match.start()
    return end


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in " \t":
        start += 1
    while start < end and text[end - 1] in " \t.,":
        end -= 1
    if start < end and (text[start], text[end - 1]) in {('"', '"'), ("«", "»"), ("“", "”")}:
        start, end = start + 1, end - 1
    return start, end


def _issuer_value(text: str, start: int) -> Candidate | None:
    end = _field_end(text, start)
    next_field = _ISSUER_NEXT_FIELD.search(text, start, end)
    if next_field is not None:
        end = next_field.start()
    if end == start + 220 and end < len(text):
        return None
    start, end = _trim(text, start, end)
    authority = _AUTHORITY.search(text, start, end)
    if authority is not None and not _MISSING_AUTHORITY.search(text, start, authority.start()):
        return start, end, "PASSPORT_ISSUER", 0.99, "explicit-passport-authority-field"
    return None


def _public_address_context(text: str, start: int) -> bool:
    prefix = text[max(0, start - 220):start]
    # Adjacent lines can continue corporate requisites; a blank line starts a
    # separate record and must not inherit ownership from the prior paragraph.
    prefix = re.split(r"\n[ \t]*\n|[;!?]", prefix)[-1]
    public = list(_PUBLIC_OWNER.finditer(prefix))
    private = list(_PRIVATE_OWNER.finditer(prefix))
    return bool(public and (not private or public[-1].start() > private[-1].start()))


def _issuer_candidates(text: str) -> Iterator[Candidate]:
    for label in _ISSUER_LABEL.finditer(text):
        candidate = _issuer_value(text, label.end())
        if candidate is not None:
            yield candidate
    for label in _REISSUE_LABEL.finditer(text):
        prefix = text[max(0, label.start() - 220):label.start()]
        if _PERSONAL_PASSPORT.search(prefix):
            candidate = _issuer_value(text, label.end())
            if candidate is not None:
                yield candidate
    for match in _ISSUER_POST.finditer(text):
        start, end = _trim(text, *match.span("value"))
        yield start, end, "PASSPORT_ISSUER", 0.99, "personal-passport-issuing-authority"


def _registration_candidates(text: str) -> Iterator[Candidate]:
    for label in _PRIVATE_ADDRESS_LABEL.finditer(text):
        start, end = label.end(), _field_end(text, label.end(), 240)
        if end == start + 240 and end < len(text):
            continue
        start, end = _trim(text, start, end)
        if _HOUSE.search(text, start, end) and _ADDRESS_COMPONENT.search(text, start, end):
            yield start, end, "ADDRESS", 0.99, "explicit-private-registration-field"


def location_candidates(text: str) -> Iterator[Candidate]:
    """Yield explicit passport issuers and complete residential field values."""
    if not text:
        return
    lower = text.lower()
    if "выда" in lower:
        yield from _issuer_candidates(text)
    if "регистрации" in lower:
        yield from _registration_candidates(text)
    if "квартира" in lower or "кв." in lower:
        for match in _RESIDENTIAL_LINE.finditer(text):
            start, end = match.span("value")
            if not _public_address_context(text, start):
                yield start, end, "ADDRESS", 0.99, "residential-address-components"
