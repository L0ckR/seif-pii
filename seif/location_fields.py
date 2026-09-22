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
    r"отдел(?:ение|ением|ом)?|управлени[ея]м?|(?:[гу]+|мо)[ \t]+мвд|умвд|[оугм]вд|[оу]{0,2}фмс|мвд)"
)
_AUTHORITY = re.compile(
    r"\b(?:(?:[гу]+|мо)[ \t]+мвд|умвд|[оугм]вд|[оу]{0,2}фмс|мвд|"
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
_ISSUER_PROSE = re.compile(
    r"\b(?:совпадает|подтвержд[её]н[аоы]?|указан[аоы]?|выдал[ао]?|"
    r"в[ \t]+(?:прошлом|текущем|этом|позапрошлом)[ \t]+году)\b|"
    r"[ \t]*\([ \t]*(?:ранее|бывш[а-яё]*)\b|"
    r"[ \t]+[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4}(?!\d)", _FLAGS,
)
_ISSUER_CONTEXT = re.compile(
    r"\b(?:выда(?:н(?:ный|ная|ное|ные|ного|а|о)?|л[ао]?)|орган[ \t]+выдачи)\b", _FLAGS,
)
_AUTHORITY_START = re.compile(rf"\b{_AUTHORITY_HEAD}\b", _FLAGS)
_AFTER_BIRTH_DATE_PLACE = re.compile(
    r"\bродил(?:ся|ась)[ \t]+[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{4}"
    r"[ \t]+в[ \t]+(?P<value>(?:г[.]|городе?)[ \t]+"
    r"[а-яё]{2,35}(?:-[а-яё]{2,35}){0,3}(?:[ \t]+[а-яё]{2,35})?)"
    r"(?=[ \t]*(?:[,;.!?\n]|$))", _FLAGS,
)
_RELATIVE_ISSUE_YEAR = re.compile(r"\bв[ \t]+(?P<value>(?:прошлом|позапрошлом|этом|текущем)[ \t]+году)\b", _FLAGS)
_PERSONAL_ISSUE = re.compile(r"\bпаспорт(?:а)?\b[^;!?\n]{0,80}\bвыдан[а-яё]*\b", _FLAGS)
_NONPERSONAL_PASSPORT = re.compile(r"\b(?:оборудовани[а-яё]*|издели[а-яё]*|станк[а-яё]*|техническ[а-яё]*)\b", _FLAGS)
_ISSUED_OWNER = re.compile(r"\b(?:(?P<personal>паспорт)|(?P<business>чек|талон|сертификат|заказ|билет))\b", _FLAGS)
_PUBLIC_BIRTH_OWNER = re.compile(r"\b(?:поэт[а-яё]*|писател[а-яё]*|композитор[а-яё]*)\b", _FLAGS)
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
_HOUSE = re.compile(r"(?<!\w)(?:дом|д)[.]?[ \t]+(?a:\d){1,4}(?!(?a:\d))", _FLAGS)
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
_STREET_TYPE = r"(?:улица|ул[.]|проспект|просп[.]|пр[.]|пр-кт|пр-т|бульвар|бул[.]|переулок|пер[.])"
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
    end = issuer_value_end(text, start, end)
    if end == start + 220 and end < len(text):
        return None
    start, end = _trim(text, start, end)
    authority = _AUTHORITY.search(text, start, end)
    if authority is not None and not _MISSING_AUTHORITY.search(text, start, authority.start()):
        return start, end, "PASSPORT_ISSUER", 0.99, "explicit-passport-authority-field"
    return None


def issuer_value_end(text: str, start: int, end: int) -> int:
    """End a recognized authority before a later field or narrative clause."""
    end = _field_end(text, start, end - start)
    prose = _ISSUER_PROSE.search(text, start, end)
    return prose.start() if prose else end


def _business_issue_context(prefix: str) -> bool:
    owners = list(_ISSUED_OWNER.finditer(prefix))
    return bool(owners and owners[-1].lastgroup == "business")


def _public_address_context(text: str, start: int) -> bool:
    prefix = text[max(0, start - 220):start]
    # Adjacent lines can continue corporate requisites; a blank line starts a
    # separate record and must not inherit ownership from the prior paragraph.
    prefix = re.split(r"\n[ \t]*\n|[;!?]", prefix)[-1]
    public = list(_PUBLIC_OWNER.finditer(prefix))
    private = list(_PRIVATE_OWNER.finditer(prefix))
    return bool(public and (not private or public[-1].start() > private[-1].start()))


def _labelled_issuer_candidates(text: str) -> Iterator[Candidate]:
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


def _issuer_candidates(text: str) -> Iterator[Candidate]:
    labelled = list(_labelled_issuer_candidates(text))
    yield from labelled
    covered = iter(sorted((candidate[0], candidate[1]) for candidate in labelled))
    next_interval = next(covered, None)
    covered_end = -1
    for authority in _AUTHORITY_START.finditer(text):
        while next_interval is not None and next_interval[0] <= authority.start():
            covered_end = max(covered_end, next_interval[1])
            next_interval = next(covered, None)
        if authority.start() < covered_end:
            continue
        prefix = re.split(r"[;!?\n]", text[max(0, authority.start() - 220):authority.start()])[-1]
        contexts = list(_ISSUER_CONTEXT.finditer(prefix))
        if (contexts and not _MISSING_AUTHORITY.search(prefix, contexts[-1].end())
                and not _NONPERSONAL_PASSPORT.search(prefix)
                and not _business_issue_context(prefix)
                and _field_end(prefix, contexts[-1].end()) == len(prefix)):
            candidate = _issuer_value(text, authority.start())
            if candidate is not None:
                covered_end = candidate[1]
                yield candidate


def _registration_candidates(text: str) -> Iterator[Candidate]:
    for label in _PRIVATE_ADDRESS_LABEL.finditer(text):
        start, end = label.end(), _field_end(text, label.end(), 240)
        if end == start + 240 and end < len(text):
            continue
        start, end = _trim(text, start, end)
        if _HOUSE.search(text, start, end) and _ADDRESS_COMPONENT.search(text, start, end):
            yield start, end, "ADDRESS", 0.99, "explicit-private-registration-field"


def _relative_issue_dates(text: str) -> Iterator[Candidate]:
    for match in _RELATIVE_ISSUE_YEAR.finditer(text):
        prefix = text[max(0, match.start() - 180):match.start()]
        issue = _PERSONAL_ISSUE.search(prefix)
        if (issue and not _NONPERSONAL_PASSPORT.search(prefix)
                and _field_end(prefix, issue.end()) == len(prefix)):
            yield *match.span("value"), "PASSPORT_DATE", 0.97, "relative-personal-document-date"


def _birth_date_places(text: str) -> Iterator[Candidate]:
    for match in _AFTER_BIRTH_DATE_PLACE.finditer(text):
        prefix = re.split(r"[;!?\n]", text[max(0, match.start() - 100):match.start()])[-1]
        if not _PUBLIC_BIRTH_OWNER.search(prefix) or _PRIVATE_OWNER.search(prefix):
            yield *match.span("value"), "BIRTH_PLACE", 0.99, "birth-date-locality-field"


def _residential_addresses(text: str) -> Iterator[Candidate]:
    for match in _RESIDENTIAL_LINE.finditer(text):
        start, end = match.span("value")
        if not _public_address_context(text, start):
            yield start, end, "ADDRESS", 0.99, "residential-address-components"


def location_candidates(text: str) -> Iterator[Candidate]:
    """Yield explicit passport issuers and complete residential field values."""
    if not text:
        return
    lower = text.lower()
    if "выда" in lower:
        yield from _issuer_candidates(text)
        yield from _relative_issue_dates(text)
    if "родил" in lower:
        yield from _birth_date_places(text)
    if "регистрации" in lower:
        yield from _registration_candidates(text)
    if "квартира" in lower or "кв." in lower:
        yield from _residential_addresses(text)
