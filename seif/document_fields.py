"""Document number formats and explicit departmental fields.

Grouped 4+6 and 2+2+6 values require document ownership; an adjacent series
and number pair provides its own structural context. Nearby business fields
veto either inference. Bare numbers remain ambiguous and are not inferred.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

Candidate = tuple[int, int, str, float, str]
_FLAGS = re.IGNORECASE
_SEPARATOR = r"[ \t/-]"
_FORMATTED_NUMBER = re.compile(
    rf"(?<![\w./-])(?P<value>(?:[0-9]{{4}}|[0-9]{{2}}{_SEPARATOR}[0-9]{{2}})"
    rf"{_SEPARATOR}[0-9]{{6}})(?!\w|[./-][0-9]|[ \t]+[0-9])",
    _FLAGS,
)
_PART = re.compile(
    r"(?<!\w)(?:(?P<series>серия|серии|серией|сер[.])[ \t]*[:=—–-]?[ \t]*"
    r"(?P<series_value>[0-9]{2}[ \t]?[0-9]{2})(?!\w)|"
    r"(?P<number>номер(?:ом)?|ном[.]|№)[ \t]*[:=—–-]?[ \t]*(?P<number_value>[0-9]{6})(?!\w))",
    _FLAGS,
)
_PART_JOIN = re.compile(r"[ \t,;./—–-]*(?:и[ \t]+)?", _FLAGS)
_OWNER = re.compile(
    r"\b(?P<license>водительск[а-яё]*[ \t]+удостоверени[а-яё]*|в[ /]?у|права)\b|"
    r"\b(?P<passport>паспорт(?:а|ом|е)?|документ(?:а|ом|е)?)\b|"
    r"\b(?P<generic_license>удостоверени[ея])(?=[ \t]*[:=])|"
    r"\b(?P<business>(?:пенсионн[а-яё]*|служебн[а-яё]*|студенческ[а-яё]*)[ \t]+удостоверени[ея]|"
    r"заказ[а-яё]*|накладн[а-яё]*|товар[а-яё]*|артикул[а-яё]*|"
    r"договор[а-яё]*|контракт[а-яё]*|сч[её]т[а-яё]*|издели[а-яё]*|оборудовани[а-яё]*|"
    r"станк[а-яё]*|партия|партии|плат[её]ж[а-яё]*|сумм[а-яё]*|сертификат[а-яё]*|"
    r"техническ[а-яё]*|транспортн[а-яё]*|телефон[а-яё]*|инн|снилс|id|sku|"
    r"удостоверени[ея]|студенческ[а-яё]*|служебн[а-яё]*|пенсионн[а-яё]*)\b",
    _FLAGS,
)
_BUSINESS_SUFFIX = re.compile(
    r"[ \t]+(?:товара|заказа|оборудования|изделия|договора|контракта|партии|сертификата)\b",
    _FLAGS,
)
_DEPARTMENT = re.compile(
    r"\b(?:код[ \t]+подразделения|к[ /]п)[ \t]*"
    r"(?:(?:стоит|указан|записан|равен|составляет)[ \t]*)?[:=—–-]?[ \t]*"
    r"(?P<value>[0-9]{3}[ \t-][0-9]{3})(?!\w|[ \t-][0-9])",
    _FLAGS,
)


def _number_kind(text: str, start: int, *, paired: bool = False) -> str | None:
    prefix = text[max(0, start - 100):start]
    # A line or completed sentence starts a new record, while abbreviated field
    # labels such as 'сер.' must stay attached to their number.
    prefix = re.split(r"[!?\n]|\.(?=[ \t]+[А-ЯЁA-Z])", prefix)[-1]
    owners = list(_OWNER.finditer(prefix))
    if not owners:
        return "PASSPORT" if paired else None
    group = owners[-1].lastgroup
    if group == "business":
        return None
    return "DRIVER_LICENSE" if group in {"license", "generic_license"} else "PASSPORT"


def _part_candidates(text: str) -> Iterator[Candidate]:
    previous = None
    for part in _PART.finditer(text):
        if previous is not None and part.start() - previous.end() <= 48:
            different_parts = bool(previous.group("series")) != bool(part.group("series"))
            gap = text[previous.end():part.start()]
            if different_parts and _PART_JOIN.fullmatch(gap) and not _BUSINESS_SUFFIX.match(text, part.end()):
                kind = _number_kind(text, previous.start(), paired=True)
                if kind is not None:
                    for item in (previous, part):
                        field = "series_value" if item.group("series") else "number_value"
                        start, end = item.span(field)
                        yield start, end, kind, 0.94, "paired-personal-document-parts"
        previous = part


def document_candidates(text: str) -> Iterator[Candidate]:
    """Yield document formats without consuming labels or surrounding prose."""
    if not any(char.isdigit() for char in text):
        return
    for number in _FORMATTED_NUMBER.finditer(text):
        start, end = number.span("value")
        kind = _number_kind(text, start)
        if kind is not None and not _BUSINESS_SUFFIX.match(text, end):
            yield start, end, kind, 0.91, "grouped-personal-document-format"
    lower = text.lower()
    if "сер" in lower and ("ном" in lower or "№" in text):
        yield from _part_candidates(text)
    if "подразделения" in lower or "к/п" in lower or "к п" in lower:
        for field in _DEPARTMENT.finditer(text):
            start, end = field.span("value")
            yield start, end, "DEPARTMENT_CODE", 0.99, "explicit-department-field"
