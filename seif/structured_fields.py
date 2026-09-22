"""Bounded recognizers for explicitly labelled personal-data field values.

These additions use field ownership instead of accepting arbitrary dates or
numbers. They preserve Unicode offsets and never normalize the original value.
Returned tuples deliberately avoid a dependency on the detector's Span class.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date

Candidate = tuple[int, int, str, float, str]
_FLAGS = re.IGNORECASE
_SPACE = r"[ \t]*"
_DIGIT2 = r"(?a:\d){2}"
_MONTH_NAMES = (
    r"янв(?:арь|аря)?", r"фев(?:раль|раля)?", r"мар(?:т|та)?",
    r"апр(?:ель|еля)?", r"ма[йя]", r"июн[ья]?", r"июл[ья]?",
    r"авг(?:уст|уста)?", r"сен(?:т|тябрь|тября)?", r"окт(?:ябрь|ября)?",
    r"ноя(?:брь|бря)?", r"дек(?:абрь|абря)?",
)
_MONTH = "(?:" + "|".join(_MONTH_NAMES) + ")"
_DATE_SEP = r"(?:[./-]|[ \t]+)"
_DATE_VALUE = (
    rf"(?:[0-9]{{4}}{_DATE_SEP}[0-9]{{1,2}}{_DATE_SEP}[0-9]{{1,2}}"
    rf"|[0-9]{{1,2}}{_DATE_SEP}[0-9]{{1,2}}{_DATE_SEP}(?:[0-9]{{4}}|[0-9]{{2}})"
    rf"|[0-9]{{1,2}}{_DATE_SEP}{_MONTH}[.]?(?:{_DATE_SEP}(?:[0-9]{{4}}|[0-9]{{2}}))?"
    rf"|[0-9]{{1,2}}{_DATE_SEP}[0-9]{{1,2}})"
    r"(?:[ \t]+(?:года|год|г[.]))?(?!\w|[./-][0-9]|[ \t]+[0-9])"
)
_DATE_START = re.compile(_DATE_VALUE, _FLAGS)
_DATE_LABEL = re.compile(
    r"\b(?P<birth>дата[ \t]+рождения)\b|"
    r"\b(?P<passport>дата[ \t]+выдачи[ \t]+паспорта)\b|"
    r"\b(?P<issued>паспорт[ \t]+выдан)\b", _FLAGS,
)
_FIELD_PREFIX = re.compile(r"[ \tа-яёА-ЯЁ()—–-]{0,96}[:=][ \t]*")
_DIRECT_PREFIX = re.compile(r"[ \t]*(?:[—–-][ \t]*)?")
_UNKNOWN_FIELD = re.compile(
    r"\b(?:не[ \t]+(?:указан|заполнен|предоставлен)[а-яё]*|неизвест[а-яё]*|"
    r"отсутств[а-яё]*|ожидается|нет)\b", _FLAGS,
)
_NEXT_DATE_FIELD = re.compile(r"\b(?:дата|релиз[а-яё]*|срок[а-яё]*)\b", _FLAGS)
_PUBLIC_OWNER = re.compile(r"\b(?:поэт[а-яё]*|писател[а-яё]*|композитор[а-яё]*|банка|компании|организации)\b", _FLAGS)
_HISTORICAL_OWNER = re.compile(r"\b(?:поэт[а-яё]*|писател[а-яё]*|композитор[а-яё]*)\b", _FLAGS)
_PRIVATE_OWNER = re.compile(
    r"\b(?:клиент[а-яё]*|пациент[а-яё]*|за[её]мщик[а-яё]*|вкладчик[а-яё]*|"
    r"сотрудник[а-яё]*|представител[а-яё]*|директор[а-яё]*|поручител[а-яё]*)\b", _FLAGS,
)
_ORDINAL_STEMS = (
    "перв", "втор", "треть", "четвёрт", "пят", "шест", "седьм", "восьм", "девят", "десят",
    "одиннадцат", "двенадцат", "тринадцат", "четырнадцат", "пятнадцат", "шестнадцат",
    "семнадцат", "восемнадцат", "девятнадцат", "двадцат",
)
_ORDINAL_VALUES = {
    form: value
    for value, stem in enumerate(_ORDINAL_STEMS, 1)
    for stem in {stem, stem.replace("ё", "е")}
    for form in (stem + ("е" if value == 3 else "ое"), stem + ("его" if value == 3 else "ого"))
}
_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
         "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90}
_ORDINAL_VALUES.update(dict(zip(
    ("тридцатого", "сорокового", "пятидесятого", "шестидесятого", "семидесятого", "восьмидесятого", "девяностого"),
    range(30, 100, 10), strict=True,
)))
_ORDINAL_VALUES["тридцатое"] = 30
_ORDINAL_PATTERN = "(?:" + "|".join(_ORDINAL_VALUES) + ")"
_TENS_PATTERN = "(?:" + "|".join(_TENS) + ")"
_WRITTEN_DATE = re.compile(
    rf"(?P<day>(?:(?:двадцать|тридцать)[ \t]+)?{_ORDINAL_PATTERN})[ \t]+"
    rf"(?P<month>{_MONTH})[ \t]+"
    r"(?P<base>тысяча[ \t]+(?:восемьсот|девятьсот)|две[ \t]+тысячи)"
    rf"(?:[ \t]+(?P<remainder>(?:{_TENS_PATTERN}[ \t]+)?{_ORDINAL_PATTERN}))?"
    r"[ \t]+года(?!\w)", _FLAGS,
)


def _written_ordinal(value: str) -> int:
    parts = value.lower().split()
    if len(parts) == 1:
        return _ORDINAL_VALUES.get(parts[0], 0)
    if len(parts) == 2 and 1 <= _ORDINAL_VALUES.get(parts[1], 0) <= 9:
        return _TENS.get(parts[0], 0) + _ORDINAL_VALUES[parts[1]]
    return 0


def _valid_written_date(match: re.Match[str]) -> bool:
    base = " ".join(match.group("base").lower().split())
    year = {"тысяча восемьсот": 1800, "тысяча девятьсот": 1900, "две тысячи": 2000}[base]
    if match.group("remainder"):
        remainder = _written_ordinal(match.group("remainder"))
        if not remainder:
            return False
        year += remainder
    month = next(index + 1 for index, pattern in enumerate(_MONTH_NAMES)
                 if re.fullmatch(pattern, match.group("month"), _FLAGS))
    try:
        date(year, month, _written_ordinal(match.group("day")))
        return True
    except ValueError:
        return False


def _date_options(numbers: list[int], month_match: int | None) -> list[tuple[int, int, int]]:
    if month_match is not None:
        if len(numbers) not in (1, 2):
            return []
        day = numbers[0]
        year = numbers[1] if len(numbers) == 2 else 2000
        if year < 100:
            year += 2000
        return [(year, month_match, day)]
    if len(numbers) in (2, 3):
        if len(numbers) == 2:
            first, second = numbers
            year = 2000
        elif numbers[0] >= 1000:
            year, first, second = numbers
        else:
            first, second, year = numbers
            if year < 100:
                year += 2000
        return [(year, first, second), (year, second, first)]
    return []


def _valid_field_date(value: str) -> bool:
    """Validate a full or partial calendar date without inferring a century.

    A missing/short year retains the February 29 possibility. Both numeric
    day-month orders are supported, matching the assignment's format variants.
    """
    numbers = [int(part) for part in re.findall(r"[0-9]+", value)]
    month_match = next((index + 1 for index, month in enumerate(_MONTH_NAMES)
                        if re.search(rf"(?<!\w){month}(?!\w)", value, _FLAGS)), None)
    for year, month, day in _date_options(numbers, month_match):
        if 1800 <= year <= 2100:
            try:
                date(year, month, day)
                return True
            except ValueError:
                continue
    return False


def _match_date_value(text: str, label, prefix, end) -> tuple[re.Match | None, bool]:
    value = _DATE_START.match(text, prefix.end(), end)
    valid = value is not None and _valid_field_date(value.group())
    if value is None and label.lastgroup == "birth":
        value = _WRITTEN_DATE.match(text, prefix.end(), end)
        valid = value is not None and _valid_written_date(value)
    return value, valid


def _date_candidates(text: str) -> Iterator[Candidate]:
    for label in _DATE_LABEL.finditer(text):
        # A bounded prefix admits owner qualifiers and '(day and month)', but
        # cannot consume another field, sentence or a date before the delimiter.
        end = min(len(text), label.end() + 160)
        prefix = _FIELD_PREFIX.match(text, label.end(), end)
        if prefix is None:
            prefix = _DIRECT_PREFIX.match(text, label.end(), end)
        assert prefix is not None
        owner = text[label.end():prefix.end()]
        if _UNKNOWN_FIELD.search(owner) or _NEXT_DATE_FIELD.search(owner):
            continue
        before = text[max(0, label.start() - 80):prefix.end()]
        before = re.split(r"[.!?;\n]", before)[-1]
        if (_PUBLIC_OWNER.search(owner) or _HISTORICAL_OWNER.search(before)) and not _PRIVATE_OWNER.search(before):
            continue
        value, valid = _match_date_value(text, label, prefix, end)
        if value is not None and valid:
            kind = "BIRTH_DATE" if label.lastgroup == "birth" else "PASSPORT_DATE"
            stop = value.end()
            if stop == end and end < len(text):
                continue
            if text[stop - 1] == "." and not value.group().lower().endswith(" г."):
                stop -= 1
            yield value.start(), stop, kind, 0.99, "explicit-personal-date-field"


# Delimiters are outside captured values, including quotes and parentheses.
_VALUE_OPEN = r"[ \t]*+(?:[:=—–-][ \t]*+)?[«\"'“(]?[ \t]*+"
_CARD_OWNER = r"(?:[ \t]+(?:(?:первой|второй|третьей|основной|дополнительной)[ \t]+)?карты)?"
_PIN_VALUE = re.compile(
    rf"(?<!\w)(?:pin|пин)(?:[ -]*(?:код|code))?{_CARD_OWNER}{_VALUE_OPEN}"
    r"(?P<value>[0-9]{4,6})(?!\w)", _FLAGS,
)
_CVV_NAME = r"(?:cvv2?|cvc2?|цвв|сvv2?|сvc2?)"
_CVV_VALUE = re.compile(
    rf"(?<!\w){_CVV_NAME}(?:[ -]+код(?:а)?)?{_CARD_OWNER}"
    rf"(?:[ \t]+(?:указан[ \t]+)?код)?{_VALUE_OPEN}(?P<value>[0-9]{{3,4}})(?!\w)", _FLAGS,
)
_CVV_REVERSE = re.compile(
    rf"\bкод[ \t]+(?P<value>[0-9]{{3,4}})(?!\w)[ \t]+"
    rf"(?:(?:неверно|ошибочно)[ \t]+)?(?:введ[её]н[ \t]+)?{_CVV_NAME}(?!\w)", _FLAGS,
)
_CARD_BACK_CODE = re.compile(
    rf"\bкод(?:а)?[ \t]+на[ \t]+(?:обороте|обратной[ \t]+стороне)[ \t]+карты{_VALUE_OPEN}"
    r"(?P<value>[0-9]{3,4})(?!\w)", _FLAGS,
)
_BANK_CARD = re.compile(
    rf"\bбанковск(?:ой|ую|ая)[ \t]+карт(?:ы|у|а|ой){_VALUE_OPEN}"
    r"(?P<value>[0-9](?:[ \t-]*[0-9]){12,18})(?!\w|[ \t-]*[0-9])", _FLAGS,
)

_LICENSE_NAME = r"(?:водительск[а-яё]*[ \t]+удостоверени[а-яё]*|в[ /]?у)"
_DOCUMENT_LABEL = re.compile(
    rf"(?<!\w)(?P<license>{_LICENSE_NAME})(?!\w)|\b(?P<passport>паспорт(?:а)?)(?!\w)", _FLAGS,
)
_DOCUMENT_OWNER = re.compile(
    r"(?:[ \t]+(?:рф|клиент[а]?|за[её]мщика|поручителя|водителя|владельца|гражданина)){0,2}"
    r"[ \t]*(?:[:,=—–-][ \t]*)?(?:№[ \t]*)?", _FLAGS,
)
_DOCUMENT_VALUE = re.compile(
    rf"(?P<value>(?:{_DIGIT2}[ \t]*{_DIGIT2}|{_DIGIT2}[ \t]+[а-яёa-z]{{2}})[ \t-]+(?a:\d){{6}})(?!\w)", _FLAGS,
)
_LICENSE_SUFFIX = re.compile(
    rf"(?<!\w)(?P<value>[0-9]{{2}}[ \t]*[0-9]{{2}}[ \t-]+[0-9]{{6}})(?!\w)"
    rf"[ \t]+[—–-][ \t]*{_LICENSE_NAME}(?!\w)", _FLAGS,
)
_DOCUMENT_PART = re.compile(
    r"\b(?:(?P<series>серия|серии)|(?P<number>номер|номером))[ \t]*(?:[:=—–-][ \t]*)?"
    r"(?P<value>[0-9]{2}[ \t]*[0-9]{2}|[0-9]{6})(?!\w)", _FLAGS,
)
_DOCUMENT_STOP = re.compile(
    r"[.!?;\n]|\b(?:паспорт|в[ /]?у|водительское|заказ[а-яё]*|товар[а-яё]*|накладн[а-яё]*|сертификат[а-яё]*|талон[а-яё]*|"
    r"договор[а-яё]*|сч[её]т[а-яё]*|телефон[а-яё]*|инн)\b", _FLAGS,
)
_NONPERSONAL_DOCUMENT = re.compile(
    r"\b(?:оборудовани[а-яё]*|издели[а-яё]*|техническ[а-яё]*|компани[а-яё]*|"
    r"организаци[а-яё]*|транспортн[а-яё]*|объект[а-яё]*|проект[а-яё]*|"
    r"станк[а-яё]*|здани[а-яё]*)\b", _FLAGS,
)


def _document_qualifier(text: str, label, end) -> str:
    qualifier_end = min(end, label.end() + 80)
    first_digit = re.search(r"[0-9]", text[label.end():qualifier_end])
    if first_digit is not None:
        qualifier_end = label.end() + first_digit.start()
    first_part = _DOCUMENT_PART.search(text, label.end(), end)
    if first_part is not None:
        qualifier_end = min(qualifier_end, first_part.start())
    return text[label.end():qualifier_end]


def _document_parts(text: str, label, end, kind) -> Iterator[Candidate]:
    for part in _DOCUMENT_PART.finditer(text, label.end(), end):
        size = sum(char.isdigit() for char in part.group("value"))
        if size == (4 if part.group("series") else 6):
            start, stop = part.span("value")
            if stop == end and stop < len(text) and (text[stop].isalnum() or text[stop] == "_"):
                continue
            yield start, stop, kind, 0.99, "explicit-personal-document-part"


def _document_candidates(text: str) -> Iterator[Candidate]:
    for label in _DOCUMENT_LABEL.finditer(text):
        kind = "DRIVER_LICENSE" if label.lastgroup == "license" else "PASSPORT"
        end = min(len(text), label.end() + 180)
        owner = _DOCUMENT_OWNER.match(text, label.end(), end)
        assert owner is not None
        qualifier = _document_qualifier(text, label, end)
        if _NONPERSONAL_DOCUMENT.search(qualifier) or _UNKNOWN_FIELD.search(qualifier):
            continue
        value = _DOCUMENT_VALUE.match(text, owner.end(), end)
        if value is not None:
            start, stop = value.span("value")
            if stop < len(text) and (text[stop].isalnum() or text[stop] == "_"):
                continue
            yield start, stop, kind, 0.99, "explicit-personal-document-field"
        stop = _DOCUMENT_STOP.search(text, label.end(), end)
        if stop is not None:
            end = stop.start()
        yield from _document_parts(text, label, end, kind)
    for value in _LICENSE_SUFFIX.finditer(text):
        start, end = value.span("value")
        yield start, end, "DRIVER_LICENSE", 0.99, "explicit-license-suffix"


def structured_candidates(text: str) -> Iterator[Candidate]:
    """Yield bounded field values with their unchanged Unicode input offsets."""
    if not text:
        return
    lower = text.lower()
    if "рождения" in lower or "выдачи" in lower or "выдан" in lower:
        yield from _date_candidates(text)
    if not any(char.isdigit() for char in text):
        return
    if any(hint in lower for hint in ("паспорт", "удостоверен", "ву", "в/у", "в у")):
        yield from _document_candidates(text)
    for pattern, kind, hints in (
        (_PIN_VALUE, "PIN", ("pin", "пин")),
        (_CVV_VALUE, "CVV", ("cvv", "cvc", "цвв", "сvv", "сvc")),
        (_CVV_REVERSE, "CVV", ("cvv", "cvc", "цвв", "сvv", "сvc")),
        (_CARD_BACK_CODE, "CVV", ("оборот", "обратной")),
        (_BANK_CARD, "CARD", ("банковск",)),
    ):
        if any(hint in lower for hint in hints):
            for match in pattern.finditer(text):
                start, end = match.span("value")
                yield start, end, kind, 0.99, "explicit-payment-security-field"
