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
_PART_LABEL_SUFFIX = re.compile(
    r"(?<!\w)(?:серия|серии|серией|сер[.]|номер(?:ом)?|ном[.]|№)[ \t]*[:=—–-]?[ \t]*$", _FLAGS,
)
_PART_JOIN = re.compile(r"[ \t,;./—–-]*(?:\r?\n[ \t]*)?(?:и[ \t]+)?", _FLAGS)
_DRIVER_CARD = (
    r"(?:карточк(?:а|и|у|ой|е|ою)[ \t]+водителя|"
    r"водительск(?:ая|ой|ую)[ \t]+карточк(?:а|и|у|ой|е|ою))"
)
_CARD_QUALIFIER = r"(?:тахографическ[а-яё]*|топливн[а-яё]*|корпоративн[а-яё]*|коммерческ[а-яё]*)"
_OWNER = re.compile(
    rf"\b(?P<other_card>{_CARD_QUALIFIER}[ \t]+{_DRIVER_CARD}|"
    rf"(?:{_CARD_QUALIFIER}[ \t]+)?карт(?:а|ы|у|ой|е)[ \t]+водителя|"
    r"водительск(?:ая|ой|ую)[ \t]+карт(?:а|ы|у|ой|е))\b|"
    rf"\b(?P<driver_card>{_DRIVER_CARD})\b|"
    r"\b(?P<license>водительск[а-яё]*[ \t]+удостоверени[а-яё]*|в[ /]?у|права)\b|"
    r"\b(?P<passport>паспортн[а-яё]*[ \t]+(?:данн[а-яё]*|реквизит[а-яё]*)|"
    r"паспорт(?:а|ом|е)?|документ(?:а|ом|е)?)\b|"
    r"\b(?P<generic_license>удостоверени[ея])(?=[ \t]*[:=])|"
    r"\b(?P<identity>удостоверени[ея](?:[ \t]+личности)?)\b"
    r"(?![ \t]+(?:сотрудник[а-яё]*|работник[а-яё]*|студент[а-яё]*|пенсионер[а-яё]*|"
    r"качества|соответствия|безопасности)\b)|"
    r"\b(?P<business>(?:пенсионн[а-яё]*|служебн[а-яё]*|студенческ[а-яё]*|"
    r"техническ[а-яё]*|транспортн[а-яё]*)[ \t]+удостоверени[ея]|"
    r"заказ[а-яё]*|накладн[а-яё]*|товар[а-яё]*|артикул[а-яё]*|"
    r"договор[а-яё]*|контракт[а-яё]*|сч[её]т[а-яё]*|издели[а-яё]*|оборудовани[а-яё]*|"
    r"станк[а-яё]*|партия|партии|плат[её]ж[а-яё]*|сумм[а-яё]*|сертификат[а-яё]*|"
    r"техническ[а-яё]*|транспортн[а-яё]*|телефон[а-яё]*|инн|снилс|id|sku|"
    r"удостоверени[ея]|студенческ[а-яё]*|служебн[а-яё]*|пенсионн[а-яё]*|"
    r"тахограф[а-яё]*|топлив[а-яё]*|оплат[а-яё]*|устройств[а-яё]*|принтер[а-яё]*|"
    r"сканер[а-яё]*|компьютер[а-яё]*|ноутбук[а-яё]*|двигател[а-яё]*|автомобил[а-яё]*)\b",
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


# A sentence/line may continue an owned record only with an explicit field
# introduction. Paragraphs, other records and unrelated narrative cannot carry
# document ownership. Numeric references such as "2. 3" are not sentence ends.
_CARD_PURPOSE = re.compile(r"\b(?:для|к)[ \t]+(?:цифров[а-яё]*[ \t]+)?тахограф[а-яё]*[ ,:—–-]*$", _FLAGS)
_OWNER_BOUNDARY = re.compile(r"[!?;]|(?:\r?\n)[ \t]*(?:\r?\n)?|[.]")
_FIELD_ABBREVIATION = re.compile(r"\b(?:сер|ном|п|стр|гл)[.]$", _FLAGS)
_OWNER_CLAUSE_END = re.compile(
    r"[ \t]*(?:(?:клиента|заявителя|водителя|владельца|представителя)[ \t]*)?"
    r"(?:(?:проверен[аоы]?|предъявлен[аоы]?|получен[аоы]?|принят[аоы]?|оформлен[аоы]?|зарегистрирован[аоы]?)[ \t]*)?[, \t]*", _FLAGS,
)
_FIELD_CONTINUATION = re.compile(
    r"[ \t]*(?:(?:в|на)[ \t]+(?:пункте|поле|графе|разделе|строке|странице|обороте|ней|нем|нём)"
    r"[ \t]*(?:№[ \t]*)?(?:[0-9]{1,3}(?:[.][ \t]*[0-9]{1,3}){0,3})?[ \t]*)?"
    r"(?:(?:указан[аоы]?|записан[аоы]?|привед[её]н[аоы]?|содержится|содержатся)[ \t]*"
    r"(?:,[ \t]*)?(?:что[ \t]+)?)?[:=—– \t]*", _FLAGS,
)
_RECORD_SWITCH = re.compile(
    r"\b(?:новая|другая|следующая)[ \t]+(?:запись|анкета|заявка|операция)\b", _FLAGS,
)
_NUMBER_PRESENTATION = re.compile(
    r"[ \t]*(?:(?:(?:его|е[её]|их|следующие)[ \t]+)?(?:данные|реквизиты|значение|номер)"
    r"|вот)[ \t]*[:=—–-][ \t]*", _FLAGS,
)
_PURPOSE_TRAILER = re.compile(r"[ \t]+(?:для|при)[ \t]+[а-яё \t-]{1,72}", _FLAGS)
_PURPOSE_PREFIX = re.compile(
    r"\b(?:для|при|с[ \t]+целью|в[ \t]+целях)[ \t]+"
    r"(?:оформления|оформлении|получения|получении|передачи|передаче|проверки|проверке|"
    r"регистрации|выдачи|выдаче|подписания|подписании|оплаты|оплате|заключения|заключении)"
    r"[ \t]+(?:[а-яё-]+[ \t]+){0,4}$", _FLAGS,
)
_REQUESTED_FIELDS = re.compile(
    r"[ \t]+(?:нужны|необходимы|требуются|потребуются)[ \t]+"
    r"(?:следующие[ \t]+)?(?:данные|реквизиты)[ \t]*[:=—–-][ \t(]*", _FLAGS,
)


def _record_boundaries(tail: str) -> Iterator[re.Match[str]]:
    for boundary in _OWNER_BOUNDARY.finditer(tail):
        if boundary.group() == ".":
            if _FIELD_ABBREVIATION.search(tail, max(0, boundary.start() - 4), boundary.end()):
                continue
            after = tail[boundary.end():].lstrip(" \t")
            if boundary.start() and tail[boundary.start() - 1].isdigit() and after[:1].isdigit():
                continue
        yield boundary


def _owner_continues(prefix: str, owner_end: int, *, paired: bool, personal: bool = False) -> bool:
    tail = prefix[owner_end:]
    if _RECORD_SWITCH.search(tail):
        return False
    boundaries = list(_record_boundaries(tail))
    if not boundaries:
        return True
    if any(b.group() in {"!", "?"} or b.group().count("\n") > 1 for b in boundaries):
        return False
    # Period + line break is one transition; another substantive clause is not.
    if any(tail[first.end():second.start()].strip() for first, second in zip(boundaries, boundaries[1:], strict=False)):
        return False
    clause_end = tail[:boundaries[0].start()]
    continuation = tail[boundaries[-1].end():]
    # Bare grouped digits need an explicit presentation cue across a sentence;
    # a series/number pair already provides that field structure itself.
    if personal and _NUMBER_PRESENTATION.fullmatch(continuation):
        return bool(_OWNER_CLAUSE_END.fullmatch(clause_end) or _PURPOSE_TRAILER.fullmatch(clause_end))
    return bool(paired and _OWNER_CLAUSE_END.fullmatch(clause_end)
                and _FIELD_CONTINUATION.fullmatch(continuation))


def _number_kind(text: str, start: int, *, paired: bool = False) -> str | None:
    begin = max(0, start - 160)
    prefix = text[begin:start]
    owners = [owner for owner in _OWNER.finditer(prefix)
              if owner.start() or not begin or not (text[begin - 1].isalnum() or text[begin - 1] == "_")]
    # An object in a purpose phrase is not the owner of subsequently requested
    # fields. Keep the established unowned paired-number policy in that case;
    # direct product labels and genitive "data of the product" still veto it.
    owners = [owner for owner in owners
              if not (owner.lastgroup == "business"
                      and _PURPOSE_PREFIX.search(prefix[:owner.start()])
                      and _REQUESTED_FIELDS.fullmatch(prefix[owner.end():]))]
    default = "PASSPORT" if paired else None
    if not owners or not _owner_continues(prefix, owners[-1].end(), paired=paired,
                                        personal=owners[-1].lastgroup not in {"business", "other_card"}):
        return default
    group = owners[-1].lastgroup
    if group == "driver_card" and _CARD_PURPOSE.search(prefix, max(0, owners[-1].start() - 48), owners[-1].start()):
        return None
    if group in {"business", "other_card"}:
        return None
    if group == "identity":
        # An identity document is personal, but its specific category is not
        # established. Only preserve the existing explicit paired-field policy.
        return default
    return "DRIVER_LICENSE" if group in {"license", "generic_license", "driver_card"} else "PASSPORT"


def document_kind_at_value(text: str, start: int) -> str | None:
    """Classify a legacy numeric match using the same bounded field ownership.

    The general passport regexp starts at the series value rather than at its
    label. Remove only that adjacent label before checking record continuation,
    so a legacy PASSPORT candidate cannot override explicit licence/business
    ownership merely because it has a higher overlap priority.
    """
    begin = max(0, start - 24)
    label = _PART_LABEL_SUFFIX.search(text, begin, start)
    owner_start = label.start() if label is not None else start
    return _number_kind(text, owner_start, paired=True)


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
