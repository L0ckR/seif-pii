"""Conservative boundary corrections for explicit field and sentence context.

These helpers never discover a name from a dictionary or normalize offsets.
They retain the candidate's type, confidence and provenance, and leave custom
recognizers untouched. External candidates must be validated by the caller.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from seif.location_fields import issuer_value_end

_FLAGS = re.IGNORECASE | re.UNICODE
# Start at the literal field name. Searching from overlapping whitespace
# quantifiers becomes cubic on a long, otherwise valid spaced document value.
_NUMBER_FIELD = re.compile(r"\bномер\b\s*+[:№=—-]?\s*+", _FLAGS)
_DOCUMENT_PART = re.compile(r"[0-9](?:[0-9 \t]*[0-9])?")
_NAME_PROSE = re.compile(
    r"(?<![\w-])(?:обратил(?:ся|ась|ись)|подтвердил(?:а|и)?|запросил(?:а|и)?|"
    r"предъявил(?:а|и)?|сообщил(?:а|и)?|подписал(?:а|и)?|получил(?:а|и)?|"
    r"оформил(?:а|и)?|заявил(?:а|и)?|вв[её]л(?:а|и)?|"
    r"код\s+назначения(?:\s+платежа)?|назначение(?=\s*:))\b",
    _FLAGS,
)
_MISSING_NAME = re.compile(r"^(?:не\s+(?:указан[аоы]?|известен|известна)|отсутствует)\b", _FLAGS)
_FIELD_SCAFFOLD = re.compile(
    r"(?:регистраци|проживани|прописк|бенефициар|указано|по\s+адресу)", _FLAGS
)
_LOCALITY_START = re.compile(r"(?:\b(?:г|ул|д|кв|пос)[.]|\b(?:город|улица|дом)\b)", _FLAGS)
_ADDRESS_PREFIX = re.compile(r"^\s*по\s+адресу\s*[:=—-]?\s*", _FLAGS)
_ENUMERATION = re.compile(r"^\s*[0-9]+[.)]\s*")
_ADDRESS_BREAK = re.compile(
    r"(?:[.]\s*|,\s*)(?:адрес\s+(?:(?:фактического\s+)?проживания|регистрации)|"
    r"(?:продавец|покупатель|за[её]мщик)\s*[—-]?\s*по\s+адресу)\s*[:=—-]?\s*",
    _FLAGS,
)
_PLACE_PROSE = re.compile(
    r"(?<![\w-])(?:подтвержден[аоы]?\s+(?:паспортом|документом)|"
    r"анкета\s+(?:проверена|подписана|заполнена)|"
    r"оформил(?:а|и)?\s+договор|совпадает\s+с\s+указанным)\b",
    _FLAGS,
)
_CORPORATE_OWNER = re.compile(
    r"\b(?:ооо|пао|оао|зао|ао|компани[яиюей]+|организаци[яиюей]+|"
    r"юрлиц[аоу]?|юридическ[а-яё]*\s+лиц[аоу]|поставщик[а-яё]*|контрагент[а-яё]*)\b",
    _FLAGS,
)
_PRIVATE_OWNER = re.compile(
    r"\b(?:клиент[а-яё]*|сотрудник[а-яё]*|физлиц[а-яё]*|"
    r"физическ[а-яё]*\s+лиц[аоу]|за[её]мщик[а-яё]*|поручител[а-яё]*|ип)\b",
    _FLAGS,
)
_INN_FIELD_END = re.compile(r"\bинн(?:\s*/\s*кпп)?\s*[:=—-]?\s*$", _FLAGS)
_COUNTRY_QUALIFIER = re.compile(r"\b(?:паспорт|гражданин|гражданка)[ \t]+$", _FLAGS)
_CORPORATE_ADDRESS = re.compile(r"\bадрес[а-яё]*[ \t]*[:=—-]?[ \t]*(?:г[.][ \t]*)?$", _FLAGS)
_NUMERIC_TAIL = re.compile(r"[.,]([0-9]{1,13})(?![0-9])")
_NUMERIC_HEAD = re.compile(r"(?<![0-9])([0-9]{1,13})[.,]$")
_RECORD_BOUNDARY = re.compile(r"[;!?]|\n[ \t]*\n|[.](?=\s|$)")
_ABBREVIATION = re.compile(r"(?:\b|\\[nr])(?:г|гор|ул|д|кв|корп|стр|обл|р-н|им|пос|тел|ао)[.]$", _FLAGS)
_OFFICE = re.compile(r"\b(?:[оуг]вд|[оу]{0,2}фмс|мвд|умвд|отдел\s+внутренних\s+дел)\b", _FLAGS)
_PERSONAL_CONTEXT = re.compile(
    r"\b(?:паспорт(?:а|у|ом|е)?|выда(?:н[аоы]?|л[аои]?)|клиент[а-яё]*|заявител[а-яё]*|"
    r"получател[а-яё]*|прожива[а-яё]*|зарегистрирован[а-яё]*|"
    r"место\s+рождения|орган\s+выдачи|адрес|мой|моя)\b",
    _FLAGS,
)


def _copy_span(span, start: int, end: int):
    return type(span)(start, end, span.type, span.confidence, span.reason)


def _segment(text: str, span, start: int, end: int):
    while start < end and text[start].isspace():
        start += 1
    while start < end and (text[end - 1].isspace() or text[end - 1] in ".,;:"):
        end -= 1
    return _copy_span(span, start, end) if start < end else None


def _document_parts(text: str, span) -> list:
    value = text[span.start:span.end]
    separator = _NUMBER_FIELD.search(value)
    if separator is None:
        return [span]
    left = value[:separator.start()].strip(" \t\r\n,;")
    right = value[separator.end():].strip()
    if not (_DOCUMENT_PART.fullmatch(left) and _DOCUMENT_PART.fullmatch(right)):
        return [span]
    # Keep both document components while exposing the explanatory field name.
    return [part for part in (
        _segment(text, span, span.start, span.start + separator.start()),
        _segment(text, span, span.start + separator.end(), span.end),
    ) if part is not None]


def _record_prefix(text: str, start: int) -> str:
    prefix = text[max(0, start - 160):start]
    offset = 0
    for boundary in _RECORD_BOUNDARY.finditer(prefix):
        if boundary.group() == "." and _ABBREVIATION.search(prefix[max(0, boundary.end() - 16):boundary.end()]):
            continue
        offset = boundary.end()
    return prefix[offset:]


def _corporate_inn(text: str, span) -> bool:
    if len(text[span.start:span.end]) != 10 or span.reason == "cis-personal-id":
        return False
    prefix = _record_prefix(text, span.start)
    if not _INN_FIELD_END.search(prefix):
        return False
    # A later private owner wins over an earlier company in the same sentence.
    corporate = list(_CORPORATE_OWNER.finditer(prefix))
    private = list(_PRIVATE_OWNER.finditer(prefix))
    return bool(corporate and (not private or corporate[-1].start() > private[-1].start()))


def _office_geography(text: str, span, personal_context: bool) -> bool:
    prefix = _record_prefix(text, span.start)
    # An institution's name alone is not a personal address or passport issuer.
    return bool(_OFFICE.search(prefix) and not personal_context)


def _geographic_scaffolding(text: str, span) -> bool:
    prefix = _record_prefix(text, span.start)
    value = text[span.start:span.end].lower()
    if value in {"рф", "россии"} and _COUNTRY_QUALIFIER.search(prefix):
        return True
    if not _CORPORATE_ADDRESS.search(prefix):
        return False
    corporate = list(_CORPORATE_OWNER.finditer(prefix))
    private = list(_PRIVATE_OWNER.finditer(prefix))
    return bool(corporate and (not private or corporate[-1].start() > private[-1].start()))


def _decimal_inn_fragment(text: str, span) -> bool:
    if span.reason != "checksum":
        return False
    after = _NUMERIC_TAIL.match(text, span.end, min(len(text), span.end + 15))
    before = _NUMERIC_HEAD.search(text[max(0, span.start - 15):span.start])
    # Two full identifiers separated by punctuation are a list, not a fractional
    # number; retain both, including lists without whitespace after a comma.
    return any(part is not None and len(part[1]) not in {10, 12} for part in (after, before))


def _name_span(text: str, span):
    value = text[span.start:span.end]
    if span.type == "CARDHOLDER" and _MISSING_NAME.match(value):
        return None
    prose = _NAME_PROSE.search(text, span.start, min(len(text), span.end + 1))
    if prose is None or prose.start() >= span.end:
        return span
    # A leading action is evidence only for the explicit cardholder field;
    # a standalone name may legitimately coincide with an ordinary word.
    if prose.start() == span.start and span.type != "CARDHOLDER":
        return span
    return _segment(text, span, span.start, prose.start())


def _place_parts(text: str, span) -> list:
    start, end = span.start, span.end
    value = text[start:end]
    prefix = _ADDRESS_PREFIX.match(value)
    if prefix:
        start += prefix.end()
    else:
        colon = value.find(":", 0, 100)
        if colon >= 0 and _FIELD_SCAFFOLD.search(value[:colon]) and not _LOCALITY_START.search(value[:colon]):
            start += colon + 1
    enumeration = _ENUMERATION.match(text[start:end])
    if enumeration:
        start += enumeration.end()
    prose = _PLACE_PROSE.search(text, start, end)
    if prose:
        end = prose.start()
    cuts = list(_ADDRESS_BREAK.finditer(text, start, end)) if span.type == "ADDRESS" else []
    parts = []
    for cut in cuts:
        part = _segment(text, span, start, cut.start())
        if part is not None:
            parts.append(part)
        start = cut.end()
    part = _segment(text, span, start, end)
    if part is not None:
        parts.append(part)
    return parts


def refine_candidates(text: str, candidates: Sequence) -> list:
    """Refine already validated candidates before ordinary overlap resolution."""
    result = []
    # Scan document-wide context once, not for every geographic candidate in a
    # large document. All remaining contextual windows are bounded per span.
    personal_context = bool(_PERSONAL_CONTEXT.search(text)) if any(
        span.type in {"CITY", "LOCATION"} for span in candidates
    ) else False
    for span in candidates:
        if span.reason == "custom-rule":
            result.append(span)
        elif span.type in {"PASSPORT", "DRIVER_LICENSE"}:
            result.extend(_document_parts(text, span))
        elif span.type in {"PERSON", "CARDHOLDER"}:
            refined = _name_span(text, span)
            if refined is not None:
                result.append(refined)
        elif span.type in {"ADDRESS", "BIRTH_PLACE"}:
            result.extend(_place_parts(text, span))
        elif span.type == "PASSPORT_ISSUER":
            refined = _segment(text, span, span.start, issuer_value_end(text, span.start, span.end))
            if refined is not None:
                result.append(refined)
        elif span.type == "INN" and (_corporate_inn(text, span) or _decimal_inn_fragment(text, span)):
            continue
        elif span.type in {"CITY", "LOCATION"} and (
            _office_geography(text, span, personal_context) or _geographic_scaffolding(text, span)
        ):
            continue
        else:
            result.append(span)
    return result
