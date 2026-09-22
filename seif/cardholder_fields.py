"""Bounded cardholder field parsing with original Unicode character offsets."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator

from seif.person_fields import Candidate, _latin_given_names, _latin_name_pair, _name_sequence

_WORD = r"[^\W\d_](?:[^\W\d_]|['’ʼ-](?=[^\W\d_])){0,39}"
_WORDS = re.compile(_WORD)
_LABEL = re.compile(
    r"(?<!\w)(?:имя[ \t]+держател[ья](?:[ \t]+карты)?|держател[ья](?:[ \t]+карты)?|"
    r"эмбоссированное[ \t]+имя|card[ \t]*holder|name[ \t]+on[ \t]+card|"
    r"карта[ \t]+(?:выпущена|оформлена)[ \t]+на[ \t]+имя)(?!\w)", re.IGNORECASE,
)
_SEPARATOR = re.compile(r"[ \t]*(?:[:=—–-][ \t]*)?[«‹“\"']?[ \t]*")
_VALUE = re.compile(rf"{_WORD}(?:[ \t]+{_WORD}){{0,3}}(?![\w'’ʼ-])")
_VALUE_BOUNDARY = re.compile(r'[ \t]*(?:[»›”"\'](?=[\s,;.!?]|$)|[,;\r\n]|[.!?](?=\s|$)|$)')
_LATIN_PATRONYMIC = re.compile(r"[a-z]{2,30}(?:ovich|evich|ovna|evna|ichna)\Z", re.IGNORECASE)
_NOT_A_NAME = frozenset(
    "не нет неизвестен неизвестна отсутствует отсутствуют указан указана указано клиент карта "
    "карты имя держатель держателя банк банка банковская банковский пластиковая пластиковой "
    "подтвердил подтвердила подтвердить обратился обратилась ввел ввёл ввела ввела указал указала "
    "получил получила запросил запросила компания организация общество ооо ао пао ип тест тестовый тестовая "
    "успешно выполнен выполнена операция операции платёж платеж перевод "
    "доступ запрещен запрещён разрешен разрешён пользователь неизвестный неизвестная "
    "неизвестно неизвестные отсутствующий новый новая другой другая сторона текст значение "
    "подпись подписи клиента обработка проверка ожидание подтверждение реквизиты юридическое "
    "card holder name unknown none unavailable null not specified confirmed confirms confirm "
    "successful operation operations payment transaction verification pending information "
    "details authorized person customer user legal entity access denied granted holdername "
    "a an the is are was were be been being has have had do did does will can must should "
    "to for from with by at of on in as and or this that account action login logout sign "
    "enter entered enter your please successfully success failed missing available ready "
    "sent said completed authorized declined cancelled canceled invalid cardnumber number "
    "received requested approved processing executed rejected declined company corporate business bank ltd llc inc".split()
)

# Explicit cardholder fields may contain given names absent from the caller's
# Russian dictionary. These common given names extend regional/foreign script
# coverage; no complete identities, family names or golden values are stored.
_FIELD_GIVEN_NAMES = frozenset(
    "әділ әлібек нұрлан айгүл айгерім ерлан болат асқар данияр азамат рустам тимур "
    "аліксандр олександр олег андрій петро микола володимир дмитро тарас богдан "
    "james robert david richard william charles anne jane sarah emily adam "
    "mohammed muhammad ali ahmed omar wei ming hua jun min lin "
    "вэй мин хуа цзюнь лин".split()
)


def _foreign_field_name(parts: list[str], given_names: Collection[str]) -> bool:
    if not all(len(part) >= 2 for part in parts):
        return False
    known = _latin_given_names(frozenset(given_names)) | _FIELD_GIVEN_NAMES
    known_count = sum(part.casefold() in given_names or part.casefold() in known for part in parts)
    # A third unfamiliar token may be action prose after a complete name.
    return known_count >= (2 if len(parts) == 3 else 1)


def _valid_parts(parts: list[str], given_names: Collection[str]) -> bool:
    if len(parts) not in (2, 3) or any(part.casefold() in _NOT_A_NAME for part in parts):
        return False
    if _name_sequence(parts, given_names) or _latin_name_pair(parts, given_names):
        return True
    if len(parts) == 3 and _name_sequence(parts[:2], given_names):
        # Once a complete two-part name is known, a third arbitrary word does
        # not become a name merely because a form renders its prose uppercase.
        return _LATIN_PATRONYMIC.fullmatch(parts[2]) is not None
    # An explicit field plus a known given name permits family names outside
    # Russian morphology without granting arbitrary uppercase prose a bypass.
    return _foreign_field_name(parts, given_names)


def _delimited_foreign_pair(text: str, label_end: int, words: list[re.Match[str]]) -> bool:
    if len(words) != 2 or any(word.group().casefold() in _NOT_A_NAME for word in words):
        return False
    if not all(len(word.group()) >= 2 for word in words):
        return False
    # Two unknown tokens need an actual form-value delimiter on both sides.
    # A nearby role word or truncated pair at the start of prose is insufficient.
    separator = text[label_end:words[0].start()]
    return bool(re.search(r"[:=«‹“\"']", separator)
                and _VALUE_BOUNDARY.match(text, words[-1].end()))


def cardholder_candidates(text: str, *, given_names: Collection[str]) -> Iterator[Candidate]:
    """Recognize each explicit field independently; never propagate a name by value."""
    for label in _LABEL.finditer(text):
        start = _SEPARATOR.match(text, label.end()).end()
        value = _VALUE.match(text, start)
        if value is None or text[value.end():value.end() + 1] == "@":
            continue
        words = list(_WORDS.finditer(text, value.start(), value.end()))
        if _delimited_foreign_pair(text, label.end(), words):
            yield words[0].start(), words[-1].end(), "CARDHOLDER", 0.96, "delimited-cardholder-field"
            continue
        for count in (3, 2):
            if len(words) < count:
                continue
            if _valid_parts([word.group() for word in words[:count]], given_names):
                yield words[0].start(), words[count - 1].end(), "CARDHOLDER", 0.98, "explicit-cardholder-field"
                break
