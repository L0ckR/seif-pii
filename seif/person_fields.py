"""Personal-name grammar for signed fields, initials and joined Russian names.

The caller supplies its existing lowercase given-name dictionary. No names or
case IDs from an evaluation corpus are embedded here. Bare single words are
accepted only as values of an explicit personal field.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterator
from itertools import permutations

Candidate = tuple[int, int, str, float, str]
_FLAGS = re.IGNORECASE
_WORD = r"[^\W\d_](?:[^\W\d_]|['’ʼ-](?=[^\W\d_])){0,39}"
_INITIAL = r"[^\W\d_][.]"
_ATOM = rf"(?:{_INITIAL}|{_WORD})"
_SEQUENCE = re.compile(rf"(?<![\w'’ʼ-]){_ATOM}(?:[ \t]+{_ATOM}){{0,3}}(?![\w'’ʼ-])", _FLAGS)
_WORDS = re.compile(rf"(?<![\w'’ʼ-]){_WORD}(?![\w'’ʼ-])", _FLAGS)
_PARTS = re.compile(_ATOM, _FLAGS)
_SURNAME = re.compile(
    r"[а-яё'-]{2,35}(?:ов|ев|ёв|ин|ын|ский|ская|цкий|цкая|енко|ко|их|ых|ович|евич|ян|дзе|швили)"
    r"(?:а|у|ым|ом|е|ой|ую|ого|ому)?\Z", _FLAGS,
)
_PATR_ENDING = r"(?:а|у|ем|е)?"
_PATRONYMIC = re.compile(
    r"[а-яё-]{2,30}(?:ович" + _PATR_ENDING + r"|евич" + _PATR_ENDING + r"|овн[аыеу]|евн[аыеу]|овной|евной|иничн[аыеу])\Z",
    _FLAGS,
)
_NON_NAMES = frozenset({
    "клиент", "клиентка", "получатель", "плательщик", "отправитель", "фио", "нет", "не",
    "указан", "указана", "указано", "неизвестен", "неизвестна", "отсутствует", "отсутствуют",
    "банк", "банка", "общество", "компания", "организация", "отдел", "ооо", "ао", "пао",
})
_ROLE = re.compile(
    r"(?<!\w)(?P<patronymic>отчество)|(?<!\w)(?P<field>фио|фамилия|имя)(?!\w)|"
    r"(?<!\w)(?P<role>клиент(?:ка|а|у)?|со?за[её]мщик(?:а)?|плательщик(?:а)?|"
    r"отправител[ья]|получател[ья]|поручител[ья]|залогодател[ья]|закладател[ья]|"
    r"бенефициар(?:а)?|доверенное[ \t]+лицо|заявител[ья]|подателем[ \t]+заявки|"
    r"доверенность[ \t]+оформлена[ \t]+на)(?!\w)", _FLAGS,
)
_ROLE_SUFFIX = re.compile(r"[ \tа-яёА-ЯЁ«»\"()-]{0,90}[:=][ \t]*[«\"'“]?[ \t]*")
_ORGANIZATION_OWNER = re.compile(
    r"\b(?:банк[а-яё]*|компани[а-яё]*|организаци[а-яё]*|юрлиц[а-яё]*|"
    r"юридическ[а-яё]*[ \t]+лиц[а-яё]*|предприяти[а-яё]*)\b", _FLAGS,
)
_OTHER_FIELD_OWNER = re.compile(
    r"\b(?:гражданство|место[ \t]+рождения|дата[ \t]+рождения|дата[ \t]+выдачи|адрес|"
    r"паспорт|телефон|инн|код|номер)[ \tа-яё«»\"()-]{0,60}$", _FLAGS,
)
_STREET_PREFIX = re.compile(
    r"\b(?:улица|улице|улицы|ул[.]|проспект|проспекте)[ \t]*+[:—–-]?[ \t]*+$", _FLAGS,
)
_INITIAL_RUN = re.compile(rf"(?<![\w'’ʼ-]){_INITIAL}(?:[ \t]*{_INITIAL})?(?!\w)", _FLAGS)
_JOINED_WORDS = re.compile(r"(?<![\w'-])[а-яё]{12,80}(?![\w'-])", _FLAGS)


def _surname(word: str) -> bool:
    return word.lower() not in _NON_NAMES and _SURNAME.fullmatch(word) is not None


def _patronymic(word: str) -> bool:
    return _PATRONYMIC.fullmatch(word) is not None


def _pair_sequence(first: str, second: str, given_names: Collection[str]) -> bool:
    return bool(
        (first.lower() in given_names and (_surname(second) or _patronymic(second)))
        or (second.lower() in given_names and (_surname(first) or _patronymic(first)))
        or (_surname(first) and _patronymic(second))
        or (_surname(second) and _patronymic(first))
    )


def _triple_sequence(full: list[str], given_names: Collection[str]) -> bool:
    return any(first.lower() in given_names and _surname(second) and _patronymic(third)
               for first, second, third in permutations(full))


def _name_sequence(words: list[str], given_names: Collection[str], *, field: bool = False) -> bool:
    """Require actual name grammar; capitalization alone cannot validate prose."""
    if not words or any(word.lower().strip(".") in _NON_NAMES for word in words):
        return False
    initials = [word for word in words if len(word) == 2 and word[1] == "."]
    full = [word for word in words if word not in initials]
    if not full:
        return False
    if len(full) == 1:
        word = full[0]
        return bool((field or initials) and (word.lower() in given_names or _surname(word) or _patronymic(word)))
    if len(full) == 2:
        return _pair_sequence(full[0], full[1], given_names)
    if len(full) == 3 and not initials:
        return _triple_sequence(full, given_names)
    return False


def _joined_split(lower: str, start: int, end: int, given_names: Collection[str]) -> bool:
    def family_parts(first: str, second: str) -> bool:
        return (_surname(first) and _patronymic(second)) or (_patronymic(first) and _surname(second))

    if start and end < len(lower):
        return family_parts(lower[:start], lower[end:])
    remaining = lower[end:] if start == 0 else lower[:start]
    return any(family_parts(remaining[:cut], remaining[cut:]) for cut in range(3, len(remaining) - 3))


def _joined_name(value: str, given_names: Collection[str], max_given_length: int) -> bool:
    lower = value.lower()
    if not any(marker in lower for marker in ("ович", "евич", "овна", "евна", "инична")):
        return False
    # Every accepted split consists of one known given name, a surname and a
    # patronymic. Dictionary membership does not depend on casing or CamelCase.
    for start in range(len(lower) - 1):
        for end in range(start + 2, min(start + max_given_length, len(lower)) + 1):
            if lower[start:end] not in given_names:
                continue
            if _joined_split(lower, start, end, given_names):
                return True
    return False


def _is_street_value(text: str, start: int) -> bool:
    return _STREET_PREFIX.search(text[max(0, start - 80):start]) is not None


def _field_value_valid(
    label, words, count, parts, given_names, max_given_length
) -> bool:
    valid = _patronymic(parts[0]) if label.lastgroup == "patronymic" and count == 1 else False
    valid = valid or _name_sequence(parts, given_names, field=True)
    joined = count == 1 and _joined_name(parts[0], given_names, max_given_length)
    valid = valid or joined
    if (count == 1 and len(words) > 1 and parts[0].lower() not in given_names
            and not _patronymic(parts[0]) and not joined):
        # A surname-shaped adjective at the start of an organization
        # name cannot become a personal surname by truncating the name.
        valid = False
    return valid


def _field_candidates(text: str, given_names: Collection[str], max_given_length: int) -> Iterator[Candidate]:
    for label in _ROLE.finditer(text):
        if _OTHER_FIELD_OWNER.search(text[max(0, label.start() - 100):label.start()]):
            continue
        suffix = _ROLE_SUFFIX.match(text, label.end())
        if suffix is None or _ORGANIZATION_OWNER.search(text[label.end():suffix.end()]):
            continue
        qualifier = text[label.end():suffix.end()].rstrip(" \t:=«\"'“")
        if _ROLE.search(qualifier) or _OTHER_FIELD_OWNER.search(qualifier):
            # Another field owns the later colon; a nearby earlier client role
            # cannot transfer its personal-name interpretation to that value.
            continue
        value = _SEQUENCE.match(text, suffix.end())
        if value is None:
            continue
        words = list(_PARTS.finditer(text, value.start(), value.end()))
        # Consider shorter complete names before action prose. Three full name
        # words need a real patronymic, never merely three capitalized words.
        for count in range(len(words), 0, -1):
            parts = [word.group() for word in words[:count]]
            if _field_value_valid(label, words, count, parts, given_names, max_given_length):
                yield words[0].start(), words[count - 1].end(), "PERSON", 0.98, "explicit-personal-name-field"
                break


def _before_candidates(text: str, initials, before) -> list[tuple[int, int]]:
    candidates = []
    for count in (1, 2):
        if len(before) >= count:
            chosen = before[-count:]
            if all(text[left.end():right.start()].isspace()
                   for left, right in zip(chosen, chosen[1:], strict=False)):
                gap = text[chosen[-1].end():initials.start()]
                if gap and gap.isspace():
                    candidates.append((chosen[0].start(), initials.end()))
    return candidates


def _after_candidates(text: str, initials, after) -> list[tuple[int, int]]:
    candidates = []
    for count in (1, 2):
        if len(after) >= count:
            chosen = after[:count]
            if all(text[left.end():right.start()].isspace()
                   for left, right in zip(chosen, chosen[1:], strict=False)):
                gap = text[initials.end():chosen[0].start()]
                if gap and gap.isspace():
                    # Lowercase city/building abbreviations are not a lone
                    # personal initial; explicit field values are handled above.
                    if count == 1 and initials.group().strip() in {"г.", "д."}:
                        continue
                    candidates.append((initials.start(), chosen[-1].end()))
    return candidates


def _initial_candidates(text: str, given_names: Collection[str]) -> Iterator[Candidate]:
    for initials in _INITIAL_RUN.finditer(text):
        if _is_street_value(text, initials.start()):
            continue
        # Attach at most two preceding/following full words, never a sentence or
        # another field. All resulting words must form a valid name sequence.
        before = list(_WORDS.finditer(text, max(0, initials.start() - 100), initials.start()))[-2:]
        after = list(_WORDS.finditer(text, initials.end(), min(len(text), initials.end() + 100)))[:2]
        candidates = _before_candidates(text, initials, before) + _after_candidates(text, initials, after)
        for start, end in candidates:
            if _is_street_value(text, start):
                continue
            words = [match.group() for match in _PARTS.finditer(text, start, end)]
            if _name_sequence(words, given_names):
                yield start, end, "PERSON", 0.96, "personal-name-with-initials"


def _surname_patronymic_pair(text: str, words) -> Iterator[Candidate]:
    for offset in range(max(0, len(words) - 1)):
        pair = words[offset:offset + 2]
        if (len(pair) == 2 and _surname(pair[0].group()) and not _patronymic(pair[0].group())
                and _patronymic(pair[1].group())
                and not _is_street_value(text, pair[0].start())):
            yield pair[0].start(), pair[-1].end(), "PERSON", 0.95, "surname-and-patronymic"


def _triple_permutation(text: str, words, given_names: Collection[str]) -> Iterator[Candidate]:
    for offset in range(max(0, len(words) - 2)):
        group = words[offset:offset + 3]
        if (len(group) == 3 and all(len(word.group()) > 2 for word in group)
                and _name_sequence([word.group() for word in group], given_names)
                and not _is_street_value(text, group[0].start())):
            yield group[0].start(), group[-1].end(), "PERSON", 0.97, "three-part-name-permutation"


def _all_candidates(text: str, given_names: Collection[str]) -> Iterator[Candidate]:
    if not text:
        return
    max_given_length = min(40, max(map(len, given_names), default=0))
    yield from _field_candidates(text, given_names, max_given_length)
    if "." in text:
        yield from _initial_candidates(text, given_names)
    for word in _JOINED_WORDS.finditer(text):
        if not _is_street_value(text, word.start()) and _joined_name(word.group(), given_names, max_given_length):
            yield word.start(), word.end(), "PERSON", 0.97, "joined-three-part-name"
    # Other permutations of three separated name parts are common in forms.
    for sequence in _SEQUENCE.finditer(text):
        words = list(_PARTS.finditer(text, sequence.start(), sequence.end()))
        yield from _surname_patronymic_pair(text, words)
        yield from _triple_permutation(text, words, given_names)


def person_candidates(
    text: str, *, given_names: Collection[str], public_name_guard: Callable[[str, int, int], bool]
) -> Iterator[Candidate]:
    """Apply the caller's lowercase name dictionary and public-name policy."""
    for candidate in _all_candidates(text, given_names):
        if not public_name_guard(text, candidate[0], candidate[1]):
            yield candidate
