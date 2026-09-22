"""Local, bounded rule-based Russian PII extraction; no model/network calls.

Offsets are Python Unicode string offsets. Context gates ambiguous numbers/dates;
checksums strengthen unlabelled identifiers. Exact originals stay outside this module.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Iterable

from seif.cardholder_fields import cardholder_candidates
from seif.context_filters import refine_candidates
from seif.location_fields import location_candidates
from seif.person_fields import person_candidates
from seif.structured_fields import structured_candidates


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    type: str
    confidence: float = 1.0
    reason: str = ""


TYPES: dict[str, str] = {
    "PERSON": "ФИО",
    "BIRTH_DATE": "Дата рождения",
    "BIRTH_PLACE": "Место рождения",
    "PASSPORT": "Серия и номер паспорта",
    "CITIZENSHIP": "Гражданство",
    "PASSPORT_ISSUER": "Орган выдачи паспорта",
    "DEPARTMENT_CODE": "Код подразделения",
    "PASSPORT_DATE": "Дата выдачи паспорта",
    "DRIVER_LICENSE": "Водительское удостоверение",
    "ADDRESS": "Адрес",
    "COUNTRY": "Страна",
    "POSTAL_CODE": "Почтовый индекс",
    "CITY": "Город",
    "STREET": "Улица",
    "HOUSE": "Дом",
    "APARTMENT": "Квартира",
    "EMAIL": "Электронная почта",
    "PHONE": "Телефон",
    "INN": "ИНН",
    "CARD": "Номер карты",
    "CVV": "Код CVV/CVC",
    "PIN": "ПИН-код",
    "CARDHOLDER": "Имя держателя карты",
    "FOREIGN_DOCUMENT": "Другой документ личности",
    "LOCATION": "Географическое название (NER)",
}
_FLAGS = re.IGNORECASE | re.UNICODE


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, _FLAGS)


_SEP = r"[ \t]*(?::|=|—|-)?[ \t]*"
_WORD = r"[а-яё][а-яё'-]{0,39}"
_NAMEWORD = r"[а-яё][а-яё'-]{1,39}"
_PATR = r"[а-яё-]{2,30}(?:ович(?:а|у|ем|е)?|евич(?:а|у|ем|е)?|ич|овн[аыеу]|евн[аыеу]|овной|евной|иничн[аыеу])"
_NAMES = set(
    "иван петр пётр александр алексей андрей антон артём артем артур борис вадим валентин валерий василий виктор виталий владимир владислав вячеслав геннадий георгий глеб григорий даниил данил денис дмитрий евгений егор иван игорь илья кирилл константин лев леонид максим марк матвей михаил никита николай олег павел пётр роман руслан семён семен сергей станислав степан тимофей тимур фёдор федор юрий ярослав алёна алена александра алина алиса алла анастасия анна валентина валерия варвара вера вероника виктория галина дарья диана ева евгения екатерина елена елизавета жанна зоя инна ирина карина кристина ксения лариса лидия любовь людмила маргарита марина мария надежда наталья наталия нина оксана олеся ольга полина раиса светлана софия софья тамара татьяна ульяна юлия яна".split()
)
_NAME_STEMS = tuple(sorted({x[:-1] if x[-1] in "аьяй" else x for x in _NAMES}, key=len, reverse=True))
_NAME_FORMS = _NAMES | {
    stem + ending for stem in _NAME_STEMS for ending in ("а", "у", "ом", "е", "ой", "ы", "и", "ю", "я")
}
_NAME_STOP = {
    "клиент",
    "клиентка",
    "фио",
    "сотрудник",
    "заёмщик",
    "заемщик",
    "пациент",
    "получатель",
    "поэт",
    "писатель",
    "зовут",
    "это",
    "и",
    "господин",
    "госпожа",
    "уважаемый",
    "уважаемая",
}
_SURNAME = _rx(
    r"(?:ов|ев|ёв|ин|ын|ский|ская|цкий|цкая|енко|ко|их|ых|ович|евич|ян|дзе|швили)(?:а|у|ым|ым|ом|е|ой|ую|ого|ому)?$"
)
_RARE_SURNAMES = {
    stem + ending
    for stem in "ким цой мороз волк лебедь заяц журавль шмидт гоголь блок даль дюма толстых белых чёрных черных".split()
    for ending in ("", "а", "у", "ом", "е")
}
_NAME_TRIPLE = _rx(rf"(?<![\w-])(?P<value>{_NAMEWORD}[ \t]+{_NAMEWORD}[ \t]+{_PATR})(?![\w-])")
_NAME_REVERSE = _rx(rf"(?<![\w-])(?P<value>{_NAMEWORD}[ \t]+{_PATR}[ \t]+{_NAMEWORD})(?![\w-])")
_NAME_PAIR = _rx(rf"(?<![\w-])(?P<value>{_NAMEWORD}[ \t]+{_NAMEWORD})(?![\w-])")
_INITIALS = _rx(rf"(?<![\w-])(?P<value>{_NAMEWORD}[ \t]+[а-яё]\.[ \t]*[а-яё]\.)(?!\w)")
_CONTEXT_NAME = _rx(
    rf"\b(?:фио|ф[.]\s*и[.]\s*о[.]|клиент(?:ка|у|а)?|за[её]мщик(?:а)?|сотрудник(?:а)?|получатель|пациент(?:ка)?|владелец|заявитель|меня зовут|имя и фамилия|фамилия и имя){_SEP}(?P<value>{_NAMEWORD}(?:[ \t]+{_NAMEWORD}){{1,2}})"
)
# Strong field labels permit regional alphabets and Latin names without a
# given-name dictionary. Bounded, delimited values do not consume free prose.
_UNICODE_NAME_WORD = r"[^\W\d_](?:[^\W\d_]|['’ʼ-](?=[^\W\d_])){0,39}"
_STRONG_NAME_FIELD = _rx(
    rf"(?<!\w)(?:фио|фамилия[ \t]+и[ \t]+имя|имя[ \t]+и[ \t]+фамилия|аты[ -]жөні)"
    rf"[ \t]*[:=][ \t]*(?P<value>{_UNICODE_NAME_WORD}(?:[ \t]+{_UNICODE_NAME_WORD}){{1,3}})"
    r"(?=[ \t]*(?:[,;\n.!?]|$))"
)
# These are role prefixes, not a global blacklist of possible personal names.
_NER_CLIENT_ROLE = _rx(r"(?<![\w'’ʼ-])клиент(?:ка|ки|ку|ке|кой|а|у|ом|е|ы|ов|ам|ами|ах)?(?![\w'’ʼ-])")
_NER_ROLE_SEPARATOR = _rx(r"[ \t]*(?P<label>[:=])[ \t]*|[ \t]+")
_NER_FIELD_SPACE = r"[^\S\r\n\u2028\u2029]"
_NER_NAME_VALUE_FIELD = _rx(
    rf"\b(?:фио|ф[.]\s*и[.]\s*о[.]|фамилия|имя|отчество|фамилия и имя|имя и фамилия|аты[ -]жөні)"
    rf"{_NER_FIELD_SPACE}*[:=—–-]{_NER_FIELD_SPACE}*"
    rf"(?:(?:\r\n|[\n\r\u2028\u2029]){_NER_FIELD_SPACE}*)?"
    rf"(?:[«‹“„\"'‘‚]{_NER_FIELD_SPACE}*)?(?:{_UNICODE_NAME_WORD}{_NER_FIELD_SPACE}+){{0,4}}\Z"
)
_PUBLIC_NAME = _rx(
    r"\b(?:поэт(?:а|у|ом)?|писател[ьяю]|стихотворени[еяю]|роман|цитат[аыу]|творчеств[ао]|памятник|имени)\b"
)
_PUBLIC_NAME_LINK = _rx(
    r"\b(?:поэт(?:а|у|ом|е)?|писател(?:ь|я|ю|ем|е)|композитор(?:а|у|ом|е)?|стихотворени[еяю]|"
    r"стих(?:и|ов)|цитат[аыу]|творчеств[ао]|памятник|имени)"
    r"[ \t]*(?:[:—–-][ \t]*)?[«\"']?[ \t]*$"
)
_PUBLIC_WORK_TITLE = _rx(r"\b(?:роман(?:[ауе]|ом)?|поэм[аыуе]|произведени[еяю])[ \t]+[«\"'][ \t]*$")
_PUBLIC_NAME_APPOSITION = _rx(
    r"^[ \t]*[—–,-][ \t]*(?:(?:русский|русская|советский|советская|великий|великая|известный|известная|знаменитый|знаменитая)[ \t]+){0,3}(?:поэт(?:есса)?|писатель(?:ница)?|композитор|драматург)\b"
)
_PERSONAL_NAME = _rx(
    r"\b(?:клиент|пациент|за[её]мщик|сотрудник|фио|зовут|заявител|владелец|договор|паспорт|получатель)"
)
_HISTORICAL_MENTION = _rx(r"\bпушкин[а-яё]*\b")
_HISTORICAL = _rx(r"^(?:(?:александр\w*|а\.)\s+)?(?:(?:сергеевич\w*|с\.)\s+)?пушкин\w*$")
_PUBLIC_ADDRESS = _rx(
    r"\b(?:адрес[а-я]*\s+)?(?:отделени[еяию]|филиал[а-я]*|офис[а-я]*|банкомат[а-я]*)\s+(?:банка|банк|сбербанка|альфа|втб)|\b(?:адрес|расположен[оа]?)\s+банка\b"
)
_PRIVATE_ADDRESS = _rx(
    r"\b(?:прожива[а-я]*|проживани[яе]|регистраци[ияю]|прописк[аи]|домашн[а-я]*|личн[а-я]*|клиент[а-я]*)\b"
)

_MONTH = r"(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])"
_DATE = rf"(?:(?:\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{4}})|(?:\d{{4}}[./-]\d{{1,2}}[./-]\d{{1,2}})|(?:\d{{1,2}}\s+{_MONTH}\s+\d{{4}})|(?:\d{{4}}(?:\s+г(?:ода|\.)?)?[, ]+\d{{1,2}}\s+{_MONTH})|(?:{_MONTH}\s+\d{{1,2}}[, ]+\d{{4}}))"
_DATE += r"(?:[ \t]+(?:года|год|г[.]))?"
_DATE_FIND = _rx(_DATE)
_BIRTH_DATE = _rx(
    rf"\b(?:дата[ \t]+рождения|д[.]?[ \t]*р[.]|родил(?:ся|ась)|рожд[её]н(?:а)?|день рождения|дата рожден[ьи]я){_SEP}(?P<value>{_DATE})(?!\d)"
)
_AFTER_BIRTH = _rx(rf"(?<!\d)(?P<value>{_DATE})(?:\s*г(?:ода|\.)?)?\s+(?:рождения|г[.]?\s*р[.])(?!\w)")
_PASS_DATE = _rx(
    rf"\b(?:дата[ \t]+выдачи(?:[ \t]+паспорта)?|паспорт[ \t]+выдан|выдан(?:а|о)?){_SEP}(?P<value>{_DATE})(?!\d)"
)
_PASS_DATE_LATE = _rx(rf"\bвыдан(?:ный|ная|ное|ные|ного|а|о)?\s+(?:[^\n;:]{{0,130}}?)[, \t]+(?P<value>{_DATE})(?!\d)")
_FIELD_END = r"(?=\s*(?:[,;\n]|$|\b(?:паспорт|телефон|email|e-mail|инн|код подразделения|дата выдачи|дата рождения|гражданство|адрес|родился|родилась|снилс|карта|фио)\b))"
_BIRTH_PLACE = _rx(rf"\b(?:место рождения|родил(?:ся|ась)\s+в\b){_SEP}(?P<value>[^;\n]{{2,140}}?){_FIELD_END}")
_CITIZENSHIP = _rx(
    rf"\b(?:гражданство|гражданин|гражданка){_SEP}(?P<value>(?:российск(?:ая\s+федерация|ое)|росси[яию]|рф|ссср|сша|беларус[ьи]|республик[аи]\s+{_WORD}|казахстан[а]?|украин[аы]|узбекистан[а]?|таджикистан[а]?|кыргызстан[а]?|киргизи[яи]|армении|германии|израиля|турции|франции))(?!\w)"
)

_EMAIL = _rx(
    r"(?<![\w.+-])(?P<value>[a-zа-яё0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@[a-zа-яё0-9](?:[a-zа-яё0-9-]{0,61}[a-zа-яё0-9])?(?:\.[a-zа-яё0-9](?:[a-zа-яё0-9-]{0,61}[a-zа-яё0-9])?){1,8})(?![\w@-])"
)
_PHONE = _rx(
    r"(?<![\w\d])(?P<value>(?:\+7|8|7)[ \t.-]*(?:\([ \t]*\d{3}[ \t]*\)|\d{3})[ \t.-]*\d{3}[ \t.-]*\d{2}[ \t.-]*\d{2})(?!\d)"
)
_INT_PHONE = _rx(r"(?<!\w)(?P<value>\+[1-9]\d{0,2}(?:[ \t().-]*\d){7,13})(?!\d)")
_LABEL_PHONE = _rx(
    rf"\b(?:телефон|тел[.]|мобильный|моб[.]|phone){_SEP}(?P<value>(?:\+?[\d(][\d() .-]{{7,24}}\d))(?!\d)"
)
_PASSPORT = _rx(
    rf"\b(?:паспорт(?:а)?(?:[ \t]+рф)?|серия){_SEP}(?:(?:серия|сер[.])\s*)?(?P<value>\d{{2}}[ \t]*\d{{2}}[ \t,-]*(?:(?:номер|ном[.]?|№)[ \t]*)?\d{{6}})(?!\d)"
)
_PASS_SERIES = _rx(rf"\bпаспорт(?:а)?(?:\s+рф)?{_SEP}(?P<value>\d{{10}})(?!\d)")
_LICENSE = _rx(
    rf"(?<!\w)(?:водительск[а-я]*[ \t]+удостоверени[а-я]*|в[ /]?у|права){_SEP}(?:серия\s*)?(?P<value>(?:\d{{2}}[ \t]*[а-яёa-z]{{2}}|\d{{2}}[ \t]*\d{{2}})[ \t,-]*(?:(?:номер|№)\s*)?\d{{6}})(?!\d)"
)
_FOREIGN_DOC = _rx(
    rf"\b(?:загранпаспорт(?:а)?|иностранный паспорт|вид на жительство|внж|свидетельство о рождении){_SEP}(?:(?:серия|номер|№)\s*)?(?P<value>(?:[IVXLCІVХ]{{1,8}}[- ]?[А-ЯЁA-Z]{{2}}[ \t№-]*\d{{6}}|\d{{2}}[ \t№-]*\d{{7}}|[A-ZА-ЯЁ]{{1,3}}[ \t-]*\d{{6,9}}|\d{{7,12}}))(?!\w)"
)
_DEPT = _rx(rf"\b(?:код(?:[ \t]+подразделения)?|к[ /]п){_SEP}(?P<value>\d{{3}}[ -]\d{{3}})(?!\d)")
_ISSUER = _rx(
    rf"\b(?:кем выдан|орган(?:,?\s+выдавший паспорт| выдачи)?|паспорт выдан|выдан){_SEP}(?:{_DATE}\s*[, ]+)?(?P<value>(?:[оугм]вд|[оу]{{0,2}}фмс|[гу]+\s+мвд|мвд|отдел(?:ом)?|управлени[ея]м?)[^\n;]{{0,150}}?)(?=\s*(?:[,;\n]|$|\b(?:код|дата|\d{{2}}[./]\d{{2}}[./]\d{{4}})\b))"
)
_ISSUER_DIRECT = _rx(
    r"\b(?P<value>(?:[оуг]вд|[оу]{0,2}фмс|гу\s+мвд|мвд)\s+(?:россии\s+)?(?:по\s+)?[а-яё0-9 .-]{3,100}?)(?=\s*(?:[,;\n]|$|\b(?:код подразделения|дата выдачи|\d{2}[./]\d{2}[./]\d{4})\b))"
)
_PUBLIC_INN_OWNER = _rx(
    r"\bинн[ \t]+(?:банка|компании|организации|юрлица|юридического[ \t]+лица)(?:[ \t]+[а-яёa-z«»\"-]{1,35}){0,3}[ \t]*[:=—-]?[ \t]*$"
)
_PRIVATE_INN_OWNER = _rx(r"\b(?:клиент[а-я]*|сотрудник[а-я]*|физлиц[а-я]*|физическ[а-я]*|за[её]мщик[а-я]*|ип)\b")
_INN_LABEL = _rx(
    rf"\bинн(?:[ \t]+(?:физлица|физического[ \t]+лица|клиента|сотрудника|за[её]мщика|поручителя|ип|банка|компании|организации|юрлица|юридического[ \t]+лица))?(?:[ \t]+банка)?{_SEP}(?P<value>\d{{4}}[ \t]+\d{{4}}[ \t]+\d{{4}}|\d{{12}}|\d{{10}})(?!\d)"
)
_INN_BARE = _rx(r"(?<![\w\d])(?P<value>\d{12}|\d{10})(?![\w\d])")
_CARD_LABEL = _rx(
    rf"\b(?:номер(?:[ \t]+банковской)?[ \t]+карты|банковская карта|карта|card|pan){_SEP}(?P<value>\d(?:[ \t-]*\d){{12,18}})(?!\d)"
)
_CARD_BARE = _rx(r"(?<![\w\d])(?P<value>[2-6]\d{2,3}(?:[ \t-]?\d){9,15})(?![\w\d])")
_CARD_SUFFIX = _rx(
    rf"\bпоследние[ \t]+(?:4|четыре)[ \t]+цифры[ \t]+(?:номера[ \t]+)?"
    rf"(?:банковской[ \t]+)?карты{_SEP}(?P<value>[0-9]{{4}})(?!\w)"
)
_CVV = _rx(
    rf"\b(?:cvv2?|cvc2?|cid|сvv|сvv2|сvc|си[ -]?ви[ -]?ви|cvv[ -]код|код безопасности)(?:[ -]*(?:код|карты))?{_SEP}(?P<value>\d{{3,4}})(?!\d)"
)
_PIN = _rx(rf"\b(?:пин|pin)(?:[ -]*код)?(?:[ \t]+карты)?{_SEP}(?P<value>\d{{4,6}})(?!\d)")
# Regional identifiers are recognized from explicit field labels, never merely
# from length. These are privacy recognizers, not government validity checks.
# Country-specific formats and official references are documented in docs/cis.md.
_IDENTIFIER_SEP = rf"{_SEP}(?:(?:номер|№|no[.]?)[ \t]*)?{_SEP}"
_KZ = r"(?:(?:республик[аи][ \t]+)?казахстан(?:а)?|рк|kz)"
_UA = r"(?:украин[аы]|україн[аи]|ukraine|ua)"
_UZ = r"(?:(?:республик[аи][ \t]+)?узбекистан(?:а)?|uzbekistan|uz)"
_KG = r"(?:кыргызстан(?:а)?|киргиз(?:ия|ии)|кыргызск(?:ая|ой)[ \t]+республик[аи]|kg)"
_BY = r"(?:(?:республик[аи][ \t]+)?беларус[ьи]|белорусси[яи]|рб|belarus|by)"
_MD = r"(?:(?:республик[аи][ \t]+)?молдов[аы]|молдави[яи]|moldova|md)"
_AZ = r"(?:(?:республик[аи][ \t]+)?азербайджан(?:а)?|azerbaijan|az)"


def _regional_field(label: str, country: str, value: str, *, country_optional: bool = False) -> re.Pattern:
    """Require an adjacent country and field, or an unambiguous field acronym.

    Country may precede the field or be its suffix/parenthesized qualifier. No
    arbitrary words or completed sentences can transfer country ownership.
    """
    country_suffix = rf"(?:[ \t]+(?:гражданина[ \t]+)?{country}|[ \t]*\([ \t]*{country}[ \t]*\))"
    if country_optional:
        country_suffix += "?"
    field = rf"(?:(?:{label}){country_suffix}|{country}[ \t]*(?:[,;:/—-][ \t]*|[ \t]+|\n[ \t]*)(?:{label}))"
    return _rx(rf"(?<!\w){field}{_IDENTIFIER_SEP}(?P<value>{value})(?!\w)")


_CIS_IDENTIFIERS = (
    (
        ("иин", "жсн"),
        _regional_field(r"(?:иин|жсн)(?:[ \t]*/[ \t]*(?:иин|жсн))?", _KZ, r"[0-9]{12}", country_optional=True),
        "INN",
    ),
    (("рнокпп",), _regional_field(r"рнокпп", _UA, r"[0-9]{10}", country_optional=True), "INN"),
    (("іпн",), _regional_field(r"іпн", _UA, r"[0-9]{10}", country_optional=True), "INN"),
    (
        ("пинфл", "jshshir", "pinfl"),
        _regional_field(r"пинфл|jshshir|pinfl", _UZ, r"[0-9]{14}", country_optional=True),
        "INN",
    ),
    (
        ("пин", "персональный"),
        _regional_field(r"пин|персональный[ \t]+идентификационный[ \t]+номер", _KG, r"[0-9]{14}"),
        "INN",
    ),
    (
        ("удостоверен", "id", "куәлік"),
        _regional_field(
            r"(?:номер[ \t]+)?(?:удостоверени[ея][ \t]+личности|id[ -]?(?:карт[аы]|card)|жеке[ \t]+куәлік)",
            _KZ,
            r"[0-9]{9}",
        ),
        "FOREIGN_DOCUMENT",
    ),
    (
        ("паспорт", "id"),
        _regional_field(r"(?:номер[ \t]+)?(?:паспорт(?:а)?|id[ -]?(?:карт[аы]|картк[аи]|card))", _UA, r"[0-9]{9}"),
        "FOREIGN_DOCUMENT",
    ),
    (
        ("личный", "идентификационный", "персональный"),
        _regional_field(r"(?:личный|идентификационный|персональный)[ \t]+номер", _BY, r"(?a:[a-z0-9]{14})"),
        "FOREIGN_DOCUMENT",
    ),
    (("idnp",), _regional_field(r"idnp", _MD, r"[0-9]{13}"), "INN"),
    (("fin", "fın", "fi\u0307n", "фин"), _regional_field(r"fin|fın|фин", _AZ, r"(?a:[a-z0-9]{7})"), "FOREIGN_DOCUMENT"),
)

_ADDRESS = _rx(
    rf"\b(?:адрес(?:[ \t]+(?:проживания|регистрации|доставки|клиента|заёмщика|заемщика|банка|отделения[ \t]+банка|филиала[ \t]+банка))?|прописан(?:а)?|зарегистрирован(?:а)?|прожива(?:ю|ет)|жив[её]т){_SEP}(?P<value>[^;\n]{{4,240}}?)(?=\s*(?:[;\n]|$|\b(?:паспорт|телефон|email|инн|дата рождения|гражданство|номер карты|фио)\b))"
)
_COUNTRY = _rx(
    rf"\bстрана(?:[ \t]+проживания)?{_SEP}(?P<value>{_WORD}(?:[ \t]+{_WORD}){{0,2}}?)(?=\s*(?:[,;.\n!?]|$|\b(?:город|индекс|адрес)\b))"
)
_POSTAL = _rx(rf"\bиндекс(?:а|у|ом|е)?{_SEP}(?P<value>\d{{6}})(?!\d)")
_CITY = _rx(
    rf"(?<!\w)(?:город(?:[ \t]+(?:проживания|рождения|регистрации))?|г[.](?![ \t]*о[.]))"
    rf"{_SEP}(?P<value>{_WORD}(?:[-]{_WORD})?)(?!\w)"
)
_STREET = _rx(
    rf"(?<!\w)(?:улица|ул[.]|проспект|пр-т|переулок|пер[.]|бульвар|бул[.]|набережная|наб[.]|шоссе){_SEP}(?P<value>(?:\d{{1,3}}[- ]?)?{_WORD}(?:[ \t]+{_WORD}){{0,2}}?)(?=\s*(?:[,;.\n!?]|$|\b(?:дом|д[.]|кв[.]|квартира|корпус|корп[.])|\d))"
)
_HOUSE = _rx(
    rf"(?<!\w)(?:дом|д[.]){_SEP}(?P<value>\d{{1,4}}(?:[/\-]\d{{1,4}})?[а-яёa-z]?(?:\s*(?:корпус|корп[.]|к[.]|строение|стр[.])\s*\d{{1,3}}[а-яёa-z]?)?)(?!\w)"
)
_APARTMENT = _rx(rf"(?<!\w)(?:квартира|кв[.]){_SEP}(?P<value>\d{{1,5}}[а-яёa-z]?)(?!\w)")
_ADDRESS_HINT = _rx(r"\b(?:ул[.]|улиц|дом|д[.]|кв[.]|квартир|проспект|пр-т|переулок|шоссе|корпус|набережн)|\b\d{6}\b")
_ADDRESS_POSTAL = _rx(r"(?<!\d)(?P<value>\d{6})(?=\s*[, ]\s*(?:г[.]|город|обл[.]|[А-ЯЁ][а-яё-]+))")

# Independently implemented component composition, inspired by Natasha/Yargy's
# address grammar. Explicit street + linked house anchors are mandatory; city,
# postcode, building and apartment components may extend the bounded span.
_ADDR_JOIN = r"(?:[ \t]*,[ \t]*(?:\n[ \t]*)?|[ \t]+|[ \t]*\n[ \t]*)"
_ADDR_NAME_WORD = (
    r"(?!(?:д|дом|корп|корпус|к|стр|строение|кв|квартира|лит|литера|г|город|"
    r"телефон|паспорт|email|инн|заказ|домофон)\b)" + _UNICODE_NAME_WORD
)
_ADDR_NAME = rf"(?:[0-9]{{1,4}}(?:-(?:я|й|ая|ый))?[ \t]+)?{_ADDR_NAME_WORD}(?:[ \t]+{_ADDR_NAME_WORD}){{0,3}}"
_ADDR_HOUSE_NUMBER = r"[0-9]{1,4}(?:[/\-][0-9]{1,4})?[а-яёa-z]?(?:[кk][0-9]{1,3})?(?!\w)"
_ADDR_STREET_HEAD = _rx(
    r"(?<!\w)(?:улица|ул|проспект|просп|пр-т|пр-кт|переулок|пер|шоссе|набережная|наб|"
    r"бульвар|бул|б-р|проезд|аллея|площадь|пл|микрорайон|мкр)(?!\w)[.]?[ \t]*"
)
_ADDR_STREET_HOUSE = _rx(
    rf"{_ADDR_NAME}{_ADDR_JOIN}(?P<label>(?:дом|д)(?!\w)[.]?[ \t]*)?"
    rf"(?:№[ \t]*)?(?P<house>{_ADDR_HOUSE_NUMBER})"
)
_ADDR_EXTRA = _rx(
    rf"{_ADDR_JOIN}(?P<label>корпус|корп|к|строение|стр|литера|литер|лит|квартира|кв)(?!\w)"
    rf"[.]?[ \t]*(?:№[ \t]*)?(?:{_ADDR_HOUSE_NUMBER}|[а-яёa-z](?!\w))"
)
_ADDR_LOCALITY = (
    rf"(?:город|г|пос[её]лок|пос|пгт|село|с|деревня|дер|д)(?!\w)[.]?[ \t]*"
    rf"(?:им[.][ \t]*)?"
    rf"{_ADDR_NAME_WORD}(?:[ \t]+{_ADDR_NAME_WORD}){{0,1}}"
)
_ADDR_LOCALITY_PREFIX = _rx(rf"(?<!\w)(?:[0-9]{{5,6}}{_ADDR_JOIN})?{_ADDR_LOCALITY}{_ADDR_JOIN}$")
_ADDR_REGION_PREFIX = _rx(
    rf"(?<!\w)(?:{_ADDR_NAME_WORD}[ \t]+(?:область|обл[.]|край|район|р-н|г[.]о[.]|с[.]п[.])"
    rf"|республика[ \t]+{_ADDR_NAME}){_ADDR_JOIN}$"
)
_ADDR_POSTAL_PREFIX = _rx(
    rf"(?<!\w)[0-9]{{5,6}}{_ADDR_JOIN}(?:{_ADDR_NAME_WORD}(?:[ \t]+{_ADDR_NAME_WORD})?{_ADDR_JOIN})?$"
)
_ADDR_LOCALITY_SUFFIX = _rx(rf"{_ADDR_JOIN}{_ADDR_LOCALITY}(?=[ \t]*(?:[,;.!?\n]|$))")
_ADDR_POSTAL_SUFFIX = _rx(rf"{_ADDR_JOIN}[0-9]{{5,6}}(?!\w)(?=[ \t]*(?:[,;.!?\n]|$))")
_ADDR_BARE_HOUSE_END = _rx(r"(?=[ \t]*(?:[,;.!?\n]|$))")

# Candidate validation and contextual invalidation are separate decisions.
# Inspired by Presidio's recognizer/context interfaces, implemented independently.
_CONTEXT_BREAK = _rx(r"[!?]|\n[ \t]*\n|\.(?=[ \t\r\n]|$)")
_CONTEXT_ABBREVIATION = _rx(r"\b(?:г|гор|ул|д|кв|корп|стр|обл|р-н|им|пос|тел|гг)\.$")
_DOCUMENT_OWNER = _rx(
    r"(?P<passport>\bпаспорт(?:а|у|ом|е)?\b)|"
    r"(?P<license>(?<!\w)(?:в[ /]?у|водительск[а-яё]*[ \t]+удостоверени[а-яё]*)(?!\w))|"
    r"(?P<other>\b(?:(?:талон|товар|чек|сертификат|заказ|билет|пропуск)(?:а|у|ом|е)?|"
    r"накладн(?:ая|ую|ой|ые|ых)|справк(?:а|у|и|е|ой))\b)"
)
_ISSUE_DATE_FIELD = _rx(r"\bдата[ \t]+выдачи(?:[ \t]+паспорта)?[ \t]*[:=—-]?[ \t]*$")
_NONPERSONAL_NUMBER_FIELD = _rx(
    r"\b(?:(?:номер|идентификатор|id)[ \t]+(?:заказа|заявки|накладной|поставки|тикета|инцидента)|"
    r"артикул(?:[ \t]+товара)?|sku|order[_ -]?id|ticket[_ -]?id|тикет|заказ|накладная)"
    r"[ \t]*(?:[:=№#—-][ \t]*){0,2}$"
)
_EXPLICIT_PERSONAL_NUMBER_FIELD = _rx(
    r"\b(?:телефон|тел|мобильн[а-яё]*|phone|инн|карта|карты|card|pan)\b"
    r"[^.;,\n\d:]{0,48}[ \t]*(?:[:=№#—-][ \t]*){0,2}$"
)


def _local_record_prefix(text: str, start: int, limit: int = 180) -> str:
    """Bounded sentence/paragraph context, retaining address abbreviations.

    One newline or a semicolon may separate fields of the same passport record;
    a completed sentence or a blank line cannot transfer document ownership.
    """
    prefix = text[max(0, start - limit) : start]
    boundary = 0
    for match in _CONTEXT_BREAK.finditer(prefix):
        if match.group() == "." and _CONTEXT_ABBREVIATION.search(prefix[max(0, match.start() - 8) : match.end()]):
            continue
        boundary = match.end()
    return prefix[boundary:]


def _has_passport_context(text: str, start: int) -> bool:
    prefix = _local_record_prefix(text, start)
    owners = list(_DOCUMENT_OWNER.finditer(prefix))
    if owners:
        # An explicit other owner beats an unqualified "дата выдачи";
        # "дата выдачи паспорта" contains the later, more specific owner.
        return owners[-1].lastgroup == "passport"
    if not _ISSUE_DATE_FIELD.search(prefix):
        return False
    # A following date field can continue a driver's licence record across a
    # sentence, but must not acquire the PASSPORT_DATE type by default.
    earlier = list(_DOCUMENT_OWNER.finditer(text[max(0, start - 180):start]))
    return not earlier or earlier[-1].lastgroup != "license"


def _license_owns_number(text: str, start: int) -> bool:
    owners = list(_DOCUMENT_OWNER.finditer(_local_record_prefix(text, start)))
    return bool(owners and owners[-1].lastgroup == "license")


def _is_nonpersonal_number_field(text: str, start: int) -> bool:
    prefix = _local_record_prefix(text, start, limit=96)
    # Explicit PII labels are stronger evidence than a nearby business identifier.
    # This veto is only used for unlabelled numeric recognizers, never their
    # explicit-field counterparts, and never excludes an equal value elsewhere.
    if _EXPLICIT_PERSONAL_NUMBER_FIELD.search(prefix):
        return False
    return bool(_NONPERSONAL_NUMBER_FIELD.search(prefix))


_PRIORITY = {
    "FOREIGN_DOCUMENT": 110,
    "PASSPORT": 105,
    "DRIVER_LICENSE": 104,
    "BIRTH_DATE": 103,
    "PASSPORT_DATE": 103,
    "DEPARTMENT_CODE": 103,
    "CARDHOLDER": 102,
    "PASSPORT_ISSUER": 101,
    "BIRTH_PLACE": 100,
    "ADDRESS": 99,
    "CARD": 95,
    "INN": 94,
    "PHONE": 90,
    "EMAIL": 90,
    "PERSON": 80,
    "LOCATION": 70,
}


def _digits(value: str) -> str:
    return "".join(c for c in value if c.isascii() and c.isdigit())


def _luhn(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) < 2:
        return False
    total = 0
    parity = len(digits) % 2
    for i, char in enumerate(digits):
        n = int(char)
        if i % 2 == parity:
            n = n * 2 - (9 if n >= 5 else 0)
        total += n
    return total % 10 == 0


def _valid_inn(value: str) -> bool:
    if len(set(value)) < 2:
        return False
    d = [int(c) for c in value]
    if len(d) == 10:
        return sum(a * b for a, b in zip(d[:9], (2, 4, 10, 3, 5, 9, 4, 6, 8), strict=True)) % 11 % 10 == d[9]
    if len(d) == 12:
        return (
            sum(a * b for a, b in zip(d[:10], (7, 2, 4, 10, 3, 5, 9, 4, 6, 8), strict=True)) % 11 % 10 == d[10]
            and sum(a * b for a, b in zip(d[:11], (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8), strict=True)) % 11 % 10 == d[11]
        )
    return False


def _valid_date(value: str) -> bool:
    # The brief explicitly permits MM.DD.YYYY and YYYY.DD.MM: accept either
    # interpretation if a real calendar date exists; do not silently normalize it.
    numbers = [int(n) for n in re.findall(r"\d+", value)]
    month_match = _rx(_MONTH).search(value)
    if month_match:
        months = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
        prefix = month_match.group().lower()[:3].replace("мая", "май")
        month = months.index(prefix) + 1
        year = next((n for n in numbers if n >= 1000), 0)
        day = next((n for n in numbers if n < 1000), 0)
        options = [(year, month, day)]
    elif len(numbers) == 3:
        if numbers[0] >= 1000:
            year, a, b = numbers
        else:
            a, b, year = numbers
        options = [(year, b, a), (year, a, b)]
    else:
        return False
    for year, month, day in options:
        if not 1800 <= year <= 2100:
            continue
        try:
            date(year, month, day)
            return True
        except ValueError:
            pass
    return False


def _given_name(word: str) -> bool:
    word = word.lower().strip(".,")
    return word in _NAME_FORMS


def _surname(word: str) -> bool:
    word = word.lower()
    return word not in _NAME_STOP and bool(_SURNAME.search(word) or word in _RARE_SURNAMES)


def _is_public_name(text: str, start: int, end: int) -> bool:
    before = _local_record_prefix(text, start, limit=65)
    # Public-reference cues must attach to this name, not merely appear in an
    # earlier clause. An unrelated discussion of a poet cannot exempt a client.
    boundary = max(before.rfind(";"), before.rfind("\n"))
    before = before[boundary + 1 :]
    value = text[start:end]
    # A personal record explicitly identifying even a famous namesake wins.
    if _PERSONAL_NAME.search(before):
        return False
    if _PUBLIC_NAME_LINK.search(before) or _PUBLIC_WORK_TITLE.search(before) or _HISTORICAL.fullmatch(value):
        return True
    after = text[end : end + 100]
    if _PUBLIC_NAME_APPOSITION.match(after):
        return True
    # A two-token candidate may be nested in a three-token public name.
    # Check exactly one following word instead of excluding the whole sentence.
    following = re.match(r"[ \t]+[а-яёА-ЯЁ'-]{2,40}", after)
    if following:
        third_word = following.group().strip()
        return bool(
            _HISTORICAL.fullmatch(value + following.group())
            or (_surname(third_word) and _PUBLIC_NAME_APPOSITION.match(after[following.end() :]))
        )
    return False


def _is_public_record(text: str, start: int) -> bool:
    before = text[max(0, start - 160) : start]
    boundary = max(before.rfind(";"), before.rfind("\n"), before.rfind("!"), before.rfind("?"))
    before = before[boundary + 1 :]
    return not _PERSONAL_NAME.search(before) and bool(_PUBLIC_NAME.search(before) or _HISTORICAL_MENTION.search(before))


def _is_public_address(text: str, start: int) -> bool:
    before = text[max(0, start - 150) : start]
    # Limit context to the local sentence/record, not an earlier bank mention.
    boundary = max(before.rfind(";"), before.rfind("\n"), before.rfind("!"), before.rfind("?"))
    before = before[boundary + 1 :]
    public = list(_PUBLIC_ADDRESS.finditer(before))
    private = list(_PRIVATE_ADDRESS.finditer(before))
    return bool(public and (not private or public[-1].start() > private[-1].start()))


def _structured_addresses(text: str) -> Iterable[Span]:
    """Compose adjacent address parts in bounded windows, retaining raw offsets.

    No gazetteer, model, normalization or arbitrary intervening prose is used.
    A bare number after a street needs a field/sentence boundary or a recognized
    building/apartment component; years and quantities in prose do not qualify.
    """
    for anchor in _ADDR_STREET_HEAD.finditer(text):
        window = text[anchor.end() : anchor.end() + 320]
        core = _ADDR_STREET_HOUSE.match(window)
        if core is None:
            continue
        stop = core.end()
        extension_count = 0
        for _ in range(6):
            extra = _ADDR_EXTRA.match(window, stop)
            if extra is None:
                break
            stop = extra.end()
            extension_count += 1
        if not core.group("label") and not extension_count and not _ADDR_BARE_HOUSE_END.match(window, stop):
            continue
        # Locality/postcode may trail the street record, in either order.
        for first, second in ((_ADDR_LOCALITY_SUFFIX, _ADDR_POSTAL_SUFFIX), (_ADDR_POSTAL_SUFFIX, _ADDR_LOCALITY_SUFFIX)):
            suffix = first.match(window, stop)
            if suffix:
                stop = suffix.end()
                suffix = second.match(window, stop)
                if suffix:
                    stop = suffix.end()
                break
        start = anchor.start()
        prefix_start = max(0, start - 160)
        prefix = text[prefix_start:start]
        preceding = _ADDR_LOCALITY_PREFIX.search(prefix) or _ADDR_POSTAL_PREFIX.search(prefix)
        if preceding:
            possible_start = prefix_start + preceding.start()
            # A sliced prefix must not create a new word/number boundary, nor
            # reinterpret a nearby order number as an address postcode.
            if (
                not (possible_start and (text[possible_start - 1].isalnum() or text[possible_start - 1] == "_"))
                and not _is_nonpersonal_number_field(text, possible_start)
            ):
                start = possible_start
        for _ in range(3):
            region_start = max(0, start - 140)
            region = _ADDR_REGION_PREFIX.search(text[region_start:start])
            if region is None:
                break
            start = region_start + region.start()
        if stop == len(window) and anchor.end() + stop < len(text):
            # Never accept a partial house/apartment value at the window edge.
            continue
        if not _is_public_address(text, start) and not _is_public_address(text, anchor.start()):
            yield Span(start, anchor.end() + stop, "ADDRESS", 0.98, "linked-address-components")


def _is_public_inn(text: str, start: int, value: str) -> bool:
    # 12-digit identifiers can belong to individuals and are never exempted.
    if len(value) != 10:
        return False
    owner = _PUBLIC_INN_OWNER.search(text[max(0, start - 120) : start])
    return bool(owner and not _PRIVATE_INN_OWNER.search(owner.group()))


def _resolve(candidates: Iterable[Span]) -> list[Span]:
    """Resolve local overlap clusters; adjacent spans remain separate."""
    ordered = sorted(set(candidates), key=lambda s: (s.start, s.end))
    result: list[Span] = []
    cluster: list[Span] = []
    end = -1

    def flush() -> None:
        if len(cluster) == 1:
            result.append(cluster[0])
            return
        accepted: list[Span] = []
        starts: list[int] = []
        for span in sorted(
            cluster,
            key=lambda s: (
                # An explicitly labelled regional personal ID can coincidentally
                # pass Luhn; its field meaning takes precedence over bare CARD.
                -max(_PRIORITY.get(s.type, 85), 96 if s.reason == "cis-personal-id" else 0),
                -(s.end - s.start),
                -s.confidence,
                s.start,
                # Exact ties must not inherit per-process set/hash ordering.
                s.type,
                s.reason,
            ),
        ):
            i = bisect_left(starts, span.start)
            if i and accepted[i - 1].end > span.start:
                continue
            if i < len(accepted) and span.end > accepted[i].start:
                continue
            accepted.insert(i, span)
            starts.insert(i, span.start)
        result.extend(accepted)

    for span in ordered:
        if cluster and span.start >= end:
            flush()
            cluster = []
        cluster.append(span)
        end = max(end, span.end)
    if cluster:
        flush()
    return result


def _trim_ner_person_role(
    text: str, span: Span, person_starts: Sequence[int], person_cover_ends: Sequence[int]
) -> Span | None:
    """Remove a leading client role while preserving explicit/core name values.

    Only NER PERSON boundaries change. A standalone role needs a visible field
    delimiter; an undelimited role needs a remaining model-recognized name.
    No given-name dictionary or capitalization requirement limits that name.
    """
    role = _NER_CLIENT_ROLE.match(text, span.start)
    if role is None or role.end() > span.end:
        return span
    covering = bisect_right(person_starts, span.start) - 1
    if covering >= 0 and person_cover_ends[covering] > span.start:
        return span
    # Preserve dotted/quoted fields and a value on the immediately next line.
    # Only name words can follow the opening quote; a completed value, later
    # sentence, another record or a blank line cannot exempt an unrelated role.
    before = text[max(0, span.start - 200) : span.start]
    if _NER_NAME_VALUE_FIELD.search(before):
        return span
    separator = _NER_ROLE_SEPARATOR.match(text, role.end(), min(len(text), span.end + 32))
    if separator is None:
        return span
    if separator.end() >= span.end:
        return None if separator.group("label") else span
    start = separator.end()
    # Repeated explicit labels are still labels. Do not repeatedly remove bare
    # words: after "Клиент: Клиент Иванович", the second word may be a surname.
    while role := _NER_CLIENT_ROLE.match(text, start):
        if role.end() > span.end:
            break
        separator = _NER_ROLE_SEPARATOR.match(text, role.end(), min(len(text), span.end + 32))
        if separator is None or not separator.group("label"):
            break
        if separator.end() >= span.end:
            return None
        start = separator.end()
    return Span(start, span.end, span.type, span.confidence, span.reason)


def _merge_ner_candidates(
    text: str, base_spans: Sequence[Span], candidates: Sequence[Span], *, allowed_types: frozenset[str]
) -> list[Span]:
    """Merge optional NER results without changing structured field classes.

    Coordinates are Unicode code-point offsets into the exact, unnormalized
    input. The caller owns model loading, deadlines and availability policy.
    Malformed external output rejects the whole merge rather than silently
    lowering protection. Scores are bounded heuristics, not probabilities.
    """
    if not isinstance(text, str) or not isinstance(base_spans, Sequence) or not isinstance(candidates, Sequence):
        raise ValueError("NER merge requires text and sequences of spans")
    if len(candidates) > min(100_000, max(1024, len(text))):
        raise ValueError("Too many NER candidates")

    def validate(span: Span, external: bool) -> None:
        if not isinstance(span, Span):
            raise ValueError("NER merge received an invalid span")
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or not 0 <= span.start < span.end <= len(text)
            or not isinstance(span.type, str)
            or not span.type
            or isinstance(span.confidence, bool)
            or not isinstance(span.confidence, (int, float))
            or not math.isfinite(span.confidence)
            or not 0 <= span.confidence <= 1
        ):
            raise ValueError("NER merge received invalid span bounds or confidence")
        if external and (span.type not in allowed_types or span.end - span.start > 200):
            raise ValueError("NER candidates need a supported type and at most 200 characters")

    # Validate every result before filtering any; a public-name exemption must
    # not hide a broken response from the configured protection component.
    for span in base_spans:
        validate(span, external=False)
    for span in candidates:
        validate(span, external=True)

    # Prefix maxima support overlapping caller-supplied base spans without a
    # quadratic scan for each NER candidate. Any core name value stays intact.
    core_people = sorted((span.start, span.end) for span in base_spans if span.type == "PERSON")
    person_starts: list[int] = []
    person_cover_ends: list[int] = []
    for start, end in core_people:
        person_starts.append(start)
        person_cover_ends.append(max(end, person_cover_ends[-1] if person_cover_ends else end))
    existing = {(span.type, span.start, span.end) for span in base_spans if span.type in allowed_types}
    accepted: dict[tuple[str, int, int], Span] = {}
    for span in candidates:
        identity = (span.type, span.start, span.end)
        if identity in existing:
            continue
        if span.type == "PERSON" and _is_public_name(text, span.start, span.end):
            continue
        if span.type == "PERSON":
            span = _trim_ner_person_role(text, span, person_starts, person_cover_ends)
            if span is None:
                continue
            identity = (span.type, span.start, span.end)
            if identity in existing:
                continue
        if span.type == "LOCATION":
            # Reuse the same public/private bank policy in this sentence only;
            # a bank in an earlier sentence cannot exempt an unrelated place.
            prefix = _local_record_prefix(text, span.start, limit=150)
            if _is_public_address(prefix, len(prefix)):
                continue
        previous = accepted.get(identity)
        if previous is None or span.confidence > previous.confidence:
            # Do not propagate free-form upstream explanations into API metadata.
            accepted[identity] = Span(
                span.start, span.end, span.type, float(span.confidence),
                "ner-person" if span.type == "PERSON" else "ner-location",
            )
    # LOCATION is less specific than every structured field, including individual
    # address components; overlaps never relabel CITY/STREET/ADDRESS as LOCATION.
    return _resolve(refine_candidates(text, [*base_spans, *accepted.values()]))


def merge_person_candidates(text: str, base_spans: Sequence[Span], candidates: Sequence[Span]) -> list[Span]:
    """Compatibility entry point for PERSON-only models and prior evaluations.

    LOCATION and all other external entity types still reject the whole merge.
    """
    return _merge_ner_candidates(text, base_spans, candidates, allowed_types=frozenset({"PERSON"}))


def merge_ner_candidates(text: str, base_spans: Sequence[Span], candidates: Sequence[Span]) -> list[Span]:
    """Merge validated PERSON/LOCATION spans with raw Unicode input offsets.

    LOCATION means a model-recognized geographical name, not a verified city or
    a complete address. It has lower overlap priority than core field types.
    The caller owns model deadlines and availability; malformed output raises
    ValueError before any public-context filtering can hide that failure.
    """
    return _merge_ner_candidates(text, base_spans, candidates, allowed_types=frozenset({"PERSON", "LOCATION"}))


@lru_cache(maxsize=128)
def _custom_pattern(pattern: str):
    import regex

    if not 1 <= len(pattern) <= 512:
        raise ValueError("Custom pattern must contain 1–512 characters")
    try:
        compiled = regex.compile(pattern, regex.IGNORECASE | regex.UNICODE)
        if compiled.search("", timeout=0.02):
            raise ValueError("Custom patterns must not match empty text")
    except regex.error as exc:
        raise ValueError("Invalid custom pattern") from exc
    return compiled


def validate_extra_rule(rule: dict) -> None:
    """Validate configuration without echoing potentially sensitive expressions."""
    if not isinstance(rule, dict) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,39}", str(rule.get("type", ""))):
        raise ValueError("Invalid custom entity type")
    if not isinstance(rule.get("pattern"), str):
        raise ValueError("Custom rule needs a pattern")
    _custom_pattern(rule["pattern"])


def detect(text: str, *, extra_rules: list[dict] | None = None) -> list[Span]:
    """Identify PII with case-insensitive rules and return nonoverlapping spans.

    Extension rules use the `regex` engine with a per-rule wall-clock deadline.
    A timeout raises ValueError: callers must fail closed, never forward plaintext.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return []
    candidates: list[Span] = []

    def add(pattern: re.Pattern, kind: str, reason: str = "context", confidence: float = 0.98, check=None) -> None:
        for match in pattern.finditer(text):
            start, end = match.span("value")
            if kind == "PERSON" and reason == "personal-record-context":
                words = list(re.finditer(_NAMEWORD, text[start:end], _FLAGS))
                if len(words) == 3 and not any(re.fullmatch(_PATR, word.group(), _FLAGS) for word in words[1:]):
                    end = start + words[1].end()
            while end > start and text[end - 1] in " \t,":
                end -= 1
            if kind in {"ADDRESS", "BIRTH_PLACE", "PASSPORT_ISSUER"}:
                while end > start and text[end - 1] in ".!?":
                    end -= 1
            value = text[start:end]
            if value and (check is None or check(value, start, end)):
                candidates.append(Span(start, end, kind, confidence, reason))

    lower = text.lower()
    if "@" in text:
        add(_EMAIL, "EMAIL", "format", 0.995)
    for hints, pattern, kind in _CIS_IDENTIFIERS:
        if any(hint in lower for hint in hints):
            add(pattern, kind, "cis-personal-id" if kind == "INN" else "cis-document-context")
    if any(c.isdigit() for c in text):
        add(
            _PHONE,
            "PHONE",
            "russian-phone-format",
            0.98,
            lambda _, start, end: not _is_nonpersonal_number_field(text, start),
        )
        if "+" in text:
            add(
                _INT_PHONE,
                "PHONE",
                "international-phone-format",
                0.98,
                lambda value, start, end: (
                    8 <= len(_digits(value)) <= 15 and not _is_nonpersonal_number_field(text, start)
                ),
            )
        if any(label in lower for label in ("тел", "моб", "phone")):
            add(_LABEL_PHONE, "PHONE", check=lambda value, *_: 7 <= len(_digits(value)) <= 15)
        if "паспорт" in lower or "серия" in lower:
            add(_PASSPORT, "PASSPORT", check=lambda _, s, e: not _license_owns_number(text, s))
            add(_PASS_SERIES, "PASSPORT")
        if any(label in lower for label in ("удостоверен", "в/у", "в у", "права")):
            add(_LICENSE, "DRIVER_LICENSE")
        if any(label in lower for label in ("паспорт", "внж", "жительств", "свидетельство")):
            add(_FOREIGN_DOC, "FOREIGN_DOCUMENT")
        if "код" in lower or "к/п" in lower:
            add(_DEPT, "DEPARTMENT_CODE")
        add(_INN_LABEL, "INN", check=lambda v, s, e: not _is_public_inn(text, s, v))
        add(
            _INN_BARE,
            "INN",
            "checksum",
            0.94,
            lambda v, s, e: (
                _valid_inn(v) and not _is_public_inn(text, s, v) and not _is_nonpersonal_number_field(text, s)
            ),
        )
        add(_CARD_LABEL, "CARD", "explicit-card-context", 0.98)
        add(_CARD_SUFFIX, "CARD", "explicit-card-suffix", 0.98)
        add(
            _CARD_BARE,
            "CARD",
            "luhn-checksum",
            0.96,
            lambda v, s, e: _luhn(v) and not _is_nonpersonal_number_field(text, s),
        )
        add(_CVV, "CVV")
        add(_PIN, "PIN")
        if any(label in lower for label in ("рожд", "родил", "д.р", "д. р", "г.р", "г. р")):
            add(_BIRTH_DATE, "BIRTH_DATE", check=lambda v, s, e: _valid_date(v) and not _is_public_record(text, s))
            add(_AFTER_BIRTH, "BIRTH_DATE", check=lambda v, s, e: _valid_date(v) and not _is_public_record(text, s))
        if "выда" in lower:
            add(
                _PASS_DATE,
                "PASSPORT_DATE",
                check=lambda v, s, e: _valid_date(v) and _has_passport_context(text, s),
            )
            if "паспорт" in lower:
                add(
                    _PASS_DATE_LATE,
                    "PASSPORT_DATE",
                    check=lambda v, s, e: _valid_date(v) and _has_passport_context(text, s),
                )
        add(_POSTAL, "POSTAL_CODE", check=lambda _, s, e: not _is_public_address(text, s))
        add(
            _ADDRESS_POSTAL, "POSTAL_CODE", "address-format", 0.95,
            lambda _, s, e: not _is_public_address(text, s) and not _is_nonpersonal_number_field(text, s),
        )
        add(_HOUSE, "HOUSE", check=lambda _, s, e: not _is_public_address(text, s))
        add(_APARTMENT, "APARTMENT", check=lambda _, s, e: not _is_public_address(text, s))

    if any(label in lower for label in ("выдан", "орган", "мвд", "увд", "фмс")):
        add(_ISSUER, "PASSPORT_ISSUER")
        if "паспорт" in lower:
            add(
                _ISSUER_DIRECT, "PASSPORT_ISSUER", confidence=0.95, check=lambda v, s, e: _has_passport_context(text, s)
            )
    if "рождения" in lower or "родил" in lower:
        add(
            _BIRTH_PLACE,
            "BIRTH_PLACE",
            check=lambda v, s, e: (
                bool(re.search(r"[а-яё]", v, _FLAGS)) and not _DATE_FIND.fullmatch(v) and not _is_public_record(text, s)
            ),
        )
    if "граждан" in lower:
        add(_CITIZENSHIP, "CITIZENSHIP")
    candidates.extend(Span(*candidate) for candidate in cardholder_candidates(text, given_names=_NAME_FORMS))
    add(_STRONG_NAME_FIELD, "PERSON", "explicit-unicode-name-field")
    add(
        _NAME_TRIPLE,
        "PERSON",
        "patronymic-name",
        0.97,
        lambda v, s, e: _surname(v.split()[0]) and not _is_public_name(text, s, e),
    )
    add(
        _NAME_REVERSE,
        "PERSON",
        "patronymic-name",
        0.97,
        lambda v, s, e: _surname(v.split()[-1]) and not _is_public_name(text, s, e),
    )
    add(
        _CONTEXT_NAME,
        "PERSON",
        "personal-record-context",
        0.96,
        lambda v, s, e: (
            any(_given_name(w) or re.fullmatch(_PATR, w, _FLAGS) for w in v.split()) and not _is_public_name(text, s, e)
        ),
    )
    add(
        _INITIALS,
        "PERSON",
        "surname-initials",
        0.94,
        lambda v, s, e: bool(_SURNAME.search(v.split()[0])) and not _is_public_name(text, s, e),
    )
    # Sliding token pairs (lookahead) avoid losing a name after an ordinary word.
    tokens = list(re.finditer(r"[а-яёА-ЯЁ][а-яёА-ЯЁ'-]{1,39}", text))
    for a, b in zip(tokens, tokens[1:], strict=False):
        if not 0 < b.start() - a.end() <= 3 or not text[a.end() : b.start()].isspace():
            continue
        av, bv = a.group(), b.group()
        if (_given_name(av) and _surname(bv)) or (_surname(av) and _given_name(bv)):
            if not _is_public_name(text, a.start(), b.end()):
                candidates.append(Span(a.start(), b.end(), "PERSON", 0.92, "given-name-and-surname"))
    if any(label in lower for label in ("адрес", "прописан", "зарегистрирован", "прожива", "живёт", "живет")):
        for match in _ADDRESS.finditer(text):
            start, end = match.span("value")
            value = text[start:end]
            if not _ADDRESS_HINT.search(value) or _PUBLIC_ADDRESS.search(value[:70]) or _is_public_address(text, start):
                continue
            # End at the last structured address component, preserving subsequent
            # sentence punctuation and prose instead of greedily masking a paragraph.
            components = [m.end("value") for rule in (_CITY, _STREET, _HOUSE, _APARTMENT) for m in rule.finditer(value)]
            if components:
                end = start + max(components)
            while end > start and text[end - 1] in " \t,.!?":
                end -= 1
            candidates.append(Span(start, end, "ADDRESS", 0.98, "structured-address-context"))
    candidates.extend(_structured_addresses(text))
    add(_COUNTRY, "COUNTRY", check=lambda _, s, e: not _is_public_address(text, s))
    add(_CITY, "CITY", check=lambda _, s, e: not _is_public_address(text, s))
    add(_STREET, "STREET", check=lambda _, s, e: not _is_public_address(text, s))

    if extra_rules:
        if len(extra_rules) > 32:
            raise ValueError("At most 32 custom rules are supported")
        for rule in extra_rules:
            validate_extra_rule(rule)
            try:
                for match in _custom_pattern(rule["pattern"]).finditer(text, timeout=0.025):
                    size = match.end() - match.start()
                    if size == 0:
                        raise ValueError("Custom patterns must not match empty text")
                    if size > 4096:
                        raise ValueError("Custom detection rule matched more than 4096 characters")
                    candidates.append(Span(match.start(), match.end(), rule["type"], 1.0, "custom-rule"))
            except TimeoutError as exc:
                raise ValueError("Custom detection rule exceeded its time budget") from exc
    candidates.extend(Span(*candidate) for candidate in structured_candidates(text))
    candidates.extend(Span(*candidate) for candidate in location_candidates(text))
    candidates.extend(Span(*candidate) for candidate in person_candidates(
        text, given_names=_NAME_FORMS, public_name_guard=_is_public_name,
    ))
    return _resolve(refine_candidates(text, candidates))
