"""Authored positive and adversarial cases for personal-name field grammar."""

import pytest

from seif.detector import _NAMES, _is_public_name
from seif.person_fields import person_candidates


def detected(text):
    return {text[start:end] for start, end, *_ in person_candidates(
        text, given_names=_NAMES, public_name_guard=_is_public_name,
    )}


@pytest.mark.parametrize("label,value", [
    ("Плательщик перевода указан: ", "Сергеев"),
    ("Отправитель СБП: ", "Сергеев Сергей Сергеевич"),
    ("Получатель перевода: ", "Сергей"),
    ("Созаемщик по кредитному договору: ", "Сергеев Сергеевич"),
    ("Поручитель по договору: ", "Сергеев С."),
    ("Залогодатель квартиры: ", "Сергей"),
    ("Закладатель сейфовой ячейки: ", "Сергеев"),
    ("Бенефициар договора: ", "С. Сергей Сергеевич"),
    ("Доверенное лицо по вкладу: ", "Сергеев С."),
    ("Подателем заявки на кредит указано: ", "Сергей"),
    ("В графе «Отчество доверенного лица» указано: ", "Сергеевич"),
    ("Доверенность оформлена на: ", "Сергей Сергеевич С."),
    ("Имя: ", "Сергей"),
    ("Отчество: ", "Сергеевич"),
])
def test_personal_field_names(label, value):
    assert value in detected(label + value + ", документ приложен.")


@pytest.mark.parametrize("value", [
    "Сергеев С.", "сергеев с.", "СЕРГЕЕВ С.", "Сергеев С. С.", "Сергеев С.С.",
    "С. Сергеев", "С. С. Сергеев", "С.С. Сергеев", "С. Сергей Сергеевич",
    "Сергей Сергеевич С.", "сергей с.", "СЕРГЕЙ СЕРГЕЕВ СЕРГЕЕВИЧ",
    "СергеевСергейСергеевич", "сергеевсергейсергеевич", "СЕРГЕЕВСЕРГЕЙСЕРГЕЕВИЧ",
    "СергейСергеевичСергеев", "СергейСергеевСергеевич",
    "СЕРГЕЕВ СЕРГЕЕВИЧ",
])
def test_initials_permutations_and_joined_names(value):
    assert value in detected(value)


@pytest.mark.parametrize("text", [
    "Сергей", "Сергеев", "Сергеевич", "Плательщик: не указан.",
    "Получатель: организация ООО Ромашка.", "Отправитель: банк Альфа.",
    "Имя банка: Сергей.", "Имя организации: Сергеев.",
    "Гражданство поручителя по договору: Российская Федерация.",
    "Место рождения клиента: Российская Федерация.",
    "Получатель: Российский союз поставщиков.",
    "Клиент ожидается — имя банка: Сергей.",
    "Клиент ожидается - имя банка: Сергей.",
    "Клиент уточнил телефон: Сергеев.",
    "Получатель перевода неизвестен. Номер заявки: Сергеев.",
    "Плательщик обратился за помощью.", "Отчество отсутствует, значение: Сергеевич.",
    "В журнале указан г. Сергеев.", "Посылка по адресу: ул. С. Сергеева.",
    "Читали произведение писателя С. Сергеева.",
    "Памятник композитору С. Сергееву.",
    "Плательщик: провёл операцию успешно.", "Бенефициар: подтвердил согласие.",
    "Товары от производителей СергеевСергей.",
    "Передача телеграфного кода: А. Б. В.",
    "Начало предложения. Последующее предложение.",
])
def test_prose_public_context_and_unlabelled_single_words(text):
    assert detected(text) == set()


def test_value_stops_before_action_prose():
    text = "Получатель: Сергей Сергеевич подтвердил получение."
    assert detected(text) == {"Сергей Сергеевич"}


def test_offsets_are_unicode_codepoints_and_repeated_fields_are_distinct():
    text = "🔐 Плательщик: Сергей; получатель: Сергей."
    values = [(start, end) for start, end, *_ in person_candidates(
        text, given_names=_NAMES, public_name_guard=_is_public_name,
    )]
    assert values == [(14, 20), (34, 40)]


@pytest.mark.parametrize("value", [
    "Kuznetsova Elena", "ELENA KUZNETSOVA", "Lebedev Sergei", "Sergey Lebedev",
    "Morozov Alexey", "Morozov Aleksei", "Yuliya Volkova", "Iuliia Volkova",
    "O’Connor John", "John O'Connor", "MacLeod George", "George McDonald",
    "Bunin Ivan", "Nikolai Ostrovskii", "Ostrovskiy Nikolay",
])
def test_transliterated_and_prefix_surnames_use_name_grammar(value):
    assert value in detected(value)


@pytest.mark.parametrize("value", ["kuznetsova elena", "o'connor john", "lebedev sergei"])
def test_lowercase_latin_names_use_the_same_name_grammar(value):
    assert value in detected(value)
    assert value in detected("ФИО: " + value + ".")


@pytest.mark.parametrize("text", [
    "Anna Login", "John Admin", "Ivan Plugin", "Anna Skin", "George Origin",
    "Anna Company", "O'Reilly Media", "MacDonald Restaurant", "Petrov Confirmed",
    "John Paul", "John", "George O'", "Annalogin", "Anna\nIvanova",
    "Читали произведение писателя John O'Connor.", "улица John O'Connor.",
])
def test_latin_prose_and_public_references_are_not_names(text):
    assert detected(text) == set()


def test_latin_offsets_repetitions_and_narrative_boundaries():
    text = "🔐 Kuznetsova Elena подтвердила операцию; получатель: KUZNETSOVA ELENA."
    spans = list(person_candidates(text, given_names=_NAMES, public_name_guard=_is_public_name))
    assert {(start, end) for start, end, *_ in spans} == {(2, 18), (53, 69)}


@pytest.mark.parametrize("value", ["Lebedev Sergei", "O’Connor John", "Elena Kuznetsova"])
@pytest.mark.parametrize("change_case", [str.lower, str.title, str.upper, str.swapcase])
def test_latin_person_case_metamorphism(value, change_case):
    transformed = change_case(value)
    assert detected("🔐 " + transformed + " подтвердил операцию.") == {transformed}


@pytest.mark.parametrize("text", [
    "Anna Login", "John Admin", "Anna Skin", "George Origin", "John Paul",
    "Please Sign In", "Client Has Logged In", "anna.ivanova@example.com",
    "sergei lebedev@example.com", "contact@sergei lebedev", "login=elena_kuznetsova",
])
@pytest.mark.parametrize("change_case", [str.lower, str.title, str.upper])
def test_latin_negative_case_metamorphism(text, change_case):
    assert detected(change_case(text)) == set()
