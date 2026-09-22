"""Authored cardholder cases include names absent from the organizer corpus."""

import pytest

from seif.cardholder_fields import cardholder_candidates
from seif.detector import _NAMES


def values(text):
    return [text[start:end] for start, end, *_ in cardholder_candidates(text, given_names=_NAMES)]


@pytest.mark.parametrize("label", [
    "Держатель карты ", "держателя ", "Имя держателя карты: ", "эмбоссированное имя «",
    "CARDHOLDER=", "card holder — ", "name on card: ", "Карта выпущена на имя ",
    "Карта оформлена на имя ",
])
@pytest.mark.parametrize("name", [
    "KUZNETSOVA ELENA", "Kuznetsova Elena", "kuznetsova elena", "ZHANG WEI", "Mary Anne Smith",
    "Сергеев Сергей Сергеевич", "сергеев сергей сергеевич", "LEBEDEV SERGEI IVANOVICH", "Қасымов Әділ", "O'Connor John",
])
def test_explicit_names(label, name):
    assert values(label + name + "». Дата операции 30.06.2026.") == [name]


@pytest.mark.parametrize("text", [
    "Держатель карты не указан.", "Держатель карты подтвердил получение.",
    "Имя держателя карты неизвестно.", "Имя держателя: банк Альфа.",
    "Cardholder Not Specified", "Cardholder UNKNOWN UNKNOWN", "Cardholder Corporate Account",
    "Держатель пластиковой карты обновлён.", "Держатель карты подтвердил ОПЕРАЦИЮ.",
    "Name On Card", "Эмбоссированное имя отсутствует", "Карта выпущена на имя банка Альфа",
    "Карта выпущена на имя неизвестной компании", "Карта выпущена на имя ООО Ромашка",
])
def test_absent_values_prose_and_organizations(text):
    assert values(text) == []


def test_mixed_case_actions_do_not_extend_cardholder_span():
    assert values("Держатель карты KUZNETSOVA ELENA подтвердила операцию") == ["KUZNETSOVA ELENA"]
    assert values("Cardholder KUZNETSOVA ELENA CONFIRMED the operation") == ["KUZNETSOVA ELENA"]


def test_each_repeated_field_has_its_own_original_offset():
    text = "🔐 Держатель карты LEBEDEV SERGEI подтвердил; карта выпущена на имя LEBEDEV SERGEI."
    spans = list(cardholder_candidates(text, given_names=_NAMES))
    assert [text[start:end] for start, end, *_ in spans] == ["LEBEDEV SERGEI", "LEBEDEV SERGEI"]
    assert len({(start, end) for start, end, *_ in spans}) == 2
    assert all(kind == "CARDHOLDER" for _, _, kind, *_ in spans)


def test_generic_name_mentions_do_not_create_cardholders():
    assert values("Имя: LEBEDEV SERGEI. Позже выпущена карта.") == []


@pytest.mark.parametrize("action", ["COMPLETED", "AUTHORIZED", "SAID", "ВОЗРАЗИЛ", "ОДОБРИЛ"])
def test_uppercase_narrative_after_complete_name_stays_visible(action):
    assert values("Держатель карты LEBEDEV SERGEI " + action) == ["LEBEDEV SERGEI"]
    assert values("Держатель карты Сергеев Сергей " + action) == ["Сергеев Сергей"]


@pytest.mark.parametrize("name", [
    "Zhang Wei", "Чжан Мин", "Цзян Хуа", "Mary Anne Smith", "Қасымов Әділ", "Ali Hassan",
    "LEBEDEV SERGEI IVANOVICH",
])
@pytest.mark.parametrize("change_case", [str.lower, str.title, str.upper, str.swapcase])
def test_regional_foreign_fields_have_case_invariant_predictions(name, change_case):
    transformed = change_case(name)
    assert values(change_case("Name on card: ") + transformed + ".") == [transformed]


@pytest.mark.parametrize("value", [
    "Was Seen", "Has Confirmed", "Please Sign In", "Login Admin", "Successful Operation",
    "Payment Approved", "Card Not Present", "Mary Smith@example.com", "anna.ivanova@example.com",
    "Has Valid Account", "Unknown Person", "Name Not Available", "Entered The Room",
])
@pytest.mark.parametrize("change_case", [str.lower, str.title, str.upper])
def test_foreign_prose_and_email_case_metamorphism(value, change_case):
    assert values(change_case("Cardholder: " + value)) == []


@pytest.mark.parametrize("name", ["Чжан Мин", "Zhang Wei", "Қасымов Әділ", "Ali Hassan"])
@pytest.mark.parametrize("change_case", [str.lower, str.title, str.upper])
def test_foreign_name_does_not_consume_unspecified_action_prose(name, change_case):
    assert values(change_case("Держатель карты " + name + " успешно выполнил перевод")) == [change_case(name)]


@pytest.mark.parametrize("name", ["Рауль Мендес", "Диего Алонсо", "Кемаль Демир", "Jean Dupont"])
@pytest.mark.parametrize("change_case", [str.lower, str.title, str.upper, str.swapcase])
@pytest.mark.parametrize("label,end", [("Держатель карты: ", ". Заявление принято."),
                                      ("Name on card=", "; payment processed"),
                                      ("Эмбоссированное имя «", "». Анкета проверена.")])
def test_unknown_foreign_pair_needs_closed_form_field(name, change_case, label, end):
    assert values(change_case(label + name + end)) == [change_case(name)]


@pytest.mark.parametrize("text", [
    "Держатель карты: Рауль Мендес подтвердил получение.",
    "Держатель карты Рауль Мендес.", "Name on card: Jean Dupont@example.com",
    "Name on card: Jean Dupont.example.com", "Name on card: Jean Dupont/path",
    "Name on card: Successful Operation", "Name on card: Payment Approved",
    "Держатель карты: Успешная операция.", "Держатель карты: Доступ запрещён.",
    "Держатель карты: Новый пользователь.", "Держатель карты: Неизвестная персона.",
    "Держатель карты: Другая сторона.", "Держатель карты: Текст отсутствует.",
    "Держатель карты: Компания Альфа.", "Держатель карты: Юридическое лицо.",
])
@pytest.mark.parametrize("change_case", [str.lower, str.title, str.upper])
def test_closed_field_does_not_validate_action_business_or_partial_values(text, change_case):
    assert values(change_case(text)) == []
