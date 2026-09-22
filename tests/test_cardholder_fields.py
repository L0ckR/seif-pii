"""Authored cardholder cases include names absent from the organizer corpus."""

import unicodedata

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
    "Name on card: Jean Dupont@example.com",
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



def test_strong_cardholder_label_does_not_require_a_colon_for_unknown_pair():
    """Policy change: a full cardholder label plus a closed value proves the field.

    This was an authored negative when the fallback required punctuation. The
    unknown name is now accepted without adding it to a given-name dictionary.
    """
    assert values("Держатель карты Рауль Мендес.") == ["Рауль Мендес"]


@pytest.mark.parametrize("name", ["Рашид Хамдани", "Haruto Yamazaki", "Zeynep Demir"])
@pytest.mark.parametrize("label", ["Держатель карты", "Имя держателя карты", "Эмбоссированное имя",
                                   "CARDHOLDER", "card holder", "name on card"])
@pytest.mark.parametrize("separator,ending", [(" ", ". Следующее поле."), (": ", "; next field"),
                                              ("=", "\n"), (" - ", "."), (" — ", "."),
                                              (" – ", "."), (" «", "»."), (' "', '".')])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.title])
def test_unknown_closed_cardholder_field_is_independent_of_dictionary_and_case(name, label, separator, ending,
                                                                              change_case):
    text = change_case(label + separator + name + ending)
    found = [text[start:end] for start, end, *_ in cardholder_candidates(text, given_names=frozenset())]
    assert found == [change_case(name)]


@pytest.mark.parametrize("text", [
    "Держатель Haruto Yamazaki.", "держателя Рашид Хамдани.",
    "Держатель карты Рашид Хамдани подтвердил данные.",
    "CARDHOLDER Haruto Yamazaki called yesterday.",
    "Держатель карты ожидает звонка.", "Держатель карты позвонил вчера.",
    "Cardholder called yesterday.", "Card holder returned today.",
    "Держатель карты: ООО Феникс.", "Name on card corporate account.",
    "Держатель карты отсутствует значение.", "Cardholder not specified.",
    "Name on card Haruto Yamazaki@example.com", "Name on card Haruto Yamazaki.example.com",
    "Name on card Haruto Yamazaki/path", "Name on card https://example.com",
])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.title])
def test_closed_value_extension_preserves_prose_business_and_network_boundaries(text, change_case):
    assert values(change_case(text)) == []


def test_unknown_name_offsets_preserve_both_closed_fields_and_intervening_text():
    text = "🔐 Держатель карты Рашид Хамдани; другое поле: значение. Name on card — Haruto Yamazaki."
    spans = list(cardholder_candidates(text, given_names=frozenset()))
    assert [(text[start:end], kind) for start, end, kind, *_ in spans] == [
        ("Рашид Хамдани", "CARDHOLDER"), ("Haruto Yamazaki", "CARDHOLDER")]
    assert spans[0][0] == text.index("Рашид")
    assert spans[1][0] == text.index("Haruto")


@pytest.mark.parametrize("value", ["неизвестной компании", "неизвестную организацию", "отсутствующего клиента"])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.title])
def test_missing_or_corporate_field_values_keep_their_inflected_meaning(value, change_case):
    assert values(change_case("Карта выпущена на имя " + value)) == []


@pytest.mark.parametrize("name", ["İlhan Durmaz", "Björk Guðmundsdóttir", "Léonie Noël"])
@pytest.mark.parametrize("normalization", ["NFC", "NFD"])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.title, str.swapcase])
def test_foreign_name_case_expansion_and_combining_accents_preserve_original_offsets(name, normalization, change_case):
    value = unicodedata.normalize(normalization, change_case(name))
    text = "🔐 Cardholder — " + value + ". Next field."
    spans = list(cardholder_candidates(text, given_names=frozenset()))
    assert [(start, end) for start, end, *_ in spans] == [(len("🔐 Cardholder — "), len("🔐 Cardholder — ") + len(value))]
    assert text[spans[0][0]:spans[0][1]] == value


@pytest.mark.parametrize("card_field", [
    "Карта 1111 2222 3333 4444, ", "Номер карты: 9876-5432-1098-7654, ",
    "CARD NUMBER=1111222233334444 ", "карта № 1111222233334, ", "card 1111222233334444555, ",
])
@pytest.mark.parametrize("name", ["Haruto Yamazaki", "Рашид Хамдани", "Zeynep Demir"])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.title])
def test_adjacent_explicit_card_number_establishes_generic_holder_field(card_field, name, change_case):
    text = change_case(card_field + "держатель " + name + ".")
    result = list(cardholder_candidates(text, given_names=frozenset()))
    assert [text[start:end] for start, end, *_ in result] == [change_case(name)]


@pytest.mark.parametrize("prefix", [
    "Карта, ", "номер карты неизвестен, ", "Карта 123456789012, ",
    "Карта 12345678901234567890, ", "Номер заказа: 1111222233334444, ",
    "Карта 1111222233334444. ", "Карта 1111222233334444; ",
    "Карта 1111222233334444,\n", "Карта 1111222233334444,\n\n",
    "Карта 1111222233334444, номер заказа 1234, ", "Карта 1111222233334444, сумма 40, ",
    "Номер счёта 1111222233334444, ", "Карта зарегистрирована для операции, ",
])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.title])
def test_generic_holder_does_not_inherit_card_context_across_records_or_fields(prefix, change_case):
    text = change_case(prefix + "держатель Haruto Yamazaki.")
    assert list(cardholder_candidates(text, given_names=frozenset())) == []


@pytest.mark.parametrize("value", ["отдел взыскания", "департамент обслуживания", "отделение банка",
                                    "управление взыскания", "служба поддержки"])
@pytest.mark.parametrize("change_case", [str.lower, str.upper, str.title])
def test_explicit_cardholder_field_does_not_turn_organization_units_into_names(value, change_case):
    assert values(change_case("Имя держателя карты: " + value + ".")) == []
