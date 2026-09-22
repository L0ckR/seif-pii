"""Context refinements preserve private values and expose field/prose words."""

import pytest

from seif.context_filters import refine_candidates
from seif.detector import Span
from seif.transform import mask, restore_exact


def candidate(text, value, kind, *, reason="test"):
    start = text.index(value)
    return Span(start, start + len(value), kind, 0.9, reason)


def protected(text, spans):
    return {i for span in spans for i in range(span.start, span.end) if text[i].isalnum()}


@pytest.mark.parametrize("kind", ["PASSPORT", "DRIVER_LICENSE"])
@pytest.mark.parametrize("separator", [" номер ", ", НОМЕР: ", "\tномер № ", ", № ", " ном. ", "№"])
def test_document_parts_preserved_without_masking_the_field_label(kind, separator):
    text = "🔐 серия 12 34" + separator + "567890"
    span = candidate(text, "12 34" + separator + "567890", kind)
    refined = refine_candidates(text, [span])
    assert [text[s.start:s.end] for s in refined] == ["12 34", "567890"]
    assert {s.type for s in refined} == {kind}
    assert {s.reason for s in refined} == {"test"}
    masked, replacements = mask(text, refined, "mask")
    assert separator in masked
    assert restore_exact({"masked": masked, "replacements": replacements}) == text


@pytest.mark.parametrize("value", ["12 34 567890", "AB номер 123456", "1234 номер документ", "1234 123456"])
def test_document_split_needs_two_numeric_components(value):
    span = candidate(value, value, "PASSPORT")
    assert refine_candidates(value, [span]) == [span]


@pytest.mark.parametrize(
    ("kind", "value", "kept"),
    [
        ("PERSON", "Иван Петров обратился", "Иван Петров"),
        ("PERSON", "АННА ЗАПРОСИЛА", "АННА"),
        ("CARDHOLDER", "IVAN PETROV подтвердил", "IVAN PETROV"),
        ("PERSON", "Петров И. Назначение", "Петров И"),
        ("PERSON", "И. И. КОД НАЗНАЧЕНИЯ ПЛАТЕЖА", "И. И"),
    ],
)
def test_name_prose_suffixes_do_not_replace_the_person(kind, value, kept):
    text = "🔐 " + value + ": запись."
    refined = refine_candidates(text, [candidate(text, value, kind)])
    assert len(refined) == 1
    assert text[refined[0].start:refined[0].end] == kept


@pytest.mark.parametrize("value", ["не указано", "не известна", "отсутствует", "ввёл пин-код"])
def test_missing_cardholder_or_action_is_not_a_name(value):
    text = "Держатель карты " + value
    assert refine_candidates(text, [candidate(text, value, "CARDHOLDER")]) == []


@pytest.mark.parametrize("value", ["Иван Иванович Петров", "Mary Anne Smith", "Не", "Подписал", "Клиент"])
def test_person_name_values_are_not_globally_blacklisted(value):
    text = "ФИО: " + value
    span = candidate(text, value, "PERSON")
    assert refine_candidates(text, [span]) == [span]


@pytest.mark.parametrize(
    ("kind", "value", "kept"),
    [
        ("ADDRESS", "по адресу: г. Омск, ул. Мира, д. 8", "г. Омск, ул. Мира, д. 8"),
        ("ADDRESS", "фактического проживания: г. Омск, ул. Мира, д. 8", "г. Омск, ул. Мира, д. 8"),
        ("ADDRESS", "(прописка): г. Омск, ул. Мира, д. 8", "г. Омск, ул. Мира, д. 8"),
        ("ADDRESS", "а регистрации за 5 лет: 1) г. Омск, ул. Мира, д. 8", "г. Омск, ул. Мира, д. 8"),
        ("BIRTH_PLACE", "» указано: г. Омск", "г. Омск"),
        ("BIRTH_PLACE", "бенефициара по договору: г. Омск", "г. Омск"),
        ("BIRTH_PLACE", "г. Омск. АНКЕТА ПРОВЕРЕНА БАНКОМ", "г. Омск"),
        ("BIRTH_PLACE", "г. Омск подтверждено паспортом", "г. Омск"),
    ],
)
def test_place_boundaries_expose_scaffolding_and_trailing_narrative(kind, value, kept):
    text = "🧭 " + value
    refined = refine_candidates(text, [candidate(text, value, kind)])
    assert len(refined) == 1
    assert text[refined[0].start:refined[0].end] == kept


@pytest.mark.parametrize("join", [". Адрес фактического проживания: ", ", продавец — по адресу "])
def test_multiple_addresses_preserve_both_values(join):
    first, second = "г. Омск, ул. Мира, д. 8", "г. Псков, ул. Полевая, д. 3"
    text = first + join + second
    refined = refine_candidates(text, [candidate(text, text, "ADDRESS")])
    assert [text[s.start:s.end] for s in refined] == [first, second]
    masked, replacements = mask(text, refined, "mask")
    assert join in masked
    assert restore_exact({"masked": masked, "replacements": replacements}) == text


@pytest.mark.parametrize("owner", ["Организация ООО «Макет»", "компания", "Реквизиты поставщика", "контрагент"])
def test_explicit_corporate_ten_digit_inn_is_public(owner):
    text = owner + ": ИНН 1234567890"
    assert refine_candidates(text, [candidate(text, "1234567890", "INN")]) == []


@pytest.mark.parametrize(
    ("text", "value", "reason"),
    [
        ("Компания заключила договор; клиент ИНН 1234567890", "1234567890", "test"),
        ("ООО: ИНН клиента 1234567890", "1234567890", "test"),
        ("ООО: ИНН 123456789012", "123456789012", "test"),
        ("ООО: ИНН 1234567890", "1234567890", "cis-personal-id"),
        ("ООО: номер заказа 1234567890", "1234567890", "test"),
        ("1234567890", "1234567890", "test"),
    ],
)
def test_personal_regional_or_unowned_identifiers_are_retained(text, value, reason):
    span = candidate(text, value, "INN", reason=reason)
    assert refine_candidates(text, [span]) == [span]


@pytest.mark.parametrize("kind", ["CITY", "LOCATION"])
def test_standalone_office_geography_is_not_a_personal_address(kind):
    text = "Отдел внутренних дел г. Омска"
    assert refine_candidates(text, [candidate(text, "Омска", kind)]) == []


@pytest.mark.parametrize("prefix", ["Паспорт выдан ", "Орган выдачи: ", "Выдала ", "Клиент проживает возле "])
def test_personal_document_or_residence_context_preserves_office_geography(prefix):
    text = prefix + "ОВД г. Омска"
    span = candidate(text, "Омска", "LOCATION")
    assert refine_candidates(text, [span]) == [span]


def test_custom_rules_are_authoritative_and_candidate_input_is_unchanged():
    text = "Держатель карты не указано"
    spans = [candidate(text, "не указано", "CARDHOLDER", reason="custom-rule")]
    before = list(spans)
    assert refine_candidates(text, spans) == before
    assert spans == before


def test_refinement_is_idempotent_with_unicode_and_multiple_candidates():
    text = "🔐 Клиент Иван Петров обратился; паспорт 12 34 номер 567890; адрес по адресу: г. Омск, ул. Мира, д. 8"
    spans = [candidate(text, "Иван Петров обратился", "PERSON"),
             candidate(text, "12 34 номер 567890", "PASSPORT"),
             candidate(text, "по адресу: г. Омск, ул. Мира, д. 8", "ADDRESS")]
    refined = refine_candidates(text, spans)
    assert refine_candidates(text, refined) == refined
    assert all(0 <= span.start < span.end <= len(text) for span in refined)
    assert protected(text, refined) <= protected(text, spans)
