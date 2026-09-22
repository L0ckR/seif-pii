"""Post-aggregate development regressions for optional PERSON+LOCATION merging.

These invented examples are independent of external benchmark text/gold. No
geographical lookup list or model predictions are used by the tests or merger.
"""

import math

import pytest

from seif.detector import TYPES, Span, detect, merge_ner_candidates, merge_person_candidates


def candidate(text, value, kind="LOCATION", confidence=0.85, reason="external"):
    start = text.index(value)
    return Span(start, start + len(value), kind, confidence, reason)


def values(text, spans):
    return [(span.type, text[span.start : span.end]) for span in spans]


def test_locations_keep_their_own_type_and_original_unicode_offsets():
    text = "🔐 Отправить посылку в Ақтау; Дина Штольц ждёт."
    spans = [candidate(text, "Ақтау"), candidate(text, "Дина Штольц", "PERSON")]
    result = merge_ner_candidates(text, [], spans)
    assert values(text, result) == [("LOCATION", "Ақтау"), ("PERSON", "Дина Штольц")]
    assert result[0].start == text.index("Ақтау")
    assert [span.reason for span in result] == ["ner-location", "ner-person"]
    assert TYPES["LOCATION"] == "Географическое название (NER)"


@pytest.mark.parametrize(
    "kind",
    ["ADDRESS", "CITY", "COUNTRY", "POSTAL_CODE", "STREET", "HOUSE", "APARTMENT",
     "BIRTH_PLACE", "PASSPORT_ISSUER", "PERSON", "EMAIL", "PHONE"],
)
def test_every_specific_core_class_beats_a_wider_location(kind):
    text = "До значения после"
    base = [candidate(text, "значения", kind, 0.6, "core")]
    assert merge_ner_candidates(text, base, [candidate(text, text, confidence=1.0)]) == base


def test_existing_complete_address_is_not_relabelled_as_location():
    text = "г. Тверь, ул. Тихая, д. 14"
    base = detect(text)
    assert merge_ner_candidates(text, base, [candidate(text, "Тверь"), candidate(text, "Тихая")]) == base
    assert values(text, base) == [("ADDRESS", text)]


@pytest.mark.parametrize(
    "text,value,excluded",
    [
        ("Адрес банка: Самара.", "Самара", True),
        ("Офис банка: Самара.", "Самара", True),
        ("Филиал банка: Самара.", "Самара", True),
        ("Адрес банка: г. Самара, ул. Тихая, д. 4.", "Самара", True),
        ("Адрес банка: г. Самара, ул. Тихая, д. 4.", "Тихая", True),
        ("Адрес клиента: Самара.", "Самара", False),
        ("Адрес клиента банка: Самара.", "Самара", False),
        ("Адрес банка: Самара; адрес клиента: Тверь.", "Тверь", False),
        ("Адрес банка: Самара. Посылка уехала в Тверь.", "Тверь", False),
        ("Обсуждали офис банка. Доставка в Тверь.", "Тверь", False),
        ("Доставка в населённый пункт без справочника: Неизвестноград.", "Неизвестноград", False),
    ],
)
def test_bank_policy_is_local_and_does_not_turn_into_a_city_allowlist(text, value, excluded):
    result = merge_ner_candidates(text, [], [candidate(text, value)])
    assert (result == []) is excluded
    if not excluded:
        assert values(text, result) == [("LOCATION", value)]


def test_public_and_private_locations_in_one_message_remain_separate():
    text = "Адрес банка: Самара; адрес клиента: Тверь."
    result = merge_ner_candidates(text, [], [candidate(text, "Самара"), candidate(text, "Тверь")])
    assert values(text, result) == [("LOCATION", "Тверь")]


def test_person_filter_and_historical_namesake_policy_are_unchanged():
    text = "Поэт Антон Чехов. Клиент Антон Чехов живёт в Твери."
    public = candidate(text, "Антон Чехов", "PERSON")
    private_start = text.rindex("Антон Чехов")
    private = Span(private_start, private_start + len("Антон Чехов"), "PERSON", 0.85)
    result = merge_ner_candidates(text, [], [public, private, candidate(text, "Твери")])
    assert values(text, result) == [("PERSON", "Антон Чехов"), ("LOCATION", "Твери")]


def test_person_only_entrypoint_preserves_prior_contract():
    text = "Дина Штольц из Твери"
    person = candidate(text, "Дина Штольц", "PERSON")
    assert merge_person_candidates(text, [], [person]) == merge_ner_candidates(text, [], [person])
    candidates = [person, candidate(text, "Твери")]
    with pytest.raises(ValueError):
        merge_person_candidates(text, [], candidates)


@pytest.mark.parametrize(
    "span",
    [Span(-1, 2, "LOCATION"), Span(0, 999, "LOCATION"), Span(0, 0, "LOCATION"),
     Span(True, 2, "LOCATION"), Span(0, 2, "LOCATION", math.nan),
     Span(0, 2, "LOCATION", math.inf), Span(0, 2, "LOCATION", -0.1),
     Span(0, 2, "LOCATION", 1.1), Span(0, 2, "LOCATION", True),
     Span(0, 2, "CITY"), Span(0, 2, "LOC"), Span(0, 2, "ORG")],
)
def test_bad_external_location_rejects_the_whole_response(span):
    text = "Адрес банка: Самара."
    candidates = [candidate(text, "Самара"), span]
    with pytest.raises(ValueError):
        merge_ner_candidates(text, [], candidates)


def test_oversized_locations_and_excessive_candidate_counts_fail_closed():
    span = Span(0, 201, "LOCATION")
    with pytest.raises(ValueError, match="200"):
        merge_ner_candidates("А" * 201, [], [span])
    spans = [Span(0, 1, "LOCATION")] * 1025
    with pytest.raises(ValueError, match="Too many"):
        merge_ner_candidates("А", [], spans)


def test_duplicate_scores_and_core_provenance_are_deterministic():
    text = "Самара"
    candidates = [candidate(text, text, confidence=0.5), candidate(text, text, confidence=0.9)]
    result = merge_ner_candidates(text, [], candidates)
    assert result == merge_ner_candidates(text, [], list(reversed(candidates)))
    assert result == [Span(0, len(text), "LOCATION", 0.9, "ner-location")]
    core = [candidate(text, text, confidence=0.6, reason="core-location")]
    assert merge_ner_candidates(text, core, candidates) == core


def test_person_wins_over_location_on_the_same_boundaries_independent_of_order():
    text = "Дина Штольц"
    candidates = [candidate(text, text), candidate(text, text, "PERSON")]
    result = merge_ner_candidates(text, [], candidates)
    assert result == merge_ner_candidates(text, [], list(reversed(candidates)))
    assert values(text, result) == [("PERSON", text)]
