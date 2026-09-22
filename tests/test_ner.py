"""Candidate-merge safety tests independent of model outputs or comparison data."""

from __future__ import annotations

import math

import pytest

from seif.detector import Span, detect, merge_person_candidates


def person(text, value, confidence=0.85, reason="external-source"):
    start = text.index(value)
    return Span(start, start + len(value), "PERSON", confidence, reason)


def test_merge_preserves_unicode_offsets_and_sanitizes_upstream_reason():
    text = "🔐 Дина Штольц, email: test@example.net"
    candidate = person(text, "Дина Штольц", reason="upstream-explanation-with-sensitive-values")
    result = merge_person_candidates(text, detect(text), [candidate])
    assert [(span.type, text[span.start : span.end]) for span in result] == [
        ("PERSON", "Дина Штольц"),
        ("EMAIL", "test@example.net"),
    ]
    assert result[0].start == 2
    assert result[0].reason == "ner-person"


def test_structured_field_class_keeps_priority_over_ner_person():
    text = "Держатель карты: NIKOLAY VOLK; паспорт 4509 123456"
    base = detect(text)
    candidates = [person(text, "NIKOLAY VOLK"), person(text, "4509 123456")]
    assert merge_person_candidates(text, base, candidates) == base


def test_longer_ner_name_extends_a_short_context_fallback():
    text = "Клиент: Дина Марковна Штольц."
    short = person(text, "Дина Марковна", 0.96, "personal-record-context")
    complete = person(text, "Дина Марковна Штольц")
    assert merge_person_candidates(text, [short], [complete]) == [
        Span(complete.start, complete.end, "PERSON", 0.85, "ner-person")
    ]


def test_short_ner_name_cannot_shrink_a_full_core_name():
    text = "Клиент: Дина Марковна Штольц."
    complete = person(text, "Дина Марковна Штольц", 0.96, "personal-record-context")
    assert merge_person_candidates(text, [complete], [person(text, "Дина Марковна")]) == [complete]


def test_equal_boundaries_keep_core_provenance_and_empty_candidates_are_stable():
    text = "Клиент Дина Штольц."
    core = person(text, "Дина Штольц", 0.96, "personal-record-context")
    assert merge_person_candidates(text, [core], [person(text, "Дина Штольц", 1.0)]) == [core]
    assert merge_person_candidates(text, [core], []) == [core]


def test_duplicate_ner_candidates_merge_deterministically():
    text = "Дина Штольц"
    candidates = [person(text, text, 0.6), person(text, text, 0.9)]
    a = merge_person_candidates(text, [], candidates)
    b = merge_person_candidates(text, [], list(reversed(candidates)))
    assert a == b == [Span(0, len(text), "PERSON", 0.9, "ner-person")]


@pytest.mark.parametrize(
    "text,value,public",
    [
        ("Расскажи о поэте Василии Жуковском.", "Василии Жуковском", True),
        ("Произведения писателя Николая Лескова.", "Николая Лескова", True),
        ("Антон Чехов — русский писатель.", "Антон Чехов", True),
        ("Клиент Антон Чехов — русский писатель.", "Антон Чехов", False),
        ("Клиент Василий Жуковский подписал заявление.", "Василий Жуковский", False),
        ("Поэт выступил на сцене. Дина Штольц подписала заявление.", "Дина Штольц", False),
        ("Читали роман, затем Дина Штольц пришла на встречу.", "Дина Штольц", False),
        ("Изучили творчество писателя, затем Дина Штольц подписала документ.", "Дина Штольц", False),
        ("Дина Штольц спросила — поэт ли её собеседник.", "Дина Штольц", False),
        ("Роман Штольц подписал заявление.", "Штольц", False),
    ],
)
def test_name_context_is_local_and_has_no_new_author_allowlist(text, value, public):
    candidate = person(text, value)
    result = merge_person_candidates(text, [], [candidate])
    assert (result == []) is public
    if not public:
        assert [(span.start, span.end) for span in result] == [(candidate.start, candidate.end)]


@pytest.mark.parametrize(
    "candidate",
    [
        Span(-1, 2, "PERSON"),
        Span(1, 1, "PERSON"),
        Span(2, 1, "PERSON"),
        Span(0, 500, "PERSON"),
        Span(0.0, 2, "PERSON"),
        Span(True, 2, "PERSON"),
        Span(0, False, "PERSON"),
        Span(0, 2, "EMAIL"),
        Span(0, 2, "person"),
        Span(0, 2, "PERSON", math.nan),
        Span(0, 2, "PERSON", math.inf),
        Span(0, 2, "PERSON", -0.01),
        Span(0, 2, "PERSON", 1.01),
        Span(0, 2, "PERSON", True),
        Span(0, 2, "PERSON", "0.9"),
        {"start": 0, "end": 2, "type": "PERSON"},
    ],
)
def test_malformed_candidate_rejects_entire_merge(candidate):
    with pytest.raises(ValueError):
        merge_person_candidates("Дина Штольц", [], [person("Дина Штольц", "Дина Штольц"), candidate])


def test_long_names_are_supported_but_oversized_model_spans_are_rejected():
    name = "Анна-Мария-Екатерина " + "Александровна-Владимировна " + "Штольц-Витгенштейн-Берген"
    assert len(name) > 70
    assert merge_person_candidates(name, [], [person(name, name)])
    with pytest.raises(ValueError, match="200"):
        merge_person_candidates("я" * 201, [], [Span(0, 201, "PERSON")])


def test_candidate_count_is_bounded_without_silent_truncation():
    with pytest.raises(ValueError, match="Too many"):
        merge_person_candidates("я", [], [Span(0, 1, "PERSON")] * 1025)


def test_public_exemption_does_not_hide_invalid_response():
    text = "Поэт Антон Чехов."
    with pytest.raises(ValueError):
        merge_person_candidates(text, [], [person(text, "Антон Чехов"), Span(0, len(text), "PERSON", math.nan)])
