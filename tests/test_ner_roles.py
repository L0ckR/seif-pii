"""Focused, self-authored regression cases for NER client-role boundary errors.

These tests use hand-supplied NER candidates, not organizer labels or predictions.
They exercise independent boundary handling; no upstream model code is copied.
"""

from __future__ import annotations

import math

import pytest

from seif.detector import Span, detect, merge_ner_candidates, merge_person_candidates


def person(text, value, reason="external-source", confidence=0.85):
    start = text.index(value)
    return Span(start, start + len(value), "PERSON", confidence, reason)


@pytest.fixture(params=[merge_person_candidates, merge_ner_candidates])
def merge(request):
    return request.param


@pytest.mark.parametrize("label", ["Клиент", "клиент", "КЛИЕНТ", "Клиентка", "Клиента"])
def test_standalone_client_field_label_is_not_a_person(merge, label):
    text = f"{label}: Дина Штольц."
    candidate = person(text, label)
    assert merge(text, [], [candidate]) == []
    core = detect(text)
    assert merge(text, core, [candidate]) == core


@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("Клиент: Дина Штольц.", "Клиент: Дина Штольц", "Дина Штольц"),
        ("Обратился клиент Дина Штольц.", "клиент Дина Штольц", "Дина Штольц"),
        ("Клиентка Дина.", "Клиентка Дина", "Дина"),
        ("Клиента Дины Штольц выслушали.", "Клиента Дины Штольц", "Дины Штольц"),
        ("🔐 Клиент: Нұрсұлтан Әбішұлы.", "Клиент: Нұрсұлтан Әбішұлы", "Нұрсұлтан Әбішұлы"),
        ("Клиент = Aoife O’Neill.", "Клиент = Aoife O’Neill", "Aoife O’Neill"),
        ("Клиент: Клиент: Дина Штольц.", "Клиент: Клиент: Дина Штольц", "Дина Штольц"),
        ("Клиент: Клиент Иванович.", "Клиент: Клиент Иванович", "Клиент Иванович"),
    ],
)
def test_trim_only_leading_role_and_keep_complete_model_name(merge, text, value, expected):
    candidate = person(text, value)
    result = merge(text, [], [candidate])
    assert result == [person(text, expected, reason="ner-person")]
    assert text[result[0].start : result[0].end] == expected


@pytest.mark.parametrize(
    "text,value",
    [
        ("Фамилия: Клиент", "Клиент"),
        ("Фамилия: Клиент Иванович", "Клиент Иванович"),
        ("ФИО: Клиент Дина Марковна", "Клиент Дина Марковна"),
        ("Ф.И.О.: Клиент Дина Марковна", "Клиент Дина Марковна"),
        ("Ф. И. О.: Клиент Дина Марковна", "Клиент Дина Марковна"),
        ("ФИО: Дина Клиент Марковна", "Клиент Марковна"),
        ("Фамилия и имя: Клиент Дина", "Клиент Дина"),
        ("Клиент: ФИО: Клиент Дина Марковна", "Клиент Дина Марковна"),
        ("Имя: Клиент", "Клиент"),
        ("Дина Клиент подписала договор.", "Дина Клиент"),
        ("Дина Клиент Марковна", "Дина Клиент Марковна"),
        ("Клиент-Штольц Дина", "Клиент-Штольц Дина"),
        ("Клиентова Дина", "Клиентова Дина"),
        ("Клиент’Штольц Дина", "Клиент’Штольц Дина"),
        ("Клиент", "Клиент"),
        ("Клиент пришёл.", "Клиент"),
    ],
)
def test_role_word_is_not_a_global_blacklist_of_names(merge, text, value):
    candidate = person(text, value)
    assert merge(text, [], [candidate]) == [person(text, value, reason="ner-person")]


@pytest.mark.parametrize(
    "prefix",
    [
        "ФИО: «",
        "ФИО: ‹",
        "ФИО: “",
        "ФИО: „",
        'ФИО: "',
        "ФИО: '",
        "ФИО: ‘",
        "ФИО: ‚",
        "ФИО:\n  ",
        "ФИО:\r\n\t",
        "ФИО:\n  «",
        "ФИО:\u2028  “",
        "Фамилия\u00a0—\u00a0",
        "Фамилия – ",
        "Фамилия - ",
        "ФИО\u00a0:\u00a0",
        "ФИО\u2009=\u202f",
        "ФИО: «Дина\u00a0",
        "Ф. И. О.:\n «",
    ],
)
def test_quoted_or_wrapped_explicit_name_keeps_role_like_surname(merge, prefix):
    value = "Клиент Дина Марковна"
    text = prefix + value
    candidate = person(text, value)
    assert merge(text, [], [candidate]) == [person(text, value, reason="ner-person")]


@pytest.mark.parametrize(
    "prefix",
    [
        "ФИО: «Дина Штольц». ",
        "ФИО: «Дина Штольц»\n",
        "ФИО: Дина Штольц\n",
        "ФИО: Дина Штольц. ",
        "ФИО:\n\n",
        "Фамилия — Штольц; ",
        "Фамилия — Штольц. ",
        "ФИО: отсутствует. ",
    ],
)
def test_explicit_name_guard_does_not_cross_a_finished_field_or_sentence(merge, prefix):
    text = prefix + "Клиент Анна Петрова"
    assert merge(text, [], [person(text, "Клиент Анна Петрова")]) == [
        person(text, "Анна Петрова", reason="ner-person")
    ]


def test_trimmed_identity_keeps_core_provenance(merge):
    text = "Клиент: Иванов Иван Иванович"
    core = detect(text)
    assert len(core) == 1
    assert merge(text, core, [person(text, text)]) == core


def test_extension_preserves_rare_name_beyond_short_core_fallback(merge):
    text = "Клиент: Дина Марковна Штольц"
    short = person(text, "Дина Марковна", reason="personal-record-context", confidence=0.96)
    complete = person(text, text)
    assert merge(text, [short], [complete]) == [person(text, "Дина Марковна Штольц", reason="ner-person")]


def test_base_person_value_prevents_shrinking_even_with_overlapping_base_spans(merge):
    text = "Дина Клиент Марковна"
    full = person(text, text, reason="explicit-unicode-name-field", confidence=0.99)
    short = person(text, "Дина", reason="context", confidence=0.96)
    external = person(text, "Клиент Марковна")
    assert merge(text, [short, full], [external]) == [full]


def test_repeated_records_and_emoji_preserve_original_unicode_offsets(merge):
    text = "🙂 Клиент: Дина Штольц\nКлиент: Айгүл Әлиева"
    candidates = [
        person(text, "Клиент: Дина Штольц"),
        person(text, "Клиент: Айгүл Әлиева"),
    ]
    result = merge(text, [], list(reversed(candidates)))
    assert result == [
        person(text, "Дина Штольц", reason="ner-person"),
        person(text, "Айгүл Әлиева", reason="ner-person"),
    ]


def test_role_filter_never_hides_a_malformed_response(merge):
    text = "Клиент: Дина Штольц"
    candidates = [person(text, "Клиент"), person(text, "Клиент", confidence=math.nan)]
    with pytest.raises(ValueError):
        merge(text, [], candidates)


def test_location_candidates_are_unchanged():
    text = "Клиент:"
    location = Span(0, 6, "LOCATION", 0.85)
    assert merge_ner_candidates(text, [], [location]) == [Span(0, 6, "LOCATION", 0.85, "ner-location")]


def test_public_name_exemption_and_private_namesake_policy_stay_unchanged(merge):
    public = "Поэт Дина Штольц."
    private = "Клиент Дина Штольц."
    assert merge(public, [], [person(public, "Дина Штольц")]) == []
    assert merge(private, [], [person(private, "Клиент Дина Штольц")]) == [
        person(private, "Дина Штольц", reason="ner-person")
    ]
