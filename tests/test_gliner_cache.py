"""Validate cached GLiNER spans against the exact input, without model imports."""
import copy

import pytest

from scripts.cache_gliner import cache_entities


def entity(text="Иван", *, start=0, confidence=0.9):
    return {"text": text, "start": start, "end": start + len(text), "confidence": confidence}


def result(*, person=(), location=(), name=None, organization=None):
    groups = {"person": list(person), "location": list(location)}
    if name is not None:
        groups["name"] = list(name)
    if organization is not None:
        groups["organization"] = list(organization)
    return {"entities": groups}


def test_cache_preserves_original_unicode_offsets_combining_marks_and_confidence():
    text = "😀 Ива\u0301н, Москва\n"
    output = result(person=[entity("Ива\u0301н", start=2, confidence=.912345)],
                    location=[entity("Москва", start=9, confidence=.678901)])
    before = copy.deepcopy(output)
    assert cache_entities(text, output) == [
        {"start": 2, "end": 7, "entity_type": "PERSON", "score": .912345},
        {"start": 9, "end": 15, "entity_type": "LOCATION", "score": .678901},
    ]
    assert output == before


@pytest.mark.parametrize("wrong_span", [
    {"text": "Ива\u0301н", "start": 3, "end": 8, "confidence": .9},  # UTF-16 coordinates.
    {"text": "Ива\u0301н", "start": 5, "end": 15, "confidence": .9},  # UTF-8 byte coordinates.
    {"text": "Иван", "start": 2, "end": 7, "confidence": .9},  # Removed combining mark.
    {"text": "ива\u0301н", "start": 2, "end": 7, "confidence": .9},  # Case-normalized text.
])
def test_cache_rejects_normalized_or_non_codepoint_coordinates(wrong_span):
    with pytest.raises(ValueError):
        cache_entities("😀 Ива\u0301н, Москва\n", result(person=[wrong_span]))


@pytest.mark.parametrize("output", [
    None, [], {}, {"entities": None}, {"entities": []}, {"entities": {}},
    {"entities": {"person": []}}, {"entities": {"location": []}},
    {"entities": {"person": [], "location": []}, "extra": {}},
    {"entities": {"person": [], "location": [], "passport number": []}},
    {"entities": {"PERSON": [], "location": []}},
    {"entities": {"person": None, "location": []}},
    {"entities": {"person": "Иван", "location": []}},
])
def test_cache_rejects_incomplete_unknown_or_malformed_label_groups(output):
    with pytest.raises(ValueError):
        cache_entities("Иван", output)


@pytest.mark.parametrize("broken", [None, [], "Иван", {},
                                     {"text": "Иван", "start": 0, "end": 4},
                                     {"start": 0, "end": 4, "confidence": .9}])
def test_cache_rejects_entire_prediction_when_any_span_is_malformed(broken):
    with pytest.raises(ValueError):
        cache_entities("Иван", result(person=[entity(), broken]))


@pytest.mark.parametrize(("field", "value"), [
    ("start", True), ("start", -1), ("start", 0.0), ("start", "0"), ("start", 4),
    ("end", False), ("end", 0), ("end", 4.0), ("end", 5),
    ("confidence", True), ("confidence", False), ("confidence", None), ("confidence", ".9"),
    ("confidence", float("nan")), ("confidence", float("inf")), ("confidence", float("-inf")),
    ("confidence", -.001), ("confidence", 1.001),
    ("text", None), ("text", "ИВАН"),
])
def test_cache_rejects_bad_offsets_nonfinite_confidence_and_non_numeric_values(field, value):
    with pytest.raises(ValueError):
        cache_entities("Иван", result(person=[{**entity(), field: value}]))


@pytest.mark.parametrize("confidence", [0, 1, 0.5])
def test_cache_preserves_valid_confidence_endpoints_without_an_extra_threshold(confidence):
    assert cache_entities("Иван", result(person=[entity(confidence=confidence)])) == [
        {"start": 0, "end": 4, "entity_type": "PERSON", "score": float(confidence)},
    ]


@pytest.mark.parametrize("schema", ["presidio-labels", "described-names"])
def test_auxiliary_organizations_are_validated_but_do_not_mask_or_veto_people(schema):
    options = {"name": []} if schema == "presidio-labels" else {}
    output = result(person=[entity(confidence=.75)], organization=[entity(confidence=1)], **options)
    assert cache_entities("Иван", output, schema) == [
        {"start": 0, "end": 4, "entity_type": "PERSON", "score": .75},
    ]
    output["entities"]["organization"][0]["text"] = "different input"
    with pytest.raises(ValueError):
        cache_entities("Иван", output, schema)


@pytest.mark.parametrize("schema", ["presidio-labels", "described-names"])
def test_declared_auxiliary_label_cannot_be_silently_missing(schema):
    options = {"name": []} if schema == "presidio-labels" else {}
    with pytest.raises(ValueError):
        cache_entities("Иван", result(person=[entity()], **options), schema)


def test_person_name_aliases_deduplicate_by_maximum_confidence_without_merging_other_types():
    output = result(
        person=[entity(confidence=.7), entity("Пётр", start=5, confidence=.8), entity(confidence=.6)],
        name=[entity(confidence=.95), entity("Пётр", start=5, confidence=.5)],
        location=[entity(confidence=.99)], organization=[],
    )
    expected = [
        {"start": 0, "end": 4, "entity_type": "LOCATION", "score": .99},
        {"start": 0, "end": 4, "entity_type": "PERSON", "score": .95},
        {"start": 5, "end": 9, "entity_type": "PERSON", "score": .8},
    ]
    assert cache_entities("Иван Пётр", output, "presidio-labels") == expected
    reversed_output = {"entities": {key: list(reversed(value))
                                    for key, value in reversed(list(output["entities"].items()))}}
    assert cache_entities("Иван Пётр", reversed_output, "presidio-labels") == expected


def test_unrecognized_entity_cannot_be_discarded_as_if_it_were_an_auxiliary_organization():
    output = result(person=[entity()], organization=[], name=[])
    output["entities"]["passport number"] = [entity()]
    with pytest.raises(ValueError):
        cache_entities("Иван", output, "presidio-labels")


def test_empty_text_and_complete_empty_schema_produce_empty_cache():
    assert cache_entities("", result()) == []
    assert cache_entities("", result(organization=[], name=[]), "presidio-labels") == []
