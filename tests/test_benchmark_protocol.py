"""Small hand-counted fixtures validate scoring mechanics, not corpus outcomes."""
import json

from scripts.evaluate_redmadrobot import (
    COMMON,
    GOLD_MAP,
    coarsen,
    merge_adjacent,
    parse_bio,
    score_scope,
)


def row(text, tokens, labels):
    return {"text": text, "tokens": json.dumps(tokens), "ner_tags": json.dumps(labels)}


def test_bio_alignment_preserves_original_unicode_and_whitespace():
    text = "😀  Дина\tШварц!"
    spans, problem = parse_bio(row(text, ["😀", "Дина", "Шварц", "!"], ["O", "B-FIRST_NAME", "B-LAST_NAME", "O"]))
    assert problem is None
    assert spans == {("FIRST_NAME", 3, 7), ("LAST_NAME", 8, 13)}
    assert merge_adjacent(text, coarsen(spans, GOLD_MAP)) == {("PERSON", 3, 13)}


def test_alignment_does_not_silently_normalize_missing_or_different_tokens():
    assert parse_bio(row("Дина", ["дина"], ["B-FIRST_NAME"])) == (None, "token_not_exactly_alignable")
    assert parse_bio(row("Дина!", ["Дина"], ["B-FIRST_NAME"])) == (None, "unaligned_text_suffix")
    assert parse_bio(row("Дина", ["Дина"], ["I-FIRST_NAME"])) == (None, "invalid_bio_transition")


def test_empty_outside_token_has_no_characters_and_keeps_bio_boundary():
    spans, error = parse_bio(row("Дина Шварц", ["Дина", "", "Шварц"], ["B-FIRST_NAME", "O", "B-LAST_NAME"]))
    assert error is None
    assert spans == {("FIRST_NAME", 0, 4), ("LAST_NAME", 5, 10)}
    assert parse_bio(row("Дина Шварц", ["Дина", "", "Шварц"], ["B-FIRST_NAME", "O", "I-FIRST_NAME"]))[1] == "invalid_bio_transition"


def test_symmetric_merge_does_not_cross_punctuation_or_entity_classes():
    assert merge_adjacent("Дина, Шварц", {("PERSON", 0, 4), ("PERSON", 6, 11)}) == {("PERSON", 0, 4), ("PERSON", 6, 11)}
    assert merge_adjacent("Дина Тула", {("PERSON", 0, 4), ("LOCATION", 5, 9)}) == {("PERSON", 0, 4), ("LOCATION", 5, 9)}
    assert merge_adjacent("Дина Шварц", {("PERSON", 0, 4), ("PERSON", 0, 10)}) == {("PERSON", 0, 10)}


def test_primary_keeps_negatives_excludes_whole_unmapped_cases_and_counts_characters():
    truth = {"name": {("PERSON", 0, 4)}, "negative": set(), "unmapped": {("UNMAPPED:SNILS", 0, 11)}}
    predicted = {"example": {"name": {("PERSON", 0, 3)}, "negative": {("PERSON", 1, 3)}, "unmapped": set()}}
    result = score_scope(truth, predicted, COMMON)
    assert result["case_ids"] == ["name", "negative"]
    char = result["systems"]["example"]["typed_character_primary"]
    assert (char["true_positive"], char["false_positive"], char["false_negative"]) == (3, 2, 1)
    exact = result["systems"]["example"]["merged_exact_span_secondary"]
    assert (exact["true_positive"], exact["false_positive"], exact["false_negative"]) == (0, 2, 1)
