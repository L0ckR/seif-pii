"""Independent examples for the shared wrapper benchmark's scoring policy."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def scorer():
    path = Path(__file__).resolve().parents[1] / "benchmarks/ner-models/wrapper-matrix-20260923/score.py"
    spec = importlib.util.spec_from_file_location("wrapper_matrix_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_symmetric_join_is_separate_from_raw_exact_score(scorer):
    text = "Дина Марковна"
    gold = scorer.checked_spans([
        {"type": "FIRST_NAME", "start": 0, "end": 4},
        {"type": "MIDDLE_NAME", "start": 5, "end": 13},
    ], text)
    predicted = scorer.checked_spans([{"type": "PERSON", "start": 0, "end": 13}], text)
    assert gold != predicted
    assert scorer._merge_adjacent(text, gold) == scorer._merge_adjacent(text, predicted)
    assert scorer.alnum_positions(text, gold) == scorer.alnum_positions(text, predicted)
    raw, joined = Counter(), Counter()
    scorer.update(raw, gold, predicted)
    scorer.update(joined, scorer._merge_adjacent(text, gold), scorer._merge_adjacent(text, predicted))
    assert scorer.metrics(raw)["f1"] == 0
    assert scorer.metrics(joined)["f1"] == 1


def test_unsupported_types_are_retained_and_offsets_checked(scorer):
    value = "A-1"
    actual = scorer.checked_spans([{"type": "UNMAPPED_PRED", "start": 0, "end": len(value)}], value)
    assert actual == {("UNMAPPED_PRED", 0, len(value))}
    assert scorer.alnum_positions(value, actual) == {0, 2}
    with pytest.raises(ValueError, match="offset"):
        scorer.checked_spans([{"type": "PERSON", "start": 0, "end": len(value) + 1}], value)
