"""Reversibility and bounded restoration for untrusted edited token replies."""

import os
import subprocess
import sys

import pytest

from seif.detector import Span, detect
from seif.transform import RestorationTooLarge, mask, restore_exact, restore_tokens


@pytest.mark.parametrize("kind", ["EMPLOYEE_ID2", "X2", "X_", "X" + "2" * 39])
def test_valid_custom_types_restore_in_edited_replies(kind):
    original = "🔒 ID-123"
    spans = detect(original, extra_rules=[{"type": kind, "pattern": r"ID-\d+"}])
    masked, replacements = mask(original, spans, "token")
    record = {"masked": masked, "replacements": replacements}
    token = replacements[0]["replacement"]
    assert restore_tokens(f"Ответ: {token}; снова {token}.", record) == "Ответ: ID-123; снова ID-123."
    assert restore_exact(record) == original


def test_unknown_digit_type_token_rejects_entire_edited_reply():
    masked, replacements = mask("mail@example.invalid", [Span(0, 20, "EMAIL")], "token")
    with pytest.raises(ValueError, match="Unknown"):
        restore_tokens(masked + " ⟦PD:ID2:0000000000000000⟧", {"replacements": replacements})


@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_exact_restoration_preserves_unicode_and_repeated_values(mode):
    original = "🔒 Имя: José; повтор: José; e\u0301\nКонец."
    first, second = original.index("José"), original.rindex("José")
    spans = [Span(first, first + 4, "PERSON"), Span(second, second + 4, "PERSON")]
    masked, replacements = mask(original, spans, mode)
    assert restore_exact({"masked": masked, "replacements": replacements}) == original


def test_exact_restoration_with_many_replacements_and_variable_lengths():
    originals = [f"Имя{i}🧪" for i in range(5000)]
    masked = "|".join("*" for _ in originals)
    replacements = [
        {"start": i * 2, "end": i * 2 + 1, "original": original, "replacement": "*"}
        for i, original in enumerate(originals)
    ]
    assert restore_exact({"masked": masked, "replacements": replacements}) == "|".join(originals)


@pytest.mark.parametrize(
    "replacements",
    [
        [{"start": -1, "end": 1, "replacement": "*", "original": "a"}],
        [{"start": False, "end": 1, "replacement": "*", "original": "a"}],
        [{"start": 0, "end": 0, "replacement": "", "original": "a"}],
        [{"start": 0, "end": 3, "replacement": "**", "original": "a"}],
        [{"start": 0, "end": 1, "replacement": "?", "original": "a"}],
        [
            {"start": 0, "end": 2, "replacement": "**", "original": "a"},
            {"start": 1, "end": 2, "replacement": "*", "original": "b"},
        ],
        [
            {"start": 1, "end": 2, "replacement": "*", "original": "a"},
            {"start": 0, "end": 1, "replacement": "*", "original": "b"},
        ],
    ],
)
def test_exact_restoration_rejects_corrupt_coordinates_or_value(replacements):
    with pytest.raises(ValueError, match="replacement"):
        restore_exact({"masked": "**", "replacements": replacements})


def test_repeated_tokens_respect_output_limit_before_expansion():
    original = "sensitive" * 400
    token, replacements = mask(original, [Span(0, len(original), "CUSTOM")], "token")
    with pytest.raises(RestorationTooLarge) as exc:
        restore_tokens(token * 1000, {"replacements": replacements}, max_output_chars=40000)
    assert original not in str(exc.value)
    assert restore_tokens(token, {"replacements": replacements}, max_output_chars=len(original)) == original


def test_output_limit_uses_final_unicode_character_count():
    originals = ("🧪" * 100, "é")
    masked, replacements = mask(" ".join(originals), [Span(0, 100, "PERSON"), Span(101, 102, "PERSON")], "token")
    first, second = (item["replacement"] for item in replacements)
    reply = first + second * 50
    assert len(reply) > 150
    assert restore_tokens(reply, {"replacements": replacements}, max_output_chars=150) == "🧪" * 100 + "é" * 50


@pytest.mark.parametrize("limit", [-1, True, 1.5, "100"])
def test_invalid_output_limits_reject(limit):
    with pytest.raises(ValueError, match="limit"):
        restore_tokens("", {"replacements": []}, max_output_chars=limit)


def test_token_restoration_does_not_reinterpret_literal_original_tokens():
    first = "⟦PD:EMAIL:0000000000000001⟧"
    second = "⟦PD:EMAIL:0000000000000002⟧"
    record = {"replacements": [
        {"replacement": first, "original": second},
        {"replacement": second, "original": "mail@example.invalid"},
    ]}
    assert restore_tokens(first + second, record, max_output_chars=100) == second + "mail@example.invalid"


@pytest.mark.parametrize("size", [4097, 10000])
def test_oversized_custom_match_fails_closed(size):
    with pytest.raises(ValueError, match="4096"):
        detect("private " + "x" * size, extra_rules=[{"type": "CUSTOM", "pattern": r"private|x+"}])


def test_custom_match_at_limit_is_protected_and_reversible():
    original = "x" * 4096
    spans = detect(original, extra_rules=[{"type": "CUSTOM", "pattern": r"x+"}])
    masked, replacements = mask(original, spans, "mask")
    assert masked == "*" * 4096
    assert restore_exact({"masked": masked, "replacements": replacements}) == original


def test_custom_zero_width_match_on_nonempty_input_fails_closed():
    with pytest.raises(ValueError, match="empty"):
        detect("private", extra_rules=[{"type": "CUSTOM", "pattern": r"(?=private)"}])


def test_equal_priority_custom_types_are_stable_across_worker_hash_seeds():
    script = (
        "from seif.detector import detect; "
        "print(detect('secret', extra_rules=["
        "{'type': 'AB', 'pattern': 'secret'}, "
        "{'type': 'CD', 'pattern': 'secret'}])[0].type)"
    )
    results = {
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        for seed in (0, 4, 5, 7)
    }
    assert results == {"AB"}
