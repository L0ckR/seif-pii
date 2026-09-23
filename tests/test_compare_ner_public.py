"""External metrics must expose unsupported labels, false positives and corrupt caches."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import compare_ner_public as public
from seif.detector import Span


def test_unfiltered_masking_keeps_unsupported_gold_and_negative_cases():
    rows = [{"key": "with-unsupported", "text": "Alice 1234", "gold": {("NAME", 0, 5), ("TOKEN", 6, 10)}},
            {"key": "negative", "text": "word", "gold": set()}]
    predictions = {"model": {"with-unsupported": [Span(0, 5, "PERSON", .9, "fixture")],
                             "negative": [Span(0, 4, "PERSON", .9, "fixture")]}}
    metrics = public.protection_metrics(rows, predictions)["systems"]["model"]
    assert metrics["true_positive"] == 5
    assert metrics["false_negative"] == 4
    assert metrics["false_positive"] == 4
    assert metrics["cases"] == 2
    assert metrics["negative_cases_with_fp"] == 1


def test_whole_case_scope_does_not_hide_unknowns_in_unfiltered_comparison():
    truth = {"common": {("NAME", 0, 5)}, "unsupported": {("TOKEN", 0, 4)}, "negative": set()}
    predictions = {"model": {"common": {("NAME", 0, 5)}, "unsupported": set(),
                             "negative": {("UNMAPPED_PRED:LOCATION", 0, 4)}}}
    filtered = public.scope_metrics(truth, predictions, {"NAME"}, True)
    unfiltered = public.scope_metrics(truth, predictions)
    assert filtered["cases"] == 2
    assert filtered["systems"]["model"]["exact_span"]["false_negative"] == 0
    assert unfiltered["cases"] == 3
    assert unfiltered["systems"]["model"]["exact_span"]["false_negative"] == 1
    assert unfiltered["systems"]["model"]["exact_span"]["false_positive"] == 1


def test_redmadrobot_location_is_mapped_and_merged_on_both_sides():
    rows = [{"key": "r", "dataset": "redmadrobot", "text": "North City",
             "gold": {("REGION", 0, 5), ("CITY", 6, 10)}}]
    predictions = {name: {"r": [Span(0, 10, "LOCATION", .9, "fixture")]} for name in public.SYSTEMS}
    truth, mapped, raw_truth, raw_pred = public.map_split(rows, predictions)
    assert truth["r"] == {("LOCATION", 0, 10)}
    assert mapped["gliner_hybrid"]["r"] == truth["r"]
    assert len(raw_truth["r"]) == 2
    assert raw_pred["gliner_hybrid"]["r"] == truth["r"]


def test_gateway_chunk_edges_are_rejected_and_offsets_remapped(monkeypatch):
    monkeypatch.setattr(public, "chunks", lambda text: iter(((0, text[:8]), (4, text[4:]))))

    class Analyzer:
        calls = 0

        def analyze(self, **_kwargs):
            self.calls += 1
            spans = ((1, 2), (7, 8)) if self.calls == 1 else ((0, 1), (3, 4))
            return [SimpleNamespace(start=start, end=end, score=.9, entity_type="PERSON") for start, end in spans]

    actual = public.infer_document(Analyzer(), "abcdefghijkl")
    assert [(e["start"], e["end"]) for e in actual] == [(1, 2), (7, 8)]


def test_model_output_cannot_silently_exceed_gateway_span_contract():
    class Analyzer:
        def analyze(self, **_kwargs):
            return [SimpleNamespace(start=0, end=201, score=.9, entity_type="PERSON")]

    analyzer = Analyzer()
    with pytest.raises((RuntimeError, ValueError)):
        public.infer_document(analyzer, "a" * 201)


def test_cache_rejects_missing_cases_even_with_valid_checksum(tmp_path):
    path = tmp_path / "spacy.jsonl"
    path.write_text("", encoding="utf-8")
    path.with_suffix(".meta.json").write_text(json.dumps({"cache_sha256": hashlib.sha256(b"").hexdigest(),
                                                        "protocol_sha256": "fixed"}))
    with pytest.raises(ValueError, match="coverage"):
        public.load_cache(path, [{"key": "required", "text": "Alice"}], "fixed")


def test_prepared_input_copy_is_immutable_and_lossless(tmp_path):
    path = tmp_path / "prepared.jsonl"
    rows = [{"key": "case", "text": "Тест\n", "gold": {("PERSON", 0, 4)}}]
    public.save_prepared_inputs(path, rows)
    protocol = {"cases": 1, "prepared_inputs_sha256": public.sha256(path)}
    assert public.load_prepared_inputs(path, protocol) == rows
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        public.load_prepared_inputs(path, protocol)
    with pytest.raises(FileExistsError):
        public.save_prepared_inputs(path, rows)
