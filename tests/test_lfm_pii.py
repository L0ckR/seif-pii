"""Guard the pinned LFM schema, one-pass decode and no-truncation contract."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from seif import lfm_pii as lfm


def schema():
    labels = {0: "O"}
    for index, kind in enumerate(sorted(lfm.NATIVE_TYPES)):
        for number, prefix in enumerate(("B", "I", "E", "S"), 1):
            labels[4 * index + number] = prefix + "-" + kind
    return SimpleNamespace(id2label=labels, num_labels=161)


@pytest.fixture
def context(monkeypatch):
    module = SimpleNamespace(context_cued_spans=lambda text: [], group_b_cue_spans=lambda text: [],
                             CONTEXT_TYPES={"identity.passport"})
    monkeypatch.setitem(sys.modules, "context_cued", module)
    return module


class Tokenizer:
    def __init__(self, count=4):
        self.count = count

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": list(range(self.count))}


class Decoder:
    def __init__(self, raw=None, hybrid=None):
        self.raw = raw if raw is not None else [{"start": 0, "end": 4, "type": "identity.person_name"}]
        self.hybrid = hybrid if hybrid is not None else self.raw
        self.calls = []

    def model_spans(self, text, tok, model):
        self.calls.append("model")
        return self.raw

    def hybrid_spans(self, text, raw):
        self.calls.append("hybrid")
        return self.hybrid


def analyzer(context, decoder=None, count=4):
    model = SimpleNamespace(config=schema(), device="cpu")
    return lfm.LfmPiiAnalyzer(Tokenizer(count), model, decoder or Decoder(), context)


def test_predicts_once_then_official_hybrid_without_scores_or_raw_text(context):
    decoder = Decoder(hybrid=[{"start": 0, "end": 4, "type": "identity.person_name", "text": "Иван"}])
    actual = analyzer(context, decoder).predict_both("Иван прибыл")
    assert decoder.calls == ["model", "hybrid"]
    assert actual == {"raw": [{"start": 0, "end": 4, "type": "identity.person_name"}],
                      "hybrid": [{"start": 0, "end": 4, "type": "identity.person_name"}]}


def test_rejects_oversize_before_neural_or_regex_inference(context):
    decoder = Decoder()
    with pytest.raises(ValueError, match="truncation"):
        analyzer(context, decoder, count=2049).predict_both("Иван")
    assert decoder.calls == []


def test_exact_token_limit_accepted(context):
    decoder = Decoder()
    analyzer(context, decoder, count=2048).predict_both("Иван")
    assert decoder.calls == ["model", "hybrid"]


def test_stale_109_label_schema_is_rejected(context):
    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "O"}, num_labels=109), device="cpu")
    with pytest.raises(ValueError, match="161-label"):
        lfm.LfmPiiAnalyzer(Tokenizer(), model, Decoder(), context)


def test_bioes_mapping_must_match_classifier_not_just_label_count(context):
    config = schema()
    config.id2label[1], config.id2label[2] = config.id2label[2], config.id2label[1]
    with pytest.raises(ValueError, match="BIOES"):
        lfm.LfmPiiAnalyzer(Tokenizer(), SimpleNamespace(config=config), Decoder(), context)


@pytest.mark.parametrize("span", [
    {"start": True, "end": 4, "type": "identity.person_name"},
    {"start": -1, "end": 4, "type": "identity.person_name"},
    {"start": 0, "end": 5, "type": "identity.person_name"},
    {"start": 2, "end": 2, "type": "identity.person_name"},
    {"start": 0, "end": 4, "type": "unknown"},
    {"start": 0, "end": 4, "type": "identity.person_name", "text": "different"},
    None,
])
def test_malformed_raw_offsets_or_labels_fail_before_hybrid(context, span):
    decoder = Decoder(raw=[span])
    with pytest.raises(ValueError):
        analyzer(context, decoder).predict_both("Иван")
    assert decoder.calls == ["model"]


def test_malformed_hybrid_offsets_rejected(context):
    with pytest.raises(ValueError):
        analyzer(context, Decoder(hybrid=[{"start": 0, "end": 100, "type": "contact.address"}])).predict_both("Иван")


def test_missing_context_layer_cannot_silently_skip_postprocessing(context, monkeypatch):
    instance = analyzer(context)
    monkeypatch.delitem(sys.modules, "context_cued")
    with pytest.raises(RuntimeError, match="context-cued"):
        instance.predict_both("Иван")
    assert instance.decoder.calls == []


def test_both_context_functions_are_mandatory(context):
    context.group_b_cue_spans = None
    with pytest.raises(RuntimeError, match="context-cued"):
        analyzer(context)


def test_mutable_raw_input_to_hybrid_does_not_change_returned_raw(context):
    decoder = Decoder()

    def mutate(text, raw):
        raw[0]["end"] = 1
        return []

    decoder.hybrid_spans = mutate
    actual = analyzer(context, decoder).predict_both("Иван")
    assert actual["raw"][0]["end"] == 4
    assert actual["hybrid"] == []


def test_loader_rejects_changed_pinned_files_before_execution(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="config.json"):
        lfm.LfmPiiAnalyzer.from_local(tmp_path)


def test_loader_rejects_unsupported_device_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="device"):
        lfm.LfmPiiAnalyzer.from_local(tmp_path, device="remote")


def test_decoder_preserves_overlaps_order_and_long_spans(context):
    raw = [{"start": 0, "end": 201, "type": "contact.address"},
           {"start": 0, "end": 4, "type": "identity.person_name"}]
    assert analyzer(context, Decoder(raw=raw)).predict_both("Иван" + "a" * 197)["raw"] == raw


def test_empty_text_runs_official_decoders_without_invented_entities(context):
    instance = analyzer(context, Decoder(raw=[], hybrid=[]))
    assert instance.predict_both("") == {"raw": [], "hybrid": []}
    assert instance.decoder.calls == ["model", "hybrid"]


class Offsets:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (1, len(rows), 2)

    def __getitem__(self, index):
        assert index == 0
        return SimpleNamespace(tolist=lambda: self.rows)


@pytest.mark.parametrize("tokens,offsets", [
    (1, [(0, 0)]),
    (2, [(0, 0), (0, 5)]),
    (2, [(0, 0), (-1, 4)]),
])
def test_tokenizer_guard_detects_truncation_and_invalid_offsets(tokens, offsets):
    encoded = {"input_ids": SimpleNamespace(shape=(1, tokens)), "offset_mapping": Offsets(offsets)}
    guard = lfm._TokenizerGuard(lambda text, **kwargs: encoded, count=2)
    with pytest.raises(ValueError):
        guard("Иван", max_length=2048)


@pytest.mark.parametrize("shape,finite", [((1, 1, 161), True), ((1, 2, 109), True), ((1, 2, 161), False)])
def test_model_guard_rejects_truncated_logits_stale_head_and_nonfinite_values(shape, finite):
    logits = SimpleNamespace(shape=shape, isfinite=lambda: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: finite)))

    class Model:
        device = "cpu"
        config = schema()

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits)

    with pytest.raises(ValueError, match="logits"):
        lfm._ModelGuard(Model(), 2)(input_ids=None)
