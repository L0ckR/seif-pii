"""Batch decoder correctness and graph ownership checks without a GPU."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from seif import rubert_batch as batch
from seif import rubert_decoder as decoder

np = pytest.importorskip("numpy")
LABELS = ["O", *(f"{prefix}-{label}" for label in sorted(decoder.NATIVE_TYPES) for prefix in ("B", "I"))]


class Encoding(dict):
    def __init__(self, ids, owners):
        super().__init__(input_ids=ids)
        self.owners = owners

    def word_ids(self, batch_index=0):
        return self.owners


class Tokenizer:
    pad_token_id = 0

    def __init__(self, counts):
        self.counts, self.pieces = counts, {}

    def __call__(self, words, **_kwargs):
        ids, owners = [], []
        for owner, word in enumerate(words):
            for subword in range(self.counts.get(word, 1)):
                token = 1000 + len(self.pieces)
                self.pieces[token] = (word, subword)
                ids.append(token)
                owners.append(owner)
        return Encoding(ids, owners)

    @staticmethod
    def build_inputs_with_special_tokens(ids):
        return [101, *ids, 102]

    @staticmethod
    def create_token_type_ids_from_sequences(ids):
        return [0] * (len(ids) + 2)


class Backend:
    def __init__(self, tokenizer, tag):
        self.tokenizer, self.tag, self.calls = tokenizer, tag, []

    def run(self, inputs):
        ids = inputs["input_ids"]
        self.calls.append(ids.copy())
        assert all(value.dtype == np.int64 and value.shape == ids.shape for value in inputs.values())
        assert np.all(inputs["token_type_ids"] == 0)
        assert np.all(inputs["attention_mask"][:, :2] == 1)
        output = np.zeros((*ids.shape, len(LABELS)), dtype=np.float32)
        for row in range(ids.shape[0]):
            first = self.tokenizer.pieces.get(int(ids[row, 1]), ("", 0))[0]
            for position, token in enumerate(ids[row]):
                word, subword = self.tokenizer.pieces.get(int(token), ("", 0))
                label = self.tag(word, subword, first)
                output[row, position, LABELS.index(label)] = 4
        return output


def runtime(tags=None, counts=None, tag=None):
    tokenizer = Tokenizer(counts or {})
    tags = tags or {}
    if tag is None:
        def tag(word, subword, first):
            return tags.get(word, "O") if subword == 0 else "O"
    return SimpleNamespace(tokenizer=tokenizer, id2label=dict(enumerate(LABELS)), backend=Backend(tokenizer, tag))


def test_matches_single_unicode_first_subwords_controls_bridges_and_input_order():
    texts = ["", "😀 Иванушка, Пётр", "\u200d Иванушка", "111:222", "Москва", "\u200d"]
    tags = {"Иванушка": "B-FIRST_NAME", "Пётр": "B-FIRST_NAME", "111": "B-PHONE", "222": "B-PHONE",
            "Москва": "B-CITY"}
    counts = {"Иванушка": 4, "\u200d": 0}
    model = runtime(tags, counts)
    actual = batch.word_predict_batch(model, texts)
    expected = [decoder.word_predict(runtime(tags, counts), text) for text in texts]
    assert actual == expected
    assert len(model.backend.calls) == 1
    assert model.backend.calls[0].shape == (4, 32)
    assert actual[1][0]["text"] == "Иванушка"
    assert actual[3][0]["text"] == "111:222"


def test_long_windows_keep_center_ownership_when_bucket_execution_reorders_them():
    text = " ".join(f"w{i}" for i in range(1000))
    prefix = " ".join(["слово"] * 180)

    def tag(word, subword, first):
        if not word:
            return "O"
        return {"w0": "B-PHONE", "w382": "B-INN", "w764": "B-SNILS"}.get(first, "O")

    model = runtime(tag=tag)
    actual = batch.word_predict_batch(model, [prefix, text])
    expected = [decoder.word_predict(runtime(tag=tag), value) for value in [prefix, text]]
    assert actual == expected
    assert [call.shape for call in model.backend.calls] == [(2, 256), (2, 512)]
    assert [item["label"] for item in actual[1]] == ["PHONE", "INN", "SNILS"]
    assert [len(item["text"].split()) for item in actual[1]] == [446, 382, 172]


def test_wide_words_and_bucket_sort_do_not_lose_overlapping_owned_words():
    texts = ["a wide Иван wide2 Пётр tail", " ".join(["слово"] * 40), "Иван"]
    counts = {"wide": 500, "wide2": 500}
    tags = {"Иван": "B-FIRST_NAME", "Пётр": "B-FIRST_NAME"}
    actual = batch.word_predict_batch(runtime(tags, counts), texts, max_batch=2)
    assert actual == [decoder.word_predict(runtime(tags, counts), text) for text in texts]


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 8, 16, 31, 32])
def test_all_batch_sizes_are_bounded_and_tail_dummy_rows_never_escape(size):
    model = runtime({"Иван": "B-FIRST_NAME"})
    output = batch.word_predict_batch(model, ["Иван"] * size, max_batch=8)
    assert len(output) == size and all(len(items) == 1 for items in output)
    assert all(call.shape[0] <= 8 and call.shape[0] in batch.BATCH_BUCKETS for call in model.backend.calls)


@pytest.mark.parametrize("texts", [None, "Иван", ("Иван",), [None], ["Иван"] * 33])
def test_invalid_batches_fail_before_inference(texts):
    model = runtime()
    with pytest.raises(ValueError, match="list of at most"):
        batch.word_predict_batch(model, texts)
    assert model.backend.calls == []


@pytest.mark.parametrize("maximum", [True, 0, -1, 3, 9, 33])
def test_invalid_batch_cap_fails(maximum):
    with pytest.raises(ValueError, match="max_batch"):
        batch.word_predict_batch(runtime(), ["Иван"], max_batch=maximum)


def test_empty_and_all_ignored_batches_skip_gpu():
    model = runtime(counts={"\u200d": 0})
    assert batch.word_predict_batch(model, []) == []
    assert batch.word_predict_batch(model, ["", " \n", "\u200d"]) == [[], [], []]
    assert model.backend.calls == []


@pytest.mark.parametrize("count", [0, 511])
def test_bad_word_prevents_any_partial_batch_response(count):
    model = runtime(counts={"wide": count})
    with pytest.raises(ValueError, match="refusing partial inference"):
        batch.word_predict_batch(model, ["Иван", "wide"])
    assert model.backend.calls == []


@pytest.mark.parametrize("change", [
    lambda value: value[:, :-1], lambda value: value[:, :, :-1], lambda value: value.astype(np.float64),
    lambda value: value.tolist(), lambda value: value * np.nan, lambda value: value + np.inf,
])
def test_invalid_logits_fail_the_whole_batch(change):
    model = runtime({"Иван": "B-FIRST_NAME"})
    original = model.backend.run
    model.backend.run = lambda inputs: change(original(inputs))
    with pytest.raises(ValueError, match="logits shape"):
        batch.word_predict_batch(model, ["Иван", "Пётр"])


class Context:
    def __init__(self, path="engine", **kwargs):
        self.path, self.options, self.calls = path, kwargs, []

    def run(self, inputs):
        self.calls.append(inputs)
        return "context"


def graph_runtime():
    fallback = Context()
    backend = SimpleNamespace(fallback=fallback, _engine_path="verified-engine", _device_index=0,
                              run=lambda inputs: "original")
    return SimpleNamespace(backend=backend)


def test_graph_contexts_are_reused_share_weights_and_cap_falls_back_without_eviction():
    model = graph_runtime()
    shared = model.backend.fallback
    graphs = batch.install_batch_backend(model, max_batch=8, max_contexts=1)
    inputs = {"input_ids": np.zeros((4, 32), dtype=np.int64)}
    assert model.backend is graphs
    assert graphs.run(inputs) == graphs.run(inputs) == "context"
    context = graphs.contexts[(4, 32)]
    assert context.options == {"device": 0, "cuda_graph": True, "_shared": shared}
    assert len(context.calls) == 2 and context.path == "verified-engine"
    assert graphs.run({"input_ids": np.zeros((8, 32), dtype=np.int64)}) == "original"
    assert graphs.run({"input_ids": np.zeros((1, 14), dtype=np.int64)}) == "original"
    assert graphs.batch_graph_shapes == [(4, 32)]


@pytest.mark.parametrize("shape", [(16, 32), (3, 32), (2, 14), (32,)])
def test_graph_wrapper_rejects_unbounded_or_nonbucket_shapes(shape):
    graphs = batch.install_batch_backend(graph_runtime(), max_batch=8)
    with pytest.raises(ValueError, match="RuBERT"):
        graphs.run({"input_ids": np.zeros(shape, dtype=np.int64)})


@pytest.mark.parametrize("cap", [-1, 26, True])
def test_graph_memory_cap_rejected(cap):
    with pytest.raises(ValueError, match="context cap"):
        batch.install_batch_backend(graph_runtime(), max_contexts=cap)


def test_install_is_idempotent_and_does_not_retain_nested_context_caches():
    model = graph_runtime()
    first = batch.install_batch_backend(model)
    assert batch.install_batch_backend(model) is first
    with pytest.raises(ValueError, match="already installed"):
        batch.install_batch_backend(model, max_batch=16)
