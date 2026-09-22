"""Decoder regressions with synthetic wordpiece IDs and logits; no model/GPU."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from seif import rubert_decoder as decoder

np = pytest.importorskip("numpy")
LABELS = ["O", *(f"{prefix}-{label}" for label in sorted(decoder.NATIVE_TYPES) for prefix in ("B", "I"))]


class Encoding(dict):
    def __init__(self, ids, owners):
        super().__init__(input_ids=ids)
        self.owners = owners

    def word_ids(self, batch_index=0):
        assert batch_index == 0
        return self.owners


class Tokenizer:
    def __init__(self, counts=None):
        self.counts = counts or {}
        self.pieces = {}
        self.calls = 0

    def __call__(self, words, **kwargs):
        assert kwargs == {
            "is_split_into_words": True, "add_special_tokens": False, "truncation": False,
            "return_attention_mask": False, "return_token_type_ids": False,
        }
        self.calls += 1
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
    def __init__(self, tokenizer, tagging):
        self.tokenizer = tokenizer
        self.tagging = tagging
        self.calls = []

    def run(self, inputs):
        assert set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
        assert all(value.dtype == np.int64 and value.shape == inputs["input_ids"].shape for value in inputs.values())
        assert np.all(inputs["attention_mask"] == 1) and np.all(inputs["token_type_ids"] == 0)
        self.calls.append(inputs["input_ids"].copy())
        logits = np.zeros((*inputs["input_ids"].shape, len(LABELS)), dtype=np.float32)
        for position, token in enumerate(inputs["input_ids"][0]):
            word, subword = self.tokenizer.pieces.get(int(token), ("", 0))
            label = self.tagging(word, subword, len(self.calls), position)
            logits[0, position, LABELS.index(label)] = 4
        return logits


def runtime(tags=None, counts=None, tagging=None):
    tokenizer = Tokenizer(counts)
    if tagging is None:
        tags = tags or {}

        def tagging(word, subword, call, position):
            return tags.get(word, "O")
    backend = Backend(tokenizer, tagging)
    return SimpleNamespace(tokenizer=tokenizer, backend=backend, id2label=dict(enumerate(LABELS)))


def values(output):
    return [(item["text"], item["label"]) for item in output]


def test_exact_unicode_offsets_and_boundary_punctuation():
    text = '😀 «Иванов...»,\tПётр\nе\u0301; a@b.ru. ..ул. ...'
    tokens = decoder.words_with_offsets(text)
    assert [word for word, _, _ in tokens] == [
        "😀", "«", "Иванов", ".", ".", ".", "»", ",", "Пётр", "е\u0301", ";",
        "a@b.ru", ".", ".", ".", "ул", ".", ".", ".", ".",
    ]
    assert all(text[start:end] == word for word, start, end in tokens)
    assert all(left[2] <= right[1] for left, right in zip(tokens, tokens[1:], strict=False))
    model = runtime({"Иванов": "B-LAST_NAME", "Пётр": "B-FIRST_NAME", "a@b.ru": "B-EMAIL"})
    output = decoder.word_predict(model, text)
    assert values(output) == [("Иванов", "LAST_NAME"), ("Пётр", "FIRST_NAME"), ("a@b.ru", "EMAIL")]
    assert all(text[item["start"]:item["end"]] == item["text"] for item in output)
    assert model.tokenizer.calls == 1


def test_first_subword_owns_whole_word_and_score_is_real_probability():
    def tagging(word, subword, call, position):
        return "B-FIRST_NAME" if word == "Иванушка" and subword == 0 else "O"

    model = runtime(counts={"Иванушка": 4}, tagging=tagging)
    output = decoder.word_predict(model, "Иванушка")
    assert values(output) == [("Иванушка", "FIRST_NAME")]
    assert output[0]["score"] == pytest.approx(np.exp(4) / (42 + np.exp(4)))
    assert output[0]["score"] != .7


def test_adjacent_same_native_type_merges_even_with_restarted_bio_tag():
    model = runtime({"Полис": "B-OMS", "ОМС": "I-OMS", "1234": "B-OMS"})
    output = decoder.word_predict(model, "Полис ОМС: 1234")
    assert values(output) == [("Полис ОМС: 1234", "OMS")]


@pytest.mark.parametrize("kind", sorted(decoder.NATIVE_TYPES))
def test_all_native_types_preserved_with_correct_punctuation_bridge_policy(kind):
    model = runtime({"лево": f"B-{kind}", "право": f"B-{kind}"})
    output = decoder.word_predict(model, "лево, право")
    if kind in decoder._UNBRIDGED:
        assert values(output) == [("лево", kind), ("право", kind)]
    else:
        assert values(output) == [("лево, право", kind)]


def test_bridge_does_not_cross_words_mixed_labels_or_multiple_unlabeled_punctuation():
    model = runtime({"111": "B-PHONE", "222": "B-SNILS", "333": "B-PHONE"})
    assert values(decoder.word_predict(model, "111:222 текст 333,,111")) == [
        ("111", "PHONE"), ("222", "SNILS"), ("333", "PHONE"), ("111", "PHONE"),
    ]


def test_long_input_owned_centers_choose_context_and_cover_final_word():
    words = [f"w{index}" for index in range(1100)]

    def tagging(word, subword, call, position):
        if not word:
            return "O"
        return {1: "B-PHONE", 2: "B-INN", 3: "B-SNILS"}[call]

    model = runtime(tagging=tagging)
    text = " ".join(words)
    result = decoder.word_predict(model, text)
    assert len(model.backend.calls) == 3
    assert model.tokenizer.calls == 1
    assert [item["label"] for item in result] == ["PHONE", "INN", "SNILS"]
    assert result[0]["text"].split() == words[:446]
    assert result[1]["text"].split() == words[446:828]
    assert result[2]["text"].split() == words[828:]
    assert result[-1]["end"] == len(text)


def test_very_wide_words_make_progress_without_skipping_following_names():
    words = ["a", "wide", "Иван", "wide2", "Пётр", "tail"]
    model = runtime({"Иван": "B-FIRST_NAME", "Пётр": "B-FIRST_NAME"}, {"wide": 500, "wide2": 500})
    result = decoder.word_predict(model, " ".join(words))
    assert values(result) == [("Иван", "FIRST_NAME"), ("Пётр", "FIRST_NAME")]
    assert 1 < len(model.backend.calls) <= len(words)


def test_exact_maximum_word_is_valid_and_fully_labeled():
    model = runtime({"wide": "B-URL"}, {"wide": 510})
    assert values(decoder.word_predict(model, "wide")) == [("wide", "URL")]
    assert model.backend.calls[0].shape == (1, 512)


@pytest.mark.parametrize("count", [0, 511, 900])
def test_unrepresentable_words_fail_before_any_partial_model_response(count):
    model = runtime({"Иван": "B-FIRST_NAME"}, {"wide": count})
    with pytest.raises(ValueError, match="refusing partial inference"):
        decoder.word_predict(model, "wide Иван")
    assert model.backend.calls == []


@pytest.mark.parametrize("text", ["", " \t\n\r\u2003"])
def test_empty_input_has_no_tokenizer_or_backend_side_effects(text):
    model = runtime()
    assert decoder.word_predict(model, text) == []
    assert model.tokenizer.calls == 0 and model.backend.calls == []


@pytest.mark.parametrize("text", [None, 42, [], b"text"])
def test_non_string_input_rejected(text):
    with pytest.raises(ValueError, match="must be a string"):
        decoder.word_predict(runtime(), text)


@pytest.mark.parametrize("change", [
    lambda value: value[:, :-1], lambda value: value[:, :, :-1], lambda value: value.astype(np.float64),
    lambda value: value.tolist(), lambda value: value * np.nan, lambda value: value + np.inf,
])
def test_malformed_logits_never_become_partial_predictions(change):
    model = runtime({"Иван": "B-FIRST_NAME"})
    original = model.backend.run
    model.backend.run = lambda inputs: change(original(inputs))
    with pytest.raises(ValueError, match="logits shape, dtype or values"):
        decoder.word_predict(model, "Иван")


@pytest.mark.parametrize("owners", [[None], [True], [-1], [2], [0, 2], [1, 0]])
def test_invalid_word_ownership_rejected(owners):
    model = runtime()
    model.tokenizer = lambda *args, **kwargs: Encoding([1000] * len(owners), owners)
    with pytest.raises(ValueError, match="alignment|omitted a word"):
        decoder.word_predict(model, "Иван Пётр")


def test_unknown_label_inventory_rejected_before_model_use():
    model = runtime()
    model.id2label[1] = "B-UNKNOWN"
    with pytest.raises(ValueError, match="label inventory"):
        decoder.word_predict(model, "Иван")
    assert model.backend.calls == []


def test_invalid_special_token_layout_rejected():
    model = runtime()
    model.tokenizer.build_inputs_with_special_tokens = lambda ids: [101, *reversed(ids), 102]
    with pytest.raises(ValueError, match="special-token layout"):
        decoder.word_predict(model, "Иван Пётр")
    assert model.backend.calls == []
