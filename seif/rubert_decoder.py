"""Word-aware decoding for the pinned RuBERT model, with original text offsets.

The decoding policy follows redmadrobot-rnd/pii-guard (Apache-2.0), revision
24230abb72949a9f85499244dd4f15a0ad0cdd9e, ner/recognizer.py: first subword labels,
center-owned windows, and structured-entity punctuation bridges. This adaptation
uses the verified NumPy backend, tokenizes only once, validates full coverage,
and reports real first-subword softmax scores instead of a fixed rule priority.
It does not require Presidio, pii-guard, or a second copy of the model weights.
"""
from __future__ import annotations

import re
import unicodedata
from itertools import accumulate

NATIVE_TYPES = frozenset({
    "PASSPORT", "CREDIT_CARD", "DRIVER_LICENSE", "INN", "SNILS", "MILITARY_ID", "BIRTH_CERTIFICATE", "OMS",
    "CITY", "COUNTRY", "DISTRICT", "EMAIL", "FIRST_NAME", "HOUSE", "IP_ADDRESS", "LAST_NAME", "MIDDLE_NAME",
    "PHONE", "REGION", "STREET", "URL",
})
_UNBRIDGED = frozenset({
    "FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "CITY", "COUNTRY", "DISTRICT", "REGION", "STREET", "HOUSE",
})
_WORD = re.compile(r'''[^\s(){}\[\]«»"“”‘’',;:!?]+|[(){}\[\]«»"“”‘’',;:!?]''')
_BRIDGE = frozenset(".,;:!?")
_BUDGET = 510
_OVERLAP = 128
_ALIGNMENT_ERROR = "Invalid RuBERT word-token alignment."


def words_with_offsets(text):
    """Split punctuation and boundary dots without normalizing Unicode/text."""
    result = []
    for match in _WORD.finditer(text):
        value = match.group()
        start = match.start()
        leading = len(value) - len(value.lstrip("."))
        trailing_start = max(leading, len(value.rstrip(".")))
        result.extend((".", start + i, start + i + 1) for i in range(leading))
        if leading < trailing_start:
            result.append((value[leading:trailing_start], start + leading, start + trailing_start))
        result.extend((".", start + i, start + i + 1) for i in range(trailing_start, len(value)))
    return result


def _label_inventory(runtime):
    labels = runtime.id2label
    expected = {"O"} | {f"{prefix}-{label}" for prefix in ("B", "I") for label in NATIVE_TYPES}
    if (not isinstance(labels, dict) or set(labels) != set(range(43))
            or any(type(key) is not int for key in labels) or set(labels.values()) != expected):
        raise ValueError("Invalid RuBERT decoder label inventory.")
    return [labels[index] for index in range(43)]


def _tokenize(tokenizer, words):
    encoding = tokenizer(
        words, is_split_into_words=True, add_special_tokens=False, truncation=False,
        return_attention_mask=False, return_token_type_ids=False,
    )
    ids = encoding["input_ids"]
    owners = encoding.word_ids(batch_index=0)
    if (not isinstance(ids, list) or not isinstance(owners, list) or len(ids) != len(owners)
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError(_ALIGNMENT_ERROR)
    counts = [0] * len(words)
    previous = -1
    for owner in owners:
        if type(owner) is not int or not 0 <= owner < len(words) or owner < previous:
            raise ValueError(_ALIGNMENT_ERROR)
        counts[owner] += 1
        previous = owner
    for word, count in zip(words, counts, strict=True):
        if count == 0 and not _bert_ignored_word(word):
            raise ValueError("RuBERT tokenizer omitted a visible word; refusing partial inference.")
    if max(counts) > _BUDGET:
        raise ValueError("RuBERT word exceeds the 510-subword window; refusing partial inference.")
    return ids, counts


def _bert_ignored_word(word):
    # The pinned fast BertNormalizer removes Unicode category C (including PDF
    # private-use bullets) and U+FFFD. Keep an O marker for an entirely removed
    # word; otherwise following IDs shift predictions to incorrect text offsets.
    return all(character == "\ufffd" or unicodedata.category(character).startswith("C") for character in word)


def _window_end(counts, start):
    total = 0
    end = start
    while end < len(counts) and total + counts[end] <= _BUDGET:
        total += counts[end]
        end += 1
    return end


def _margin_words(counts, threshold):
    total = 0
    consumed = 0
    for consumed, count in enumerate(counts, 1):
        total += count
        if total >= threshold:
            return consumed
    return consumed


def _ownership(counts, start, end, next_unwritten):
    window = counts[start:end]
    left = 0 if start == 0 else _margin_words(window, _OVERLAP // 2)
    right = len(window)
    if end < len(counts):
        right -= _margin_words(reversed(window), _OVERLAP // 2) - 1
    # A wide word can move the next window beyond the previous owned center.
    # Never omit a word merely because it lies within the left context margin.
    left = max(0, min(left, next_unwritten - start))
    return left, right


def _model_inputs(tokenizer, token_ids):
    import numpy as np

    inputs = tokenizer.build_inputs_with_special_tokens(token_ids)
    types = tokenizer.create_token_type_ids_from_sequences(token_ids)
    if (len(inputs) != len(token_ids) + 2 or not 2 <= len(inputs) <= 512
            or inputs[1:-1] != token_ids or len(types) != len(inputs)
            or any(type(item) is not int or item < 0 for item in inputs)
            or any(type(item) is not int or item != 0 for item in types)):
        raise ValueError("Invalid RuBERT special-token layout.")
    ids = np.asarray([inputs], dtype=np.int64)
    return {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}


def _predict_window(runtime, token_ids, first_positions, labels):
    import numpy as np

    inputs = _model_inputs(runtime.tokenizer, token_ids)
    logits = runtime.backend.run(inputs)
    expected = (1, len(token_ids) + 2, len(labels))
    if (not isinstance(logits, np.ndarray) or logits.shape != expected
            or logits.dtype not in (np.dtype("float16"), np.dtype("float32")) or not np.isfinite(logits).all()):
        raise ValueError("Invalid RuBERT decoder logits shape, dtype or values.")
    first_logits = logits[0, first_positions].astype(np.float64)
    winners = first_logits.argmax(axis=-1)
    probabilities = np.exp(first_logits - first_logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    scores = probabilities[np.arange(len(winners)), winners]
    return [(labels[int(winner)], float(score)) for winner, score in zip(winners, scores, strict=True)]


def _predict_words(runtime, ids, counts, labels):
    boundaries = [0, *accumulate(counts)]
    output = [None] * len(counts)
    start = 0
    next_unwritten = 0
    while start < len(counts):
        end = _window_end(counts, start)
        offset = boundaries[start]
        positions = [boundaries[index] - offset + 1 for index in range(start, end)]
        predicted = _predict_window(runtime, ids[offset:boundaries[end]], positions, labels)
        left, right = _ownership(counts, start, end, next_unwritten)
        output[start + left:start + right] = predicted[left:right]
        next_unwritten = max(next_unwritten, start + right)
        if end == len(counts):
            break
        overlap = _margin_words(reversed(counts[start:end]), _OVERLAP)
        start = max(end - overlap, start + 1)
    if any(item is None for item in output):
        raise ValueError("RuBERT window ownership omitted a word; refusing partial inference.")
    return output


def _bridge_punctuation(words, predicted):
    result = list(predicted)
    for index in range(1, len(words) - 1):
        if result[index][0] != "O" or not set(words[index]).issubset(_BRIDGE):
            continue
        previous, previous_score = result[index - 1]
        following, following_score = result[index + 1]
        label = previous[2:]
        if previous != "O" and following != "O" and label == following[2:] and label not in _UNBRIDGED:
            # A bridge has no model probability for this label: use the weaker
            # neighboring probability as conservative imputed confidence.
            result[index] = (f"I-{label}", min(previous_score, following_score))
    return result


def _entities(text, tokens, predicted):
    result = []
    active = None
    scores = []
    for (_, start, end), (tag, score) in zip(tokens, predicted, strict=True):
        label = None if tag == "O" else tag[2:]
        if active is not None and active["label"] != label:
            result.append({**active, "score": sum(scores) / len(scores), "text": text[active["start"]:active["end"]]})
            active = None
            scores = []
        if label is not None:
            if active is None:
                active = {"start": start, "end": end, "label": label}
            active["end"] = end
            scores.append(score)
    if active is not None:
        result.append({**active, "score": sum(scores) / len(scores), "text": text[active["start"]:active["end"]]})
    return result


def word_predict(runtime, text):
    """Return all native labels; fail on any truncated or unrepresentable input.

    Known BERT-ignored control-only words retain explicit O markers. Missing
    visible words fail rather than accepting misaligned or partial predictions.
    Uses argmax without a probability threshold. Scores are means of the winning
    first-subword softmax probabilities, with conservative punctuation imputation;
    they are not calibrated probabilities that an entire entity is correct.
    The caller must serialize access to the shared TensorRT backend.
    """
    if not isinstance(text, str):
        raise ValueError("RuBERT input must be a string.")
    tokens = words_with_offsets(text)
    if not tokens:
        return []
    labels = _label_inventory(runtime)
    words = [item[0] for item in tokens]
    ids, counts = _tokenize(runtime.tokenizer, words)
    active = [index for index, count in enumerate(counts) if count]
    predicted = [("O", 1.0)] * len(words)
    if active:
        visible = _predict_words(runtime, ids, [counts[index] for index in active], labels)
        for index, prediction in zip(active, visible, strict=True):
            predicted[index] = prediction
    return _entities(text, tokens, _bridge_punctuation(words, predicted))
