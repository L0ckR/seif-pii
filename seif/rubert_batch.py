"""Opt-in bounded batching of the corrected RuBERT word decoder.

Documents retain their original tokenization, overlapping-window ownership and
Unicode offsets. Only independent model windows are grouped by padded length.
The caller must serialize calls and installation with the analyzer's model lock.
No document, prediction or input-dependent cache survives a call.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import accumulate

from seif import rubert_decoder as decoder

LENGTH_BUCKETS = (32, 64, 128, 256, 512)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)
MAX_DOCUMENTS = 32
MAX_GRAPH_CONTEXTS = 25


def _validate_batch_size(max_batch):
    if type(max_batch) is not int or max_batch not in BATCH_BUCKETS:
        raise ValueError("RuBERT max_batch must be one of 1, 2, 4, 8, 16, 32.")


@dataclass
class _Document:
    text: str
    tokens: list
    words: list
    active: list = field(default_factory=list)
    visible: list = field(default_factory=list)


@dataclass
class _Window:
    document: _Document
    token_ids: list
    positions: list
    start: int
    left: int
    right: int
    predicted: list | None = None


def _document_windows(runtime, text):
    tokens = decoder.words_with_offsets(text)
    words = [item[0] for item in tokens]
    document = _Document(text, tokens, words)
    if not tokens:
        return document, []
    ids, counts = decoder._tokenize(runtime.tokenizer, words)
    document.active = [index for index, count in enumerate(counts) if count]
    counts = [counts[index] for index in document.active]
    document.visible = [None] * len(counts)
    boundaries = [0, *accumulate(counts)]
    windows = []
    start = next_unwritten = 0
    while start < len(counts):
        end = decoder._window_end(counts, start)
        offset = boundaries[start]
        positions = [boundaries[index] - offset + 1 for index in range(start, end)]
        left, right = decoder._ownership(counts, start, end, next_unwritten)
        windows.append(_Window(document, ids[offset:boundaries[end]], positions, start, left, right))
        next_unwritten = max(next_unwritten, start + right)
        if end == len(counts):
            break
        overlap = decoder._margin_words(reversed(counts[start:end]), decoder._OVERLAP)
        start = max(end - overlap, start + 1)
    return document, windows


def _batch_inputs(tokenizer, windows, length):
    import numpy as np

    size = next(bucket for bucket in BATCH_BUCKETS if bucket >= len(windows))
    pad = tokenizer.pad_token_id
    if type(pad) is not int or pad < 0:
        raise ValueError("Invalid RuBERT pad token.")
    ids = np.full((size, length), pad, dtype=np.int64)
    attention = np.zeros_like(ids)
    types = np.zeros_like(ids)
    for row in range(size):
        # Valid dummy rows keep padded batch slots numerically well-defined.
        tokens = windows[row].token_ids if row < len(windows) else []
        item = decoder._model_inputs(tokenizer, tokens)
        count = item["input_ids"].shape[1]
        ids[row, :count] = item["input_ids"][0]
        attention[row, :count] = 1
    return {"input_ids": ids, "attention_mask": attention, "token_type_ids": types}


def _predict_batch(runtime, windows, length, labels):
    import numpy as np

    inputs = _batch_inputs(runtime.tokenizer, windows, length)
    logits = runtime.backend.run(inputs)
    expected = (*inputs["input_ids"].shape, len(labels))
    if (not isinstance(logits, np.ndarray) or logits.shape != expected
            or logits.dtype not in (np.dtype("float16"), np.dtype("float32")) or not np.isfinite(logits).all()):
        raise ValueError("Invalid RuBERT batch logits shape, dtype or values.")
    for row, window in enumerate(windows):
        first_logits = logits[row, window.positions].astype(np.float64)
        winners = first_logits.argmax(axis=-1)
        probabilities = np.exp(first_logits - first_logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        scores = probabilities[np.arange(len(winners)), winners]
        window.predicted = [(labels[int(winner)], float(score))
                            for winner, score in zip(winners, scores, strict=True)]


def _finish(document):
    if any(item is None for item in document.visible):
        raise ValueError("RuBERT batch ownership omitted a word; refusing partial inference.")
    predicted = [("O", 1.0)] * len(document.words)
    for index, prediction in zip(document.active, document.visible, strict=True):
        predicted[index] = prediction
    return decoder._entities(document.text, document.tokens, decoder._bridge_punctuation(document.words, predicted))


def word_predict_batch(runtime, texts, *, max_batch=8):
    """Return one native prediction list per input, with at most 32 documents.

    Model calls use power-of-two batch sizes and fixed length buckets. Replaying
    window ownership in original order preserves the sequential decoder even
    when differently sized windows were inferred in a different order.
    """
    _validate_batch_size(max_batch)
    if (not isinstance(texts, list) or len(texts) > MAX_DOCUMENTS
            or any(not isinstance(text, str) for text in texts)):
        raise ValueError("RuBERT batch requires a list of at most 32 strings.")
    documents, windows = [], []
    for text in texts:
        document, prepared = _document_windows(runtime, text)
        documents.append(document)
        windows.extend(prepared)
    labels = decoder._label_inventory(runtime) if any(document.tokens for document in documents) else []
    if not windows:
        return [[] for _ in documents]
    grouped = defaultdict(list)
    for window in windows:
        bucket = next(length for length in LENGTH_BUCKETS if length >= len(window.token_ids) + 2)
        grouped[bucket].append(window)
    for length, group in grouped.items():
        for start in range(0, len(group), max_batch):
            _predict_batch(runtime, group[start:start + max_batch], length, labels)
    for window in windows:
        if window.predicted is None:
            raise ValueError("RuBERT batch window has no predictions.")
        window.document.visible[window.start + window.left:window.start + window.right] = (
            window.predicted[window.left:window.right]
        )
    return [_finish(document) for document in documents]


class BatchGraphBackend:
    """A capped collection of fixed-shape contexts sharing verified TRT weights.

    Batch-one calls retain the original graph implementation. Additional graph
    contexts own separate streams and reusable buffers, but calls still require
    the analyzer lock. Once the cap is reached, uncaptured shapes use the pinned
    dynamic fallback. The cap bounds retention without evicting live CUDA graphs.
    """

    def __init__(self, backend, *, max_batch=8, max_contexts=12):
        _validate_batch_size(max_batch)
        if type(max_contexts) is not int or not 0 <= max_contexts <= MAX_GRAPH_CONTEXTS:
            raise ValueError("RuBERT batch graph context cap must be between 0 and 25.")
        original = getattr(backend, "backend", backend)
        if not all(hasattr(original, name) for name in ("fallback", "_engine_path", "_device_index")):
            raise ValueError("RuBERT batch graphs require the pinned TensorRT graph backend.")
        self.backend = backend
        self.original = original
        self.max_batch, self.max_contexts = max_batch, max_contexts
        self.contexts = {}

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def batch_graph_shapes(self):
        return sorted(self.contexts)

    def run(self, inputs):
        shape = inputs["input_ids"].shape
        if len(shape) != 2:
            raise ValueError("RuBERT batch inputs require two dimensions.")
        batch, length = shape
        if batch not in BATCH_BUCKETS or batch > self.max_batch:
            raise ValueError("RuBERT batch exceeds the configured graph profile.")
        if batch == 1:
            return self.backend.run(inputs)
        if length not in LENGTH_BUCKETS:
            raise ValueError("RuBERT batched sequence requires a fixed length bucket.")
        if shape not in self.contexts:
            if len(self.contexts) >= self.max_contexts:
                return self.backend.run(inputs)
            shared = self.original.fallback
            self.contexts[shape] = type(shared)(self.original._engine_path,
                                              device=self.original._device_index, cuda_graph=True, _shared=shared)
        return self.contexts[shape].run(inputs)


def install_batch_backend(runtime, *, max_batch=8, max_contexts=12):
    """Install once after verification; lazy capture occurs under the caller lock."""
    if isinstance(runtime.backend, BatchGraphBackend):
        if (runtime.backend.max_batch, runtime.backend.max_contexts) != (max_batch, max_contexts):
            raise ValueError("RuBERT batch backend is already installed with another profile.")
        return runtime.backend
    backend = BatchGraphBackend(runtime.backend, max_batch=max_batch, max_contexts=max_contexts)
    runtime.backend = backend
    return backend
