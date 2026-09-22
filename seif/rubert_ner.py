"""Pinned local RuBERT TensorRT with word decoding and complete PII transport.

The explicit published/person-location profile remains available for historical
experiments. The service defaults to word decoding and preserves every native
type through a lossless mapping of label names and original span boundaries.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from threading import Lock

from seif.gliner_ner import GlinerSpan
from seif.ner_contract import LEGACY_NER_TYPES, MAX_ENTITIES, NER_ENTITY_TYPES, max_entity_chars

MODEL_NAME = "lockR/rubert-base-pii-ner-tensorrt"
MODEL_REVISION = "73be581047bf123dac6505e7b3900ec292942296"
NATIVE_TYPES = frozenset({
    "PASSPORT", "CREDIT_CARD", "DRIVER_LICENSE", "INN", "SNILS", "MILITARY_ID", "BIRTH_CERTIFICATE", "OMS",
    "CITY", "COUNTRY", "DISTRICT", "EMAIL", "FIRST_NAME", "HOUSE", "IP_ADDRESS", "LAST_NAME", "MIDDLE_NAME",
    "PHONE", "REGION", "STREET", "URL",
})
PERSON_TYPES = frozenset({"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"})
LOCATION_TYPES = frozenset({"CITY", "COUNTRY", "DISTRICT", "REGION", "STREET", "HOUSE"})
NATIVE_MAPPING = {kind: "PERSON" if kind in PERSON_TYPES else "LOCATION" if kind in LOCATION_TYPES
                  else "CARD" if kind == "CREDIT_CARD" else kind for kind in NATIVE_TYPES}
PINNED_FILES = {
    "pii_ner.py": "4591fc8c5580c39dc5121ca649f4a2a4936276ea4c9e82ae144e5c467eb5a5ca",
    "trt_backend.py": "eafc144a9b698e94dfe155074c7a3ab1063ee7817ab1a285b69639db1c07b424",
    "model.fp16.engine": "f4d3f4eaa72ed8543b81ef4f6acf216f95be93fe4da8c3be27c8513cb9e80b35",
    "config.json": "f4eeba88d1503a56d629195c9307d42c1acf4ce53ab88fdaecfc754fd0745cf9",
    "runtime_config.json": "841420d76f5e7ec4ec7dd781b79987fe0f3facb25cc68896e7c88012f7af1b5d",
    "tokenizer.json": "f61eee9d0dc18f14c195321a37eadf9e5aeb3a323bc278dec689adeb89dec1b9",
    "tokenizer_config.json": "ffe381e31042ba94cb077bf7d5b9cee944dbd0c2a85079b095bea7e5a5a6058b",
}
INVALID_NATIVE = "Invalid RuBERT native output."
GATEWAY_LIMIT = "RuBERT gateway profile exceeds the span length/count contract."


def fingerprint_checkpoint(model_path):
    """Verify only pinned inference inputs, excluding generated files/caches."""
    if not model_path:
        raise ValueError("A pinned local RuBERT checkpoint directory is required.")
    directory = Path(model_path).expanduser().resolve()
    result = {}
    for name, expected in PINNED_FILES.items():
        candidate = directory / name
        if not candidate.is_file():
            raise ValueError(f"Missing pinned RuBERT file: {name}")
        with candidate.open("rb") as handle:
            result[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if result[name] != expected:
            raise ValueError(f"Pinned RuBERT SHA256 mismatch: {name}")
    return result


def validate_native(text, output):
    """Validate every native type before any gateway-only filtering occurs."""
    if not isinstance(text, str) or not isinstance(output, list):
        raise ValueError(INVALID_NATIVE)
    result = []
    for item in output:
        if not isinstance(item, dict) or set(item) != {"start", "end", "label", "score", "text"}:
            raise ValueError(INVALID_NATIVE)
        start, end, score, label = item["start"], item["end"], item["score"], item["label"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(label, str) or label not in NATIVE_TYPES
                or not isinstance(item["text"], str) or item["text"] != text[start:end]):
            raise ValueError(INVALID_NATIVE)
        result.append({**item, "score": float(score)})
    result.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    if any(right["start"] < left["end"] for left, right in zip(result, result[1:], strict=False)):
        # The published overlap resolver already returns non-overlapping spans.
        raise ValueError(INVALID_NATIVE)
    return result


def _legacy_gateway(text, native):
    result = []
    for span in native:
        label = span["label"]
        if label not in PERSON_TYPES | LOCATION_TYPES:
            continue
        kind = "PERSON" if label in PERSON_TYPES else "LOCATION"
        item = {"start": span["start"], "end": span["end"], "entity_type": kind, "score": span["score"]}
        if result:
            previous = result[-1]
            gap = text[previous["end"]:item["start"]]
            if previous["entity_type"] == kind and (not gap or gap.isspace()):
                previous["end"] = item["end"]
                previous["score"] = max(previous["score"], item["score"])
                continue
        result.append(item)
    if len(result) > 2048 or any(item["end"] - item["start"] > 200 for item in result):
        raise ValueError(GATEWAY_LIMIT)
    return result


def _gateway_from_validated(text, native, profile):
    if profile == "person-location":
        return _legacy_gateway(text, native)
    if profile != "native":
        raise ValueError("Unknown RuBERT gateway profile.")
    result = [{"start": item["start"], "end": item["end"], "entity_type": NATIVE_MAPPING[item["label"]],
               "score": item["score"]} for item in native]
    if len(result) > MAX_ENTITIES or any(
        item["end"] - item["start"] > max_entity_chars(item["entity_type"]) for item in result
    ):
        raise ValueError(GATEWAY_LIMIT)
    return result


def gateway_entities(text, native, *, profile="person-location"):
    """Validate all native predictions, then apply the explicitly selected profile."""
    return _gateway_from_validated(text, validate_native(text, native), profile)


class _ValidatedBackend:
    """Prevent published zip-based decoding from accepting truncated logits."""

    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def run(self, inputs):
        import numpy as np

        output = self.backend.run(inputs)
        expected = (*inputs["input_ids"].shape, 43)
        if (not isinstance(output, np.ndarray) or output.shape != expected
                or output.dtype not in (np.dtype("float16"), np.dtype("float32")) or not np.isfinite(output).all()):
            raise ValueError("Invalid RuBERT logits shape, dtype or values.")
        return output


def _load_runtime(directory):
    name = "_seif_pinned_rubert_" + MODEL_REVISION
    spec = importlib.util.spec_from_file_location(name, directory / "pii_ner.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load verified local RuBERT runtime.")
    module = importlib.util.module_from_spec(spec)
    # No sys.path or sys.modules mutation: the verified published runtime also
    # resolves its sibling TensorRT module explicitly through importlib.
    spec.loader.exec_module(module)
    return module.PiiNER(model_dir=directory, backend="trt-graph", min_confidence=0.3, batch_size=1)


def _validate_runtime(runtime, directory):
    expected = {int(key): value for key, value in json.loads((directory / "config.json").read_text())["id2label"].items()}
    inventory = {"O"} | {f"{prefix}-{kind}" for prefix in ("B", "I") for kind in NATIVE_TYPES}
    if (runtime.id2label != expected or set(expected.values()) != inventory or len(expected) != 43
            or runtime.backend_name != "trt-graph" or runtime.min_confidence != 0.3 or runtime.batch_size != 1):
        raise ValueError("Unexpected RuBERT label inventory or runtime configuration.")


def _package_versions():
    result = {}
    for name in ("torch", "numpy", "transformers", "tokenizers", "tensorrt-cu12", "tensorrt-cu12-bindings"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class RubertAnalyzer:
    """Presidio-compatible experimental analyzer; native predictions accessible."""

    model_name = MODEL_NAME

    def __init__(self, runtime, *, runtime_metadata=None, decoder="published", profile="person-location", batch_size=1):
        if decoder not in {"published", "word"} or profile not in {"person-location", "native"}:
            raise ValueError("Unknown RuBERT decoder or gateway profile.")
        self.runtime = runtime
        if type(batch_size) is not int or batch_size not in (1, 2, 4, 8, 16, 32) or (batch_size > 1 and decoder != "word"):
            raise ValueError("Batched RuBERT requires word decoding and a supported batch size.")
        self.batch_size = batch_size
        self.decoder, self.profile = decoder, profile
        self.supported_entities = NER_ENTITY_TYPES if profile == "native" else LEGACY_NER_TYPES
        self._metadata = {} if runtime_metadata is None else deepcopy(runtime_metadata)
        self._model_lock = Lock()
        self._warmed = False

    @classmethod
    def from_local(cls, model_path, *, device="cuda", warmup=True, decoder="word", profile="native", batch_size=1):  # noqa: PLR0913 - preserve existing explicit factory options
        if device != "cuda" or type(warmup) is not bool:
            raise ValueError("RuBERT TensorRT requires device='cuda' and boolean warmup.")
        files = fingerprint_checkpoint(model_path)
        directory = Path(model_path).expanduser().resolve()
        runtime = _load_runtime(directory)
        _validate_runtime(runtime, directory)
        runtime.backend = _ValidatedBackend(runtime.backend)
        if batch_size > 1:
            from seif.rubert_batch import install_batch_backend

            install_batch_backend(runtime, max_batch=batch_size)
        metadata = {
            "model": MODEL_NAME, "revision": MODEL_REVISION, "files_sha256": files,
            "backend": "trt-graph", "device": "cuda", "min_confidence": None if decoder == "word" else 0.3,
            "decoder": decoder, "gateway_profile": profile, "batch_size": batch_size,
            "max_tokens": 512, "overlap_tokens": 128, "native_types": sorted(NATIVE_TYPES),
            "person_labels": sorted(PERSON_TYPES), "location_labels": sorted(LOCATION_TYPES),
            "gateway_merge": ("none; original native boundaries" if profile == "native" else
                              "same mapped type, empty/whitespace gap, maximum constituent score"),
            "location_scope": "six address-component types, broader than geographic proper names",
            "native_predictions": "all21 labels retained before coarse gateway mapping",
            "precision": "published TensorRT FP16 engine with FP32 normalization accumulation; TF32 disabled at build",
            "packages": _package_versions(), "logits_shape_validation": "exact [batch, padded_tokens,43]",
        }
        analyzer = cls(runtime, runtime_metadata=metadata, decoder=decoder, profile=profile, batch_size=batch_size)
        if warmup:
            analyzer.warmup()
        return analyzer

    @classmethod
    def from_env(cls):
        try:
            cpu_threads = int(os.getenv("SEIF_RUBERT_CPU_THREADS", "4"))
        except ValueError:
            raise ValueError("SEIF_RUBERT_CPU_THREADS must be an integer between 1 and 32.") from None
        if not 1 <= cpu_threads <= 32:
            raise ValueError("SEIF_RUBERT_CPU_THREADS must be an integer between 1 and 32.")
        import torch

        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return cls.from_local(os.getenv("SEIF_RUBERT_MODEL_PATH"),
                              decoder=os.getenv("SEIF_RUBERT_DECODER", "word"),
                              profile=os.getenv("SEIF_RUBERT_PROFILE", "native"),
                              batch_size=int(os.getenv("SEIF_NER_BATCH_SIZE", "1")))

    def warmup(self):
        with self._model_lock:
            if not self._warmed:
                self.runtime.warmup()
                if self.batch_size > 1:
                    self._warmup_batches()
                self._warmed = True

    def _warmup_batches(self):
        import numpy as np

        tokenizer = self.runtime.tokenizer
        # Warm the short-document shapes before readiness. Longer shapes are
        # still supported, with bounded lazy capture/dynamic fallback.
        for size in (2, 4, 8, 16, 32):
            if size > self.batch_size:
                break
            for length in (32, 64):
                ids = np.full((size, length), tokenizer.pad_token_id, dtype=np.int64)
                ids[:, :2] = [tokenizer.cls_token_id, tokenizer.sep_token_id]
                attention = np.zeros_like(ids)
                attention[:, :2] = 1
                logits = self.runtime.backend.run({"input_ids": ids, "attention_mask": attention,
                                                   "token_type_ids": np.zeros_like(ids)})
                if logits.shape != (size, length, 43) or not np.isfinite(logits).all():
                    raise ValueError("Invalid RuBERT batch warmup logits.")

    def predict_native(self, text):
        if not isinstance(text, str):
            raise ValueError("RuBERT input must be a string.")
        with self._model_lock:
            if self.decoder == "word":
                from seif.rubert_decoder import word_predict

                output = word_predict(self.runtime, text)
            else:
                output = self.runtime.predict(text)
            return validate_native(text, output)

    def predict_both(self, text):
        native = self.predict_native(text)
        try:
            gateway = _gateway_from_validated(text, native, self.profile)
        except ValueError as error:
            return {"native": native, "gateway": None, "gateway_error": str(error)}
        return {"native": native, "gateway": gateway, "gateway_error": None}

    def predict_native_batch(self, texts):
        from seif.rubert_batch import word_predict_batch

        if (self.decoder != "word" or not isinstance(texts, list)
                or not 1 <= len(texts) <= self.batch_size or any(not isinstance(text, str) for text in texts)):
            raise ValueError("Unsupported RuBERT batch request.")
        with self._model_lock:
            outputs = word_predict_batch(self.runtime, texts, max_batch=self.batch_size)
            if not isinstance(outputs, list) or len(outputs) != len(texts):
                raise ValueError("Invalid RuBERT batch count.")
            return [validate_native(text, output) for text, output in zip(texts, outputs, strict=True)]

    def analyze_batch(self, *, texts, language, entities, score_threshold):
        if (language != "ru" or not isinstance(entities, list)
                or any(not isinstance(kind, str) or kind not in self.supported_entities for kind in entities)
                or score_threshold != 0.0):
            raise ValueError("Unsupported RuBERT analyzer request.")
        native = self.predict_native_batch(texts)
        return [[GlinerSpan(**item) for item in _gateway_from_validated(text, output, self.profile)
                 if item["entity_type"] in entities] for text, output in zip(texts, native, strict=True)]

    def analyze(self, *, text, language, entities, score_threshold):
        if (not isinstance(text, str) or language != "ru" or not isinstance(entities, list)
                or any(not isinstance(kind, str) or kind not in self.supported_entities for kind in entities)
                or score_threshold != 0.0):
            raise ValueError("Unsupported RuBERT analyzer request.")
        native = self.predict_native(text)
        return [GlinerSpan(**item) for item in _gateway_from_validated(text, native, self.profile)
                if item["entity_type"] in entities]

    def metadata(self):
        result = deepcopy(self._metadata)
        result["warmup_complete"] = self._warmed
        result["graph_buckets_captured"] = sorted(getattr(self.runtime.backend, "_bucket_backends", {}))
        result["batch_graph_shapes"] = getattr(self.runtime.backend, "batch_graph_shapes", [])
        return result
