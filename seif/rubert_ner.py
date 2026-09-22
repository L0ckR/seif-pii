"""Pinned RuBERT TensorRT experiment using the published local PiiNER runtime.

The 21 native types remain available independently. The gateway profile maps
name parts to PERSON and six address components to LOCATION, then joins only
whitespace-adjacent spans of the same mapped type. This explicit coarse profile
is broader than geographic-name NER and never rewrites native predictions.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from threading import Lock

from seif.gliner_ner import GlinerSpan

MODEL_NAME = "lockR/rubert-base-pii-ner-tensorrt"
MODEL_REVISION = "73be581047bf123dac6505e7b3900ec292942296"
NATIVE_TYPES = frozenset({
    "PASSPORT", "CREDIT_CARD", "DRIVER_LICENSE", "INN", "SNILS", "MILITARY_ID", "BIRTH_CERTIFICATE", "OMS",
    "CITY", "COUNTRY", "DISTRICT", "EMAIL", "FIRST_NAME", "HOUSE", "IP_ADDRESS", "LAST_NAME", "MIDDLE_NAME",
    "PHONE", "REGION", "STREET", "URL",
})
PERSON_TYPES = frozenset({"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"})
LOCATION_TYPES = frozenset({"CITY", "COUNTRY", "DISTRICT", "REGION", "STREET", "HOUSE"})
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


def _gateway_from_validated(text, native):
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


def gateway_entities(text, native):
    """Map complete validated native output into the frozen gateway profile."""
    return _gateway_from_validated(text, validate_native(text, native))


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

    def __init__(self, runtime, *, runtime_metadata=None):
        self.runtime = runtime
        self._metadata = {} if runtime_metadata is None else deepcopy(runtime_metadata)
        self._model_lock = Lock()
        self._warmed = False

    @classmethod
    def from_local(cls, model_path, *, device="cuda", warmup=True):
        if device != "cuda" or type(warmup) is not bool:
            raise ValueError("RuBERT TensorRT requires device='cuda' and boolean warmup.")
        files = fingerprint_checkpoint(model_path)
        directory = Path(model_path).expanduser().resolve()
        runtime = _load_runtime(directory)
        _validate_runtime(runtime, directory)
        runtime.backend = _ValidatedBackend(runtime.backend)
        metadata = {
            "model": MODEL_NAME, "revision": MODEL_REVISION, "files_sha256": files,
            "backend": "trt-graph", "device": "cuda", "min_confidence": 0.3, "batch_size": 1,
            "max_tokens": 512, "overlap_tokens": 128, "native_types": sorted(NATIVE_TYPES),
            "person_labels": sorted(PERSON_TYPES), "location_labels": sorted(LOCATION_TYPES),
            "gateway_merge": "same mapped type, empty/whitespace gap, maximum constituent score; no punctuation expansion",
            "location_scope": "six address-component types, broader than geographic proper names",
            "native_predictions": "all21 labels retained before coarse gateway mapping",
            "precision": "published TensorRT FP16 engine with FP32 normalization accumulation; TF32 disabled at build",
            "packages": _package_versions(), "logits_shape_validation": "exact [batch, padded_tokens,43]",
        }
        analyzer = cls(runtime, runtime_metadata=metadata)
        if warmup:
            analyzer.warmup()
        return analyzer

    def warmup(self):
        with self._model_lock:
            if not self._warmed:
                self.runtime.warmup()
                self._warmed = True

    def predict_native(self, text):
        if not isinstance(text, str):
            raise ValueError("RuBERT input must be a string.")
        with self._model_lock:
            return validate_native(text, self.runtime.predict(text))

    def predict_both(self, text):
        native = self.predict_native(text)
        try:
            gateway = _gateway_from_validated(text, native)
        except ValueError as error:
            return {"native": native, "gateway": None, "gateway_error": str(error)}
        return {"native": native, "gateway": gateway, "gateway_error": None}

    def analyze(self, *, text, language, entities, score_threshold):
        if not isinstance(text, str) or language != "ru" or entities != ["PERSON", "LOCATION"] or score_threshold != 0.0:
            raise ValueError("Unsupported RuBERT analyzer request.")
        native = self.predict_native(text)
        return [GlinerSpan(**item) for item in _gateway_from_validated(text, native)]

    def metadata(self):
        result = deepcopy(self._metadata)
        result["warmup_complete"] = self._warmed
        result["graph_buckets_captured"] = sorted(getattr(self.runtime.backend, "_bucket_backends", {}))
        return result
