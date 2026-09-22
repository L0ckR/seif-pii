"""Local pinned LFM PII inference with the publisher's unchanged decoders.

This experimental wrapper preserves the checkpoint's forty native entity types.
It neither assigns probability scores nor maps labels to the gateway taxonomy.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from threading import Lock

MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M-PII-Detector"
MODEL_REVISION = "b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2"
MAX_TOKENS = 2048
NATIVE_TYPES = frozenset((
    "contact.address", "contact.email", "contact.ip_address", "contact.phone", "contact.postal_code",
    "credential.api_key", "credential.connection_string", "credential.jwt", "credential.password", "credential.private_key",
    "developer.device_id", "developer.login_credentials", "device.imei", "device.mac_address",
    "financial.amount", "financial.bank_account", "financial.credit_card", "financial.crypto_wallet",
    "financial.iban", "financial.swift_bic", "healthcare.condition", "healthcare.health_plan_id",
    "healthcare.medical_record", "healthcare.medication", "identity.date_of_birth", "identity.drivers_license",
    "identity.national_id", "identity.passport", "identity.person_name", "identity.ssn", "identity.tax_id",
    "legal.case_number", "location.gps_coordinates", "online.url", "online.username", "org.company_name",
    "special.health_status", "special.orientation", "special.political", "special.religion",
))
PINNED_FILES = {
    "config.json": "ced1b2917ec78c766b309233a38dd3351b6bd1cb4ba22c18363ce47c190679ac",
    "tokenizer.json": "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711",
    "tokenizer_config.json": "2892f908f7cc397e412a92be1526234079542d139b3c2bc7e1b06921b6f55cd0",
    "modeling_phase2_tc.py": "e48f084a44f25a43389a153dba6c94d53cb8f8fa4afd9f95b36ea697bf24d863",
    "pii_hybrid_decode.py": "798ee81182e1a5a68307d01e8bb350bcd36fa68ebe68e58bb5bacdc2a6c95ed5",
    "context_cued.py": "2a65a71e7625267b6dae4afd9a3256faa02ec3a724a754622f5c295d6252c33c",
    "model.safetensors": "fbfec8b59db250a1d35b4ddc0d73571777f7088946ee22a5d7962e37c02ea6a8",
}
_MODULE_LOCK = Lock()


def _validate_schema(config):
    labels = getattr(config, "id2label", None)
    if not isinstance(labels, dict):
        raise ValueError("LFM checkpoint must expose its trained BIOES label mapping")
    expected = {0: "O"}
    for index, kind in enumerate(sorted(NATIVE_TYPES)):
        expected.update({4 * index + n + 1: prefix + "-" + kind for n, prefix in enumerate(("B", "I", "E", "S"))})
    if labels != expected or getattr(config, "num_labels", None) != 161:
        raise ValueError("LFM checkpoint requires the pinned 161-label BIOES schema, not label_schema.json")


def fingerprint_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("LFM requires an installed local checkpoint directory")
    hashes = {}
    for name, expected in PINNED_FILES.items():
        with (path / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if hashes[name] != expected:
            raise ValueError(f"Pinned LFM checkpoint file differs: {name}")
    return hashes


def _load_module(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path.resolve():
            raise RuntimeError("A different LFM helper is already loaded")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned LFM helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _context_ready(context):
    if (sys.modules.get("context_cued") is not context
            or not callable(getattr(context, "context_cued_spans", None))
            or not callable(getattr(context, "group_b_cue_spans", None))
            or not isinstance(getattr(context, "CONTEXT_TYPES", None), (set, frozenset))
            or not context.CONTEXT_TYPES or not context.CONTEXT_TYPES <= NATIVE_TYPES):
        raise RuntimeError("Both pinned context-cued decoding layers must remain available")


def _load_helpers(path):
    with _MODULE_LOCK:
        context = _load_module("context_cued", path / "context_cued.py")
        _context_ready(context)
        decoder = _load_module("_seif_lfm_hybrid_" + MODEL_REVISION, path / "pii_hybrid_decode.py")
    if not callable(getattr(decoder, "model_spans", None)) or not callable(getattr(decoder, "hybrid_spans", None)):
        raise RuntimeError("Required official LFM decoders are missing")
    return decoder, context


def _validate_spans(spans, text):
    if not isinstance(spans, list):
        raise ValueError("LFM decoder must return a list of spans")
    result = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("Malformed LFM decoder span")
        start, end, kind = span.get("start"), span.get("end"), span.get("type")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(kind, str) or kind not in NATIVE_TYPES
                or ("text" in span and span["text"] != text[start:end])):
            raise ValueError("LFM decoder returned invalid original-text offsets or native type")
        result.append({"start": start, "end": end, "type": kind})
    return result


class _TokenizerGuard:
    def __init__(self, tokenizer, count):
        self.tokenizer, self.count = tokenizer, count

    def __call__(self, text, **kwargs):
        encoded = self.tokenizer(text, **kwargs)
        if (kwargs.get("max_length") != MAX_TOKENS or encoded["input_ids"].shape != (1, self.count)
                or encoded["offset_mapping"].shape != (1, self.count, 2)):
            raise ValueError("LFM helper truncated input or returned inconsistent tokenizer dimensions")
        for start, end in encoded["offset_mapping"][0].tolist():
            if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text):
                raise ValueError("LFM tokenizer offsets do not index the original input")
        return encoded


class _ModelGuard:
    def __init__(self, model, count):
        self.model, self.count = model, count
        self.device, self.config = model.device, model.config

    def __call__(self, **kwargs):
        output = self.model(**kwargs)
        logits = output.logits
        if logits.shape != (1, self.count, 161) or not logits.isfinite().all().item():
            raise ValueError("LFM logits have invalid shape or non-finite values")
        return output


class LfmPiiAnalyzer:
    """One neural inference per input, then raw and official hybrid decoding."""

    def __init__(self, tokenizer, model, decoder, context, *, metadata=None):
        _validate_schema(model.config)
        _context_ready(context)
        self.tokenizer, self.model = tokenizer, model
        self.decoder, self.context = decoder, context
        self._metadata = metadata or {}
        self._lock = Lock()

    @classmethod
    def from_local(cls, model_path, *, device="cpu"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("LFM device must be cpu or cuda")
        path = Path(model_path).expanduser().resolve()
        hashes = fingerprint_checkpoint(path)
        decoder, context = _load_helpers(path)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        if not tokenizer.is_fast:
            raise ValueError("LFM requires the pinned fast tokenizer for original Unicode offsets")
        model = AutoModelForTokenClassification.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=True, dtype=torch.float32,
        ).to(device).eval()
        metadata = {"model": MODEL_ID, "revision": MODEL_REVISION, "file_sha256": hashes,
                    "device": str(model.device), "dtype": str(next(model.parameters()).dtype),
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "tokenizers", "safetensors")},
                    "decoder": "Official model_spans once, followed by unchanged hybrid_spans.",
                    "max_tokens_including_special": MAX_TOKENS, "label_count": 161,
                    "native_types": sorted(NATIVE_TYPES), "scores": "No confidence scores produced."}
        return cls(tokenizer, model, decoder, context, metadata=metadata)

    def metadata(self):
        return json.loads(json.dumps(self._metadata))

    def token_count(self, text):
        if not isinstance(text, str):
            raise ValueError("LFM input must be text")
        ids = self.tokenizer(text, add_special_tokens=True, truncation=False,
                             return_attention_mask=False)["input_ids"]
        if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("LFM tokenizer returned malformed untruncated token IDs")
        return len(ids)

    def predict_both(self, text):
        with self._lock:
            count = self.token_count(text)
            if count > MAX_TOKENS:
                raise ValueError("LFM input exceeds 2048 tokens; refusing silent truncation")
            _context_ready(self.context)
            _validate_schema(self.model.config)
            raw = self.decoder.model_spans(text, _TokenizerGuard(self.tokenizer, count), _ModelGuard(self.model, count))
            raw = _validate_spans(raw, text)
            hybrid = self.decoder.hybrid_spans(text, [dict(span) for span in raw])
            hybrid = _validate_spans(hybrid, text)
            _context_ready(self.context)
            return {"raw": raw, "hybrid": hybrid}
