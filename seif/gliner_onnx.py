"""Pinned ONNX experiment with the native GLiNER2 processor and decoder.

Only learned encoder/boundary inference runs in ONNX Runtime. Torch remains a
CPU preprocessing/decoding dependency, and startup loads the matching native
checkpoint before replacing its two learned modules. Host NumPy I/O also means
CUDA inference includes copies between the two sessions. This is deliberately
an exact-preprocessing experiment, not a Torch-free deployment implementation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

from seif.gliner_ner import GlinerAnalyzer

MODEL_NAME = "DanKau/gliner2.5-multi-v1-onnx"
MODEL_REVISION = "481ad683a5420349c20f7ccc992efd25f9f3b809"
NATIVE_REVISION = "a221b77a8baf4a613b8f8652661d41fa10a5641e"
ONNX_FILES = {
    "onnx/encoder.onnx": "ab5f77ea6df6de0c8e5be3038d3bd55bb2dfe28b37dfbd2b1248cf17c6073579",
    "onnx/boundary.onnx": "131f64aa024ea46a991de880562ffc94c02f8fa7886b8afe03c55d927c9e33e2",
    "config.json": "dca03c3cfc6144e1c08409ac5be48377edfaea7bc48dbda5b57dd8fd9f5d4fbb",
    "gliner2_config.json": "96089e65559fbd5d4484063cd165c50522ba65954d9710e3f44459233cb083db",
    "tokenizer.json": "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a",
    "tokenizer_config.json": "db8ff95e236160a65fb6139babdf12dcaec4759bd0225af09b26eca8aa97d017",
}
NATIVE_FILES = {
    "config.json": "8b59a0f426a65859c89cd1ea850c3529c09aa3be3a6fafd8eddfdd17b1bf0146",
    "encoder_config/config.json": "fa4f9ef2903b5369ab172333aae4574e6a476511d7465845cf59f8360ee18716",
    "model.safetensors": "c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395",
    "tokenizer.json": "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a",
    "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
}
BOUNDARY_INPUTS = {"token_states", "text_mask", "query_states", "query_mask"}
BOUNDARY_OUTPUTS = {"pair_logits", "candidate_indices", "candidate_valid", "null_logits"}


def _verify_files(directory, expected, label):
    if not directory:
        raise ValueError(f"{label} requires an installed local checkpoint directory.")
    path = Path(directory).expanduser().resolve()
    for name, digest in expected.items():
        candidate = path / name
        if not candidate.is_file():
            raise ValueError(f"Missing pinned {label} file: {name}")
        with candidate.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Pinned {label} SHA256 mismatch: {name}")
    return path


def _validate_options(device, cpu_threads):
    if device not in {"cpu", "cuda"}:
        raise ValueError("ONNX device must be cpu or cuda.")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")


def _session(ort, path, *, device, cpu_threads, profile_dir):
    required = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if required not in ort.get_available_providers():
        raise RuntimeError(f"Requested ONNX provider is unavailable: {required}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_dir is not None:
        directory = Path(profile_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(directory / path.stem)
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, (required, {"device_id": 0, "use_tf32": 0}))
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    # ORT otherwise retries a runtime provider error on a fallback EP. CPU
    # shape/control ops within an explicitly CUDA-backed graph remain allowed.
    session.disable_fallback()
    if required not in session.get_providers():
        raise RuntimeError(f"ONNX session failed to activate requested provider: {required}")
    return session


def _validate_signatures(encoder, boundary):
    if {item.name for item in encoder.get_inputs()} != {"input_ids", "attention_mask"}:
        raise RuntimeError("Unsupported encoder ONNX input signature.")
    if len(encoder.get_outputs()) != 1:
        raise RuntimeError("Unsupported encoder ONNX output signature.")
    if {item.name for item in boundary.get_inputs()} != BOUNDARY_INPUTS:
        raise RuntimeError("Unsupported boundary ONNX input signature.")
    if {item.name for item in boundary.get_outputs()} != BOUNDARY_OUTPUTS:
        raise RuntimeError("Unsupported boundary ONNX output signature.")


def _numpy_input(tensor, dtype, name):
    if tensor.device.type != "cpu" or tensor.dtype != dtype:
        raise ValueError(f"ONNX shim requires CPU tensor of the expected dtype: {name}")
    return tensor.detach().contiguous().numpy()


def _check_boundary_arrays(arrays, text_mask, query_mask):
    import numpy as np

    pair, indices, valid, null = (arrays[name] for name in
                                ("pair_logits", "candidate_indices", "candidate_valid", "null_logits"))
    batch, queries = query_mask.shape
    expected = (batch, queries, 192)
    if (pair.shape != expected or indices.shape != (*expected, 2) or valid.shape != expected
            or null.shape != (batch, queries) or pair.dtype != np.float32 or null.dtype != np.float32
            or indices.dtype != np.int64 or valid.dtype != np.bool_):
        raise ValueError("Invalid ONNX boundary output shape or dtype.")
    if not np.isfinite(pair[valid]).all() or not np.isfinite(null[query_mask]).all():
        raise ValueError("Nonfinite ONNX boundary output for a valid candidate/query.")
    starts, ends = indices[..., 0], indices[..., 1]
    lengths = text_mask.sum(axis=1)[:, None, None]
    if np.any(valid & ((starts < 0) | (ends <= starts) | (ends > lengths))):
        raise ValueError("Invalid ONNX candidate word boundaries.")
    if np.any(valid & ~query_mask[..., None]):
        raise ValueError("ONNX returned candidates for a padded query.")


def _make_shims(encoder_session, boundary_session):
    """Import optional heavy dependencies only when constructing the backend."""
    import numpy as np
    import torch
    from gliner2.models.outputs import CandidateTensorBatch

    class Encoder(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask):
            ids = _numpy_input(input_ids, torch.int64, "input_ids")
            attention = _numpy_input(attention_mask, torch.int64, "attention_mask")
            if ids.ndim != 2 or ids.shape[0] != 1 or attention.shape != ids.shape:
                raise ValueError("ONNX encoder requires one equally shaped ID/mask sequence.")
            hidden, = encoder_session.run(None, {"input_ids": ids, "attention_mask": attention})
            if hidden.shape != (*ids.shape, 768) or hidden.dtype != np.float32 or not np.isfinite(hidden).all():
                raise ValueError("Invalid ONNX encoder output shape, dtype or values.")
            return SimpleNamespace(last_hidden_state=torch.from_numpy(hidden))

    class Boundary(torch.nn.Module):
        def forward(self, token_states, text_mask, query_states, query_mask, *, return_candidates=True):
            if return_candidates is not True:
                raise ValueError("ONNX adapter only supports entity candidate inference.")
            feed = {
                "token_states": _numpy_input(token_states, torch.float32, "token_states"),
                "text_mask": _numpy_input(text_mask, torch.bool, "text_mask"),
                "query_states": _numpy_input(query_states, torch.float32, "query_states"),
                "query_mask": _numpy_input(query_mask, torch.bool, "query_mask"),
            }
            if (token_states.ndim != 3 or token_states.shape[0] != 1 or token_states.shape[-1] != 768
                    or query_states.ndim != 3 or query_states.shape[0] != 1 or query_states.shape[-1] != 768
                    or text_mask.shape != token_states.shape[:2] or query_mask.shape != query_states.shape[:2]):
                raise ValueError("Invalid ONNX boundary input shapes.")
            arrays = dict(zip((item.name for item in boundary_session.get_outputs()),
                              boundary_session.run(None, feed), strict=True))
            _check_boundary_arrays(arrays, feed["text_mask"], feed["query_mask"])
            candidates = CandidateTensorBatch(
                indices=torch.from_numpy(arrays["candidate_indices"]), proposal_logits=None,
                pair_logits=torch.from_numpy(arrays["pair_logits"]),
                valid_mask=torch.from_numpy(arrays["candidate_valid"]), query_mask=query_mask,
            )
            # Adaptive thresholds are disabled in the pinned native config.
            # Abstention is enabled: its null logits MUST reach native decode.
            return SimpleNamespace(candidates=candidates, count_log_rates=None,
                                   null_logits=torch.from_numpy(arrays["null_logits"]))

    return Encoder(), Boundary()


def _session_metadata(session):
    return {
        "providers": session.get_providers(), "provider_options": session.get_provider_options(),
        "inputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in session.get_inputs()],
        "outputs": [{"name": item.name, "type": item.type, "shape": item.shape} for item in session.get_outputs()],
    }


class GlinerOnnxAnalyzer(GlinerAnalyzer):
    """Experimental selected schema (.8), preserving the native gateway API."""

    model_name = MODEL_NAME

    def __init__(self, extractor, *, sessions, runtime_metadata, profile_enabled=False):
        super().__init__(extractor, schema="described-names", threshold=0.8)
        self._sessions = sessions
        self._runtime_metadata = runtime_metadata
        self._profile_enabled = profile_enabled
        self._profile_files = None

    @classmethod
    def from_local(cls, onnx_path, native_path, *, device="cpu", cpu_threads=4, profile_dir=None):
        _validate_options(device, cpu_threads)
        onnx_path = _verify_files(onnx_path, ONNX_FILES, "ONNX")
        native_path = _verify_files(native_path, NATIVE_FILES, "native")
        if importlib.metadata.version("gliner2") != "2.0.0":
            raise RuntimeError("Exact ONNX preprocessing requires gliner2==2.0.0.")
        import onnxruntime as ort
        import torch
        from gliner2 import AutoExtractor

        # Importing Torch preloads the CUDA runtime shared by the pinned ORT
        # installation. Torch operations still remain on CPU for this adapter.
        torch.set_num_threads(cpu_threads)
        encoder = _session(ort, onnx_path / "onnx/encoder.onnx", device=device,
                           cpu_threads=cpu_threads, profile_dir=profile_dir)
        boundary = _session(ort, onnx_path / "onnx/boundary.onnx", device=device,
                            cpu_threads=cpu_threads, profile_dir=profile_dir)
        _validate_signatures(encoder, boundary)
        extractor = AutoExtractor.from_pretrained(str(native_path), map_location="cpu", quantize=False,
                                                 compile=False, local_files_only=True)
        if (extractor.architecture != "boundary" or extractor.config.token_pooling != "first"  # noqa: S105 -- pooling enum
                or extractor.boundary_settings.adaptive_threshold):
            raise RuntimeError("Native model settings are incompatible with the pinned ONNX export.")
        extractor.encoder, extractor.boundary_head = _make_shims(encoder, boundary)
        extractor.eval()
        metadata = {
            "model": MODEL_NAME, "revision": MODEL_REVISION, "native_revision": NATIVE_REVISION,
            "onnx_files_sha256": dict(ONNX_FILES), "native_files_sha256": dict(NATIVE_FILES),
            "schema": "described-names", "threshold": 0.8, "overlap_policy": "flat",
            "device": device, "dtype": "float32", "cpu_threads": cpu_threads, "inter_op_threads": 1,
            "torch_threads": torch.get_num_threads(), "cuda_tf32": False,
            "runtime": "ORT encoder + boundary; native GLiNER2 CPU Torch preprocessing and decoding",
            "io": "CPU NumPy inputs/outputs; CUDA includes host/device copies at both sessions",
            "native_checkpoint_required_at_startup": True,
            "classification_records_relations_supported": False,
            "encoder": _session_metadata(encoder), "boundary": _session_metadata(boundary),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("gliner2", "torch", "onnxruntime-gpu", "numpy", "transformers", "tokenizers")},
        }
        return cls(extractor, sessions=(encoder, boundary), runtime_metadata=metadata,
                   profile_enabled=profile_dir is not None)

    @classmethod
    def from_env(cls):
        raise RuntimeError("Experimental ONNX loading requires explicit pinned local paths via from_local().")

    def metadata(self):
        from copy import deepcopy

        return deepcopy(self._runtime_metadata)

    def finish_profiling(self):
        with self._model_lock:
            if self._profile_files is None:
                self._profile_files = ([session.end_profiling() for session in self._sessions]
                                       if self._profile_enabled else [])
            return list(self._profile_files)
