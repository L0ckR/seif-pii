"""ONNX contract tests: fake sessions and tiny CPU tensors; no model downloads."""
from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import pytest

from seif.gliner_ner import CHUNK_OVERLAP, CHUNK_WORDS, SCHEMAS
from seif.gliner_onnx import (
    BOUNDARY_INPUTS,
    BOUNDARY_OUTPUTS,
    GlinerOnnxAnalyzer,
    _check_boundary_arrays,
    _make_shims,
    _session,
    _validate_options,
    _validate_signatures,
    _verify_files,
)


def test_pinned_files_reject_changes_and_ignore_generated_cache(tmp_path):
    config = tmp_path / "config.json"
    config.write_bytes(b"pinned configuration")
    expected = {config.name: hashlib.sha256(config.read_bytes()).hexdigest()}
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "generated.pyc").write_bytes(b"not model input")
    assert _verify_files(tmp_path, expected, "native") == tmp_path
    config.write_bytes(b"different configuration")
    with pytest.raises(ValueError, match="SHA256 mismatch: config.json"):
        _verify_files(tmp_path, expected, "native")
    config.unlink()
    with pytest.raises(ValueError, match="Missing pinned"):
        _verify_files(tmp_path, expected, "native")


@pytest.mark.parametrize(("device", "threads"), [("gpu", 4), ("cuda:0", 4), ("cpu", True), ("cpu", 0),
                                                ("cpu", -1), ("cpu", 4.0)])
def test_invalid_runtime_configuration_rejected(device, threads):
    with pytest.raises(ValueError):
        _validate_options(device, threads)


class FakeOrt:
    SessionOptions = SimpleNamespace
    ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")

    def __init__(self, available, active=None):
        self.available = available
        self.active = available if active is None else active
        self.calls = []

    def get_available_providers(self):
        return self.available

    def InferenceSession(self, path, *, sess_options, providers):
        session = SimpleNamespace(get_providers=lambda: self.active, disable_fallback=self.disable_fallback)
        self.calls.append((path, sess_options, providers))
        return session

    def disable_fallback(self):
        self.fallback_disabled = True


@pytest.mark.parametrize(("available", "active"), [(["CPUExecutionProvider"], None),
    (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"])])
def test_cuda_unavailable_or_silent_cpu_fallback_rejected(tmp_path, available, active):
    ort = FakeOrt(available, active)
    with pytest.raises(RuntimeError, match="provider"):
        _session(ort, tmp_path / "encoder.onnx", device="cuda", cpu_threads=4, profile_dir=None)


def test_cuda_allows_cpu_shape_ops_but_disables_tf32_and_runtime_fallback(tmp_path):
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    _session(ort, tmp_path / "encoder.onnx", device="cuda", cpu_threads=4, profile_dir=tmp_path / "profiles")
    _, options, providers = ort.calls[0]
    assert providers == [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"]
    assert options.intra_op_num_threads == 4
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == "sequential"
    assert options.enable_profiling
    assert options.profile_file_prefix.endswith("profiles/encoder")
    assert ort.fallback_disabled


def test_cpu_session_never_requests_cuda(tmp_path):
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"])
    _session(ort, tmp_path / "encoder.onnx", device="cpu", cpu_threads=4, profile_dir=None)
    assert ort.calls[0][2] == ["CPUExecutionProvider"]


class FakeSession:
    def __init__(self, inputs, outputs, arrays=None):
        self.inputs = [SimpleNamespace(name=name) for name in inputs]
        self.outputs = [SimpleNamespace(name=name) for name in outputs]
        self.arrays = arrays
        self.feeds = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, names, feed):
        assert names is None
        self.feeds.append(feed)
        return [self.arrays[item.name] for item in self.outputs]


def test_graph_contract_accepts_actual_hidden_states_name_and_rejects_missing_null_gate():
    encoder = FakeSession(["input_ids", "attention_mask"], ["hidden_states"])
    boundary = FakeSession(BOUNDARY_INPUTS, BOUNDARY_OUTPUTS)
    _validate_signatures(encoder, boundary)
    boundary.outputs = [item for item in boundary.outputs if item.name != "null_logits"]
    with pytest.raises(RuntimeError, match="boundary ONNX output"):
        _validate_signatures(encoder, boundary)


@pytest.fixture
def numeric():
    # These are optional backend dependencies, absent from the service env.
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("gliner2")
    return np, torch


def boundary_arrays(np):
    indices = np.full((1, 3, 192, 2), -1, dtype=np.int64)
    valid = np.zeros((1, 3, 192), dtype=np.bool_)
    pair = np.full((1, 3, 192), -np.inf, dtype=np.float32)
    # Two disjoint .85 candidates have more total score than overlapping .95.
    indices[0, 0, :3] = [[0, 3], [0, 1], [1, 3]]
    valid[0, 0, :3] = True
    pair[0, 0, :3] = np.log(np.array([.95, .85, .85]) / np.array([.05, .15, .15]))
    return {"candidate_indices": indices, "candidate_valid": valid, "pair_logits": pair,
            "null_logits": np.array([[2.0, -3.0, 0.0]], dtype=np.float32)}


def test_shims_keep_exact_ids_states_null_logits_and_native_weighted_overlap(numeric):
    np, torch = numeric
    from gliner2.models.boundary.engine import _resolve_flat_spans

    hidden = np.arange(1 * 5 * 768, dtype=np.float32).reshape(1, 5, 768)
    encoder_session = FakeSession(["input_ids", "attention_mask"], ["hidden_states"], {"hidden_states": hidden})
    arrays = boundary_arrays(np)
    boundary_session = FakeSession(BOUNDARY_INPUTS, sorted(BOUNDARY_OUTPUTS, reverse=True), arrays)
    encoder, boundary = _make_shims(encoder_session, boundary_session)
    ids = torch.tensor([[9, 7, 42, 4, 0]], dtype=torch.int64)
    attention = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.int64)
    output = encoder(input_ids=ids, attention_mask=attention)
    assert torch.equal(output.last_hidden_state, torch.from_numpy(hidden))
    np.testing.assert_array_equal(encoder_session.feeds[0]["input_ids"], ids.numpy())
    np.testing.assert_array_equal(encoder_session.feeds[0]["attention_mask"], attention.numpy())
    result = boundary(output.last_hidden_state[:, :3], torch.ones((1, 3), dtype=torch.bool),
                      output.last_hidden_state[:, :3], torch.ones((1, 3), dtype=torch.bool))
    assert result.count_log_rates is None
    assert torch.equal(result.null_logits, torch.tensor([[2., -3., 0.]]))
    assert torch.equal(result.candidates.indices, torch.from_numpy(arrays["candidate_indices"]))
    probabilities = torch.sigmoid(result.candidates.pair_logits[0, 0, :3])
    resolved = _resolve_flat_spans([(float(p), int(span[0]), int(span[1])) for p, span in
                                   zip(probabilities, result.candidates.indices[0, 0, :3], strict=True)])
    assert [(start, end) for _, start, end in resolved] == [(0, 1), (1, 3)]
    assert torch.sigmoid(result.null_logits)[0, 0] > .5  # native abstention must see this


@pytest.mark.parametrize("failure", ["nan", "end", "reverse", "float_indices", "pair_shape", "null_shape", "padded_query"])
def test_boundary_invalid_outputs_reject_instead_of_dropping_spans(numeric, failure):
    np, _ = numeric
    arrays = boundary_arrays(np)
    if failure == "nan":
        arrays["pair_logits"][0, 0, 0] = np.nan
    elif failure == "end":
        arrays["candidate_indices"][0, 0, 0, 1] = 4
    elif failure == "reverse":
        arrays["candidate_indices"][0, 0, 0] = [2, 1]
    elif failure == "float_indices":
        arrays["candidate_indices"] = arrays["candidate_indices"].astype(np.float32)
    elif failure == "pair_shape":
        arrays["pair_logits"] = arrays["pair_logits"][:, :, :3]
    elif failure == "null_shape":
        arrays["null_logits"] = arrays["null_logits"][:, :1]
    text_mask = np.ones((1, 3), dtype=np.bool_)
    query_mask = np.ones((1, 3), dtype=np.bool_)
    if failure == "padded_query":
        query_mask[0, 0] = False
    with pytest.raises(ValueError):
        _check_boundary_arrays(arrays, text_mask, query_mask)


def test_invalid_padding_not_mistaken_for_a_real_nonfinite_candidate(numeric):
    np, _ = numeric
    arrays = boundary_arrays(np)
    assert not np.isfinite(arrays["pair_logits"][0, 0, -1])
    _check_boundary_arrays(arrays, np.ones((1, 3), dtype=np.bool_), np.ones((1, 3), dtype=np.bool_))


@pytest.mark.parametrize("pool_size", [0, 1, 2, 33, 40, 73, 191, 192])
def test_dynamic_candidate_pool_keeps_short_input_candidates(numeric, pool_size):
    np, torch = numeric
    arrays = boundary_arrays(np)
    for name in ("pair_logits", "candidate_indices", "candidate_valid"):
        arrays[name] = arrays[name][:, :, :pool_size]
    session = FakeSession(BOUNDARY_INPUTS, sorted(BOUNDARY_OUTPUTS), arrays)
    _, boundary = _make_shims(None, session)
    result = boundary(torch.zeros((1, 3, 768)), torch.ones((1, 3), dtype=torch.bool),
                      torch.zeros((1, 3, 768)), torch.ones((1, 3), dtype=torch.bool))
    assert result.candidates.indices.shape == (1, 3, pool_size, 2)
    assert torch.equal(result.candidates.indices, torch.from_numpy(arrays["candidate_indices"]))
    assert torch.equal(result.candidates.valid_mask, torch.from_numpy(arrays["candidate_valid"]))
    assert torch.equal(result.candidates.pair_logits, torch.from_numpy(arrays["pair_logits"]))
    assert torch.equal(result.null_logits, torch.from_numpy(arrays["null_logits"]))


def test_oversized_pool_rejects_even_when_all_output_shapes_agree(numeric):
    np, _ = numeric
    arrays = {"candidate_indices": np.zeros((1, 3, 193, 2), dtype=np.int64),
              "candidate_valid": np.zeros((1, 3, 193), dtype=np.bool_),
              "pair_logits": np.zeros((1, 3, 193), dtype=np.float32),
              "null_logits": np.zeros((1, 3), dtype=np.float32)}
    with pytest.raises(ValueError, match="pool dimension or budget"):
        _check_boundary_arrays(arrays, np.ones((1, 3), dtype=np.bool_), np.ones((1, 3), dtype=np.bool_))


def test_export_runtime_failure_propagates_without_empty_prediction_or_native_fallback(numeric):
    _, torch = numeric

    class BrokenSession:
        def get_outputs(self):
            return [SimpleNamespace(name=name) for name in BOUNDARY_OUTPUTS]

        def run(self, *_):
            raise RuntimeError("export cannot broadcast Where_14")

    _, boundary = _make_shims(None, BrokenSession())
    with pytest.raises(RuntimeError, match="cannot broadcast"):
        boundary(torch.zeros((1, 1, 768)), torch.ones((1, 1), dtype=torch.bool),
                 torch.zeros((1, 3, 768)), torch.ones((1, 3), dtype=torch.bool))


@pytest.mark.parametrize("malformed", ["float64", "nan", "width"])
def test_invalid_encoder_outputs_fail_closed(numeric, malformed):
    np, torch = numeric
    hidden = np.ones((1, 2, 768), dtype=np.float32)
    if malformed == "float64":
        hidden = hidden.astype(np.float64)
    elif malformed == "nan":
        hidden[0, 0, 0] = np.nan
    else:
        hidden = hidden[:, :, :767]
    encoder, _ = _make_shims(FakeSession([], ["hidden_states"], {"hidden_states": hidden}), None)
    with pytest.raises(ValueError, match="encoder output"):
        encoder(input_ids=torch.ones((1, 2), dtype=torch.int64), attention_mask=torch.ones((1, 2), dtype=torch.int64))


class FakeExtractor:
    def __init__(self):
        self.processor = SimpleNamespace(word_splitter=lambda text, **_: re.finditer(r"\S+", text))
        self.calls = []

    def extract_entities(self, text, labels, **options):
        self.calls.append((text, labels, options))
        start = text.index("Иван")
        return {"entities": {"person": [{"text": "Иван", "start": start, "end": start + 4, "confidence": .91}],
                             "location": [], "organization": []}}

    extract_entities_long = extract_entities


@pytest.mark.parametrize("prefix", [" 😀 е\u0301 ", "слово " * (CHUNK_WORDS + 1)])
def test_adapter_inherits_exact_unicode_and_long_chunk_contract(prefix):
    extractor = FakeExtractor()
    adapter = GlinerOnnxAnalyzer(extractor, sessions=(), runtime_metadata={})
    text = prefix + "Иван\n"
    result = adapter.analyze(text=text, language="ru", entities=["PERSON", "LOCATION"], score_threshold=0.0)
    assert text[result[0].start:result[0].end] == "Иван"
    original, labels, options = extractor.calls[0]
    assert original == text
    assert labels == SCHEMAS["described-names"]
    assert options["threshold"] == .8 and options["overlap_policy"] == "flat"
    if len(prefix) > 100:
        assert options["chunk_size"] == CHUNK_WORDS
        assert options["chunk_overlap"] == CHUNK_OVERLAP
        assert options["batch_size"] == 1
    else:
        assert options["max_len"] == CHUNK_WORDS + 1


def test_profile_finish_is_idempotent_and_metadata_copy_cannot_change_record():
    paths = iter(["encoder.json", "boundary.json"])
    session = SimpleNamespace(end_profiling=lambda: next(paths))
    adapter = GlinerOnnxAnalyzer(None, sessions=(session, session), runtime_metadata={"nested": {"x": 1}},
                                 profile_enabled=True)
    returned = adapter.metadata()
    returned["nested"]["x"] = 7
    assert adapter.metadata() == {"nested": {"x": 1}}
    assert adapter.finish_profiling() == ["encoder.json", "boundary.json"]
    assert adapter.finish_profiling() == ["encoder.json", "boundary.json"]
