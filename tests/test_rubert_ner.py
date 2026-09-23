"""Synthetic contract checks for the optional RuBERT adapter; no GPU/model load."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.ner_service import build_analyzer
from seif import rubert_ner as adapter


def test_rubert_is_default_ner_backend(monkeypatch):
    expected = object()
    monkeypatch.delenv("SEIF_NER_BACKEND", raising=False)
    monkeypatch.setattr(adapter.RubertAnalyzer, "from_env", classmethod(lambda cls: expected))
    assert build_analyzer() is expected


def span(text, value, label, score=.8, start=None):
    start = text.index(value) if start is None else start
    return {"start": start, "end": start + len(value), "label": label, "score": score, "text": value}


class FakeRuntime:
    def __init__(self, output):
        self.output = output
        self.inputs = []
        self.warmups = 0
        self.backend = SimpleNamespace(_bucket_backends={})

    def predict(self, text):
        self.inputs.append(text)
        return self.output

    def warmup(self):
        self.warmups += 1
        self.backend._bucket_backends = dict.fromkeys((32, 64, 128, 256, 512))


def test_one_inference_keeps_all_native_types_and_merges_coarse_names_by_offsets():
    text = "😀 Иванов\tПётр Сергеевич, Москва Россия; a@b.ru"
    raw = [span(text, "Иванов", "LAST_NAME", .7), span(text, "Пётр", "FIRST_NAME", .9),
           span(text, "Сергеевич", "MIDDLE_NAME", .8), span(text, "Москва", "CITY", .6),
           span(text, "Россия", "COUNTRY", .7), span(text, "a@b.ru", "EMAIL", .95)]
    saved = [item.copy() for item in raw]
    runtime = FakeRuntime(raw)
    result = adapter.RubertAnalyzer(runtime).predict_both(text)
    assert runtime.inputs == [text]
    assert result["native"] == saved
    assert raw == saved
    assert result["gateway_error"] is None
    assert [(text[item["start"]:item["end"]], item["entity_type"], item["score"]) for item in result["gateway"]] == [
        ("Иванов\tПётр Сергеевич", "PERSON", .9), ("Москва Россия", "LOCATION", .7)]


def test_mapping_does_not_expand_across_punctuation_or_non_target_entity():
    text = "Иван, Пётр a@b.ru Анна. Москва, Россия"
    native = [span(text, "Иван", "FIRST_NAME"), span(text, "Пётр", "FIRST_NAME"),
              span(text, "a@b.ru", "EMAIL"), span(text, "Анна", "FIRST_NAME"),
              span(text, "Москва", "CITY"), span(text, "Россия", "COUNTRY")]
    result = adapter.gateway_entities(text, native)
    assert [text[item["start"]:item["end"]] for item in result] == ["Иван", "Пётр", "Анна", "Москва", "Россия"]


def test_analyze_returns_gateway_compatible_spans_and_keeps_original_unicode_input():
    text = "😀 е\u0301  Иванов Иван\n"
    raw = [span(text, "Иванов", "LAST_NAME"), span(text, "Иван", "FIRST_NAME", start=text.rindex("Иван"))]
    runtime = FakeRuntime(raw)
    model = adapter.RubertAnalyzer(runtime)
    result = model.analyze(text=text, language="ru", entities=["PERSON", "LOCATION"], score_threshold=0.)
    assert runtime.inputs == [text]
    assert len(result) == 1
    assert text[result[0].start:result[0].end] == "Иванов Иван"
    assert result[0].entity_type == "PERSON"


@pytest.mark.parametrize("label", sorted(adapter.NATIVE_TYPES))
def test_every_native_class_preserved_even_when_outside_gateway(label):
    text = "Синтетическое значение"
    item = span(text, text, label)
    result = adapter.RubertAnalyzer(FakeRuntime([item])).predict_both(text)
    assert result["native"] == [item]
    mapped = label in adapter.PERSON_TYPES | adapter.LOCATION_TYPES
    assert len(result["gateway"]) == int(mapped)


@pytest.mark.parametrize(("field", "value"), [
    ("start", True), ("start", -1), ("start", 1.), ("end", False), ("end", 5), ("end", 0),
    ("score", True), ("score", "0.8"), ("score", float("nan")), ("score", float("inf")),
    ("score", -.1), ("score", 1.01), ("text", "иван"), ("text", None), ("label", "UNKNOWN"), ("label", []),
])
def test_invalid_native_cannot_hide_behind_gateway_filtering(field, value):
    text = "Иван"
    item = span(text, text, "EMAIL")
    item[field] = value
    with pytest.raises(ValueError, match="Invalid RuBERT native output"):
        adapter.gateway_entities(text, [item])


@pytest.mark.parametrize("output", [None, {}, [None], [{}], [{"start": 0, "end": 1}]])
def test_malformed_native_container_fails(output):
    with pytest.raises(ValueError, match="Invalid RuBERT native output"):
        adapter.validate_native("Иван", output)


def test_overlapping_native_rows_reject_instead_of_redecoding_published_output():
    text = "Иванов"
    rows = [span(text, text, "LAST_NAME"), span(text, "Иван", "FIRST_NAME")]
    with pytest.raises(ValueError, match="Invalid RuBERT native output"):
        adapter.validate_native(text, rows)


def test_native_retained_when_coarse_merge_exceeds_gateway_limit():
    text = "А" * 100 + " " + "Б" * 100
    native = [span(text, "А" * 100, "FIRST_NAME"), span(text, "Б" * 100, "LAST_NAME")]
    runtime = FakeRuntime(native)
    model = adapter.RubertAnalyzer(runtime)
    result = model.predict_both(text)
    assert runtime.inputs == [text]
    assert result["native"] == native
    assert result["gateway"] is None
    assert result["gateway_error"] == adapter.GATEWAY_LIMIT
    with pytest.raises(ValueError, match="span length/count contract"):
        model.analyze(text=text, language="ru", entities=["PERSON", "LOCATION"], score_threshold=0.)


def test_gateway_count_limit_rejects_without_dropping_native():
    text = ",".join("А" for _ in range(2049))
    native = [span(text, "А", "FIRST_NAME", start=index) for index in range(0, len(text), 2)]
    result = adapter.RubertAnalyzer(FakeRuntime(native)).predict_both(text)
    assert len(result["native"]) == 2049
    assert result["gateway"] is None
    assert result["gateway_error"] == adapter.GATEWAY_LIMIT


def test_unmapped_long_entity_preserved_without_failing_gateway():
    text = "x" * 201
    result = adapter.RubertAnalyzer(FakeRuntime([span(text, text, "URL")])).predict_both(text)
    assert result["native"][0]["end"] == 201
    assert result["gateway"] == []
    assert result["gateway_error"] is None


def test_empty_and_long_input_pass_to_published_windowing_without_adapter_truncation():
    runtime = FakeRuntime([])
    model = adapter.RubertAnalyzer(runtime)
    for text in ("", "слово " * 1600 + "Иван"):
        assert model.predict_native(text) == []
    assert runtime.inputs == ["", "слово " * 1600 + "Иван"]


def test_warmup_calls_published_all_bucket_routine_once_and_metadata_isolated():
    runtime = FakeRuntime([])
    model = adapter.RubertAnalyzer(runtime, runtime_metadata={"nested": {"test": True}})
    model.warmup()
    model.warmup()
    assert runtime.warmups == 1
    metadata = model.metadata()
    assert metadata["warmup_complete"]
    assert metadata["graph_buckets_captured"] == [32, 64, 128, 256, 512]
    metadata["nested"]["test"] = False
    assert model.metadata()["nested"]["test"]


def test_pin_validation_happens_before_executing_model_code(tmp_path, monkeypatch):
    local_file = tmp_path / "pii_ner.py"
    local_file.write_bytes(b"pinned")
    monkeypatch.setattr(adapter, "PINNED_FILES", {"pii_ner.py": hashlib.sha256(b"pinned").hexdigest()})
    assert adapter.fingerprint_checkpoint(tmp_path) == adapter.PINNED_FILES
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "file.pyc").write_bytes(b"generated")
    assert adapter.fingerprint_checkpoint(tmp_path) == adapter.PINNED_FILES
    local_file.write_bytes(b"changed")
    monkeypatch.setattr(adapter, "_load_runtime", lambda _: pytest.fail("unverified source executed"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        adapter.RubertAnalyzer.from_local(tmp_path)


def test_isolated_import_passes_fixed_backend_options_without_import_path_mutation(tmp_path):
    (tmp_path / "pii_ner.py").write_text("class PiiNER:\n    def __init__(self, **kwargs):\n        self.options=kwargs\n")
    before_path = list(sys.path)
    before_modules = dict(sys.modules)
    runtime = adapter._load_runtime(tmp_path)
    assert runtime.options == {"model_dir": tmp_path, "backend": "trt-graph", "min_confidence": .3, "batch_size": 1}
    assert sys.path == before_path
    assert sys.modules == before_modules


def test_runtime_inventory_requires_all_43_bio_labels(tmp_path):
    labels = dict(enumerate(["O", *sorted(
        f"{prefix}-{kind}" for prefix in ("B", "I") for kind in adapter.NATIVE_TYPES)]))
    (tmp_path / "config.json").write_text(json.dumps({"id2label": labels}))
    runtime = SimpleNamespace(id2label=labels.copy(), backend_name="trt-graph", min_confidence=.3, batch_size=1)
    adapter._validate_runtime(runtime, tmp_path)
    runtime.id2label[0] = "UNKNOWN"
    with pytest.raises(ValueError, match="label inventory"):
        adapter._validate_runtime(runtime, tmp_path)


@pytest.mark.parametrize("device", ["cpu", "cuda:0", None])
def test_cpu_or_implicit_fallback_not_allowed(device):
    with pytest.raises(ValueError, match="requires device"):
        adapter.RubertAnalyzer.from_local("unused", device=device)


@pytest.fixture
def numpy():
    return pytest.importorskip("numpy")


@pytest.mark.parametrize("bad", ["short_sequence", "wrong_labels", "missing_batch", "float64", "nan"])
def test_logits_guard_blocks_silent_zip_truncation_and_nonfinite_values(numpy, bad):
    np = numpy
    output = np.zeros((1, 8, 43), dtype=np.float32)
    if bad == "short_sequence":
        output = output[:, :7]
    elif bad == "wrong_labels":
        output = output[:, :, :42]
    elif bad == "missing_batch":
        output = output[0]
    elif bad == "float64":
        output = output.astype(np.float64)
    else:
        output[0, 0, 0] = np.nan
    wrapped = adapter._ValidatedBackend(SimpleNamespace(run=lambda _: output))
    inputs = {"input_ids": np.zeros((1, 8), dtype=np.int64)}
    with pytest.raises(ValueError, match="logits shape"):
        wrapped.run(inputs)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_logits_guard_keeps_exact_backend_array_and_buckets(numpy, dtype):
    np = numpy
    output = np.zeros((1, 8, 43), dtype=dtype)
    backend = SimpleNamespace(run=lambda _: output, buckets=(32, 64, 128, 256, 512))
    wrapped = adapter._ValidatedBackend(backend)
    assert wrapped.run({"input_ids": np.zeros((1, 8), dtype=np.int64)}) is output
    assert wrapped.buckets == backend.buckets


@pytest.mark.parametrize(("configured", "expected"), [(None, 4), ("1", 1), ("4", 4), ("32", 32)])
def test_from_env_applies_bounded_cpu_threads_before_loading_local_model(monkeypatch, configured, expected):
    calls = []
    fake_torch = SimpleNamespace(set_num_threads=calls.append, backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)), cudnn=SimpleNamespace(allow_tf32=True),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("SEIF_RUBERT_CPU_THREADS", raising=False)
    if configured is not None:
        monkeypatch.setenv("SEIF_RUBERT_CPU_THREADS", configured)
    monkeypatch.setenv("SEIF_RUBERT_MODEL_PATH", "/fixture/model")
    monkeypatch.setenv("SEIF_RUBERT_DECODER", "word")
    monkeypatch.setenv("SEIF_RUBERT_PROFILE", "native")
    monkeypatch.setenv("SEIF_NER_BATCH_SIZE", "16")

    def load(model_path, **options):
        assert calls == [expected]
        return model_path, options

    monkeypatch.setattr(adapter.RubertAnalyzer, "from_local", staticmethod(load))
    assert adapter.RubertAnalyzer.from_env() == (
        "/fixture/model", {"decoder": "word", "profile": "native", "batch_size": 16},
    )
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False


@pytest.mark.parametrize("configured", ["0", "-1", "33", "999", "1.5", "nan", "true", ""])
def test_from_env_rejects_bad_cpu_threads_before_importing_torch(monkeypatch, configured):
    monkeypatch.setenv("SEIF_RUBERT_CPU_THREADS", configured)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(adapter.RubertAnalyzer, "from_local", lambda *_args, **_kwargs: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="SEIF_RUBERT_CPU_THREADS must be an integer between 1 and 32"):
        adapter.RubertAnalyzer.from_env()
