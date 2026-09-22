"""GLiNER protocol/boundary tests using fake extractors, without Torch."""
from __future__ import annotations

import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from scripts.ner_service import NerSettings, build_analyzer, create_app, infer
from seif.gliner_ner import CHUNK_OVERLAP, CHUNK_WORDS, MODEL_NAME, SCHEMAS, GlinerAnalyzer


def split_words(text, lower=False):
    for match in re.finditer(r"\S+", text):
        yield match.group().lower() if lower else match.group(), match.start(), match.end()


class FakeExtractor:
    def __init__(self, output=None):
        self.output = output if output is not None else {"entities": {"person": [], "location": []}}
        self.processor = SimpleNamespace(word_splitter=split_words)
        self.calls = []

    def extract_entities(self, text, labels, **options):
        self.calls.append(("short", text, labels, options))
        return self.output

    def extract_entities_long(self, text, labels, **options):
        self.calls.append(("long", text, labels, options))
        return self.output


def raw_span(text, value, confidence=0.87):
    start = text.index(value)
    return {"text": value, "start": start, "end": start + len(value), "confidence": confidence}


def test_maps_unicode_offsets_and_preserves_real_confidence_and_original_text():
    text = "  😀 İванов е\u0301, Москва\n"
    person = raw_span(text, "İванов е\u0301", confidence=0.912345)
    location = raw_span(text, "Москва", confidence=0.678901)
    extractor = FakeExtractor({"entities": {"location": [location], "person": [person]}})
    output = infer(GlinerAnalyzer(extractor), text)
    assert output == {"entities": [
        {"start": person["start"], "end": person["end"], "score": 0.912345, "entity_type": "PERSON"},
        {"start": location["start"], "end": location["end"], "score": 0.678901, "entity_type": "LOCATION"},
    ]}
    assert extractor.calls == [("short", text, ["person", "location"], {
        "threshold": 0.5, "include_spans": True, "include_confidence": True, "overlap_policy": "flat",
        "max_len": CHUNK_WORDS + 1,
    })]


def test_long_input_reaches_official_chunk_api_with_end_of_document_entity():
    text = "  " + "слово " * (CHUNK_WORDS + 20) + "Москва\n"
    entity = raw_span(text, "Москва")
    extractor = FakeExtractor({"entities": {"person": [], "location": [entity]}})
    result = infer(GlinerAnalyzer(extractor), text)
    assert result["entities"][0]["end"] == len(text) - 1
    assert extractor.calls == [("long", text, ["person", "location"], {
        "threshold": 0.5, "include_spans": True, "include_confidence": True, "overlap_policy": "flat",
        "chunk_size": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP, "batch_size": 1, "num_workers": 0,
    })]


def test_exact_word_limit_reserves_upstream_period_without_rewriting_input():
    text = "слово " * (CHUNK_WORDS - 1) + "Москва"
    extractor = FakeExtractor({"entities": {"person": [], "location": [raw_span(text, "Москва")]}})
    assert infer(GlinerAnalyzer(extractor), text)["entities"][0]["end"] == len(text)
    assert extractor.calls[0][0] == "short"
    assert extractor.calls[0][1] == text
    assert extractor.calls[0][3]["max_len"] == CHUNK_WORDS + 1


@pytest.mark.parametrize("output", [
    None, [], {}, {"entities": []}, {"entities": None}, {"entities": {}}, {"entities": {}, "other": []},
    {"entities": {"person": []}}, {"entities": {"location": []}},
    {"entities": {"organization": []}}, {"entities": {"PERSON": []}},
    {"entities": {"person": None, "location": []}}, {"entities": {"person": "Иван", "location": []}},
    {"entities": {"person": [None], "location": []}}, {"entities": {"person": [{}], "location": []}},
])
def test_malformed_outputs_fail_closed(output):
    extractor = FakeExtractor()
    extractor.output = output
    with pytest.raises(ValueError, match="Invalid GLiNER model output"):
        infer(GlinerAnalyzer(extractor), "Иван")


@pytest.mark.parametrize(("field", "value"), [
    ("start", True), ("start", -1), ("start", 1.0), ("start", 4),
    ("end", 0), ("end", 5), ("end", "4"), ("end", False),
    ("confidence", True), ("confidence", "0.9"), ("confidence", None),
    ("confidence", float("nan")), ("confidence", float("inf")),
    ("confidence", -0.1), ("confidence", 1.1),
    ("text", "иван"), ("text", "Иван."), ("text", None),
])
def test_rejects_invalid_span_types_bounds_confidence_and_text(field, value):
    entity = {**raw_span("Иван", "Иван"), field: value}
    extractor = FakeExtractor({"entities": {"person": [entity], "location": []}})
    with pytest.raises(ValueError, match="Invalid GLiNER model output"):
        infer(GlinerAnalyzer(extractor), "Иван")


def test_one_invalid_entity_rejects_entire_prediction_instead_of_partial_success():
    extractor = FakeExtractor({"entities": {"person": [raw_span("Иван", "Иван"), {}], "location": []}})
    with pytest.raises(ValueError, match="Invalid GLiNER model output"):
        infer(GlinerAnalyzer(extractor), "Иван")


@pytest.mark.parametrize("label", ["person", "location"])
def test_partial_label_groups_reject_even_valid_predictions(label):
    text = "Иван в Москве"
    value = "Иван" if label == "person" else "Москве"
    extractor = FakeExtractor({"entities": {label: [raw_span(text, value)]}})
    with pytest.raises(ValueError, match="Invalid GLiNER model output"):
        infer(GlinerAnalyzer(extractor), text)


def test_empty_text_does_not_invoke_model():
    extractor = FakeExtractor()
    assert infer(GlinerAnalyzer(extractor), "") == {"entities": []}
    assert extractor.calls == []


@pytest.mark.parametrize("changes", [
    {"language": "en"}, {"entities": ["PERSON"]}, {"entities": ["PERSON", "ORGANIZATION"]},
    {"score_threshold": 0.9}, {"text": None},
])
def test_analyzer_protocol_does_not_silently_change(changes):
    args = {"text": "Иван", "language": "ru", "entities": ["PERSON", "LOCATION"], "score_threshold": 0.0}
    with pytest.raises(ValueError):
        GlinerAnalyzer(FakeExtractor()).analyze(**(args | changes))


def test_local_loader_checks_files_before_import_and_forces_local_unquantized_model(tmp_path, monkeypatch):
    calls = []

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **options):
            calls.append((path, options))
            return SimpleNamespace(eval=lambda: calls.append("eval"))

    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(AutoExtractor=Loader))
    for file in ["config.json", "encoder_config/config.json", "tokenizer_config.json", "tokenizer.json"]:
        target = tmp_path / file
        target.parent.mkdir(exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="local GLiNER checkpoint files"):
        GlinerAnalyzer.from_local(tmp_path)
    assert calls == []
    (tmp_path / "model.safetensors").write_bytes(b"fake checkpoint fixture")
    analyzer = GlinerAnalyzer.from_local(tmp_path, device="cuda")
    assert analyzer.model_name == MODEL_NAME
    assert calls == [(str(tmp_path.resolve()), {
        "map_location": "cuda", "quantize": False, "compile": False, "local_files_only": True,
    }), "eval"]
    with pytest.raises(ValueError, match="cpu or cuda"):
        GlinerAnalyzer.from_local(tmp_path, device="other")


@pytest.mark.parametrize("path", ["", "fastino/gliner2.5-multi-v1", "https://example.invalid/model"])
def test_missing_model_path_cannot_become_remote_download(path):
    with pytest.raises(RuntimeError):
        GlinerAnalyzer.from_local(path)


def test_environment_backend_selection_is_explicit(monkeypatch):
    expected = object()
    calls = []

    def fake_load(cls, model_path, *, device, schema, threshold):
        calls.append((model_path, device, schema, threshold))
        return expected

    monkeypatch.setattr(GlinerAnalyzer, "from_local", classmethod(fake_load))
    monkeypatch.setenv("SEIF_NER_BACKEND", "gliner")
    monkeypatch.setenv("SEIF_GLINER_MODEL_PATH", "/local/prepared-checkpoint")
    monkeypatch.setenv("SEIF_GLINER_DEVICE", "cuda")
    monkeypatch.delenv("SEIF_GLINER_SCHEMA", raising=False)
    monkeypatch.delenv("SEIF_GLINER_THRESHOLD", raising=False)
    assert build_analyzer() is expected
    assert calls == [("/local/prepared-checkpoint", "cuda", "person-location", 0.5)]
    monkeypatch.setenv("SEIF_GLINER_SCHEMA", "described-names")
    monkeypatch.setenv("SEIF_GLINER_THRESHOLD", "0.95")
    assert build_analyzer() is expected
    assert calls[-1] == ("/local/prepared-checkpoint", "cuda", "described-names", 0.95)
    monkeypatch.setenv("SEIF_NER_BACKEND", "unknown")
    with pytest.raises(RuntimeError, match="SEIF_NER_BACKEND"):
        build_analyzer()


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("threshold", [0.5, 0.8, 0.95])
def test_schema_presets_and_threshold_reach_model_and_keep_gateway_targets(schema, threshold):
    text = "Иван, Москва, Компания"
    groups = {label: [] for label in SCHEMAS[schema]}
    groups["person"] = [raw_span(text, "Иван", confidence=0.88)]
    groups["location"] = [raw_span(text, "Москва", confidence=0.99)]
    if "organization" in groups:
        groups["organization"] = [raw_span(text, "Компания", confidence=0.97)]
    extractor = FakeExtractor({"entities": groups})
    result = infer(GlinerAnalyzer(extractor, schema=schema, threshold=threshold), text)
    assert result == {"entities": [
        {"start": 0, "end": 4, "score": 0.88, "entity_type": "PERSON"},
        {"start": 6, "end": 12, "score": 0.99, "entity_type": "LOCATION"},
    ]}
    assert extractor.calls[0][2] == SCHEMAS[schema]
    assert extractor.calls[0][3]["threshold"] == threshold
    assert extractor.calls[0][3]["overlap_policy"] == "flat"


@pytest.mark.parametrize(("schema", "missing"), [
    (schema, label) for schema, labels in SCHEMAS.items() for label in labels
])
def test_every_selected_schema_group_is_required_even_when_empty(schema, missing):
    groups = {label: [] for label in SCHEMAS[schema] if label != missing}
    analyzer = GlinerAnalyzer(FakeExtractor({"entities": groups}), schema=schema)
    with pytest.raises(ValueError, match="Invalid GLiNER model output"):
        infer(analyzer, "обычный текст")


@pytest.mark.parametrize("schema", ["presidio-labels", "described-names"])
def test_organization_is_validated_before_excluding_it_from_gateway(schema):
    groups = {label: [] for label in SCHEMAS[schema]}
    groups["organization"] = [{"text": "Компания", "start": 0, "end": 99, "confidence": 0.9}]
    analyzer = GlinerAnalyzer(FakeExtractor({"entities": groups}), schema=schema)
    with pytest.raises(ValueError, match="Invalid GLiNER model output"):
        infer(analyzer, "Компания")


def test_person_and_name_duplicates_preserve_highest_score_without_merging_different_types():
    text = "Иван"
    extractor = FakeExtractor({"entities": {
        "person": [raw_span(text, text, 0.7), raw_span(text, text, 0.6)],
        "name": [raw_span(text, text, 0.95)],
        "location": [raw_span(text, text, 0.8)],
        "organization": [raw_span(text, text, 0.99)],
    }})
    assert infer(GlinerAnalyzer(extractor, schema="presidio-labels"), text) == {"entities": [
        {"start": 0, "end": 4, "score": 0.8, "entity_type": "LOCATION"},
        {"start": 0, "end": 4, "score": 0.95, "entity_type": "PERSON"},
    ]}


@pytest.mark.parametrize("threshold", [None, True, False, "0.5", 0, 1, -0.1, 1.1,
                                       float("nan"), float("inf"), float("-inf")])
def test_invalid_threshold_rejected_before_loading_or_inference(threshold):
    with pytest.raises(ValueError, match="SEIF_GLINER_THRESHOLD"):
        GlinerAnalyzer(FakeExtractor(), threshold=threshold)
    with pytest.raises(ValueError, match="SEIF_GLINER_THRESHOLD"):
        GlinerAnalyzer.from_local("/nonexistent-checkpoint", threshold=threshold)


@pytest.mark.parametrize("schema", [None, "", "unknown", [], {}])
def test_unknown_schema_rejected_before_loading_or_inference(schema):
    with pytest.raises(ValueError, match="SEIF_GLINER_SCHEMA"):
        GlinerAnalyzer(FakeExtractor(), schema=schema)
    with pytest.raises(ValueError, match="SEIF_GLINER_SCHEMA"):
        GlinerAnalyzer.from_local("/nonexistent-checkpoint", schema=schema)


def test_long_api_preserves_configured_descriptions_and_threshold():
    extractor = FakeExtractor({"entities": {"person": [], "location": [], "organization": []}})
    text = "слово " * (CHUNK_WORDS + 1)
    assert infer(GlinerAnalyzer(extractor, schema="described-names", threshold=0.8), text) == {"entities": []}
    assert extractor.calls[0][0] == "long"
    assert extractor.calls[0][2] == SCHEMAS["described-names"]
    assert extractor.calls[0][3]["threshold"] == 0.8


def test_http_health_reports_actual_model_and_prediction_errors_are_redacted():
    extractor = FakeExtractor({"entities": {"person": [{"text": "private@example.invalid"}], "location": []}})
    analyzer = GlinerAnalyzer(extractor)
    with TestClient(create_app(NerSettings(demo=True), analyzer_factory=lambda: analyzer)) as client:
        assert client.get("/health").json()["model"] == MODEL_NAME
        response = client.post("/analyze", json={"text": "private@example.invalid"})
        assert response.status_code == 503
        assert "private@" not in response.text
        extractor.output = {"entities": {"person": [], "location": []}}
        assert client.post("/analyze", json={"text": "обычный текст"}).json() == {"entities": []}


def test_stub_health_keeps_presidio_model_default():
    with TestClient(create_app(NerSettings(demo=True), analyzer_factory=object)) as client:
        assert client.get("/health").json()["model"] == "ru_core_news_sm"


def test_concurrent_calls_share_one_model_lock():
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    extractor = FakeExtractor()
    active = []

    def extract(text, labels, **options):
        active.append(text)
        entered.set()
        assert release.wait(3)
        return {"entities": {"person": [], "location": []}}

    extractor.extract_entities = extract
    analyzer = GlinerAnalyzer(extractor)

    def second_call():
        second_started.set()
        return infer(analyzer, "второй")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(infer, analyzer, "первый")
        try:
            assert entered.wait(3)
            second = pool.submit(second_call)
            assert second_started.wait(3)
            assert not second.done()
            assert active == ["первый"]
        finally:
            release.set()
        assert first.result(3) == second.result(3) == {"entities": []}
    assert active == ["первый", "второй"]
