"""Network-free checks for the experimental HTTP measurement, no real model."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

bench = importlib.import_module("benchmarks.ner-models.gliner25-onnx.http_benchmark")


def example_case():
    return {"case_id": "synthetic", "text": "Иван", "expected": "****",
            "entities": [(0, 4, "PERSON")], "types": ["PERSON"]}


def example_body(payload_id="synthetic-id"):
    return {"result": "****", "payload_id": payload_id,
            "entities": [{"start": 0, "end": 4, "type": "PERSON", "confidence": .9, "reason": "ner-person"}],
            "types": ["PERSON"], "mode": "mask", "latency_ms": 1, "masking_enabled": True}


def test_response_parity_uses_mask_and_typed_offsets_not_float_equality():
    body = example_body()
    body["entities"][0]["confidence"] = .90000001
    result = bench.compare_response(example_case(), "synthetic-id", body)
    assert result == {"mask_match": True, "entities_match": True, "types_match": True,
                      "unexpected_unmasked": False}
    body["entities"][0]["end"] = 3
    assert not bench.compare_response(example_case(), "synthetic-id", body)["entities_match"]


def test_unmasked_success_is_reported_as_disagreement_and_possible_fail_open():
    body = example_body()
    body.update(result="Иван", entities=[], types=[])
    result = bench.compare_response(example_case(), "synthetic-id", body)
    assert not result["mask_match"] and not result["entities_match"] and not result["types_match"]
    assert result["unexpected_unmasked"]


def test_comparison_accepts_actual_public_mask_contract(monkeypatch):
    import seif.app
    from seif.config import Settings
    from seif.detector import Span

    class FakeNerClient:
        def __init__(self, *_args):
            pass

        async def detect(self, _text):
            return [Span(0, 4, "PERSON", .9, "presidio-ru-ner")]

        async def health(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(seif.app, "NerClient", FakeNerClient)
    app = seif.app.create_app(Settings(demo=True, ner_url="http://127.0.0.1:12345", ner_token="test-token"))
    with TestClient(app) as client:
        response = client.post("/v1/mask", json={"payload": "Иван", "payload_id": "synthetic-id"})
    assert response.status_code == 200
    result = bench.compare_response(example_case(), "synthetic-id", response.json())
    assert result["mask_match"] and result["entities_match"] and result["types_match"]


@pytest.mark.parametrize("replacement", [float("nan"), True, -1, 1.1])
def test_response_rejects_invalid_scores(replacement):
    body = example_body()
    body["entities"][0]["confidence"] = replacement
    with pytest.raises(bench.smoke.SmokeFailure):
        bench.compare_response(example_case(), "synthetic-id", body)


def test_summary_keeps_http_errors_and_does_not_credit_skipped_ner():
    good = {"case_id": "first", "payload_id": "private-id", "masked": "private-value", "status": 200,
            "latency_ms": 20, "mask_match": True, "entities_match": True, "types_match": True,
            "unexpected_unmasked": False}
    failed = {"case_id": "second", "status": 503, "error": "http_503", "latency_ms": 40}
    report = bench.phase_summary([good, failed], 1, 4, {"model_started": 1, "model_completed": 1, "http_200": 1}, 1)
    assert report["successful_mask_rps"] == 1
    assert report["attempted_mask_rps"] == 2
    assert report["errors"] == {"http_503": 1}
    assert report["requests_without_ner_stage"] == 1
    assert not report["valid_successful_measurement"]
    encoded = json.dumps(report)
    assert "private-id" not in encoded and "private-value" not in encoded
    report = bench.phase_summary([good], 1, 1, {"model_started": 0, "model_completed": 0, "http_200": 0}, 0)
    assert report["reference_match_ratio"] == 1
    assert not report["every_mask_completed_real_ner"]


def test_counted_real_service_factory_distinguishes_calls_errors_and_http_rejections(monkeypatch):
    import seif.gliner_ner

    class FakeAnalyzer:
        model_name = bench.smoke.MODEL_NAME
        fail = False

        def analyze(self, **_kwargs):
            if self.fail:
                raise ValueError("Sensitive exception text must never be echoed")
            return []

    analyzer = FakeAnalyzer()
    torch_stub = SimpleNamespace(set_num_threads=lambda _: None, manual_seed=lambda _: None,
                                 backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                                          cudnn=SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(seif.gliner_ner.GlinerAnalyzer, "from_local", lambda *_args, **_kwargs: analyzer)
    for name, value in {"SEIF_NER_TOKEN": "test-token", "SEIF_NER_DEMO": "0", "SEIF_NER_MAX_MODEL_JOBS": "4",
                        "SEIF_HTTP_BENCH_BACKEND": "torch", "SEIF_HTTP_BENCH_NATIVE": "/unused",
                        "SEIF_HTTP_BENCH_DEVICE": "cpu"}.items():
        monkeypatch.setenv(name, value)
    app = bench.create_ner_app()
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(app) as client:
        assert client.get("/benchmark-stats").status_code == 401
        assert client.post("/analyze", json={"text": "пример"}, headers=headers).status_code == 200
        assert client.post("/analyze", json={"text": 42}, headers=headers).status_code == 422
        analyzer.fail = True
        response = client.post("/analyze", json={"text": "пример"}, headers=headers)
        assert response.status_code == 503 and "Sensitive" not in response.text
        counts = client.get("/benchmark-stats", headers=headers).json()["counts"]
    assert counts["model_started"] == 2
    assert counts["model_completed"] == counts["model_failed"] == 1
    assert counts["http_started"] == counts["http_completed"] == 3
    assert counts["model_active"] == counts["http_active"] == 0
    assert counts["http_200"] == counts["http_422"] == counts["http_503"] == 1


def test_closed_loop_phases_use_fresh_ids_and_restore_outside_timed_counters():
    counts = bench.Counts()

    class FakeNer:
        def call(self, path):
            assert path == "/benchmark-stats"
            return 200, {"counts": counts.snapshot(), "model": {}}

    class FakeApi:
        timeout = 1

        def __init__(self):
            self.ids = []
            self.restores = 0

        def call(self, path, body=None, *, raw=False):
            if path == "/metrics":
                assert raw
                return 200, f'seif_stage_duration_seconds_count{{stage="ner"}} {len(self.ids)}\n'
            if path == "/v1/unmask":
                self.restores += 1
                assert body["payload_id"] in self.ids
                return 200, {"result": "Иван"}
            assert path == "/v1/mask"
            self.ids.append(body["payload_id"])
            counts.add(model_started=1, model_completed=1, http_started=1, http_completed=1, http_200=1)
            return 200, example_body(body["payload_id"])

    api, ner = FakeApi(), FakeNer()
    first = bench.run_phase(api, ner, [example_case()], 1)
    second = bench.run_phase(api, ner, [example_case()], 4)
    assert len(api.ids) == len(set(api.ids)) == 2
    assert api.restores == 2
    assert first["valid_successful_measurement"] and second["valid_successful_measurement"]
    assert first["ner_stage_attempts"] == second["ner_stage_attempts"] == 1
    assert first["restore"] == {"checked": 1, "exact": 1, "excluded_from_mask_timing": True}


def test_cli_preserves_virtualenv_interpreter_symlinks(monkeypatch, tmp_path):
    binary = tmp_path / "base-python"
    binary.touch()
    interpreter = tmp_path / "venv-python"
    interpreter.symlink_to(binary)
    monkeypatch.setattr(sys, "argv", ["http_benchmark.py", "--backend", "torch", "--model-path", str(tmp_path),
                                    "--ner-python", str(interpreter), "--api-python", str(interpreter),
                                    "--output", str(tmp_path / "report.json")])
    args = bench.parse_args()
    assert args.ner_python == interpreter and args.api_python == interpreter
    assert args.ner_python != binary
