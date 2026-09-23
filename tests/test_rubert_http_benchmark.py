"""RuBERT HTTP benchmark protocol checks without CUDA or model libraries."""
import hashlib
import importlib
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

bench = importlib.import_module("benchmarks.ner-models.rubert-tensorrt.http_benchmark")


@pytest.fixture
def reference(tmp_path):
    dataset = bench.ROOT / "datasets/golden/organizer-v1/cases.jsonl"
    cases = [json.loads(line) for line in dataset.read_text().splitlines()]
    cache = tmp_path / "rubert-organizer.jsonl"
    rows = [{"case_id": case["case_id"], "text_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
             "entities": []} for case in cases]
    cache.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    metadata = {"model": bench.MODEL_ID, "model_revision": bench.MODEL_REVISION,
                "cache_sha256": bench.file_digest(cache), "dataset_sha256": bench.file_digest(dataset)}
    cache.with_suffix(".meta.json").write_text(json.dumps(metadata))
    return SimpleNamespace(dataset=dataset, reference_cache=cache, model_path=tmp_path), cases, rows


def test_reference_expectations_cover_all_cases_without_loading_model(reference):
    args, cases, _rows = reference
    prepared = bench.prepare_cases(args)
    assert len(prepared) == 446
    assert [case["case_id"] for case in prepared] == [case["case_id"] for case in cases]
    assert all(set(case) == {"case_id", "text", "expected", "entities", "types"} for case in prepared)
    assert any(case["expected"] != case["text"] for case in prepared)


@pytest.mark.parametrize("field,value", [("model", "other/model"), ("model_revision", "other-revision"),
                                       ("dataset_sha256", "wrong-digest")])
def test_reference_rejects_other_model_revision_or_dataset(reference, field, value):
    args, _cases, _rows = reference
    path = args.reference_cache.with_suffix(".meta.json")
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(bench.smoke.SmokeFailure, match="frozen_rubert_reference_identity"):
        bench.prepare_cases(args)


def test_reference_obeys_real_private_ner_span_length_contract(reference):
    from seif.ner import NerUnavailable

    args, cases, rows = reference
    index = next(index for index, case in enumerate(cases) if len(case["text"]) > 200)
    rows[index]["entities"] = [{"start": 0, "end": 201, "entity_type": "PERSON", "score": .9}]
    args.reference_cache.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    path = args.reference_cache.with_suffix(".meta.json")
    metadata = json.loads(path.read_text())
    metadata["cache_sha256"] = bench.file_digest(args.reference_cache)
    path.write_text(json.dumps(metadata))
    with pytest.raises(NerUnavailable):
        bench.prepare_cases(args)


def test_native_factory_warms_graphs_before_readiness_and_counts_actual_calls(monkeypatch, tmp_path):
    calls = []

    class FakeRubert:
        model_name = bench.MODEL_ID

        @classmethod
        def from_local(cls, model_path, **options):
            calls.append((model_path, options))
            return cls()

        def analyze(self, **_kwargs):
            return []

        def metadata(self):
            return {"model": self.model_name, "backend": "trt-graph", "warmed": True}

    torch = SimpleNamespace(set_num_threads=lambda _: None, manual_seed=lambda _: None,
                            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                                     cudnn=SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "seif.rubert_ner", SimpleNamespace(RubertAnalyzer=FakeRubert))
    monkeypatch.setenv("SEIF_RUBERT_HTTP_MODEL", str(tmp_path))
    monkeypatch.setenv("SEIF_NER_TOKEN", "test-token")
    monkeypatch.setenv("SEIF_NER_DEMO", "0")
    monkeypatch.setenv("SEIF_NER_MAX_MODEL_JOBS", "4")
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(bench.create_ner_app()) as client:
        assert calls == [(tmp_path, {"device": "cuda", "warmup": True})]
        assert client.get("/health").json()["model"] == bench.MODEL_ID
        assert client.get("/benchmark-stats").status_code == 401
        assert client.post("/analyze", json={"text": "пример"}, headers=headers).status_code == 200
        stats = client.get("/benchmark-stats", headers=headers).json()
    assert stats["model"]["warmed"]
    assert "uvicorn" in stats["server_environment"]["packages"]
    assert stats["server_environment"]["python_executable"] == sys.executable
    assert stats["counts"]["model_started"] == stats["counts"]["model_completed"] == 1
    assert stats["counts"]["http_200"] == 1
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_selected_phases_use_shared_checked_runner_and_unique_warmup_ids(monkeypatch):
    offered, phases = [], []

    class FakeApi:
        def call(self, path, body):
            assert path == "/v1/mask"
            offered.append(body["payload_id"])
            return 200, {"result": "synthetic"}

    def fake_phase(_api, _ner, cases, concurrency, *, phase_reports):
        phases.append((len(cases), concurrency))
        phase = {"ner_drained": True, "successful_mask_rps": 1,
                 "reference_match_ratio": 1, "every_mask_completed_real_ner": True,
                 "valid_successful_measurement": True}
        phase_reports.append(phase)
        return phase

    monkeypatch.setattr(bench.common, "run_phase", fake_phase)
    monkeypatch.setattr(bench.common, "ner_snapshot", lambda _: {
        "model": {"backend": "trt-graph"}, "server_environment": {"python_version": "test-runtime"},
    })
    report = {}
    assert bench.run_phases(SimpleNamespace(concurrency=[1, 4, 8, 16]), FakeApi(), object(), [None] * 446, report)
    assert phases == [(446, 1), (446, 4), (446, 8), (446, 16)]
    assert len(offered) == len(set(offered)) == 5
    assert len(report["phases"]) == 4
    assert report["ner_server_environment"]["python_version"] == "test-runtime"


@pytest.mark.parametrize("failure_status,expected", [(503, "PASS"), (200, "FAIL")])
def test_service_stops_own_ner_then_requires_fail_closed_and_cleans_up(monkeypatch, tmp_path, failure_status, expected):
    stopped, closed = [], []

    class FakeClient:
        def call(self, path, _body=None):
            assert path in {"/v1/mask", "/health"}
            return failure_status, {}

        def close(self):
            closed.append(self)

    def launch(_args, _temporary, children, _logs):
        children.extend([("ner", object()), ("api", object())])
        return FakeClient(), FakeClient(), {"api": {}, "ner": {}}

    def phases(_args, _api, _ner, _cases, report):
        report["phases"] = [{"valid_successful_measurement": True}]
        return True

    def terminate(children):
        stopped.append([label for label, _process in children])
        return [{"service": label, "exited": True} for label, _process in children]

    monkeypatch.setattr(bench, "launch", launch)
    monkeypatch.setattr(bench, "run_phases", phases)
    monkeypatch.setattr(bench.smoke, "terminate_children", terminate)
    report = bench.run_service(SimpleNamespace(logs_dir=tmp_path / "logs"), [])
    assert report["status"] == expected
    assert stopped == [["ner"], ["ner", "api"]]
    assert len(closed) == 2
    if expected == "FAIL":
        assert report["failed_check"] == "missing_ner_fails_closed"
        assert len(report["failure_logs"]) == 2


def test_cli_keeps_venv_symlinks_and_bounded_ordered_concurrency(monkeypatch, tmp_path):
    base = tmp_path / "python-base"
    base.touch()
    interpreter = tmp_path / "venv-python"
    interpreter.symlink_to(base)
    argv = ["http_benchmark.py", "--model-path", str(tmp_path), "--reference-cache", str(tmp_path / "cache.jsonl"),
            "--ner-python", str(interpreter), "--api-python", str(interpreter),
            "--output", str(tmp_path / "report.json")]
    monkeypatch.setattr(sys, "argv", argv)
    args = bench.parse_args()
    assert args.ner_python == args.api_python == interpreter
    assert args.concurrency == [1, 4, 8, 16]
    assert args.repeats == 1
    monkeypatch.setattr(sys, "argv", [*argv, "--concurrency", "4", "1"])
    with pytest.raises(SystemExit):
        bench.parse_args()
    monkeypatch.setattr(sys, "argv", [*argv, "--repeats", "101"])
    with pytest.raises(SystemExit):
        bench.parse_args()


def test_repeated_workload_preserves_expected_masks_and_uses_fresh_ids(monkeypatch):
    case = {"case_id": "same-source", "text": "Иван", "expected": "****",
            "entities": [(0, 4, "PERSON")], "types": ["PERSON"]}
    monkeypatch.setattr(bench, "prepare_cases", lambda _: [case])
    cases = bench.prepare_workload(SimpleNamespace(repeats=3))
    assert len(cases) == 3
    assert all(row["expected"] == "****" for row in cases)
    counts, ids = bench.common.Counts(), []

    class FakeNer:
        def call(self, _path):
            return 200, {"counts": counts.snapshot(), "model": {}}

    class FakeApi:
        timeout = 1

        def call(self, path, body=None, *, raw=False):
            if path == "/metrics":
                assert raw
                return 200, f'seif_stage_duration_seconds_count{{stage="ner"}} {len(ids)}\n'
            if path == "/v1/unmask":
                assert body["payload_id"] in ids
                return 200, {"result": "Иван"}
            assert path == "/v1/mask"
            assert body["payload_id"] not in ids
            ids.append(body["payload_id"])
            counts.add(model_started=1, model_completed=1, http_started=1, http_completed=1, http_200=1)
            return 200, {"result": "****", "payload_id": body["payload_id"], "mode": "mask", "latency_ms": 1,
                         "masking_enabled": True, "types": ["PERSON"],
                         "entities": [{"start": 0, "end": 4, "type": "PERSON", "confidence": .9, "reason": "ner"}]}

    report = bench.common.run_phase(FakeApi(), FakeNer(), cases, 1)
    assert len(set(ids)) == report["mask_requests"] == report["ner_counts"]["model_completed"] == 3
    assert report["valid_successful_measurement"]


def test_provenance_includes_reused_helpers_and_verified_model_hashes(monkeypatch, reference):
    args, _cases, _rows = reference
    checked = []

    def fingerprint(path):
        checked.append(path)
        return {"model.fp16.engine": "verified-sha"}

    monkeypatch.setitem(sys.modules, "seif.rubert_ner", SimpleNamespace(fingerprint_checkpoint=fingerprint))
    report = bench.provenance(args)
    assert checked == [args.model_path]
    assert report["model_file_sha256"] == {"model.fp16.engine": "verified-sha"}
    assert "benchmarks/ner-models/gliner25-onnx/http_benchmark.py" in report["source_sha256"]
    assert "benchmarks/ner-models/rubert-tensorrt/http_benchmark.py" in report["source_sha256"]
