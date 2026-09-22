"""Security policy types and stalled uploads must fail closed."""

import asyncio
import os

import pytest
import yaml
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from seif.app import Boundary, create_app
from seif.config import Policy, Settings


@pytest.mark.parametrize("field", ["enabled", "allow_unmask"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_access_flags_require_actual_booleans(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        Policy(**{field: value})


@pytest.mark.parametrize("field", ["enabled", "allow_unmask"])
def test_quoted_yaml_false_never_enables_an_access_flag(tmp_path, monkeypatch, field):
    for name in os.environ:
        if name.startswith("SEIF_"):
            monkeypatch.delenv(name)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({"systems": {"demo": {field: "false"}}}))
    monkeypatch.setenv("SEIF_CONFIG", str(path))
    monkeypatch.setenv("SEIF_DEMO", "1")
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        Settings.from_env()
    path.write_text(yaml.safe_dump({"systems": {"demo": {field: False}}}))
    assert getattr(Settings.from_env().policies["demo"], field) is False


@pytest.mark.parametrize("field", ["min_types", "rps"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1000", float("inf")])
def test_invalid_policy_limits_cannot_bypass_cooccurrence_or_rate_control(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        Policy(**{field: value})


@pytest.mark.parametrize("field", [
    "ttl_seconds", "max_records", "max_store_bytes", "max_payload_chars", "max_body_bytes",
    "max_inflight_body_bytes", "max_inflight", "cpu_workers",
])
@pytest.mark.parametrize("value", [0, -1, True, float("inf")])
def test_invalid_resource_limits_fail_at_configuration(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        Settings(**{field: value})


@pytest.mark.parametrize("value", [0, -1, True, "30", float("nan"), float("inf")])
def test_invalid_body_deadline_is_rejected(value):
    with pytest.raises(ValueError, match="positive finite number"):
        Settings(request_body_timeout_seconds=value)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_demo_access_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match="demo must be a boolean"):
        Settings(demo=value)


def test_invalid_policy_does_not_allocate_clients_before_lifespan_cleanup(monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Invalid policies must be rejected before allocating managed resources")

    monkeypatch.setattr("seif.app.NerClient", unexpected_allocation)
    monkeypatch.setattr("seif.app.Vault", unexpected_allocation)
    monkeypatch.setattr("seif.app.ThreadPoolExecutor", unexpected_allocation)
    settings = Settings(demo=True, ner_url="http://ner.test", ner_token="test-token",
                        policies={"demo": Policy(types=("UNKNOWN_TYPE",))})
    with pytest.raises(ValueError, match="Unknown entity type"):
        create_app(settings)


def test_incomplete_body_times_out_and_releases_slot_and_shared_budget():
    async def scenario():
        output, received = [], []

        async def downstream(scope, receive, send):
            received.append(await receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        settings = Settings(max_inflight=1, max_inflight_body_bytes=8, request_body_timeout_seconds=0.02)
        boundary = Boundary(downstream, settings, CollectorRegistry())
        chunks = asyncio.Queue()
        chunks.put_nowait({"type": "http.request", "body": b"private", "more_body": True})

        async def send(message):
            output.append(message)

        def scope():
            return {"type": "http", "method": "POST", "path": "/process", "headers": []}

        await asyncio.wait_for(boundary(scope(), chunks.get, send), timeout=1)
        assert output[0]["status"] == 408
        assert b"request_timeout" in output[1]["body"]
        assert b"private" not in output[1]["body"]
        assert (b"cache-control", b"no-store") in output[0]["headers"]
        assert boundary.inflight == boundary.buffered_bytes == 0
        assert not received
        output.clear()
        chunks.put_nowait({"type": "http.request", "body": b"ok", "more_body": False})
        await boundary(scope(), chunks.get, send)
        assert output[0]["status"] == 200
        assert len(received) == 1
        assert boundary.inflight == boundary.buffered_bytes == 0

    asyncio.run(scenario())


def test_body_deadline_does_not_timeout_processing_after_upload():
    async def scenario():
        output = []

        async def downstream(scope, receive, send):
            await receive()
            await asyncio.sleep(0.04)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return {"type": "http.request", "body": b"ok", "more_body": False}

        async def send(message):
            output.append(message)

        boundary = Boundary(downstream, Settings(request_body_timeout_seconds=0.01), CollectorRegistry())
        scope = {"type": "http", "method": "POST", "path": "/process", "headers": []}
        await boundary(scope, receive, send)
        assert output[0]["status"] == 200
        assert boundary.inflight == boundary.buffered_bytes == 0

    asyncio.run(scenario())


def test_repeated_tokens_cannot_expand_beyond_payload_limit_and_mapping_stays_usable():
    source = "CLT-" + "x" * 200
    policy = Policy(extra_rules=({"type": "CLIENT_ID", "pattern": r"\bCLT-x{200}\b"},))
    app = create_app(Settings(demo=True, max_payload_chars=256, policies={"demo": policy}))
    with TestClient(app) as client:
        masked = client.post("/v1/mask", json={"payload": source, "payload_id": "amplified", "mode": "token"})
        assert masked.status_code == 200
        token = masked.json()["result"]
        assert token != source
        edited = " ".join([token] * 4)
        assert len(edited) < 256
        rejected = client.post("/v1/unmask", json={"payload": edited, "payload_id": "amplified"})
        assert rejected.status_code == 413
        assert rejected.json()["error"]["code"] == "too_large"
        assert source not in rejected.text
        restored = client.post("/v1/unmask", json={"payload": token, "payload_id": "amplified"})
        assert restored.status_code == 200
        assert restored.json()["result"] == source
