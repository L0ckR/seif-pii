"""Independent behavioral and security tests using fictional input only."""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from seif.app import Boundary, create_app
from seif.config import Policy, Settings


@pytest.fixture
def app():
    return create_app(Settings(demo=True, policies={"demo": Policy(rps=10_000)}))


@pytest.fixture
def client(app):
    with TestClient(app) as session:
        yield session


def process(client, text, payload_id="roundtrip", **kwargs):
    return client.post("/process", json={"payload": text, "payload_id": payload_id}, **kwargs)


def test_contract_roundtrip_and_retries_are_idempotent(client):
    original = "Клиент Иванов Иван Иванович; email ivan@example.net."
    first = process(client, original)
    assert first.status_code == 200
    assert set(first.json()) == {"result"}
    masked = first.json()["result"]
    assert "ivan@example.net" not in masked
    assert masked != original
    # An HTTP retry of masking must not toggle into the reverse direction.
    assert process(client, original).json() == first.json()
    assert process(client, masked).json() == {"result": original}
    assert process(client, masked).json() == {"result": original}
    assert process(client, original).json() == first.json()


def test_plain_text_and_empty_input_roundtrip(client):
    for number, original in enumerate(("", "Объясни, почему небо голубое.")):
        response = process(client, original, f"plain-{number}")
        assert response.status_code == 200
        assert process(client, response.json()["result"], f"plain-{number}").json() == {"result": original}


def test_unicode_and_punctuation_are_preserved(client):
    original = "«Email: hello@example.net» — 😊;\nТелефон: +7 (999) 123-45-67!\tе\u0308."
    response = process(client, original, "unicode")
    assert response.status_code == 200
    masked = response.json()["result"]
    assert len(masked) == len(original)
    for index, character in enumerate(original):
        if not character.isalnum():
            assert masked[index] == character
    assert process(client, masked, "unicode").json()["result"] == original


def test_same_id_different_input_is_conflict(client):
    process(client, "first@example.net", "collision").raise_for_status()
    response = process(client, "second@example.net", "collision")
    assert response.status_code == 409
    assert "first@example.net" not in response.text
    assert "second@example.net" not in response.text


@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_modes_roundtrip_without_plain_values_in_entity_metadata(client, mode):
    original = "Email: client@example.net; телефон: +7 (999) 123-45-67."
    payload_id = f"mode-{mode}"
    response = client.post("/v1/mask", json={"payload": original, "payload_id": payload_id, "mode": mode})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == mode
    assert "client@example.net" not in data["result"]
    assert "client@example.net" not in str(data["entities"])
    restored = client.post("/v1/unmask", json={"payload": data["result"], "payload_id": payload_id})
    assert restored.status_code == 200
    assert restored.json()["result"] == original


def test_token_restoration_in_edited_llm_reply(client, caplog):
    response = client.post("/v1/mask", json={"payload": "alice@example.net", "payload_id": "llm", "mode": "token"})
    assert response.status_code == 200
    token = response.json()["result"]
    reply = f"Ответ модели: {token}. Повтор: {token}!"
    with caplog.at_level(logging.INFO, logger="seif.audit"):
        restored = client.post("/v1/unmask", json={"payload": reply, "payload_id": "llm"})
    assert restored.status_code == 200
    assert restored.json()["result"] == "Ответ модели: alice@example.net. Повтор: alice@example.net!"
    audit = [json.loads(record.message) for record in caplog.records if record.name == "seif.audit"]
    assert {"authorize", "fingerprint", "vault_read", "restore"} <= set(audit[-1]["stages_ms"])


def test_unknown_token_cannot_be_demasked(client):
    client.post("/v1/mask", json={"payload": "alice@example.net", "payload_id": "unknown-token", "mode": "token"}).raise_for_status()
    response = client.post("/v1/unmask", json={"payload": "⟦PD:EMAIL:0000000000000000⟧", "payload_id": "unknown-token"})
    assert 400 <= response.status_code < 500
    assert "alice@example.net" not in response.text


def test_tenant_authentication_and_isolation():
    settings = Settings(policies={
        "alpha": Policy(api_key="alpha-secret", rps=10_000),
        "beta": Policy(api_key="beta-secret", rps=10_000),
        "disabled": Policy(api_key="disabled-secret", enabled=False),
    })
    with TestClient(create_app(settings)) as client:
        original = "alice@example.net"
        for headers in ({}, {"X-System-ID": "alpha"}, {"X-System-ID": "alpha", "X-API-Key": "wrong"},
                        {"X-System-ID": "unknown", "X-API-Key": "alpha-secret"},
                        {"X-System-ID": "disabled", "X-API-Key": "disabled-secret"}):
            response = process(client, original, headers=headers)
            assert response.status_code in {401, 403}
            assert original not in response.text
        alpha = {"X-System-ID": "alpha", "X-API-Key": "alpha-secret"}
        beta = {"X-System-ID": "beta", "X-API-Key": "beta-secret"}
        first = process(client, original, "shared-id", headers=alpha)
        assert first.status_code == 200
        masked = first.json()["result"]
        denied = client.post("/v1/unmask", json={"payload": masked, "payload_id": "shared-id"}, headers=beta)
        assert denied.status_code in {403, 404, 410}
        assert original not in denied.text
        second = process(client, "bob@example.net", "shared-id", headers=beta)
        assert second.status_code == 200
        assert process(client, masked, "shared-id", headers=alpha).json()["result"] == original
        assert process(client, second.json()["result"], "shared-id", headers=beta).json()["result"] == "bob@example.net"


@pytest.mark.parametrize("body", [
    {"payload": {"private": "do-not-reflect@example.net"}, "payload_id": "invalid"},
    {"payload": "do-not-reflect@example.net", "payload_id": None},
    {"payload": "do-not-reflect@example.net", "payload_id": "invalid", "unexpected": "private"},
    {"payload": 12345, "payload_id": "invalid"},
])
def test_invalid_request_never_reflects_input(client, body):
    response = client.post("/process", json=body)
    assert response.status_code in {400, 422}
    assert "do-not-reflect@example.net" not in response.text
    assert "12345" not in response.text
    assert "error" in response.json()


def test_malformed_json_does_not_reflect_input(client):
    response = client.post("/process", content='{"payload": "do-not-reflect@example.net",', headers={"Content-Type": "application/json"})
    assert response.status_code in {400, 422}
    assert "do-not-reflect@example.net" not in response.text


def test_bad_mode_is_not_silently_accepted(client):
    response = client.post("/v1/mask", json={"payload": "alice@example.net", "payload_id": "bad-mode", "mode": "plaintext"})
    assert response.status_code in {400, 422}
    assert "alice@example.net" not in response.text


def test_system_policy_types_and_unmask_permission():
    settings = Settings(demo=True, policies={"demo": Policy(types=("EMAIL",), allow_unmask=False)})
    with TestClient(create_app(settings)) as client:
        source = "Email: alice@example.net. Телефон: +7 (999) 123-45-67."
        masked_response = process(client, source, "policy")
        assert masked_response.status_code == 200
        masked = masked_response.json()["result"]
        assert "alice@example.net" not in masked
        assert "+7 (999) 123-45-67" in masked
        denied = process(client, masked, "policy")
        assert denied.status_code == 403
        assert "alice@example.net" not in denied.text


def test_cooccurrence_policy():
    settings = Settings(demo=True, policies={"demo": Policy(min_types=2, required_types=("CARD",))})
    with TestClient(create_app(settings)) as client:
        pin_only = "ПИН-код карты: 4321."
        assert process(client, pin_only, "pin-only").json()["result"] == pin_only
        both = "Номер карты: 4111 1111 1111 1111; ПИН-код карты: 4321."
        response = process(client, both, "both")
        assert response.status_code == 200
        assert "4111" not in response.json()["result"]
        assert "4321" not in response.json()["result"]


def test_originals_are_encrypted_in_vault(app, client):
    source = "alice-private@example.net"
    process(client, source, "encrypted").raise_for_status()
    assert app.state.vault.records
    for key, (_, encrypted) in app.state.vault.records.items():
        assert source not in key
        assert source.encode() not in encrypted


def test_expired_mapping_cannot_be_restored(app, client, monkeypatch):
    import seif.vault as vault_module

    original = "expired@example.net"
    masked = process(client, original, "expiry").json()["result"]
    real_now = vault_module.time.monotonic()
    monkeypatch.setattr(vault_module, "time", SimpleNamespace(monotonic=lambda: real_now + app.state.vault.settings.ttl_seconds + 1))
    response = client.post("/v1/unmask", json={"payload": masked, "payload_id": "expiry"})
    assert response.status_code == 410
    assert original not in response.text


def test_capacity_rejection_is_retryable_and_keeps_existing_mapping():
    settings = Settings(demo=True, max_records=1)
    with TestClient(create_app(settings)) as client:
        original = "one@example.net"
        first = process(client, original, "one")
        assert first.status_code == 200
        second = process(client, "two@example.net", "two")
        assert second.status_code == 429
        assert "Retry-After" in second.headers
        assert process(client, first.json()["result"], "one").json()["result"] == original


def test_large_text_100000_words_roundtrips(client):
    # Word count is explicit; model-specific BPE token counts are not inferred.
    original = "слово " * 100_000 + "email: large@example.net"
    first = process(client, original, "large")
    assert first.status_code == 200
    assert "large@example.net" not in first.json()["result"]
    assert process(client, first.json()["result"], "large").json()["result"] == original


def test_concurrent_retries_commit_one_mapping(client):
    original = "Concurrency: alice@example.net"
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: process(client, original, "concurrent"), range(16)))
    assert all(response.status_code == 200 for response in responses)
    assert len({response.json()["result"] for response in responses}) == 1
    assert process(client, responses[0].json()["result"], "concurrent").json()["result"] == original


def test_ciphertext_tampering_fails_closed(app, client):
    original = "tampering@example.net"
    masked = process(client, original, "tampering").json()["result"]
    key, (expiry, blob) = next(iter(app.state.vault.records.items()))
    corrupted = blob[:-1] + bytes([blob[-1] ^ 1])
    app.state.vault.records[key] = (expiry, corrupted)
    response = process(client, masked, "tampering")
    assert response.status_code == 503
    assert original not in response.text
    assert masked not in response.text


def test_logs_and_metrics_do_not_contain_payload_or_correlation(client, caplog):
    original = "audit-private@example.net"
    correlation = "private-correlation@example.net"
    with caplog.at_level(logging.INFO, logger="seif.audit"):
        process(client, original, correlation).raise_for_status()
    assert original not in caplog.text
    assert correlation not in caplog.text
    assert "EMAIL" in caplog.text
    audit = [json.loads(record.message) for record in caplog.records if record.name == "seif.audit"]
    assert {"authorize", "fingerprint", "vault_read", "detect", "transform", "vault_write"} <= set(audit[-1]["stages_ms"])
    assert all(isinstance(value, (int, float)) and value >= 0 for value in audit[-1]["stages_ms"].values())
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert original not in metrics.text
    assert correlation not in metrics.text
    assert "seif_http_duration_seconds" in metrics.text


def test_detector_failure_does_not_expose_input(client, monkeypatch, caplog):
    def broken_detector(*args, **kwargs):
        raise RuntimeError("secret-exception@example.net")

    monkeypatch.setattr("seif.app.detect", broken_detector)
    with caplog.at_level(logging.INFO, logger="seif.audit"):
        response = process(client, "request-private@example.net", "fault")
    assert response.status_code == 503
    assert "secret-exception@example.net" not in caplog.text + response.text
    assert "request-private@example.net" not in caplog.text + response.text


def test_unpaired_surrogate_is_rejected_without_server_failure(client):
    response = client.post("/process", content=b'{"payload":"private@example.net\\ud800","payload_id":"surrogate"}',
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "private@example.net" not in response.text


def test_payload_size_limit_has_no_input_echo():
    settings = Settings(demo=True, max_payload_chars=16)
    with TestClient(create_app(settings)) as client:
        response = process(client, "oversized-private@example.net", "oversized")
        assert response.status_code == 413
        assert "oversized-private@example.net" not in response.text


def test_rate_limit_returns_retry_after():
    settings = Settings(demo=True, policies={"demo": Policy(rps=1)})
    with TestClient(create_app(settings)) as client:
        process(client, "a@example.net", "a").raise_for_status()
        response = process(client, "b@example.net", "b")
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1


def test_mode_change_for_same_correlation_is_conflict(client):
    base = {"payload": "alice@example.net", "payload_id": "change-mode"}
    first = client.post("/v1/mask", json={**base, "mode": "mask"})
    assert first.status_code == 200
    second = client.post("/v1/mask", json={**base, "mode": "token"})
    assert second.status_code == 409


@pytest.mark.parametrize("policy", [
    Policy(types=("EMIAL",)),
    Policy(required_types=("EMIAL",)),
    Policy(extra_rules=({"type": "CLIENT_ID", "pattern": "("},)),
    Policy(extra_rules=({"type": "CLIENT_ID", "pattern": ".*"},)),
])
def test_invalid_rules_and_unknown_types_fail_at_startup(policy):
    with pytest.raises(ValueError):
        create_app(Settings(demo=True, policies={"demo": policy}))


def test_configured_extra_type_has_same_protection_and_roundtrip():
    policy = Policy(types=("CLIENT_ID",), extra_rules=({"type": "CLIENT_ID", "pattern": r"\bCLT-\d{8}\b"},))
    with TestClient(create_app(Settings(demo=True, policies={"demo": policy}))) as client:
        original = "Внутренний номер: CLT-12345678."
        first = process(client, original, "extension")
        assert first.status_code == 200
        assert "CLT-12345678" not in first.json()["result"]
        assert process(client, first.json()["result"], "extension").json() == {"result": original}


def test_large_job_cancellation_keeps_cpu_slot_until_thread_finishes(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def detector(text, **kwargs):
        if len(text) > 16000:
            calls.append(text)
            entered.set()
            try:
                if not release.wait(3):
                    raise RuntimeError("Test detector release timed out")
            finally:
                finished.set()
        return []

    monkeypatch.setattr("seif.app.detect", detector)
    app = create_app(Settings(demo=True, cpu_workers=1))

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/process", json={"payload": "слово " * 3000, "payload_id": "cancel-first"}))
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    # Cancelling HTTP must not free a slot while the CPU job runs.
                    for number in range(4):
                        response = await client.post("/process", json={"payload": "слово " * 3000, "payload_id": f"cancel-next-{number}"})
                        assert response.status_code == 429
                        assert response.json()["error"]["code"] == "cpu_busy"
                        assert response.headers["Retry-After"] == "1"
                        assert response.headers["Cache-Control"] == "no-store"
                    assert len(calls) == 1
                    assert (await client.get("/health")).status_code == 200
                    small = await client.post("/process", json={"payload": "Без личных данных.", "payload_id": "small-during-cpu"})
                    assert small.status_code == 200
                    release.set()
                    assert await asyncio.to_thread(finished.wait, 2)
                    await asyncio.sleep(0)
                    recovered = await client.post("/process", json={"payload": "слово " * 3000, "payload_id": "after-cancel"})
                    assert recovered.status_code == 200
                    assert len(calls) == 2
                finally:
                    release.set()
                    if not first.done():
                        first.cancel()
                        await asyncio.gather(first, return_exceptions=True)

    asyncio.run(scenario())


def test_large_detector_failure_releases_capacity_and_fails_closed(monkeypatch, caplog):
    from seif.detector import detect as real_detect

    def detector(text, **kwargs):
        if text.startswith("failure-private@example.net"):
            raise RuntimeError("internal-private@example.net")
        return real_detect(text, **kwargs)

    monkeypatch.setattr("seif.app.detect", detector)
    app = create_app(Settings(demo=True, cpu_workers=1))
    with TestClient(app) as client, caplog.at_level(logging.INFO, logger="seif.audit"):
        failed = process(client, "failure-private@example.net " + "слово " * 3000, "large-failure")
        assert failed.status_code == 503
        assert "failure-private@example.net" not in failed.text + caplog.text
        assert "internal-private@example.net" not in failed.text + caplog.text
        source = "слово " * 3000 + " recovery@example.net"
        recovered = process(client, source, "large-recovered")
        assert recovered.status_code == 200
        assert "recovery@example.net" not in recovered.json()["result"]
        assert process(client, recovered.json()["result"], "large-recovered").json()["result"] == source


def test_cancelled_cpu_failure_does_not_reach_unhandled_exception_hook(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def detector(text, **kwargs):
        entered.set()
        try:
            if not release.wait(3):
                raise RuntimeError("Test detector release timed out")
            raise RuntimeError("detached-private@example.net")
        finally:
            finished.set()

    monkeypatch.setattr("seif.app.detect", detector)
    app = create_app(Settings(demo=True, cpu_workers=1))

    async def scenario():
        unhandled = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                request_task = asyncio.create_task(client.post("/process", json={"payload": "слово " * 3000, "payload_id": "detached-failure"}))
                assert await asyncio.to_thread(entered.wait, 2)
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
                del request_task
                release.set()
                assert await asyncio.to_thread(finished.wait, 2)
                await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_lifespan_drains_cancelled_cpu_job_without_blocking_event_loop(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    worker_timed_out = []

    def detector(text, **kwargs):
        entered.set()
        if not release.wait(1):
            worker_timed_out.append(True)
        return []

    monkeypatch.setattr("seif.app.detect", detector)
    app = create_app(Settings(demo=True, cpu_workers=1))

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/process", json={"payload": "слово " * 3000, "payload_id": "shutdown"}))
                assert await asyncio.to_thread(entered.wait, 2)
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                # Only the event loop can release the worker; blocking shutdown fails.
                asyncio.get_running_loop().call_later(0.05, release.set)
        assert release.is_set()
        assert not worker_timed_out

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_health_reports_actual_python_threading_runtime(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["python"] == sys.version.split()[0]
    assert body["free_threaded"] is bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    assert body["gil_enabled"] is getattr(sys, "_is_gil_enabled", lambda: True)()


@pytest.mark.parametrize("compiled_free_threaded,gil_enabled", [(False, True), (True, True)])
def test_required_free_threading_fails_startup_and_closes_resources(monkeypatch, compiled_free_threaded, gil_enabled):
    real_get = sysconfig.get_config_var
    monkeypatch.setenv("SEIF_REQUIRE_FREE_THREADING", "1")
    monkeypatch.setattr("seif.app.sysconfig.get_config_var", lambda key: compiled_free_threaded if key == "Py_GIL_DISABLED" else real_get(key))
    monkeypatch.setattr("seif.app.sys._is_gil_enabled", lambda: gil_enabled, raising=False)
    app = create_app(Settings(demo=True))
    closed = []

    async def close():
        closed.append(True)

    monkeypatch.setattr(app.state.vault, "close", close)
    with pytest.raises(RuntimeError, match="requires a free-threaded Python"):
        with TestClient(app):
            pass
    assert closed == [True]


def test_storage_startup_failure_closes_resources(monkeypatch):
    app = create_app(Settings(demo=True))
    closed = []

    async def ping():
        raise ConnectionError("Storage unavailable")

    async def close():
        closed.append(True)

    monkeypatch.setattr(app.state.vault, "ping", ping)
    monkeypatch.setattr(app.state.vault, "close", close)
    with pytest.raises(ConnectionError):
        with TestClient(app):
            pass
    assert closed == [True]


def test_concurrent_chunked_requests_enforce_shared_body_budget():
    async def scenario():
        async def downstream(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        boundary = Boundary(downstream, Settings(max_inflight_body_bytes=10), CollectorRegistry())
        queues = [asyncio.Queue(), asyncio.Queue()]
        output = [[], []]

        async def run(index):
            async def send(message):
                output[index].append(message)
            scope = {"type": "http", "method": "POST", "path": "/process", "headers": []}
            await boundary(scope, queues[index].get, send)

        for queue in queues:
            queue.put_nowait({"type": "http.request", "body": b"1234", "more_body": True})
        first, second = asyncio.create_task(run(0)), asyncio.create_task(run(1))
        await asyncio.sleep(0)
        assert boundary.buffered_bytes == 8
        queues[0].put_nowait({"type": "http.request", "body": b"567", "more_body": False})
        await first
        assert output[0][0]["status"] == 429
        assert boundary.buffered_bytes == 4
        assert (b"retry-after", b"1") in output[0][0]["headers"]
        queues[1].put_nowait({"type": "http.request", "body": b"56", "more_body": False})
        await second
        assert output[1][0]["status"] == 200
        assert boundary.buffered_bytes == 0
        assert boundary.inflight == 0

    asyncio.run(scenario())


def test_body_budget_rejects_content_length_before_reading_and_releases_on_cancel():
    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def downstream(scope, receive, send):
            await receive()
            entered.set()
            await finish.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        boundary = Boundary(downstream, Settings(max_inflight_body_bytes=10), CollectorRegistry())

        async def receive():
            return {"type": "http.request", "body": b"123456", "more_body": False}

        async def discard(message):
            pass

        def scope():
            return {"type": "http", "method": "POST", "path": "/process", "headers": [(b"content-length", b"6")]}
        first = asyncio.create_task(boundary(scope(), receive, discard))
        await entered.wait()
        assert boundary.buffered_bytes == 6
        output = []

        async def must_not_receive():
            pytest.fail("Known excessive body should be rejected before reading")

        async def collect(message):
            output.append(message)

        await boundary(scope(), must_not_receive, collect)
        assert output[0]["status"] == 429
        assert boundary.buffered_bytes == 6
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert boundary.buffered_bytes == 0
        assert boundary.inflight == 0
        finish.set()
        output.clear()
        await boundary(scope(), receive, collect)
        assert output[0]["status"] == 200
        assert boundary.buffered_bytes == 0

    asyncio.run(scenario())


def test_cancelled_partial_body_releases_aggregate_budget():
    async def scenario():
        async def downstream(scope, receive, send):
            pytest.fail("A cancelled partial request must not reach the application")

        boundary = Boundary(downstream, Settings(max_inflight_body_bytes=10), CollectorRegistry())
        chunks = asyncio.Queue()
        chunks.put_nowait({"type": "http.request", "body": b"12345678", "more_body": True})

        async def send(message):
            pass

        task = asyncio.create_task(boundary({"type": "http", "method": "POST", "path": "/process", "headers": []}, chunks.get, send))
        await asyncio.sleep(0)
        assert boundary.buffered_bytes == 8
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert boundary.buffered_bytes == 0
        assert boundary.inflight == 0

    asyncio.run(scenario())
