"""Optional NER boundary tests; no spaCy/model installation is required."""
from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from scripts.ner_service import MAX_BODY_BYTES, Boundary, NerSettings, create_app


class StubAnalyzer:
    def __init__(self, callback=None):
        self.callback = callback or (lambda _text: [])
        self.calls = 0

    def analyze(self, *, text, language, entities, score_threshold):
        assert language == "ru"
        assert entities == ["PERSON", "LOCATION"]
        assert score_threshold == 0.0
        self.calls += 1
        return self.callback(text)


def span(start, end, kind="PERSON", score=0.85):
    return SimpleNamespace(start=start, end=end, entity_type=kind, score=score)


def make_app(analyzer=None, **settings):
    analyzer = analyzer or StubAnalyzer()
    return create_app(NerSettings(**{"demo": True, **settings}), analyzer_factory=lambda: analyzer)


def test_authentication_health_and_typed_unicode_offsets():
    original = "😀 Клиент Дина Шварц, Москва."
    start = original.index("Дина")
    analyzer = StubAnalyzer(lambda _: [span(original.index("Москва"), len(original) - 1, "LOCATION"),
                                       span(start, start + len("Дина Шварц"))])
    with TestClient(make_app(analyzer, demo=False, token="test-service-key")) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer test-service-key"}).status_code == 200
        assert client.post("/analyze", json={"text": original}).status_code == 401
        assert client.post("/analyze", json={"text": original}, headers={"Authorization": "Bearer wrong"}).status_code == 401
        result = client.post("/analyze", json={"text": original}, headers={"Authorization": "Bearer test-service-key"})
        assert result.json() == {"entities": [
            {"start": start, "end": start + len("Дина Шварц"), "score": 0.85, "entity_type": "PERSON"},
            {"start": original.index("Москва"), "end": len(original) - 1, "score": 0.85, "entity_type": "LOCATION"},
        ]}
        assert result.headers["cache-control"] == "no-store"
        assert "Дина" not in result.text
        assert analyzer.calls == 1


def test_startup_requires_token_and_redacts_model_initialization_errors():
    settings = NerSettings()
    with pytest.raises(RuntimeError, match="SEIF_NER_TOKEN"):
        with TestClient(create_app(settings, analyzer_factory=StubAnalyzer)):
            pass

    def broken():
        raise ValueError("private-value@example.invalid")

    settings = NerSettings(demo=True)
    with pytest.raises(RuntimeError, match="initialization failed") as exc:
        with TestClient(create_app(settings, analyzer_factory=broken)):
            pass
    assert "private-value" not in str(exc.value)
    assert exc.value.__suppress_context__


@pytest.mark.parametrize("body", [
    {}, {"text": 42}, {"text": None}, {"text": ["secret@example.invalid"]},
    {"text": "secret@example.invalid", "extra": "secret@example.invalid"},
    {"text": "x" * 20_001},
])
def test_invalid_inputs_are_rejected_without_echo(body):
    analyzer = StubAnalyzer()
    with TestClient(make_app(analyzer)) as client:
        result = client.post("/analyze", json=body)
    assert result.status_code == 422
    assert set(result.json()) == {"error"}
    assert "secret@" not in result.text
    assert analyzer.calls == 0


@pytest.mark.parametrize("body", [b'{"text":"secret@example.invalid"', b'\xff\x00'])
def test_malformed_json_is_redacted(body):
    with TestClient(make_app()) as client:
        result = client.post("/analyze", content=body, headers={"Content-Type": "application/json"})
    assert result.status_code in {400, 422}
    assert "secret@" not in result.text


def test_limits_count_unicode_characters_and_reject_bytes_before_inference():
    analyzer = StubAnalyzer()
    with TestClient(make_app(analyzer)) as client:
        assert client.post("/analyze", json={"text": "😀" * 20_000}).status_code == 200
        assert client.post("/analyze", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413
        assert client.post("/analyze", json={"text": ""}).json() == {"entities": []}
    assert analyzer.calls == 1


def test_chunked_body_is_bounded_without_content_length():
    async def run():
        analyzer = StubAnalyzer()
        app = make_app(analyzer)

        async def body():
            yield b'{"text":"'
            yield b"x" * (MAX_BODY_BYTES // 2)
            yield b"x" * (MAX_BODY_BYTES // 2)

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ner") as client:
                response = await client.post("/analyze", content=body())
                assert response.status_code == 413
                assert analyzer.calls == 0
                assert (await client.post("/analyze", json={"text": "valid"})).status_code == 200

    asyncio.run(run())


def test_auth_failure_does_not_read_body():
    async def run():
        sent = []

        async def forbidden(*_args):
            raise AssertionError("Unauthenticated body or app must not be read.")

        async def send(message):
            sent.append(message)

        boundary = Boundary(forbidden, NerSettings(token="test-service-key"))
        await boundary({"type": "http", "path": "/analyze", "headers": []}, forbidden, send)
        assert sent[0]["status"] == 401
        assert boundary.inflight == 0

    asyncio.run(run())


def test_anonymous_health_post_does_not_reserve_capacity_or_read_body():
    async def run():
        sent = []

        async def forbidden(*_args):
            raise AssertionError("Anonymous health POST must not read its body or call the app.")

        async def send(message):
            sent.append(message)

        boundary = Boundary(forbidden, NerSettings(token="test-service-key"))
        await boundary({"type": "http", "path": "/health", "method": "POST", "headers": []}, forbidden, send)
        assert sent[0]["status"] == 401
        assert boundary.inflight == 0

    asyncio.run(run())


@pytest.mark.parametrize("drip_feed", [False, True])
def test_incomplete_authenticated_upload_times_out_and_releases_capacity(monkeypatch, caplog, drip_feed):
    monkeypatch.setattr("scripts.ner_service.BODY_READ_TIMEOUT_SECONDS", 0.02)

    async def run():
        sent, received = [], 0
        calls = 0

        async def app(_scope, _receive, send):
            nonlocal calls
            calls += 1
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def stalled_receive():
            nonlocal received
            received += 1
            if received == 1:
                return {"type": "http.request", "body": b'{"text":"private@example.invalid', "more_body": True}
            if drip_feed:
                await asyncio.sleep(0.005)
                return {"type": "http.request", "body": b"a", "more_body": True}
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        boundary = Boundary(app, NerSettings(token="test-service-key", max_http_inflight=1))
        await asyncio.wait_for(boundary({
            "type": "http", "path": "/analyze", "method": "POST",
            "headers": [(b"authorization", b"Bearer test-service-key")],
        }, stalled_receive, send), timeout=1)
        assert sent[0]["status"] == 408
        assert calls == boundary.inflight == 0
        assert b"private@" not in sent[1]["body"]
        sent.clear()
        await boundary({"type": "http", "path": "/health", "method": "GET", "headers": []}, stalled_receive, send)
        assert sent[0]["status"] == 200
        assert calls == 1
        assert boundary.inflight == 0

    asyncio.run(run())
    assert "private@" not in caplog.text


def test_ner_settings_repr_does_not_include_shared_credential():
    assert "test-service-key" not in repr(NerSettings(token="test-service-key"))


def test_upload_deadline_cannot_interrupt_an_oversized_body_response(monkeypatch):
    monkeypatch.setattr("scripts.ner_service.BODY_READ_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("scripts.ner_service.MAX_BODY_BYTES", 8)

    async def run():
        sent = []

        async def forbidden(*_args):
            raise AssertionError("An oversized upload must not call the app.")

        async def receive():
            return {"type": "http.request", "body": b"x" * 9, "more_body": True}

        async def slow_send(message):
            sent.append(message)
            if message["type"] == "http.response.start" and message["status"] == 413:
                await asyncio.sleep(0.03)

        boundary = Boundary(forbidden, NerSettings(token="test-service-key"))
        await asyncio.wait_for(boundary({
            "type": "http", "path": "/analyze", "method": "POST",
            "headers": [(b"authorization", b"Bearer test-service-key")],
        }, receive, slow_send), timeout=1)
        assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [413]
        assert sent[-1]["type"] == "http.response.body"
        assert boundary.inflight == 0

    asyncio.run(run())


@pytest.mark.parametrize("bad_span", [span(-1, 2), span(0, 100), span(0, 2, score=float("nan")),
                                     span(True, 2), span(0, 2, score=1.2)])
def test_invalid_model_offsets_and_scores_fail_closed(bad_span):
    with TestClient(make_app(StubAnalyzer(lambda _: [bad_span]))) as client:
        result = client.post("/analyze", json={"text": "secret@example.invalid"})
    assert result.status_code == 503
    assert "secret@" not in result.text


def test_model_exception_is_redacted_and_next_request_recovers(caplog):
    def callback(text):
        if text == "secret@example.invalid":
            raise RuntimeError(text)
        return [span(0, len(text))]

    with caplog.at_level(logging.WARNING):
        with TestClient(make_app(StubAnalyzer(callback))) as client:
            assert client.post("/analyze", json={"text": "secret@example.invalid"}).status_code == 503
            assert client.post("/analyze", json={"text": "Дина Шварц"}).status_code == 200
    assert "secret@" not in caplog.text


def test_cancelled_http_does_not_release_running_model_capacity(caplog):
    async def run():
        started, release = threading.Event(), threading.Event()
        loop_errors = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        def callback(text):
            if text.startswith("private"):
                started.set()
                if not release.wait(3):
                    raise RuntimeError("Test synchronization timed out.")
                raise RuntimeError(text)
            return []

        app = make_app(StubAnalyzer(callback), max_model_jobs=1)
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ner") as client:
                    task = asyncio.create_task(client.post("/analyze", json={"text": "private@example.invalid"}))
                    assert await asyncio.to_thread(started.wait, 2)
                    assert (await client.get("/health")).status_code == 200
                    assert (await client.post("/analyze", json={"text": "next"})).status_code == 429
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert app.state.model_inflight == 1
                    assert (await client.post("/analyze", json={"text": "next"})).status_code == 429
                    release.set()
                    for _ in range(100):
                        if app.state.model_inflight == 0:
                            break
                        await asyncio.sleep(0.01)
                    assert app.state.model_inflight == 0
                    assert (await client.post("/analyze", json={"text": "next"})).status_code == 200
            assert not app.state.ready
            assert app.state.pool._shutdown
            assert not loop_errors
        finally:
            release.set()
            loop.set_exception_handler(previous)

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    assert "private@" not in caplog.text


def test_unexpected_boundary_exception_cannot_reach_server_logging(caplog):
    async def run():
        sent = []

        async def broken(*_args):
            raise ValueError("sensitive-request-value")

        async def send(message):
            sent.append(message)

        boundary = Boundary(broken, NerSettings(demo=True))
        await boundary({"type": "http", "path": "/health", "method": "GET", "headers": []}, broken, send)
        assert sent[0]["status"] == 503
        assert boundary.inflight == 0

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    assert "sensitive-request-value" not in caplog.text


def test_model_queue_is_bounded_and_cancelled_queued_job_never_runs():
    async def run():
        started, release = threading.Event(), threading.Event()
        observed = []

        def callback(text):
            observed.append(text)
            if text == "first":
                started.set()
                if not release.wait(3):
                    raise RuntimeError("Test synchronization timed out.")
            return []

        app = make_app(StubAnalyzer(callback), max_model_jobs=4)
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ner") as client:
                    first = asyncio.create_task(client.post("/analyze", json={"text": "first"}))
                    assert await asyncio.to_thread(started.wait, 2)
                    queued = [asyncio.create_task(client.post("/analyze", json={"text": f"queued-{index}"})) for index in range(3)]
                    for _ in range(100):
                        if app.state.model_inflight == 4:
                            break
                        await asyncio.sleep(0.01)
                    assert app.state.model_inflight == 4
                    assert (await client.post("/analyze", json={"text": "overflow"})).status_code == 429
                    queued[1].cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await queued[1]
                    for _ in range(100):
                        if app.state.model_inflight == 3:
                            break
                        await asyncio.sleep(0.01)
                    assert app.state.model_inflight == 3
                    assert observed == ["first"]
                    release.set()
                    responses = await asyncio.gather(first, queued[0], queued[2])
                    assert all(response.status_code == 200 for response in responses)
                    assert observed == ["first", "queued-0", "queued-2"]
                    assert app.state.model_inflight == 0
        finally:
            release.set()

    asyncio.run(run())
