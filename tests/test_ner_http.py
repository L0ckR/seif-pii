"""Actual loopback HTTP checks of the opt-in aiohttp NER session; no GPU."""
from __future__ import annotations

import asyncio
import json
import os
import ssl
from contextlib import asynccontextmanager

import httpx
import pytest

from seif.config import Settings
from seif.ner import NerClient, NerUnavailable
from seif.ner_http import AiohttpNerSession

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402


@asynccontextmanager
async def server(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


def compressed(body):
    response = web.Response(body=body, content_type="application/json")
    response.enable_compression(force=web.ContentCoding.gzip)
    return response


def test_real_unicode_auth_compression_keepalive_and_no_proxy_or_cookie(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    text = "😀 е\u0301 Иван"
    connections, paths = set(), []

    async def handler(request):
        connections.add(id(request.transport))
        paths.append(request.path)
        assert request.headers["Authorization"] == "Bearer fixture-secret"
        assert "Cookie" not in request.headers
        if request.method == "GET":
            return compressed(b'{"status":"ok"}')
        assert await request.json() == {"text": text}
        start = text.index("Иван")
        body = {"entities": [{"start": start, "end": len(text), "score": .8, "entity_type": "PERSON"}]}
        response = compressed(json.dumps(body).encode())
        response.set_cookie("ignored", "fixture-value")
        return response

    async def scenario():
        async with server(handler) as url:
            client = NerClient(url + "/prefix", "fixture-secret", backend="aiohttp", max_concurrency=8)
            assert client.client._session is None
            try:
                await client.health()
                for _ in range(2):
                    found = await client.detect(text)
                    assert [(item.type, text[item.start:item.end]) for item in found] == [("PERSON", "Иван")]
                session = client.client._session
                assert session.connector.limit == session.connector.limit_per_host == 9
                assert isinstance(session.cookie_jar, aiohttp.DummyCookieJar)
                assert session.trust_env is False
            finally:
                await client.close()
            assert client.client._session.closed
        assert paths == ["/prefix/health", "/prefix/analyze", "/prefix/analyze"]
        assert len(connections) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("health", [False, True])
def test_decoded_response_size_limit_is_preserved_for_gzip(health):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return compressed(b" " * (5000 if health else 270000) + b"{}")
        return web.json_response({"status": "ok"} if request.method == "GET" else {"entities": []})

    async def scenario():
        async with server(handler) as url:
            client = NerClient(url, "fixture", backend="aiohttp")
            try:
                operation = client.health() if health else client.detect("Иван")
                with pytest.raises(NerUnavailable):
                    await operation
                await client.health()
            finally:
                await client.close()
        assert calls == 2

    asyncio.run(scenario())


def test_redirect_is_rejected_without_following_it_and_429_retries_stay_bounded():
    paths = []
    mode = "redirect"

    async def handler(request):
        paths.append(request.path)
        return (web.Response(status=302, headers={"Location": "/unexpected"})
                if mode == "redirect" else web.Response(status=429))

    async def scenario():
        nonlocal mode
        async with server(handler) as url:
            client = NerClient(url, "fixture", backend="aiohttp")
            try:
                with pytest.raises(NerUnavailable):
                    await client.detect("Иван")
                assert paths == ["/analyze"]
                mode = "overload"
                with pytest.raises(NerUnavailable):
                    await client.detect("Иван")
                assert paths == ["/analyze"] * 5
            finally:
                await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_stream_timeout_or_cancellation_releases_connection_and_preserves_safe_error(cancel):
    entered, release = None, None
    transports = []

    async def handler(request):
        if request.path == "/health":
            return web.json_response({"status": "ok"})
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        transports.append(request.transport)
        entered.set()
        await release.wait()
        return response

    async def scenario():
        nonlocal entered, release
        entered, release = asyncio.Event(), asyncio.Event()
        async with server(handler) as url:
            client = NerClient(url, "fixture", timeout=.03 if not cancel else 1,
                               backend="aiohttp", max_concurrency=1)
            try:
                job = asyncio.create_task(client.detect("private fixture"))
                await entered.wait()
                if cancel:
                    job.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await job
                else:
                    with pytest.raises(NerUnavailable) as caught:
                        await asyncio.wait_for(job, 1)
                    assert "private fixture" not in str(caught.value)
                release.set()
                await client.health()
                assert transports[0].is_closing()
            finally:
                release.set()
                await client.close()

    asyncio.run(scenario())


def test_aiohttp_outer_deadline_includes_admission_before_session_creation():
    async def scenario():
        client = NerClient("http://127.0.0.1:1", "fixture", timeout=.01, max_concurrency=1, backend="aiohttp")
        await client.capacity.acquire()
        try:
            with pytest.raises(NerUnavailable):
                await client.detect("Иван")
            assert client.client._session is None
        finally:
            client.capacity.release()
            await client.close()

    asyncio.run(scenario())


def test_tls_context_keeps_certificate_and_hostname_verification(monkeypatch):
    contexts = []
    original = ssl.create_default_context

    def context(**kwargs):
        value = original(**kwargs)
        contexts.append((kwargs, value))
        return value

    monkeypatch.setattr(ssl, "create_default_context", context)

    async def scenario():
        client = AiohttpNerSession("https://ner.invalid", "fixture", 20, 4)
        try:
            client._get_session()
            assert len(contexts) == 1
            options, value = contexts[0]
            assert options["cafile"]
            assert value.check_hostname is True
            assert value.verify_mode == ssl.CERT_REQUIRED
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_selected_backend_does_not_replace_injected_httpx_transport():
    async def scenario():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"entities": []}))
        client = NerClient("http://ner", "fixture", backend="aiohttp", transport=transport)
        try:
            assert isinstance(client.client, httpx.AsyncClient)
            assert await client.detect("Иван") == []
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["", "requests", "AIOHTTP", None])
def test_invalid_backend_rejected(backend):
    with pytest.raises(ValueError, match="ner_http_backend"):
        Settings(ner_http_backend=backend)
    with pytest.raises(ValueError, match="NER HTTP backend"):
        NerClient("http://ner", "fixture", backend=backend)


def test_backend_config_default_and_env(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("SEIF_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SEIF_DEMO", "1")
    monkeypatch.delenv("SEIF_NER_HTTP_BACKEND", raising=False)
    assert Settings.from_env().ner_http_backend == "httpx"
    monkeypatch.setenv("SEIF_NER_HTTP_BACKEND", "aiohttp")
    assert Settings.from_env().ner_http_backend == "aiohttp"
    monkeypatch.setenv("SEIF_MAX_INFLIGHT", "256")
    settings = Settings.from_env()
    assert settings.max_inflight == 256
    assert settings.max_inflight_body_bytes == 64 * 1024 * 1024
    assert settings.ner_max_concurrency == 4
