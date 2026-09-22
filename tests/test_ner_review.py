"""Regressions for false readiness and unbounded NER health responses."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from scripts.ner_service import NerSettings, create_app
from seif.ner import NerClient, NerUnavailable


def test_private_client_readiness_rejects_mismatched_shared_token():
    async def run():
        app = create_app(NerSettings(token="server-only-test-key"), analyzer_factory=object)
        async with app.router.lifespan_context(app):
            invalid = NerClient("http://ner", "mismatched-test-key", transport=httpx.ASGITransport(app=app))
            valid = NerClient("http://ner", "server-only-test-key", transport=httpx.ASGITransport(app=app))
            try:
                with pytest.raises(NerUnavailable) as caught:
                    await invalid.health()
                assert "mismatched-test-key" not in str(caught.value)
                await valid.health()
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ner") as probe:
                    assert (await probe.get("/health")).status_code == 200
            finally:
                await invalid.close()
                await valid.close()

    asyncio.run(run())


def test_health_cannot_accept_an_unbounded_json_body():
    async def run():
        client = NerClient("http://ner", "test-key", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"status": "ok", "padding": "x" * 262_144})))
        try:
            with pytest.raises(NerUnavailable):
                await client.health()
        finally:
            await client.close()

    asyncio.run(run())


def test_health_has_total_deadline_even_when_data_keeps_arriving():
    async def run():
        class SlowHealth(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b'{"status":"ok"'
                # An unfinished JSON document keeps each individual read alive.
                while True:
                    await asyncio.sleep(0.01)
                    yield b" "

            async def aclose(self):
                self.closed = True

        stream = SlowHealth()
        client = NerClient("http://ner", "test-key", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream)))
        try:
            health = client.health()
            with pytest.raises(NerUnavailable):
                await asyncio.wait_for(health, timeout=3)
            assert stream.closed
        finally:
            await client.close()

    asyncio.run(run())
