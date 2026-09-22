"""Upload refactors preserve cancellation, disconnect, replay and capture state."""
from __future__ import annotations

import asyncio

import pytest
from prometheus_client import CollectorRegistry

from scripts.ner_service import Boundary as NerBoundary
from scripts.ner_service import NerSettings
from seif.app import Boundary
from seif.config import Settings


class CaptureStub:
    def __init__(self):
        self.records = []

    def submit(self, metadata, body):
        self.records.append((metadata, body))


def make_boundary(kind, app, capture=None):
    if kind == "api":
        return Boundary(app, Settings(max_inflight=1), CollectorRegistry(), capture=capture)
    return NerBoundary(app, NerSettings(demo=True, max_http_inflight=1))


def request_scope():
    return {"type": "http", "method": "POST", "path": "/process", "headers": []}


@pytest.mark.parametrize("kind", ["api", "ner"])
def test_partial_disconnect_releases_capacity_without_response_or_complete_capture(kind):
    async def run():
        sent, received = [], 0
        capture = CaptureStub()

        async def forbidden(*_args):
            pytest.fail("A disconnected partial upload must not reach the app")

        boundary = make_boundary(kind, forbidden, capture)

        async def receive():
            nonlocal received
            received += 1
            if received == 1:
                return {"type": "http.request", "body": b"partial", "more_body": True}
            if kind == "api":
                assert boundary.buffered_bytes == 7
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await boundary(request_scope(), receive, send)
        assert boundary.inflight == 0
        assert sent == []
        if kind == "api":
            assert boundary.buffered_bytes == 0
            assert len(capture.records) == 1
            metadata, body = capture.records[0]
            assert metadata["body_complete"] is False
            assert metadata["status_code"] == 500
            assert body == b""

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["api", "ner"])
def test_cancelled_upload_releases_capacity_and_preserves_incomplete_capture(kind):
    async def run():
        chunks, capture = asyncio.Queue(), CaptureStub()
        chunks.put_nowait({"type": "http.request", "body": b"partial", "more_body": True})

        async def forbidden(*_args):
            pytest.fail("A cancelled partial upload must not reach the app or send a response")

        boundary = make_boundary(kind, forbidden, capture)
        task = asyncio.create_task(boundary(request_scope(), chunks.get, forbidden))
        await asyncio.sleep(0)
        assert boundary.inflight == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert boundary.inflight == 0
        if kind == "api":
            assert boundary.buffered_bytes == 0
            assert len(capture.records) == 1
            assert capture.records[0][0]["body_complete"] is False
            assert capture.records[0][1] == b""

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["api", "ner"])
def test_empty_complete_upload_replays_once_then_forwards_disconnect(kind):
    async def run():
        reads, downstream, sent = 0, [], []
        capture = CaptureStub()

        async def receive():
            nonlocal reads
            reads += 1
            if reads == 1:
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.disconnect"}

        async def app(_scope, bounded_receive, send):
            downstream.append(await bounded_receive())
            downstream.append(await bounded_receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def send(message):
            sent.append(message)

        boundary = make_boundary(kind, app, capture)
        await boundary(request_scope(), receive, send)
        assert reads == 2
        assert downstream == [{"type": "http.request", "body": b"", "more_body": False},
                              {"type": "http.disconnect"}]
        assert sent[0]["status"] == 200
        assert boundary.inflight == 0
        if kind == "api":
            assert boundary.buffered_bytes == 0
            assert capture.records[0][0]["body_complete"] is True
            assert capture.records[0][1] == b""

    asyncio.run(run())


@pytest.mark.parametrize(("max_body", "max_buffered", "status", "code"),
                         [(8, 100, 413, b"too_large"), (100, 8, 429, b"body_budget")])
def test_api_upload_deadline_does_not_interrupt_body_rejection_response(max_body, max_buffered, status, code):
    async def run():
        sent, capture = [], CaptureStub()

        async def forbidden(*_args):
            pytest.fail("An excessive upload must not reach the app")

        async def receive():
            return {"type": "http.request", "body": b"x" * 9, "more_body": True}

        async def slow_send(message):
            sent.append(message)
            if message["type"] == "http.response.start":
                await asyncio.sleep(0.03)

        settings = Settings(max_body_bytes=max_body, max_inflight_body_bytes=max_buffered,
                            request_body_timeout_seconds=0.01)
        boundary = Boundary(forbidden, settings, CollectorRegistry(), capture=capture)
        await asyncio.wait_for(boundary(request_scope(), receive, slow_send), timeout=1)
        assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [status]
        assert code in sent[-1]["body"]
        assert boundary.buffered_bytes == boundary.inflight == 0
        assert capture.records[0][0]["body_complete"] is False
        assert capture.records[0][0]["status_code"] == status
        assert capture.records[0][1] == b""

    asyncio.run(run())
