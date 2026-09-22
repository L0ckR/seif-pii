"""Immediate framework callbacks preserve dispatch, binding, and failures."""

import asyncio
import inspect
import threading

import httpx
import pytest
from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from seif.async_callbacks import immediate_response


class WorkerExit(BaseException):
    """Verify failures outside Exception also stay owned by the caller."""


def test_immediate_response_preserves_signature_binding_and_loop_turn():
    class Responder:
        def __init__(self):
            self.calls = []

        @immediate_response
        def reply(self, value: int = 7):
            self.calls.append((value, asyncio.current_task(), threading.get_ident()))
            return {"value": value}

    responder = Responder()
    assert inspect.signature(Responder.reply) == inspect.signature(Responder.reply.__wrapped__)
    assert inspect.iscoroutinefunction(responder.reply)

    async def scenario():
        loop = asyncio.get_running_loop()
        other_turns = []
        loop.call_soon(other_turns.append, "scheduled")
        pending = responder.reply(9)
        assert pending.done()
        assert await pending == {"value": 9}
        assert responder.calls == [(9, asyncio.current_task(), threading.get_ident())]
        assert other_turns == []

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [ValueError, WorkerExit, asyncio.CancelledError])
def test_callback_failure_keeps_identity_without_unhandled_logging(failure_type, caplog):
    failure = failure_type("synthetic-private-value@example.invalid")

    @immediate_response
    def broken():
        raise failure

    async def scenario():
        unhandled = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        with pytest.raises(failure_type) as caught:
            await broken()
        assert caught.value is failure
        assert unhandled == []

    asyncio.run(scenario())
    assert "synthetic-private-value" not in caplog.text


def test_fastapi_and_starlette_await_callbacks_without_threadpool(monkeypatch):
    def reject_threadpool(*_args, **_kwargs):
        raise AssertionError("Threadpool dispatch must not handle immediate responses")

    monkeypatch.setattr("fastapi.routing.run_in_threadpool", reject_threadpool)
    monkeypatch.setattr("starlette._exception_handler.run_in_threadpool", reject_threadpool)
    monkeypatch.setattr("starlette.middleware.errors.run_in_threadpool", reject_threadpool)

    async def scenario():
        app = FastAPI()
        owner_task, owner_thread = asyncio.current_task(), threading.get_ident()
        observed = []

        @immediate_response
        def endpoint(value: int = 7):
            observed.append((asyncio.current_task(), threading.get_ident()))
            return {"value": value}

        @immediate_response
        def http_error(_request, exc):
            observed.append((asyncio.current_task(), threading.get_ident()))
            return JSONResponse({"error": "synthetic"}, status_code=exc.status_code)

        app.get("/response")(endpoint)
        app.add_exception_handler(HTTPException, http_error)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            valid = await client.get("/response?value=11")
            assert valid.status_code == 200
            assert valid.json() == {"value": 11}
            missing = await client.get("/missing")
            assert missing.status_code == 404
            assert missing.json() == {"error": "synthetic"}
        assert observed == [(owner_task, owner_thread), (owner_task, owner_thread)]

    asyncio.run(scenario())
