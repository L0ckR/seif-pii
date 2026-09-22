"""Worker exceptions retain request ownership without leaking to loop hooks."""

import asyncio
import threading
from concurrent.futures import CancelledError as WorkerCancelledError
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from seif.app import ProcessRequest, _AppContext
from seif.config import Policy, Settings


class WorkerExit(BaseException):
    """A worker failure outside Exception must retain its original identity."""


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_worker_base_exception_remains_owned_by_request(monkeypatch, cancel_waiter):
    entered, release = threading.Event(), threading.Event()
    failure = WorkerExit("private-worker-value@example.net")

    def detector(*args, **kwargs):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("Test worker timed out")
        raise failure

    monkeypatch.setattr("seif.app.detect", detector)

    async def scenario():
        context = _AppContext(Settings(demo=True, cpu_workers=1))
        loop = asyncio.get_running_loop()
        unhandled = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, message: unhandled.append(message))
        try:
            async with context.lifespan(None):
                body = ProcessRequest(payload="neutral " * 3000, payload_id="worker-exit")
                pending = asyncio.create_task(context._detect_large(Policy(), body))
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    if cancel_waiter:
                        pending.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await pending
                        assert context.large_inflight == 1
                    release.set()
                    if not cancel_waiter:
                        with pytest.raises(WorkerExit) as caught:
                            await pending
                        assert caught.value is failure
                finally:
                    release.set()
            # Lifespan drains the actual worker, including cancelled waiters.
            await asyncio.sleep(0)
            assert context.large_inflight == 0
            assert unhandled == []
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_cancelled_worker_future_transfers_worker_cancellation():
    completed = Future()
    completed.cancel()
    context = SimpleNamespace(
        large_inflight=0,
        settings=Settings(cpu_workers=1),
        cpu_pool=SimpleNamespace(submit=lambda _: completed),
    )
    body = ProcessRequest(payload="neutral " * 3000, payload_id="cancelled-worker")

    async def scenario():
        with pytest.raises(WorkerCancelledError):
            await _AppContext._detect_large(context, Policy(), body)
        assert context.large_inflight == 0

    asyncio.run(scenario())
