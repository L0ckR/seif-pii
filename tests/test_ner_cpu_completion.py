"""NER worker completion preserves exception ownership and real-job capacity."""

import asyncio
import threading
from concurrent.futures import CancelledError as WorkerCancelledError
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from scripts import ner_service


class WorkerExit(BaseException):
    """Model failures outside Exception still belong to the awaiting request."""


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_ner_worker_failure_never_escapes_to_event_loop(monkeypatch, cancel_waiter):
    entered, release = threading.Event(), threading.Event()
    failure = WorkerExit("private-model-input@example.net")

    def infer(_analyzer, _text):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("Test model worker timed out")
        raise failure

    monkeypatch.setattr(ner_service, "infer", infer)

    async def scenario():
        app = SimpleNamespace(state=SimpleNamespace(model_inflight=0))
        loop = asyncio.get_running_loop()
        unhandled = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, message: unhandled.append(message))
        lifespan = ner_service._lifespan(ner_service.NerSettings(demo=True), object)
        try:
            async with lifespan(app):
                pending = asyncio.create_task(ner_service._run_model(app.state, "synthetic test input"))
                try:
                    assert await asyncio.to_thread(entered.wait, 2)
                    if cancel_waiter:
                        pending.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await pending
                        assert app.state.model_inflight == 1
                    release.set()
                    if not cancel_waiter:
                        with pytest.raises(WorkerExit) as caught:
                            await pending
                        assert caught.value is failure
                finally:
                    release.set()
            await asyncio.sleep(0)
            assert app.state.model_inflight == 0
            assert unhandled == []
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_ner_cancelled_worker_future_preserves_worker_cancellation_type():
    completed = Future()
    completed.cancel()
    state = SimpleNamespace(model_inflight=1)

    async def scenario():
        waiter = asyncio.get_running_loop().create_future()
        ner_service._complete_model_job(state, waiter, completed)
        with pytest.raises(WorkerCancelledError):
            await waiter
        assert state.model_inflight == 0

    asyncio.run(scenario())
