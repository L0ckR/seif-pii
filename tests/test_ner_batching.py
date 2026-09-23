"""Concurrency guarantees of opt-in request micro-batching (no model needed)."""
import asyncio
import gc
import time
import weakref
from threading import Event

import httpx
import pytest

from scripts.ner_service import NerSettings, create_app
from seif.config import Settings
from seif.ner import NerClient
from seif.ner_batching import BatchExecutor


def test_batch_keeps_order_and_transfers_failure_without_killing_worker():
    calls = []

    def invoke(texts):
        calls.append(texts)
        if "bad" in texts:
            raise ValueError("model error")
        return [text.upper() for text in texts]

    pool = BatchExecutor(invoke, batch_size=4, wait_ms=20, capacity=8)
    try:
        jobs = [pool.submit(text) for text in ("a", "b", "c", "d")]
        assert [job.result(2) for job in jobs] == ["A", "B", "C", "D"]
        assert calls == [["a", "b", "c", "d"]]
        bad_job = pool.submit("bad")
        with pytest.raises(ValueError, match="model error"):
            bad_job.result(2)
        assert pool.submit("next").result(2) == "NEXT"
    finally:
        pool.shutdown(cancel_futures=True)


def test_running_job_retains_ownership_and_shutdown_cancels_queued_jobs():
    entered, release = Event(), Event()

    def invoke(texts):
        entered.set()
        assert release.wait(3)
        return texts

    pool = BatchExecutor(invoke, batch_size=2, wait_ms=0, capacity=2)
    first = pool.submit("running")
    try:
        assert entered.wait(2)
        assert first.cancel() is False
        queued = [pool.submit("queued") for _ in range(2)]
        with pytest.raises(RuntimeError):
            pool.submit("overflow")
        pool.shutdown(wait=False, cancel_futures=True)
        assert all(job.cancelled() for job in queued)
        assert not first.done()
        with pytest.raises(RuntimeError):
            pool.submit("after shutdown")
    finally:
        release.set()
        pool.shutdown()
    assert first.result() == "running"


def test_wrong_batch_count_fails_all_waiters_and_worker_recovers():
    pool = BatchExecutor(lambda _texts: [], batch_size=2, wait_ms=20, capacity=2)
    try:
        jobs = [pool.submit("a"), pool.submit("b")]
        for job in jobs:
            with pytest.raises(ValueError, match="batch response count"):
                job.result(2)
    finally:
        pool.shutdown()


def test_shutdown_cancels_batch_already_collected_but_not_running():
    collected, release = Event(), Event()

    class ControlledExecutor(BatchExecutor):
        def _process_batch(self, batch):
            collected.set()
            assert release.wait(3)
            return super()._process_batch(batch)

    calls = []
    pool = ControlledExecutor(lambda texts: calls.append(texts) or texts, batch_size=2, wait_ms=0, capacity=2)
    job = pool.submit("queued")
    try:
        assert collected.wait(2)
        pool.shutdown(wait=False, cancel_futures=True)
    finally:
        release.set()
        pool.shutdown()
    assert job.cancelled()
    assert calls == []


def test_idle_executor_does_not_retain_last_input():
    class Input:
        pass

    value = Input()
    pointer = weakref.ref(value)
    pool = BatchExecutor(lambda texts: [None for _ in texts], batch_size=2, wait_ms=0, capacity=2)
    try:
        job = pool.submit(value)
        del value
        assert job.result(2) is None
        deadline = time.monotonic() + 1
        while pointer() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(.01)
        assert pointer() is None
    finally:
        pool.shutdown()


def test_service_batches_distinct_requests_and_serial_mode_still_available():
    class Analyzer:
        supported_entities = ["PERSON", "LOCATION"]

        def __init__(self):
            self.batches = []

        def analyze_batch(self, *, texts, **_kwargs):
            self.batches.append(texts)
            return [[] for _ in texts]

    async def scenario():
        analyzer = Analyzer()
        app = create_app(NerSettings(demo=True, batch_size=4, batch_wait_ms=20), lambda: analyzer)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ner") as client:
                results = await asyncio.gather(*(client.post("/analyze", json={"text": str(i)}) for i in range(4)))
            assert all(response.status_code == 200 for response in results)
            assert analyzer.batches == [["0", "1", "2", "3"]]
            assert app.state.model_inflight == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("capacity", [0, -1, 257, True, 2.0])
def test_invalid_api_concurrency_rejected(capacity):
    with pytest.raises(ValueError, match="ner_max_concurrency"):
        Settings(ner_max_concurrency=capacity)
    with pytest.raises(ValueError, match="NER concurrency"):
        NerClient("http://ner", "secret", max_concurrency=capacity)


def test_configurable_client_capacity_does_not_leave_default_bottleneck():
    async def scenario():
        client = NerClient("http://ner", "secret", max_concurrency=16)
        try:
            for _ in range(16):
                await asyncio.wait_for(client.capacity.acquire(), .1)
            acquire = client.capacity.acquire()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(acquire, .01)
        finally:
            await client.close()

    asyncio.run(scenario())
