"""Pool topology and HTTP semantics using fake transports; no network or GPU."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from seif.ner_transport import ShardedNerTransport


@pytest.fixture
def pools(monkeypatch):
    instances, contexts = [], []
    context = object()

    def make_context(**kwargs):
        contexts.append(kwargs)
        return context

    class Pool:
        def __init__(self, **options):
            self.index = len(instances)
            self.options = options
            self.requests = []
            self.closes = 0
            self.failure = None
            self.close_failure = None
            self.close_gate = None
            self.close_started = None
            self.close_completed = False
            self.gate = None
            instances.append(self)

        async def handle_async_request(self, request):
            self.requests.append(request)
            if self.gate is not None:
                await self.gate.wait()
            if self.failure is not None:
                raise self.failure
            return httpx.Response(200, json={"pool": self.index}, request=request)

        async def aclose(self):
            self.closes += 1
            if self.close_started is not None:
                self.close_started.set()
            if self.close_gate is not None:
                await self.close_gate.wait()
            await asyncio.sleep(0)
            self.close_completed = True
            if self.close_failure is not None:
                raise self.close_failure

    monkeypatch.setattr(httpx, "create_ssl_context", make_context)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", Pool)
    return instances, contexts, context


@pytest.mark.parametrize(("capacity", "shards", "connections"), [
    (1, 1, 2), (4, 1, 5), (5, 2, 4), (16, 4, 5), (32, 8, 5), (64, 8, 9), (256, 8, 33),
])
def test_capacity_is_bounded_and_all_pools_share_one_default_verified_context(pools, capacity, shards, connections):
    instances, contexts, context = pools
    transport = ShardedNerTransport(capacity)
    assert (transport.shard_count, transport.connections_per_shard) == (shards, connections)
    assert len(instances) == shards
    assert contexts == [{"trust_env": False}]
    for pool in instances:
        assert pool.options["verify"] is context
        assert pool.options["trust_env"] is False and pool.options["retries"] == 0
        limits = pool.options["limits"]
        assert limits.max_connections == limits.max_keepalive_connections == connections
    asyncio.run(transport.aclose())


@pytest.mark.parametrize("capacity", [0, -1, 257, True, 1.0, None, "16"])
def test_bad_capacity_is_rejected_before_constructing_context_or_pools(pools, capacity):
    instances, contexts, _ = pools
    with pytest.raises(ValueError, match="capacity"):
        ShardedNerTransport(capacity)
    assert instances == contexts == []


def test_requests_round_robin_without_changing_body_auth_timeouts_or_response(pools):
    instances, _, _ = pools

    async def scenario():
        transport = ShardedNerTransport(16)
        async with httpx.AsyncClient(transport=transport, base_url="https://ner.internal",
                                    headers={"Authorization": "Bearer fixture-secret"}, timeout=7,
                                    trust_env=False, follow_redirects=False) as client:
            replies = [await client.post("/analyze", json={"text": str(i)}) for i in range(9)]
            assert [reply.json()["pool"] for reply in replies] == [0, 1, 2, 3, 0, 1, 2, 3, 0]
            for pool in instances:
                for request in pool.requests:
                    assert request.headers["authorization"] == "Bearer fixture-secret"
                    assert request.url == "https://ner.internal/analyze"
                    assert request.content.startswith(b'{"text":')
                    assert request.extensions["timeout"] == {"connect": 7, "read": 7, "write": 7, "pool": 7}
        assert all(pool.closes == 1 for pool in instances)

    asyncio.run(scenario())


def test_concurrent_dispatch_advances_before_await_and_cancellation_is_not_retried(pools):
    instances, _, _ = pools

    async def scenario():
        transport = ShardedNerTransport(16)
        gate = asyncio.Event()
        for pool in instances:
            pool.gate = gate
        jobs = [asyncio.create_task(transport.handle_async_request(httpx.Request("POST", "http://ner/analyze")))
                for _ in range(8)]
        try:
            await asyncio.sleep(0)
            assert [len(pool.requests) for pool in instances] == [2, 2, 2, 2]
            jobs[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await jobs[0]
            gate.set()
            replies = await asyncio.gather(*jobs[1:])
            assert [reply.json()["pool"] for reply in replies] == [1, 2, 3, 0, 1, 2, 3]
            assert sum(len(pool.requests) for pool in instances) == 8
        finally:
            gate.set()
            await transport.aclose()

    asyncio.run(scenario())


def test_transport_exception_is_preserved_without_a_retry_or_pool_replacement(pools):
    instances, _, _ = pools

    async def scenario():
        transport = ShardedNerTransport(8)
        request = httpx.Request("POST", "http://ner/analyze")
        failure = httpx.ReadTimeout("fixture timeout", request=request)
        instances[0].failure = failure
        try:
            with pytest.raises(httpx.ReadTimeout) as caught:
                await transport.handle_async_request(request)
            assert caught.value is failure
            assert [len(pool.requests) for pool in instances] == [1, 0]
            response = await transport.handle_async_request(request)
            assert response.json()["pool"] == 1
            assert len(instances) == 2
        finally:
            await transport.aclose()

    asyncio.run(scenario())


def test_close_attempts_every_pool_once_and_preserves_failure_for_later_callers(pools):
    instances, _, _ = pools

    async def scenario():
        transport = ShardedNerTransport(32)
        failure = httpx.CloseError("fixture close failure")
        instances[0].close_failure = failure
        with pytest.raises(httpx.CloseError) as caught:
            await transport.aclose()
        assert caught.value is failure
        assert len(instances) == 8 and all(pool.closes == 1 for pool in instances)
        with pytest.raises(httpx.CloseError) as repeated:
            await transport.aclose()
        assert repeated.value is failure
        assert all(pool.closes == 1 for pool in instances)
        with pytest.raises(RuntimeError, match="closed"):
            await transport.handle_async_request(httpx.Request("GET", "http://ner/health"))

    asyncio.run(scenario())


def test_cancelled_close_caller_does_not_cancel_cleanup_and_another_caller_waits(pools):
    instances, _, _ = pools

    async def scenario():
        transport = ShardedNerTransport(8)
        release = asyncio.Event()
        for pool in instances:
            pool.close_gate = release
            pool.close_started = asyncio.Event()
        first = asyncio.create_task(transport.aclose())
        await asyncio.gather(*(pool.close_started.wait() for pool in instances))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert all(pool.closes == 1 and not pool.close_completed for pool in instances)
        second = asyncio.create_task(transport.aclose())
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await asyncio.wait_for(second, 1)
        assert all(pool.closes == 1 and pool.close_completed for pool in instances)
        await transport.aclose()
        assert all(pool.closes == 1 for pool in instances)

    asyncio.run(scenario())


def test_cleanup_failure_after_caller_cancellation_is_available_to_later_caller(pools):
    instances, _, _ = pools

    async def scenario():
        transport = ShardedNerTransport(8)
        release = asyncio.Event()
        started = asyncio.Event()
        failure = httpx.CloseError("fixture delayed close failure")
        instances[0].close_gate = release
        instances[0].close_started = started
        instances[0].close_failure = failure
        first = asyncio.create_task(transport.aclose())
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        with pytest.raises(httpx.CloseError) as caught:
            await transport.aclose()
        assert caught.value is failure
        assert all(pool.closes == 1 and pool.close_completed for pool in instances)

    asyncio.run(scenario())
