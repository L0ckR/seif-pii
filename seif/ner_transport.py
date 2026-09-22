"""Small independent HTTP pools for the private NER endpoint.

The owning AsyncClient and this transport must run on the same event loop.
Sharding reduces connection-pool bookkeeping at high concurrency without
changing request retries, authentication, timeouts or response streaming.
"""
from __future__ import annotations

import asyncio

import httpx


class ShardedNerTransport(httpx.AsyncBaseTransport):
    """Round-robin at most eight bounded pools, sharing one verified TLS context."""

    def __init__(self, capacity):
        if type(capacity) is not int or not 1 <= capacity <= 256:
            raise ValueError("NER transport capacity must be an integer between 1 and 256.")
        self.shard_count = min(8, (capacity + 3) // 4)
        # One extra connection per pool permits health probes during inference.
        self.connections_per_shard = (capacity + self.shard_count - 1) // self.shard_count + 1
        context = httpx.create_ssl_context(trust_env=False)
        limits = httpx.Limits(max_connections=self.connections_per_shard,
                              max_keepalive_connections=self.connections_per_shard)
        self._pools = [httpx.AsyncHTTPTransport(verify=context, trust_env=False, retries=0, limits=limits)
                       for _ in range(self.shard_count)]
        self._next = 0
        self._closed = False
        self._close_task = None

    async def handle_async_request(self, request):
        if self._closed:
            raise RuntimeError("NER transport is closed.")
        pool = self._pools[self._next]
        # No await separates selection and increment: same-loop callers cannot
        # choose the same counter position through task interleaving.
        self._next = (self._next + 1) % self.shard_count
        return await pool.handle_async_request(request)

    async def aclose(self):
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_pools())
            # If every caller is cancelled, consume a later cleanup failure so
            # it does not become an unhandled background-task traceback. Future
            # callers can still await the task and receive the same exception.
            self._close_task.add_done_callback(self._consume_close_error)
        await asyncio.shield(self._close_task)

    @staticmethod
    def _consume_close_error(task):
        if not task.cancelled():
            task.exception()

    async def _close_pools(self):
        # A failed pool close must not prevent closing the remaining pools.
        outcomes = await asyncio.gather(*(pool.aclose() for pool in self._pools), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
