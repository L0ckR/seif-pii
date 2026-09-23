"""Optional aiohttp session for the bounded private NER HTTP contract.

The owner retains the overall operation deadline, including admission, retries
and all chunks. aiohttp's connect timeout also includes connector waiting/DNS;
unlike HTTPX it has no separate write timeout. This is intentionally a narrow
NER session, not a generally interchangeable HTTPX client.
"""
from __future__ import annotations

import asyncio
import json
import math
import ssl
from contextlib import asynccontextmanager


class NerHttpError(RuntimeError):
    """Input-free HTTP failure accepted by the private NER boundary."""


def _encode_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class _Response:
    def __init__(self, response, client_error):
        self.response = response
        self.client_error = client_error
        self.status_code = response.status

    async def aiter_bytes(self):
        try:
            # aiohttp decodes Content-Encoding; the existing response reader
            # bounds these decoded bytes before parsing JSON.
            async for part in self.response.content.iter_chunked(16384):
                yield part
        except (self.client_error, TimeoutError, OSError):
            raise NerHttpError("NER HTTP response failed.") from None


class AiohttpNerSession:
    """One lazily created, loop-owned session with bounded persistent sockets."""

    def __init__(self, url, token, timeout, capacity):
        import aiohttp

        if type(capacity) is not int or not 1 <= capacity <= 256:
            raise ValueError("NER HTTP capacity must be an integer between 1 and 256.")
        self._aiohttp = aiohttp
        self._url = url.rstrip("/") + "/"
        self._headers = {"Authorization": "Bearer " + token}
        self._timeout, self._capacity = timeout, capacity
        self._session = None
        self._loop = None
        self._closed = False
        self._close_task = None

    def _get_session(self):
        if self._closed:
            raise NerHttpError("NER HTTP session is closed.")
        loop = asyncio.get_running_loop()
        if self._session is None:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
            connector = self._aiohttp.TCPConnector(
                limit=self._capacity + 1, limit_per_host=self._capacity + 1,
                ssl=context, keepalive_timeout=5,
            )
            self._session = self._aiohttp.ClientSession(
                base_url=self._url, connector=connector, headers=self._headers,
                cookie_jar=self._aiohttp.DummyCookieJar(), trust_env=False,
                raise_for_status=False, auto_decompress=True, json_serialize=_encode_json,
            )
            self._loop = loop
        if self._loop is not loop:
            raise NerHttpError("NER HTTP session belongs to another event loop.")
        return self._session

    @asynccontextmanager
    async def stream(self, method, path, *, json=None, timeout=None):
        limit = self._timeout if timeout is None else timeout
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("NER HTTP timeout must be positive and finite.")
        deadline = self._aiohttp.ClientTimeout(total=None, connect=min(limit, 2.0),
                                                sock_connect=min(limit, 2.0), sock_read=limit,
                                                ceil_threshold=math.inf)
        try:
            async with self._get_session().request(method, path, json=json, timeout=deadline,
                                                    allow_redirects=False) as response:
                yield _Response(response, self._aiohttp.ClientError)
        except (self._aiohttp.ClientError, TimeoutError, OSError):
            raise NerHttpError("NER HTTP request failed.") from None

    async def aclose(self):
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_session())
            self._close_task.add_done_callback(self._consume_close_error)
        await asyncio.shield(self._close_task)

    @staticmethod
    def _consume_close_error(task):
        if not task.cancelled():
            task.exception()

    async def _close_session(self):
        if self._session is not None:
            await self._session.close()
