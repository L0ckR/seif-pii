"""Optional private NER transport; no fallback to incomplete protection on failure."""

from __future__ import annotations

import asyncio
import json
import math
from urllib.parse import urlsplit

import httpx

from .detector import Span

NER_UNAVAILABLE = "NER is unavailable"
INVALID_NER_RESPONSE = "Invalid NER response"
NER_INCOMPLETE = "NER did not complete protection"


class NerUnavailable(RuntimeError):
    """Safe, input-free error for the HTTP boundary."""


def validate_ner_settings(url: str, token: str, timeout: float) -> None:
    if not url:
        return
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and (parsed.port is None or 1 <= parsed.port <= 65535)
        )
    except ValueError:
        valid = False
    if not valid or not token or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError("NER requires a valid HTTP(S) endpoint, a shared token and a bounded timeout")


def chunks(text: str, size: int = 16000, overlap: int = 256):
    """Keep Unicode offsets and a bounded overlap for names across boundaries."""
    if size <= overlap or overlap < 0:
        raise ValueError("Invalid NER chunk dimensions")
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        yield start, text[start:end]
        if end == len(text):
            break
        start = end - overlap


async def _response_json(response, limit: int, error_message: str):
    content = bytearray()
    async for part in response.aiter_bytes():
        if len(content) + len(part) > limit:
            raise NerUnavailable(error_message)
        content.extend(part)
    return json.loads(content)


def _entity_values(item: dict, text_length: int) -> tuple[int, int, float, str]:
    if not isinstance(item, dict) or set(item) != {"start", "end", "score", "entity_type"}:
        raise NerUnavailable(INVALID_NER_RESPONSE)
    start, end, score = item["start"], item["end"], item["score"]
    kind = item["entity_type"]
    if (
        kind not in ("PERSON", "LOCATION")
        or type(start) is not int or type(end) is not int
        or not 0 <= start < end <= text_length or end - start > 200
        or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
    ):
        raise NerUnavailable(INVALID_NER_RESPONSE)
    return start, end, score, kind


class NerClient:
    def __init__(self, url: str, token: str, timeout: float = 20.0, *, transport=None):
        validate_ner_settings(url, token, timeout)
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + token},
            timeout=httpx.Timeout(timeout, connect=min(timeout, 2.0)),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )
        # A process-wide bound, shared by all requests using this app instance.
        self.capacity = asyncio.Semaphore(4)

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> None:
        try:
            async with asyncio.timeout(2.0):
                async with self.client.stream("GET", "health", timeout=2.0) as response:
                    if response.status_code != 200:
                        raise NerUnavailable(NER_UNAVAILABLE)
                    body = await _response_json(response, 4096, NER_UNAVAILABLE)
                    if body.get("status") != "ok":
                        raise NerUnavailable(NER_UNAVAILABLE)
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError, RecursionError):
            raise NerUnavailable(NER_UNAVAILABLE) from None

    async def _analyze_chunk(self, text: str) -> dict:
        for attempt in range(4):
            async with self.client.stream("POST", "analyze", json={"text": text}) as response:
                if response.status_code == 429 and attempt < 3:
                    retry = True
                elif response.status_code != 200:
                    raise NerUnavailable(NER_INCOMPLETE)
                else:
                    retry = False
                    body = await _response_json(response, 262144, INVALID_NER_RESPONSE)
            if not retry:
                return body
            await asyncio.sleep(0.005 * (2**attempt))
        raise NerUnavailable(NER_INCOMPLETE)

    def _parse_entities(self, body: dict, text: str, offset: int, total_length: int) -> list[Span]:
        if not isinstance(body, dict) or set(body) != {"entities"}:
            raise NerUnavailable(INVALID_NER_RESPONSE)
        entities = body["entities"]
        if not isinstance(entities, list) or len(entities) > 2048:
            raise NerUnavailable(INVALID_NER_RESPONSE)
        result = []
        for item in entities:
            start, end, score, kind = _entity_values(item, len(text))
            # Do not accept a name visibly clipped by an artificial chunk edge.
            # The overlapping neighbouring chunk contains the complete name.
            if (offset and start == 0) or (offset + len(text) < total_length and end == len(text)):
                continue
            result.append(Span(offset + start, offset + end, kind, score, "presidio-ru-ner"))
        return result

    async def _chunk(self, offset: int, text: str, total_length: int) -> list[Span]:
        async with self.capacity:
            body = await self._analyze_chunk(text)
            return self._parse_entities(body, text, offset, total_length)

    async def _detect_batch(self, pending: list[tuple[int, str]], total_length: int) -> list[Span]:
        if len(pending) == 1:
            offset, part = pending[0]
            return await self._chunk(offset, part, total_length)
        # TaskGroup cancels sibling chunks on failure; no model requests may
        # outlive a failed protection operation. The comprehension creates one
        # task per chunk (2–4 here); the single-chunk path returns above.
        # A single create_task call site does not mean a single runtime task.
        async with asyncio.TaskGroup() as group:
            jobs = [group.create_task(self._chunk(offset, part, total_length)) for offset, part in pending]
        return [span for job in jobs for span in job.result()]

    async def detect(self, text: str) -> list[Span]:
        result = []
        pending = []
        try:
            # Includes waiting for capacity, all chunks and bounded 429 retries.
            async with asyncio.timeout(self.timeout):
                for offset, part in chunks(text):
                    pending.append((offset, part))
                    if len(pending) == 4:
                        result.extend(await self._detect_batch(pending, len(text)))
                        pending.clear()
                if pending:
                    result.extend(await self._detect_batch(pending, len(text)))
        except Exception:
            # Do not propagate HTTP URLs, response bodies or model exceptions.
            raise NerUnavailable(NER_INCOMPLETE) from None
        return result
