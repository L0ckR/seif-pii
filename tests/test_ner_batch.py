"""NER batching preserves concurrency, ordered results, and sibling cancellation."""

import asyncio
from types import SimpleNamespace

import pytest

from seif.ner import NerClient


def test_four_chunk_tasks_start_together_and_preserve_input_order():
    async def scenario():
        entered = []
        all_started = asyncio.Event()

        async def chunk(offset, _part, _total_length):
            entered.append(offset)
            if len(entered) == 4:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return [offset]

        client = SimpleNamespace(_chunk=chunk)
        result = await NerClient._detect_batch(client, [(index, "x") for index in range(4)], 4)
        assert sorted(entered) == [0, 1, 2, 3]
        assert result == [0, 1, 2, 3]

    asyncio.run(scenario())


def test_chunk_failure_awaits_sibling_cancellation():
    async def scenario():
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()
        failure = RuntimeError("synthetic chunk failure")

        async def chunk(offset, _part, _total_length):
            if offset == 0:
                await sibling_started.wait()
                raise failure
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_cancelled.set()

        client = SimpleNamespace(_chunk=chunk)
        with pytest.raises(ExceptionGroup) as caught:
            await NerClient._detect_batch(client, [(0, "a"), (1, "b")], 2)
        assert caught.value.exceptions == (failure,)
        assert sibling_cancelled.is_set()

    asyncio.run(scenario())


def test_single_chunk_uses_the_calling_task():
    async def scenario():
        owner = asyncio.current_task()

        async def chunk(offset, _part, _total_length):
            assert asyncio.current_task() is owner
            return [offset]

        client = SimpleNamespace(_chunk=chunk)
        result = await NerClient._detect_batch(client, [(0, "single")], 6)
        assert result == [0]

    asyncio.run(scenario())
