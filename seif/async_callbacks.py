"""Inline awaitable responses for bounded FastAPI/Starlette callbacks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def immediate_response(callback: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    """Run nonblocking response construction on the awaiting framework's loop.

    FastAPI/Starlette use the coroutine marker to await the returned, completed
    Future directly instead of dispatching to a worker thread. No event-loop
    turn is introduced. Exceptions keep their identity and request ownership.

    Only use for bounded callbacks invoked as ``await callback(...)``. This is
    not a general coroutine API: its Future cannot be passed to ``create_task``.
    Blocking I/O, model inference, and expensive computation do not belong here.
    """
    @wraps(callback)
    @inspect.markcoroutinefunction
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future[R]:
        loop = asyncio.get_running_loop()
        result = callback(*args, **kwargs)
        completed = loop.create_future()
        completed.set_result(result)
        return completed

    return wrapped
