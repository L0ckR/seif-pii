"""Experimental framework adapter; never imported by production source."""
import asyncio
import inspect
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')


def immediate_response(callback: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    """Execute bounded response construction inline and return a done Future."""
    @wraps(callback)
    @inspect.markcoroutinefunction
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future[R]:
        loop = asyncio.get_running_loop()
        result = callback(*args, **kwargs)
        completed = loop.create_future()
        completed.set_result(result)
        return completed

    return wrapped
