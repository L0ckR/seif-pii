import asyncio
import inspect
import json
import threading
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.dependencies.models import _is_coroutine_callable
from starlette._utils import is_async_callable
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from adapter import immediate_response


async def verify():
    owner_task = asyncio.current_task()
    owner_thread = threading.get_ident()
    observed = []
    failure = ValueError('synthetic failure')

    def response(value: int = 7):
        observed.append((asyncio.current_task() is owner_task, threading.get_ident() == owner_thread))
        return value

    wrapped = immediate_response(response)
    future = wrapped(9)
    assert isinstance(future, asyncio.Future)
    assert future.done()
    assert await future == 9
    assert inspect.signature(wrapped) == inspect.signature(response)
    assert inspect.iscoroutinefunction(wrapped)
    assert is_async_callable(wrapped)
    assert _is_coroutine_callable(wrapped)

    def broken():
        raise failure

    try:
        await immediate_response(broken)()
    except ValueError as caught:
        assert caught is failure
    else:
        raise AssertionError('Exception identity was lost')

    app = FastAPI()

    @immediate_response
    def endpoint(value: int = 7):
        return {'value': response(value)}

    @immediate_response
    def http_error(_request, exc):
        response()
        return JSONResponse({'error': 'synthetic'}, status_code=exc.status_code)

    app.get('/response')(endpoint)
    app.add_exception_handler(HTTPException, http_error)

    def reject_threadpool(*_args, **_kwargs):
        raise AssertionError('Threadpool dispatch is forbidden for adapter')

    with patch('fastapi.routing.run_in_threadpool', reject_threadpool), patch('starlette._exception_handler.run_in_threadpool', reject_threadpool), patch('starlette.middleware.errors.run_in_threadpool', reject_threadpool):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://local.test') as client:
            valid = await client.get('/response?value=11')
            assert valid.status_code == 200
            assert valid.json() == {'value': 11}
            missing = await client.get('/missing')
            assert missing.status_code == 404
            assert missing.json() == {'error': 'synthetic'}
    assert all(same_task and same_thread for same_task, same_thread in observed)
    return {
        'fastapi_async_recognition': True,
        'starlette_async_recognition': True,
        'signature_preserved': True,
        'completed_future_without_scheduling': True,
        'exception_identity_preserved': True,
        'same_caller_task_and_loop_thread': True,
        'no_threadpool_dispatch_endpoint_and_handler': True,
        'python': __import__('sys').version,
    }


report = asyncio.run(verify())
Path(__file__).with_name('verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
