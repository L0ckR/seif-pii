# Immediate-response adapter prototype

Production callbacks were restored to native `async def`; only the explicit TaskGroup creation loop remains changed. This directory explores an internal `inspect.markcoroutinefunction` adapter: a synchronous bounded callback runs immediately on the event loop and its value is placed in an already-completed Future. `functools.wraps` preserves the callable signature. No threadpool, task scheduling, sleep, artificial yield, or custom Coroutine implementation is involved.

The installed FastAPI and Starlette recognize the marker. A real framework route and exception handler were exercised while threadpool dispatch functions were patched to raise. Signature, caller Task/thread, completed Future, result and exception identity checks passed (`verify.py`).

Six alternating microbenchmark rounds measured native async at 63 ns/call and the adapter at 437 ns/call: a real ~0.374 μs cost plus one Future allocation. Repeated ASGI runs no longer show the 0.2 ms sync-handler scheduling penalty, but are too noisy to claim zero endpoint overhead or production RPS equality. `/process` itself was not adapted.

The prototype is mechanically valid specifically for frameworks doing `await callback(...)`. Its returned Future is not a Coroutine for `asyncio.create_task`. Native async is the simpler and faster implementation; introducing this adapter trades a small real cost and an extra abstraction for explicit analyzer-readable bounded response callbacks. No production adapter was installed.

Artifacts: `verification.json`, `callback-cost.json`, `assessment.json`, four ASGI result files and their reproducible scripts. No public requests were made.
