import asyncio
import json
import statistics
import time
from pathlib import Path
from adapter import immediate_response

PAYLOAD = {'types': {'EMAIL': 'Email'}}

async def native_response():
    return PAYLOAD

@immediate_response
def completed_response():
    return PAYLOAD

async def run():
    rounds = {'native_async': [], 'completed_future_adapter': []}
    for _ in range(6):
        for name, callback in [('native_async', native_response), ('completed_future_adapter', completed_response)]:
            start = time.perf_counter_ns()
            for _ in range(200000):
                result = await callback()
            elapsed = time.perf_counter_ns() - start
            if result is not PAYLOAD:
                raise AssertionError('Response changed')
            rounds[name].append(elapsed / 200000)
    return {name: {'per_call_ns': values, 'median_ns': statistics.median(values)} for name, values in rounds.items()}

report = asyncio.run(run())
report['method'] = '6 alternating rounds of 200000 awaited fixed responses; no framework or scheduling'
Path(__file__).with_name('callback-cost.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
