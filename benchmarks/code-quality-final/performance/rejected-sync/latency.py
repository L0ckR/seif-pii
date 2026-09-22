"""Local in-process HTTP latency; no network, NER model, or public traffic."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
for name in tuple(os.environ):
    if name.startswith('SEIF_REQUEST_CAPTURE_'):
        os.environ.pop(name)
import httpx
from seif.app import create_app
from seif.config import Policy, Settings
from scripts.ner_service import NerSettings, create_app as create_ner_app

class EmptyAnalyzer:
    def analyze(self, **kwargs):
        return []

async def failing_endpoint():
    raise RuntimeError('synthetic benchmark failure')

async def measure():
    logging.disable(logging.CRITICAL)
    app = create_app(Settings(demo=True, policies={'demo': Policy(rps=1000000)}))
    ner = create_ner_app(NerSettings(demo=True), analyzer_factory=EmptyAnalyzer)
    ner.get('/benchmark-failure')(failing_endpoint)
    cases = [
        ('api_types', 'api', 'GET', '/v1/types', None, 200),
        ('api_invalid_control', 'api', 'POST', '/process', {'payload': 1, 'payload_id': 'invalid'}, 422),
        ('api_process', 'api', 'POST', '/process', {'payload': 'Email: synthetic@example.invalid', 'payload_id': 'replace'}, 200),
        ('ner_validation', 'ner', 'POST', '/analyze', {'text': 1}, 422),
        ('ner_http_error', 'ner', 'GET', '/missing-route', None, 404),
        ('ner_unexpected_error', 'ner', 'GET', '/benchmark-failure', None, 503),
    ]
    results = {}
    async with app.router.lifespan_context(app), ner.router.lifespan_context(ner):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://local.test') as api_client, httpx.AsyncClient(transport=httpx.ASGITransport(app=ner, raise_app_exceptions=False), base_url='http://local.test') as ner_client:
            for name, service, method, url, body, expected in cases:
                client = api_client if service == 'api' else ner_client
                samples = []
                for number in range(1260):
                    payload = body
                    if name == 'api_process':
                        payload = {**body, 'payload_id': f'local-latency-{number}'}
                    started = time.perf_counter_ns()
                    response = await client.request(method, url, json=payload)
                    elapsed = (time.perf_counter_ns() - started) / 1000
                    if response.status_code != expected:
                        raise AssertionError(f'{name}: unexpected HTTP {response.status_code}')
                    if number >= 60:
                        samples.append(elapsed)
                ordered = sorted(samples)
                results[name] = {'requests': len(samples), 'mean_us': statistics.mean(samples), 'median_us': statistics.median(samples), 'p95_us': ordered[int(len(ordered) * .95)]}
    return results

report = {
    'method': 'sequential httpx ASGITransport; 60 warmup + 1200 measured requests per path; in-memory vault, synthetic email, stub NER',
    'python': sys.version,
    'source_sha256': {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest() for name in ['seif/app.py', 'seif/ner.py', 'scripts/ner_service.py']},
    'cases': asyncio.run(measure()),
    'limitations': ['Local latency only; excludes network/Redis/model and concurrent load.', 'Not an HTTP RPS capacity benchmark.', 'Other system activity can affect timings.'],
}
with args.output.open('x') as target:
    json.dump(report, target, indent=2)
print(json.dumps(report['cases']))
