"""Private NER boundary: completeness, Unicode offsets and failure semantics."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from seif.app import create_app
from seif.config import Settings
from seif.detector import Span
from seif.ner import NerClient, NerUnavailable, chunks, validate_ner_settings


def test_chunking_covers_long_unicode_text_without_gaps():
    text = '🙂' * 17000 + 'Альма Вайс' + 'я' * 22000
    pieces = list(chunks(text))
    reconstructed = pieces[0][1] + ''.join(part[256:] for _, part in pieces[1:])
    assert reconstructed == text
    assert all(len(part) <= 16000 for _, part in pieces)
    assert list(chunks('')) == []


@pytest.mark.parametrize('kind,name', [('PERSON', 'Альма Вайс'), ('LOCATION', 'Караганда')])
def test_ner_recovers_name_across_chunk_boundary_and_scopes_token(kind, name):
    async def scenario():
        text = '🙂' * 15995 + name + ' ' * 17000
        seen = []

        async def handler(request):
            assert request.headers['authorization'] == 'Bearer test-private-ner-token'
            part = json.loads(request.content)['text']
            seen.append(part)
            offset = part.find(name)
            return httpx.Response(200, json={'entities': [] if offset < 0 else [
                {'entity_type': kind, 'start': offset, 'end': offset + len(name), 'score': .85}
            ]})

        client = NerClient('http://ner.internal:8770', 'test-private-ner-token', transport=httpx.MockTransport(handler))
        try:
            found = await client.detect(text)
            assert len(seen) == 3
            assert {(s.start, s.end, text[s.start:s.end]) for s in found} == {(15995, 15995 + len(name), name)}
            assert all(s.type == kind for s in found)
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('body', [
    {'entities': [{'entity_type': 'PERSON', 'start': -1, 'end': 2, 'score': .8}]},
    {'entities': [{'entity_type': 'PERSON', 'start': True, 'end': 2, 'score': .8}]},
    {'entities': [{'entity_type': 'PERSON', 'start': 0, 'end': 999, 'score': .8}]},
    {'entities': [{'entity_type': 'PERSON', 'start': 0, 'end': 2, 'score': 'nan'}]},
    {'entities': [{'entity_type': 'PERSON', 'start': 0, 'end': 2, 'score': 2}]},
    {'entities': [{'entity_type': 'PERSON', 'start': 0, 'end': 2, 'score': .8, 'text': 'private-value'}]},
    {'entities': [{'entity_type': 'ORGANIZATION', 'start': 0, 'end': 2, 'score': .8}]},
    {'entities': [{'entity_type': [], 'start': 0, 'end': 2, 'score': .8}]},
    {'entities': [{'start': 0, 'end': 2, 'score': .8}]},
    {'entities': 'bad'},
    {'entities': [], 'raw_text': 'private-value'},
])
def test_bad_model_response_is_an_error_not_empty_detection(body):
    async def scenario():
        client = NerClient('http://ner.internal', 'secret', transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=body)))
        try:
            with pytest.raises(NerUnavailable, match='did not complete protection') as caught:
                await client.detect('private-value')
            assert 'private-value' not in str(caught.value)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_model_capacity_retries_are_bounded():
    async def scenario():
        calls = 0

        def handler(_):
            nonlocal calls
            calls += 1
            return httpx.Response(429)

        client = NerClient('http://ner.internal', 'secret', transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(NerUnavailable):
                await client.detect('Альма Вайс')
            assert calls == 4
        finally:
            await client.close()

    asyncio.run(scenario())


def test_total_deadline_includes_capacity_wait():
    async def scenario():
        client = NerClient('http://ner.internal', 'secret', timeout=.015,
                           transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'entities': []})))
        for _ in range(4):
            await client.capacity.acquire()
        try:
            with pytest.raises(NerUnavailable):
                await client.detect('Альма Вайс')
        finally:
            for _ in range(4):
                client.capacity.release()
            await client.close()

    asyncio.run(scenario())


def test_nested_health_json_is_reported_as_unavailable():
    async def scenario():
        body = b"[" * 1500 + b"0" + b"]" * 1500
        client = NerClient("http://ner.internal", "secret", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=body)))
        try:
            with pytest.raises(NerUnavailable, match="NER is unavailable"):
                await client.health()
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('url,token,timeout', [
    ('ftp://ner.internal', 'secret', 1),
    ('http://user:password@ner.internal', 'secret', 1),
    ('http://ner.internal?secret=value', 'secret', 1),
    ('http://ner.internal', '', 1),
    ('http://ner.internal', 'secret', float('nan')),
    ('http://ner.internal', 'secret', -1),
])
def test_invalid_private_ner_configuration_rejected(url, token, timeout):
    with pytest.raises(ValueError):
        validate_ner_settings(url, token, timeout)


@pytest.mark.parametrize('kind,value', [('PERSON', 'Альма Вайс'), ('LOCATION', 'Караганда')])
def test_hybrid_api_restores_exactly_and_does_not_reanalyze_retries(monkeypatch, kind, value):
    class FakeNer:
        calls = 0

        def __init__(self, *args, max_concurrency=4, backend="httpx"):
            pass

        async def health(self):
            pass

        async def close(self):
            pass

        async def detect(self, text):
            type(self).calls += 1
            start = text.index(value)
            return [Span(start, start + len(value), kind, .85)]

    monkeypatch.setattr('seif.app.NerClient', FakeNer)
    app = create_app(Settings(demo=True, ner_url='http://ner.internal', ner_token='secret'))
    body = {'payload': 'В заявлении указано: ' + value + '.', 'payload_id': 'hybrid-contract'}
    with TestClient(app) as client:
        first = client.post('/process', json=body)
        assert first.status_code == 200
        assert set(first.json()) == {'result'}
        assert value not in first.json()['result']
        assert client.post('/process', json=body).json() == first.json()
        restored = client.post('/process', json={**body, 'payload': first.json()['result']})
        assert restored.json() == {'result': body['payload']}
        assert FakeNer.calls == 1
        assert client.get('/health').json()['detector_profile'] == 'hybrid'


def test_model_failure_never_returns_plaintext_success(monkeypatch, caplog):
    class FailedNer:
        def __init__(self, *args, max_concurrency=4, backend="httpx"):
            pass

        async def health(self):
            pass

        async def close(self):
            pass

        async def detect(self, text):
            raise NerUnavailable(text)

    monkeypatch.setattr('seif.app.NerClient', FailedNer)
    app = create_app(Settings(demo=True, ner_url='http://ner.internal', ner_token='secret'))
    sensitive = 'private@example.net'
    with TestClient(app) as client:
        response = client.post('/process', json={'payload': sensitive, 'payload_id': 'ner-fail'})
        assert response.status_code == 503
        assert sensitive not in response.text
        assert sensitive not in caplog.text
        assert not app.state.vault.records
