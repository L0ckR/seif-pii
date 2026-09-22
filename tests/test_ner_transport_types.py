"""Expanded NER capabilities survive both HTTP boundaries and consumer policy."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from scripts.ner_service import NerSettings, infer
from scripts.ner_service import create_app as create_ner_app
from seif.app import create_app
from seif.config import Policy, Settings
from seif.detector import Span
from seif.ner import NerClient, NerUnavailable
from seif.ner_contract import NER_ENTITY_TYPES, analyzer_entities, max_entity_chars
from seif.rubert_ner import NATIVE_MAPPING, RubertAnalyzer


@pytest.mark.parametrize("native,kind", sorted(NATIVE_MAPPING.items()))
def test_native_type_survives_adapter_and_authenticated_private_http(native, kind):
    text = "😀 синтетика"
    runtime = SimpleNamespace(predict=lambda _: [{"start": 2, "end": len(text), "label": native,
                                                 "text": text[2:], "score": .83}])
    analyzer = RubertAnalyzer(runtime, profile="native")
    app = create_ner_app(NerSettings(token="test-private-key"), lambda: analyzer)
    with TestClient(app) as client:
        assert client.get("/health").json()["entities"] == list(NER_ENTITY_TYPES)
        assert client.post("/analyze", json={"text": text}).status_code == 401
        response = client.post("/analyze", json={"text": text},
                               headers={"Authorization": "Bearer test-private-key"})
    assert response.status_code == 200
    entities = response.json()["entities"]
    assert entities == [{"start": 2, "end": len(text), "entity_type": kind, "score": .83}]
    received = NerClient._parse_entities(None, {"entities": entities}, text, 0, len(text))
    assert [(s.start, s.end, s.type) for s in received] == [(2, len(text), kind)]


def test_full_gateway_retains_separate_native_name_components():
    text = "Анна Иванова"
    raw = [{"start": 0, "end": 4, "label": "FIRST_NAME", "text": "Анна", "score": .9},
           {"start": 5, "end": 12, "label": "LAST_NAME", "text": "Иванова", "score": .8}]
    analyzer = RubertAnalyzer(SimpleNamespace(predict=lambda _: raw), profile="native")
    result = infer(analyzer, text)["entities"]
    assert [(s["start"], s["end"]) for s in result] == [(0, 4), (5, 12)]


@pytest.mark.parametrize("kind", NER_ENTITY_TYPES)
def test_extended_types_keep_bounded_transport(kind):
    length = max_entity_chars(kind)
    row = {"start": 0, "end": length, "score": .8, "entity_type": kind}
    assert NerClient._parse_entities(None, {"entities": [row]}, "x" * length, 0, length)[0].end == length
    with pytest.raises(NerUnavailable):
        NerClient._parse_entities(None, {"entities": [{**row, "end": length + 1}]}, "x" * (length + 1), 0, length + 1)


@pytest.mark.parametrize("capabilities", [("EMAIL", "UNKNOWN"), ("PERSON", "PERSON"), [], "PERSON", [[]]])
def test_invalid_advertised_capabilities_fail_before_inference(capabilities):
    with pytest.raises(ValueError, match="capabilities"):
        analyzer_entities(SimpleNamespace(supported_entities=capabilities))


@pytest.mark.parametrize("length,offset", [(923, 15800), (2048, 13952)])
def test_long_url_at_chunk_boundary_is_not_silently_lost(length, offset):
    async def scenario():
        prefix = "https://example.invalid/"
        value = prefix + "a" * (length - len(prefix))
        text = " " * offset + value + " " * 16000

        def handler(request):
            part = json.loads(request.content)["text"]
            start = part.find(value)
            entities = [] if start < 0 else [{"start": start, "end": start + len(value), "score": .95,
                                             "entity_type": "URL"}]
            return httpx.Response(200, json={"entities": entities})

        client = NerClient("http://ner.internal", "test", transport=httpx.MockTransport(handler))
        try:
            result = await client.detect(text)
        finally:
            await client.close()
        assert {(span.start, span.end, span.type) for span in result} == {(offset, offset + len(value), "URL")}

    asyncio.run(scenario())


@pytest.mark.parametrize("allowed,protected", [((), True), (("SNILS",), True), (("EMAIL",), False)])
def test_new_type_obeys_consumer_policy_and_restores_exact_original(monkeypatch, allowed, protected):
    text = "🙂 СНИЛС: 123-456-789 01"
    start = text.index("123")

    class FakeNer:
        def __init__(self, *_args):
            pass

        async def health(self):
            pass

        async def close(self):
            pass

        async def detect(self, _text):
            return [Span(start, len(text), "SNILS", .9)]

    monkeypatch.setattr("seif.app.NerClient", FakeNer)
    settings = Settings(demo=True, ner_url="http://ner.internal", ner_token="test",
                        policies={"demo": Policy(types=allowed)})
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/mask", json={"payload": text, "payload_id": "new-type-policy"})
        assert response.status_code == 200
        assert ("123-456-789 01" not in response.json()["result"]) == protected
        restored = client.post("/v1/unmask", json={"payload": response.json()["result"], "payload_id": "new-type-policy"})
        assert restored.status_code == 200
        assert restored.json()["result"] == text
