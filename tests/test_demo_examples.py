"""The actual demo examples keep whole values with component-level NER output."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from seif.app import create_app
from seif.config import Policy, Settings
from seif.detector import Span
from seif.transform import shape_mask

_APP_JS = Path(__file__).resolve().parents[1] / "web" / "app.js"
_EXPECTED = {
    "client": [
        ("PERSON", "Иванов Иван Иванович"), ("BIRTH_DATE", "15.04.1990"),
        ("PASSPORT", "4510 123456"), ("PHONE", "+7 (900) 123-45-67"),
        ("EMAIL", "ivan.petrov@example.org"), ("ADDRESS", "г. Москва, ул. Лесная, д. 12, кв. 34"),
    ],
    "context": [],
    "foreign": [
        ("PERSON", "Петрова Анна Сергеевна"), ("BIRTH_DATE", "12.08.1985"),
        ("CITIZENSHIP", "Республика Беларусь"), ("FOREIGN_DOCUMENT", "AB1234567"),
        ("FOREIGN_DOCUMENT", "75 1234567"), ("FOREIGN_DOCUMENT", "123456789"),
        ("DRIVER_LICENSE", "77 12 345678"), ("EMAIL", "anna.petrova@example.org"),
        ("PHONE", "+1 (202) 555-0147"),
    ],
}
_MODEL_PARTS = {
    "client": [("PERSON", word) for word in ("Иванов", "Иван", "Иванович")] + [
        ("LOCATION", value) for value in ("Москва", "ул. Лесная", "д. 12, кв. 34")
    ],
    "context": [("PERSON", word) for word in ("Александр", "Сергеевич", "Пушкин", "Евгений", "Онегин")],
    "foreign": [("PERSON", word) for word in ("Петрова", "Анна", "Сергеевна")],
}


def _example(case):
    block = re.search(r"const examples = \{\n(.*?)\n\};", _APP_JS.read_text(), re.S)[1]
    return json.loads(re.search(rf'^\s*{case}: ("(?:[^"\\]|\\.)*")', block, re.M)[1])


@pytest.mark.parametrize("case", _EXPECTED)
@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_demo_examples_with_fragmented_model(monkeypatch, case, mode):
    text = _example(case)
    candidates = []
    cursor = 0
    for kind, value in _MODEL_PARTS[case]:
        start = text.index(value, cursor)
        candidates.append(Span(start, start + len(value), kind, .9, "model"))
        cursor = start + len(value)

    class FakeNer:
        def __init__(self, *_args, **_kwargs):
            pass

        async def health(self):
            pass

        async def close(self):
            pass

        async def detect(self, value):
            assert value == text
            return candidates

    monkeypatch.setattr("seif.app.NerClient", FakeNer)
    settings = Settings(demo=True, ner_url="http://ner.internal", ner_token="test",
                        policies={"demo": Policy(mode=mode)})
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/mask", json={"payload": text, "payload_id": "demo-regression"})
        assert response.status_code == 200
        body = response.json()
        assert [(s["type"], text[s["start"]:s["end"]]) for s in body["entities"]] == _EXPECTED[case]
        masked = body["result"]
        if case == "context":
            assert masked == text
        elif mode == "token":
            assert masked.count("⟦PD:PERSON:") == 1
            assert masked.count("⟦PD:ADDRESS:") == (case == "client")
            assert masked.count("⟦PD:") == len(_EXPECTED[case])
        elif mode == "synthetic":
            assert masked.count("Макет Макетович") == 1
            assert masked.count("г. Макетный, ул. Тестовая") == (case == "client")
        else:
            expected = text
            for _, value in _EXPECTED[case]:
                expected = expected.replace(value, shape_mask(value))
            assert masked == expected
        restored = client.post("/v1/unmask", json={"payload": masked, "payload_id": "demo-regression"})
        assert restored.status_code == 200
        assert restored.json()["result"] == text
