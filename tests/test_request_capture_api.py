"""Behavioral checks for explicitly enabled local capture using fictional data."""
import json
from datetime import datetime

from fastapi.testclient import TestClient

from seif.app import capture_origin, create_app
from seif.config import Policy, Settings


def read_records(directory):
    return [json.loads(line) for file in directory.glob("requests-*.jsonl") for line in file.read_text().splitlines()]


def test_capture_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SEIF_REQUEST_CAPTURE_DIR", raising=False)
    with TestClient(create_app(Settings(demo=True))) as client:
        assert client.app.state.capture is None
        assert client.post("/process", json={"payload": "example@example.net", "payload_id": "off"}).status_code == 200
    assert not list(tmp_path.iterdir())


def test_capture_keeps_exact_requests_and_origin_without_credentials(monkeypatch, tmp_path):
    directory = tmp_path / "capture"
    monkeypatch.setenv("SEIF_REQUEST_CAPTURE_DIR", str(directory))
    original = "Тест 😊: fictional@example.net\nе\u0308\tконец"
    headers = {
        "CF-Connecting-IP": "203.0.113.42",
        "CF-Ray": "0123456789abcdef-AMS",
        "X-Forwarded-For": "203.0.113.42, 192.0.2.1",
        "User-Agent": "seif-capture-test",
        "X-SEIF-Capture-Probe": "unit-probe",
        "Authorization": "Bearer credential-must-not-be-saved",
        "X-API-Key": "api-key-must-not-be-saved",
        "Cookie": "cookie-must-not-be-saved",
    }
    with TestClient(create_app(Settings(demo=True, policies={"demo": Policy(rps=10000)}))) as client:
        first = client.post("/process", json={"payload": original, "payload_id": "pair"}, headers=headers)
        assert first.status_code == 200
        second = client.post("/process", json={"payload": first.json()["result"], "payload_id": "pair"}, headers=headers)
        assert second.status_code == 200
        assert second.json()["result"] == original
        assert client.get("/health", headers=headers).status_code == 200
    rows = read_records(directory)
    assert len(rows) == 2
    assert [row["operation"] for row in rows] == ["mask", "unmask"]
    assert rows[0]["payload"] == original
    assert rows[1]["payload"] == first.json()["result"]
    assert rows[0]["request_id"] == first.headers["X-Request-ID"]
    for row in rows:
        assert row["payload_id"] == "pair"
        assert row["status_code"] == 200
        assert row["route"] == "/process"
        assert row["body_complete"] is True
        assert row["cf_connecting_ip"] == "203.0.113.42"
        assert row["source_ip"] == "203.0.113.42"
        assert row["forwarded_for"] == "203.0.113.42, 192.0.2.1"
        assert row["capture_probe"] == "unit-probe"
        assert row["organizer_identity_verified"] is False
        assert datetime.fromisoformat(row["received_at"]) <= datetime.fromisoformat(row["completed_at"])
    serialized = json.dumps(rows)
    assert "credential-must-not-be-saved" not in serialized
    assert "api-key-must-not-be-saved" not in serialized
    assert "cookie-must-not-be-saved" not in serialized


def test_capture_preserves_failed_valid_requests_and_malformed_status(monkeypatch, tmp_path):
    directory = tmp_path / "capture"
    monkeypatch.setenv("SEIF_REQUEST_CAPTURE_DIR", str(directory))
    with TestClient(create_app(Settings(demo=True))) as client:
        rejected = client.post("/process", json={"payload": "invalid@example.net", "payload_id": "bad", "extra": True})
        assert rejected.status_code == 422
        malformed = client.post("/process", content='{"payload": "never-print@example.net",')
        assert malformed.status_code == 422
    rows = read_records(directory)
    assert len(rows) == 2
    assert all(row["status_code"] == 422 for row in rows)
    assert rows[0]["payload"] == "invalid@example.net"
    assert rows[0]["request"]["extra"] is True
    assert "never-print@example.net" not in json.dumps(rows[1])


def test_capture_origin_rejects_invalid_ip_headers():
    origin = capture_origin({"client": ("127.0.0.1", 1234)}, {
        b"cf-connecting-ip": b"not-an-ip",
        b"x-forwarded-for": b"203.0.113.42, not-an-ip",
    })
    assert origin["source_ip"] == "127.0.0.1"
    assert origin["cf_connecting_ip"] is None
    assert origin["forwarded_for"] is None
    assert origin["organizer_identity_verified"] is False
