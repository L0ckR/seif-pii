"""System access and replacement policy are separate, explicit decisions."""

import base64
import json
import logging

import pytest
import yaml
from fastapi.testclient import TestClient

import seif.app as app_module
from seif.app import create_app
from seif.config import Policy, Settings
from seif.ner import NerUnavailable

TEXT = "Email: invented@example.net"


def request(client, route="/process", payload=TEXT, payload_id="policy", **kwargs):
    return client.post(route, json={"payload": payload, "payload_id": payload_id}, **kwargs)


def test_disabled_access_still_denies_even_when_masking_is_disabled():
    settings = Settings(demo=True, policies={"demo": Policy(enabled=False, masking_enabled=False)})
    with TestClient(create_app(settings)) as client:
        response = request(client)
        assert response.status_code == 403
        assert TEXT not in response.text


def test_explicit_masking_opt_out_keeps_detection_and_returns_no_protected_entities(monkeypatch, caplog):
    calls = []
    original_detect = app_module.detect

    def observe_detect(text, **kwargs):
        calls.append(text)
        return original_detect(text, **kwargs)

    monkeypatch.setattr(app_module, "detect", observe_detect)
    app = create_app(Settings(demo=True, policies={"demo": Policy(masking_enabled=False)}))
    with TestClient(app) as client, caplog.at_level(logging.INFO, logger="seif.audit"):
        response = request(client, "/v1/mask")
        assert response.status_code == 200
        body = response.json()
        assert body["result"] == TEXT
        assert body["entities"] == body["types"] == []
        assert body["masking_enabled"] is False
        assert calls == [TEXT]
        audit = [json.loads(item.message) for item in caplog.records if item.name == "seif.audit"]
        assert audit[-1]["masking_enabled"] is False
        assert audit[-1]["types"] == []
        assert audit[-1]["detected_types"] == ["EMAIL"]
        assert TEXT not in str(audit)
        assert "detect" in audit[-1]["stages_ms"]
        key = app.state.vault.key("demo", "policy")
        record = client.portal.call(app.state.vault.get, key)
        assert record["masking_enabled"] is False
        assert record["entities"] == record["replacements"] == []
        assert record["detected_types"] == ["EMAIL"]


@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_process_contract_and_explicit_unmask_roundtrip_when_replacements_are_disabled(mode):
    app = create_app(Settings(demo=True, policies={"demo": Policy(masking_enabled=False, mode=mode)}))
    with TestClient(app) as client:
        first = request(client)
        assert first.status_code == 200
        assert first.json() == {"result": TEXT}
        assert request(client).json() == first.json()
        assert request(client, payload=first.json()["result"]).json() == first.json()
        restored = request(client, "/v1/unmask", payload=first.json()["result"])
        assert restored.status_code == 200
        assert restored.json()["result"] == TEXT
        assert restored.json()["masking_enabled"] is False


def test_unmask_permission_is_independent_of_the_masking_switch():
    app = create_app(Settings(demo=True, policies={"demo": Policy(masking_enabled=False, allow_unmask=False)}))
    with TestClient(app) as client:
        assert request(client).json() == {"result": TEXT}
        assert request(client).json() == {"result": TEXT}  # The same input is an idempotent retry.
        response = request(client, "/v1/unmask")
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "unmask_disabled"


@pytest.mark.parametrize("first_enabled", [False, True])
def test_existing_id_retains_saved_masking_fact_after_policy_change(first_enabled, caplog):
    settings = Settings(demo=True, policies={"demo": Policy(masking_enabled=first_enabled)})
    with TestClient(create_app(settings)) as client, caplog.at_level(logging.INFO, logger="seif.audit"):
        first = request(client, "/v1/mask").json()
        settings.policies["demo"] = Policy(masking_enabled=not first_enabled)
        retry = request(client, "/v1/mask").json()
        assert retry["result"] == first["result"]
        assert retry["entities"] == first["entities"]
        assert retry["masking_enabled"] is first_enabled
        audit = [json.loads(item.message) for item in caplog.records if item.name == "seif.audit"]
        assert audit[-1]["masking_enabled"] is first_enabled
        restored = request(client, "/v1/unmask", payload=first["result"]).json()
        assert restored["result"] == TEXT
        assert restored["masking_enabled"] is first_enabled
        fresh = request(client, "/v1/mask", payload_id="new-policy").json()
        assert fresh["masking_enabled"] is (not first_enabled)
        assert (fresh["result"] != TEXT) is (not first_enabled)


def test_legacy_records_default_to_historically_enabled_masking_without_reprocessing():
    app = create_app(Settings(demo=True, policies={"demo": Policy()}))
    with TestClient(app) as client:
        first = request(client, "/v1/mask").json()
        vault = app.state.vault
        key = vault.key("demo", "policy")
        expiry, blob = vault.records[key]
        old_record = vault._open(key, blob)
        del old_record["masking_enabled"]
        legacy_blob = vault._seal(key, old_record)
        vault.records[key] = (expiry, legacy_blob)
        vault.total_bytes += len(legacy_blob) - len(blob)
        app.state.settings.policies["demo"] = Policy(masking_enabled=False)
        retry = request(client, "/v1/mask").json()
        assert retry["result"] == first["result"]
        assert retry["masking_enabled"] is True
        assert request(client, "/v1/unmask", payload=first["result"]).json()["result"] == TEXT


def test_other_systems_and_their_correlation_records_are_isolated():
    settings = Settings(policies={
        "inspection": Policy(api_key="test-inspection", masking_enabled=False),
        "protected": Policy(api_key="test-protected", masking_enabled=True),
    })
    with TestClient(create_app(settings)) as client:
        raw_headers = {"X-System-ID": "inspection", "X-API-Key": "test-inspection"}
        protected_headers = {"X-System-ID": "protected", "X-API-Key": "test-protected"}
        raw = request(client, "/v1/mask", headers=raw_headers).json()
        protected = request(client, "/v1/mask", headers=protected_headers).json()
        assert raw["result"] == TEXT
        assert protected["result"] != TEXT
        assert raw["masking_enabled"] is False
        assert protected["masking_enabled"] is True
        assert protected["types"] == ["EMAIL"]
        assert request(client, headers=raw_headers).json() == {"result": TEXT}
        assert request(client, payload=protected["result"], headers=protected_headers).json() == {"result": TEXT}
        assert request(client, headers={"X-System-ID": "inspection"}).status_code == 401


def test_detection_failure_is_not_bypassed_by_disabling_replacements(monkeypatch):
    def broken_detector(*args, **kwargs):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr(app_module, "detect", broken_detector)
    app = create_app(Settings(demo=True, policies={"demo": Policy(masking_enabled=False)}))
    with TestClient(app) as client:
        response = request(client)
        assert response.status_code == 503
        assert TEXT not in response.text
        assert not app.state.vault.records


def test_ner_failure_is_not_bypassed_by_disabling_replacements(monkeypatch):
    calls = []

    class BrokenNer:
        def __init__(self, *args, max_concurrency=4, backend="httpx"):
            pass

        async def health(self):
            pass

        async def close(self):
            pass

        async def detect(self, text):
            calls.append(text)
            raise NerUnavailable("synthetic NER failure")

    monkeypatch.setattr(app_module, "NerClient", BrokenNer)
    settings = Settings(demo=True, ner_url="http://ner.test", ner_token="test-token",
                        policies={"demo": Policy(masking_enabled=False)})
    app = create_app(settings)
    with TestClient(app) as client:
        response = request(client)
        assert calls == [TEXT]
        assert response.status_code == 503
        assert TEXT not in response.text
        assert not app.state.vault.records


@pytest.mark.parametrize("value", ["false", "true", "", None, 0, 1, [], {}])
def test_programmatic_policy_rejects_non_booleans(value):
    with pytest.raises(ValueError, match="masking_enabled must be a boolean"):
        Policy(masking_enabled=value)


@pytest.mark.parametrize("value", ["false", "true", None, 0, 1, []])
def test_yaml_policy_rejects_non_booleans(tmp_path, monkeypatch, value):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({"systems": {"demo": {"masking_enabled": value}}}))
    monkeypatch.setenv("SEIF_CONFIG", str(path))
    monkeypatch.setenv("SEIF_DEMO", "1")
    monkeypatch.setenv("SEIF_MASTER_KEY", base64.b64encode(b"t" * 32).decode())
    monkeypatch.delenv("SEIF_SENTINELS", raising=False)
    monkeypatch.delenv("SEIF_REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="masking_enabled must be a boolean"):
        Settings.from_env()


@pytest.mark.parametrize("value", [True, False])
def test_yaml_boolean_roundtrip_and_default_stays_enabled(tmp_path, monkeypatch, value):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({"systems": {"demo": {"masking_enabled": value}}}))
    monkeypatch.setenv("SEIF_CONFIG", str(path))
    monkeypatch.setenv("SEIF_DEMO", "1")
    monkeypatch.setenv("SEIF_MASTER_KEY", base64.b64encode(b"t" * 32).decode())
    monkeypatch.delenv("SEIF_SENTINELS", raising=False)
    monkeypatch.delenv("SEIF_REDIS_URL", raising=False)
    assert Settings.from_env().policies["demo"].masking_enabled is value
    assert Policy().masking_enabled is True
