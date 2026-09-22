"""Exercise capture limits and privacy through a synthetic read-only replica."""
import json
from types import SimpleNamespace

import pytest

from scripts import capture_requests
from seif.config import Settings
from seif.vault import Vault


class Replica:
    def __init__(self, records, *, fail=False):
        self.records = records
        self.fail = fail
        self.closed = False
        self.reads = []

    def info(self, section):
        if self.fail:
            raise RuntimeError("sensitive upstream detail must not reach the console")
        return {"role": "slave", "master_link_status": "up", "master_sync_in_progress": 0}

    def scan(self, **_kwargs):
        return 0, list(self.records)

    def mget(self, keys):
        self.reads.extend(keys)
        return [self.records[key] for key in keys]

    def close(self):
        self.closed = True


def make_session(tmp_path, monkeypatch, *, max_bytes=100_000, max_records=100, fail=False):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(capture_requests.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(capture_requests.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    settings = Settings(encryption_key=bytes(range(32)))
    vault = Vault(settings)
    original = "Synthetic capture fixture without personal data."
    record = {"masked": original, "replacements": [], "entities": [], "original_hash": vault.digest(original)}
    records = {}
    for identifier in ("first", "duplicate"):
        key = vault.key("capture-unit-test", identifier)
        records[key.encode()] = vault._seal(key, record)
    records[b"expired"] = None
    records[b"unreadable"] = b"invalid authenticated ciphertext"
    client = Replica(records, fail=fail)
    args = SimpleNamespace(duration=1, max_bytes=max_bytes, max_records=max_records)
    session = capture_requests._CaptureSession(args, settings, client, tmp_path)
    return session, client


def test_capture_deduplicates_and_accounts_for_unavailable_records(tmp_path, monkeypatch):
    session, client = make_session(tmp_path, monkeypatch)
    session.run()
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(rows) == 1
    assert manifest["counts"]["source_records_seen"] == 4
    assert manifest["counts"]["duplicate_originals"] == 1
    assert manifest["counts"]["expired_before_read"] == 1
    assert manifest["counts"]["unrecoverable_records"] == 1
    assert manifest["source_records_per_case"] == {"capture_000001": 2}
    assert manifest["complete"] is True
    assert manifest["stop_reason"] == "duration"
    assert client.closed is True


@pytest.mark.parametrize(("bounds", "reason", "reads"), [
    ({"max_bytes": 1}, "byte_limit", 4),
    ({"max_records": 3}, "record_limit", 0),
])
def test_capture_limit_does_not_publish_a_partial_record(tmp_path, monkeypatch, bounds, reason, reads):
    session, client = make_session(tmp_path, monkeypatch, **bounds)
    session.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (tmp_path / "requests.jsonl").read_bytes() == b""
    assert manifest["stop_reason"] == reason
    assert len(client.reads) == reads
    assert client.closed is True


def test_capture_read_failure_records_only_safe_diagnostics(tmp_path, monkeypatch, capsys):
    session, client = make_session(tmp_path, monkeypatch, fail=True)
    session.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stop_reason"] == "capture_read_error"
    assert manifest["counts"]["read_failures"] == 1
    assert "sensitive upstream detail" not in capsys.readouterr().out
    assert client.closed is True
