"""The capture writer is tested with synthetic local fixtures only."""

import json
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from seif.request_capture import RequestCapture


def rows(directory):
    return [json.loads(line) for path in directory.glob("requests-*.jsonl") for line in path.read_text().splitlines()]


def test_disabled_without_explicit_directory(monkeypatch):
    monkeypatch.delenv("SEIF_REQUEST_CAPTURE_DIR", raising=False)
    assert RequestCapture.from_env() is None


def test_exact_unicode_full_request_and_private_files(tmp_path):
    directory = tmp_path / "local-data" / "capture"
    capture = RequestCapture(directory)
    request = {"payload": "Тестовый текст 🧪\nстрока", "payload_id": "fixture", "extra": [True, 42]}
    assert capture.submit({"client_ip": "192.0.2.1", "direction": "mask"}, json.dumps(request).encode())
    capture.close()
    row, = rows(directory)
    assert row["request"] == request
    assert row["payload"] == request["payload"]
    assert row["payload_id"] == "fixture"
    assert row["client_ip"] == "192.0.2.1"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in directory.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    status = json.loads(next(directory.glob("worker-*.json")).read_text())
    assert status["written"] == status["submitted"] == 1
    assert status["queued_bytes"] == status["queued_records"] == 0
    assert status["closed"] is True
    assert status["bytes"] == next(directory.glob("requests-*.jsonl")).stat().st_size


def test_malformed_request_does_not_retain_raw_body_or_log_it(tmp_path, caplog):
    capture = RequestCapture(tmp_path)
    assert capture.submit({"status": 400}, b"invalid-sensitive-body")
    capture.close()
    row, = rows(tmp_path)
    assert row == {"status": 400, "parse_error": True}
    assert "invalid-sensitive-body" not in caplog.text
    assert "invalid-sensitive-body" not in next(tmp_path.glob("requests-*.jsonl")).read_text()


def test_parallel_producers_do_not_interleave_or_lose_records(tmp_path):
    capture = RequestCapture(tmp_path)

    def producer(index):
        for sequence in range(100):
            assert capture.submit({"producer": index, "sequence": sequence}, b'{"payload":"synthetic"}')

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(producer, range(8)))
    capture.close()
    captured = rows(tmp_path)
    assert len(captured) == 800
    assert capture.snapshot()["dropped"] == 0
    for producer in range(8):
        assert [r["sequence"] for r in captured if r["producer"] == producer] == list(range(100))


def test_queue_byte_bound_rejects_before_accepting(tmp_path):
    capture = RequestCapture(tmp_path, max_queued_bytes=512)
    assert not capture.submit({}, b"large")
    capture.close()
    assert capture.snapshot()["submitted"] == 0
    assert capture.snapshot()["dropped"] == 1
    assert rows(tmp_path) == []


def test_record_queue_limit_and_nonblocking_submit(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = RequestCapture._write

    def delayed(self, metadata, body):
        entered.set()
        assert release.wait(5)
        original(self, metadata, body)

    monkeypatch.setattr(RequestCapture, "_write", delayed)
    capture = RequestCapture(tmp_path, max_records=1)
    try:
        assert capture.submit({}, b"{}")
        assert entered.wait(5)
        assert capture.submit({}, b"{}")
        started = time.monotonic()
        assert not capture.submit({}, b"{}")
        assert time.monotonic() - started < 0.1
    finally:
        release.set()
        capture.close()
    assert capture.snapshot()["written"] == 2
    assert capture.snapshot()["dropped"] == 1


def test_disk_budget_leaves_valid_complete_lines(tmp_path):
    capture = RequestCapture(tmp_path, max_disk_bytes=40)
    for _ in range(10):
        assert capture.submit({}, b"{}")
    capture.close()
    status = capture.snapshot()
    assert 0 < status["bytes"] <= 40
    assert status["disk_limit_reached"]
    assert status["written"] + status["dropped"] == 10
    assert len(rows(tmp_path)) == status["written"]


def test_flush_and_status_arrive_while_recorder_is_open(tmp_path):
    capture = RequestCapture(tmp_path)
    try:
        assert capture.submit({}, b'{"payload":"fixture"}')
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = json.loads(next(tmp_path.glob("worker-*.json")).read_text())
            if status["written"] == 1:
                break
            time.sleep(0.02)
        assert status["written"] == 1
        assert len(rows(tmp_path)) == 1
        assert not status["closed"]
    finally:
        capture.close()


def test_close_is_idempotent_and_rejects_new_records(tmp_path):
    capture = RequestCapture(tmp_path)
    capture.close()
    capture.close()
    assert not capture.submit({}, b"{}")
    assert capture.snapshot()["dropped"] == 1


@pytest.mark.parametrize("metadata", [{"nested": {}}, {"long": "x" * 4097}, {"number": float("nan")}])
def test_unsupported_metadata_is_safely_rejected(tmp_path, metadata):
    capture = RequestCapture(tmp_path)
    assert not capture.submit(metadata, b"{}")
    capture.close()
    assert capture.snapshot()["dropped"] == 1


def test_directory_symlink_is_rejected_without_changing_target(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir(mode=0o755)
    (tmp_path / "link").symlink_to(destination, target_is_directory=True)
    with pytest.raises(OSError):
        RequestCapture(tmp_path / "link" / "capture")
    assert not (destination / "capture").exists()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755


def test_invalid_environment_fails_before_writer_starts(monkeypatch, tmp_path):
    monkeypatch.setenv("SEIF_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SEIF_REQUEST_CAPTURE_MAX_RECORDS", "invalid-sensitive-value")
    with pytest.raises(ValueError, match="SEIF_REQUEST_CAPTURE_MAX_RECORDS") as caught:
        RequestCapture.from_env()
    assert "invalid-sensitive-value" not in str(caught.value)
    assert not list(tmp_path.iterdir())


def test_writer_failure_is_counted_without_payload_or_exception_logging(tmp_path, monkeypatch, caplog):
    def fail(self, metadata, body):
        raise OSError("synthetic-sensitive-exception")

    monkeypatch.setattr(RequestCapture, "_write", fail)
    capture = RequestCapture(tmp_path)
    assert capture.submit({}, b'{"payload":"synthetic-sensitive-body"}')
    assert capture.submit({}, b"{}")
    capture.close()
    assert capture.snapshot()["write_errors"] == 2
    assert capture.snapshot()["dropped"] == 2
    assert "synthetic-sensitive" not in caplog.text
    assert len(caplog.records) == 1
