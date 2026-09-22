"""Opt-in local request capture; the HTTP path only enqueues bounded memory.

Files contain original request data and must live in an ignored private directory.
Neither HTTP credentials nor arbitrary headers should be passed as metadata.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from pathlib import Path

_LOG = logging.getLogger(__name__)


def _positive_env(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        if not 1 <= value <= maximum:
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a positive integer no greater than {maximum}") from None


def _private_directory(path: Path) -> int:
    """Pin every path component by descriptor, rejecting symbolic links."""
    absolute = Path(os.path.abspath(path))
    if absolute == Path(absolute.anchor):
        raise ValueError("Capture directory cannot be a filesystem root")
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in absolute.parts[1:]:
            try:
                os.mkdir(part, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if os.fstat(descriptor).st_uid != os.getuid():
            raise ValueError("Capture directory must belong to the current user")
        os.fchmod(descriptor, 0o700)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


class RequestCapture:
    """One private JSONL stream and bounded writer queue per worker process.

    ``submitted`` counts accepted queue entries; ``dropped`` also includes entries
    later rejected by the disk limit or a writer error. ``bytes`` counts JSONL
    bytes, excluding the small aggregate status file. Metadata must be flat scalars.
    """

    @classmethod
    def from_env(cls) -> RequestCapture | None:
        directory = os.environ.get("SEIF_REQUEST_CAPTURE_DIR", "")
        if not directory:
            return None
        options = {
            "max_queued_bytes": _positive_env("SEIF_REQUEST_CAPTURE_MAX_QUEUED_BYTES", 64 << 20, 1 << 30),
            "max_records": _positive_env("SEIF_REQUEST_CAPTURE_MAX_RECORDS", 8192, 1_000_000),
            "max_disk_bytes": _positive_env("SEIF_REQUEST_CAPTURE_MAX_DISK_BYTES", 2 << 30, 1 << 40),
        }
        try:
            return cls(Path(directory), **options)
        except (OSError, ValueError):
            raise ValueError("Request capture directory could not be opened safely") from None

    def __init__(
        self, directory: Path, *, max_queued_bytes: int = 64 << 20,
        max_records: int = 8192, max_disk_bytes: int = 2 << 30,
    ) -> None:
        if any(type(value) is not int or value < 1 for value in (max_queued_bytes, max_records, max_disk_bytes)):
            raise ValueError("Capture limits must be positive integers")
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=max_records)
        self._stop = threading.Event()
        self._closed = False
        self._warned = False
        self._max_queued_bytes = max_queued_bytes
        self._max_disk_bytes = max_disk_bytes
        self._queued_bytes = 0
        self._counts = {"submitted": 0, "written": 0, "dropped": 0, "write_errors": 0, "bytes": 0}
        self._disk_limit_reached = False
        self._pid = os.getpid()
        self._filename = f"requests-{self._pid}-{uuid.uuid4().hex}.jsonl"
        self._dir_fd = _private_directory(Path(directory))
        try:
            fd = os.open(
                self._filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=self._dir_fd,
            )
            self._stream = os.fdopen(fd, "wb", buffering=65536)
            self._publish_status()
        except Exception:
            if hasattr(self, "_stream"):
                self._stream.close()
            os.close(self._dir_fd)
            raise
        self._thread = threading.Thread(target=self._run, name="seif-request-capture", daemon=True)
        self._thread.start()

    def submit(self, metadata: dict, body: bytes) -> bool:
        """Return immediately; do not parse, serialize, log, or access disk here."""
        try:
            if type(body) is not bytes or type(metadata) is not dict or len(metadata) > 64:
                raise ValueError
            copied = {}
            size = len(body) + 512
            for key, value in metadata.items():
                if type(key) is not str or len(key) > 128:
                    raise ValueError
                if value is not None and type(value) not in (str, int, float, bool):
                    raise ValueError
                if isinstance(value, str) and len(value) > 4096:
                    raise ValueError
                if type(value) is int and value.bit_length() > 256:
                    raise ValueError
                if type(value) is float and not math.isfinite(value):
                    raise ValueError
                copied[key] = value
                size += 128 + len(key) * 4 + (len(value) * 4 if isinstance(value, str) else 32)
            with self._lock:
                if self._closed or self._queued_bytes + size > self._max_queued_bytes:
                    self._counts["dropped"] += 1
                    return False
                try:
                    self._queue.put_nowait((copied, body, size))
                except queue.Full:
                    self._counts["dropped"] += 1
                    return False
                self._queued_bytes += size
                self._counts["submitted"] += 1
            return True
        except Exception:
            with self._lock:
                self._counts["dropped"] += 1
            return False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                **self._counts, "pid": self._pid, "data_file": self._filename,
                "queued_bytes": self._queued_bytes, "queued_records": self._queue.qsize(),
                "max_queued_bytes": self._max_queued_bytes, "max_disk_bytes": self._max_disk_bytes,
                "disk_limit_reached": self._disk_limit_reached, "closed": self._closed,
                "updated_at_unix": time.time(),
            }

    def _error(self) -> None:
        with self._lock:
            self._counts["write_errors"] += 1
            warn = not self._warned
            self._warned = True
        if warn:
            _LOG.warning("Local request capture encountered a write error; inspect aggregate capture status")

    def _write(self, metadata: dict, body: bytes) -> None:
        record = dict(metadata)
        try:
            request = json.loads(body)
        except (ValueError, RecursionError):
            record["parse_error"] = True
        else:
            record["request"] = request
            if isinstance(request, dict):
                for key in ("payload", "payload_id"):
                    if key in request:
                        record[key] = request[key]
        encoded = (json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        with self._lock:
            if self._counts["bytes"] + len(encoded) > self._max_disk_bytes:
                self._disk_limit_reached = True
                self._counts["dropped"] += 1
                return
        self._stream.write(encoded)
        with self._lock:
            self._counts["written"] += 1
            self._counts["bytes"] += len(encoded)

    def _publish_status(self) -> None:
        temporary = f".worker-{self._pid}-{uuid.uuid4().hex}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._dir_fd)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(json.dumps(self.snapshot(), separators=(",", ":")).encode())
            os.replace(temporary, f"worker-{self._pid}.json", src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self._dir_fd)
            except FileNotFoundError:
                pass

    def _run(self) -> None:
        published = time.monotonic()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                metadata, body, size = self._queue.get(timeout=0.2)
            except queue.Empty:
                pass
            else:
                try:
                    self._write(metadata, body)
                except Exception:
                    with self._lock:
                        self._counts["dropped"] += 1
                    self._error()
                finally:
                    with self._lock:
                        self._queued_bytes -= size
                    self._queue.task_done()
            if time.monotonic() - published >= 0.8:
                try:
                    self._stream.flush()
                    self._publish_status()
                except Exception:
                    self._error()
                published = time.monotonic()
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except Exception:
            self._error()
        finally:
            try:
                self._stream.close()
            except Exception:
                self._error()
            try:
                self._publish_status()
            except Exception:
                self._error()
            os.close(self._dir_fd)

    def close(self) -> None:
        """Stop accepting records, drain accepted records, flush and fsync."""
        with self._lock:
            self._closed = True
            self._stop.set()
        self._thread.join()
