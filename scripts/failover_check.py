#!/usr/bin/env python3
"""Check a saved /process mapping across an operator-controlled failover.

1. prepare --baseurl URL --state /tmp/seif-failover.json
2. Restart/replace the API replica or perform Redis Sentinel failover separately.
3. verify --baseurl URL --state /tmp/seif-failover.json

Headers are read from environment variables, never CLI secret values. Only
synthetic data is sent. Reports contain no payload, mask, API key or response
body. The state file is private (0600) and contains a synthetic mask and ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

SOURCE = "Клиент Тестов Иван Иванович; email failover@example.net; паспорт 4509 123456."
SOURCE_SHA256 = hashlib.sha256(SOURCE.encode()).hexdigest()


class CheckFailed(Exception):
    """Only fixed non-sensitive reasons are allowed in user-visible errors."""


def _response_body(response):
    try:
        body = response.json()
    except ValueError:
        raise CheckFailed("invalid_json_response") from None
    if not isinstance(body, dict) or set(body) != {"result"} or not isinstance(body["result"], str):
        raise CheckFailed("invalid_contract_response")
    return body


def request(client: httpx.Client, payload: str, payload_id: str, deadline: float) -> tuple[str, dict]:
    started = time.perf_counter()
    attempts = 0
    errors: Counter[str] = Counter()
    while time.perf_counter() - started < deadline:
        attempts += 1
        delay = min(0.25 * attempts, 2.0)
        try:
            response = client.post("/process", json={"payload": payload, "payload_id": payload_id})
            if response.status_code == 200:
                body = _response_body(response)
                return body["result"], {
                    "attempts": attempts,
                    "transient_errors": dict(errors),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            if response.status_code not in {429, 502, 503, 504}:
                raise CheckFailed(f"http_{response.status_code}")
            errors[f"http_{response.status_code}"] += 1
            with suppress(ValueError):
                delay = max(delay, min(float(response.headers.get("Retry-After", "0")), 2.0))
        except httpx.TimeoutException:
            errors["timeout"] += 1
        except httpx.HTTPError:
            errors["transport"] += 1
        remaining = deadline - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(min(delay, remaining))
    raise CheckFailed("service_not_ready_before_deadline")


def save_state(path: Path, masked: str, payload_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "source_sha256": SOURCE_SHA256,
        "payload_id": payload_id,
        "masked": masked,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    # Refuse to overwrite or follow an existing file/symlink. Repeat checks use a new state path.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)
        handle.write("\n")


def load_state(path: Path) -> dict:
    try:
        if path.stat().st_size > 16_384:
            raise CheckFailed("invalid_state_file")
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            state.get("schema_version") != 1
            or state.get("source_sha256") != SOURCE_SHA256
            or not isinstance(state.get("payload_id"), str)
            or not state["payload_id"]
            or not isinstance(state.get("masked"), str)
        ):
            raise CheckFailed("invalid_state_file")
        return state
    except (OSError, ValueError, AttributeError):
        raise CheckFailed("invalid_state_file") from None


def run(args: argparse.Namespace) -> dict:
    parsed = urlsplit(args.baseurl)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise CheckFailed("baseurl_must_be_http_url_without_credentials_or_query")
    headers = {}
    system = os.getenv(args.system_env, "")
    api_key = os.getenv(args.api_key_env, "")
    if system:
        headers["X-System-ID"] = system
    if api_key:
        headers["X-API-Key"] = api_key
    with httpx.Client(
        base_url=args.baseurl.rstrip("/"), headers=headers, timeout=args.timeout, trust_env=False
    ) as client:
        if args.phase == "prepare":
            if args.state.exists() or args.state.is_symlink():
                raise CheckFailed("state_already_exists_choose_new_path")
            payload_id = "failover-" + uuid.uuid4().hex
            masked, metrics = request(client, SOURCE, payload_id, args.deadline)
            if masked == SOURCE or "failover@example.net" in masked or "4509 123456" in masked:
                raise CheckFailed("synthetic_source_was_not_protected")
            try:
                save_state(args.state, masked, payload_id)
            except OSError:
                raise CheckFailed("cannot_create_private_state_file") from None
            return {
                "phase": "prepare",
                "status": "pass",
                "synthetic_data_only": True,
                "mapping_saved": True,
                **metrics,
                "next_step": "Perform the operator-controlled change, then run verify with the same state file before mapping TTL expires.",
            }
        state = load_state(args.state)
        restored, metrics = request(client, state["masked"], state["payload_id"], args.deadline)
        if restored != SOURCE:
            raise CheckFailed("mapping_not_preserved_or_result_mismatch")
        return {
            "phase": "verify",
            "status": "pass",
            "synthetic_data_only": True,
            "exact_roundtrip": True,
            **metrics,
            "limitation": "One preserved mapping does not establish zero data loss for asynchronous replication or measure failover RTO.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("prepare", "verify"))
    parser.add_argument("--baseurl", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--system-env", default="SEIF_TEST_SYSTEM")
    parser.add_argument("--api-key-env", default="SEIF_TEST_API_KEY")
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--deadline", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.timeout <= 0 or args.deadline <= 0:
        parser.error("timeout and deadline must be positive")
    try:
        report = run(args)
        exit_code = 0
    except CheckFailed as exc:
        report = {"phase": args.phase, "status": "fail", "reason": str(exc), "synthetic_data_only": True}
        exit_code = 1
    except Exception:
        report = {
            "phase": args.phase,
            "status": "fail",
            "reason": "unexpected_client_error",
            "synthetic_data_only": True,
        }
        exit_code = 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
