"""Sentinel configuration, discovery wiring and optional real failover integration.

Run the integration test with SEIF_TEST_REDIS_SERVER=/path/to/redis-server.
It starts isolated local Redis/Sentinel processes and never touches an existing
cluster. Replicas are checked before failure: this proves rediscovery and
retention of replicated records, not a zero-loss durability guarantee.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import socket
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from seif.config import Settings, parse_sentinels
from seif.vault import Vault


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SEIF_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SEIF_DEMO", "1")
    monkeypatch.setenv("SEIF_CONFIG", str(tmp_path / "missing.yaml"))
    return monkeypatch


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", ()),
        ("redis-0.sentinel.local:26379", (("redis-0.sentinel.local", 26379),)),
        ("127.0.0.1:26379, localhost:26380", (("127.0.0.1", 26379), ("localhost", 26380))),
        ("[::1]:26379", (("::1", 26379),)),
        ("[2001:db8::1234]:26379", (("2001:db8::1234", 26379),)),
    ],
)
def test_sentinel_endpoints(value, expected):
    assert parse_sentinels(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        " ",
        "localhost",
        "localhost:",
        ":26379",
        "localhost:0",
        "localhost:65536",
        "localhost:-1",
        "localhost:abc",
        "localhost:26379,",
        "bad host:26379",
        "redis://localhost:26379",
        "user:secret@localhost:26379",
        "-host:26379",
        "host..local:26379",
        "999.999.999.999:26379",
        "[invalid]:26379",
        "::1:26379",
        "host:26379,host:26379",
        "[::1]:26379/path",
        ",".join(f"node{i}:26379" for i in range(17)),
    ],
)
def test_invalid_sentinel_endpoints_do_not_echo_input(value):
    with pytest.raises(ValueError) as error:
        parse_sentinels(value)
    assert "secret" not in str(error.value)


def test_sentinel_env_defaults_and_secrets(clean_env):
    clean_env.setenv("SEIF_MASTER_KEY", base64.b64encode(b"k" * 32).decode())
    clean_env.setenv("SEIF_SENTINELS", "redis-0.redis:26379,redis-1.redis:26379")
    clean_env.setenv("SEIF_REDIS_PASSWORD", "synthetic-redis-secret")
    settings = Settings.from_env()
    assert settings.sentinels == (("redis-0.redis", 26379), ("redis-1.redis", 26379))
    assert settings.sentinel_master == "seif-master"
    assert settings.sentinel_password == settings.redis_password == "synthetic-redis-secret"
    assert "synthetic-redis-secret" not in repr(settings)
    assert "kkkkkkkk" not in repr(settings)
    clean_env.setenv("SEIF_SENTINEL_PASSWORD", "separate-sentinel-secret")
    assert Settings.from_env().sentinel_password == "separate-sentinel-secret"
    clean_env.setenv("SEIF_SENTINEL_PASSWORD", "")
    assert Settings.from_env().sentinel_password == ""


@pytest.mark.parametrize(
    "shared_env,value",
    [
        ("SEIF_SENTINELS", "localhost:26379"),
        ("SEIF_REDIS_URL", "redis://localhost:6379/0"),
    ],
)
def test_shared_storage_requires_stable_key_even_in_demo(clean_env, shared_env, value):
    clean_env.setenv(shared_env, value)
    with pytest.raises(ValueError, match="stable SEIF_MASTER_KEY"):
        Settings.from_env()


def test_conflicting_discovery_sources_rejected(clean_env):
    clean_env.setenv("SEIF_MASTER_KEY", base64.b64encode(b"k" * 32).decode())
    clean_env.setenv("SEIF_SENTINELS", "localhost:26379")
    clean_env.setenv("SEIF_REDIS_URL", "redis://localhost:6379/0")
    with pytest.raises(ValueError, match="not both"):
        Settings.from_env()


@pytest.mark.parametrize("name", ["", "two words", "name/secret", ":bad", "a" * 129])
def test_invalid_sentinel_service_name(clean_env, name):
    clean_env.setenv("SEIF_SENTINEL_MASTER", name)
    with pytest.raises(ValueError, match="service name"):
        Settings.from_env()


def test_vault_uses_master_discovery_and_closes_all_pools(monkeypatch):
    primary = MagicMock()
    primary.aclose = AsyncMock()
    first, second = MagicMock(), MagicMock()
    first.aclose, second.aclose = AsyncMock(), AsyncMock()
    manager = MagicMock()
    manager.master_for.return_value = primary
    manager.sentinels = [first, second]
    factory = MagicMock(return_value=manager)
    monkeypatch.setattr("redis.asyncio.sentinel.Sentinel", factory)
    settings = Settings(
        sentinels=(("node-0", 26379), ("node-1", 26379)),
        sentinel_master="seif-test",
        redis_password="redis-secret",
        sentinel_password="sentinel-secret",
    )
    vault = Vault(settings)
    assert vault.distributed
    assert vault.redis is primary
    manager.master_for.assert_called_once_with("seif-test")
    options = factory.call_args.kwargs
    assert options["password"] == "redis-secret"
    assert options["sentinel_kwargs"]["password"] == "sentinel-secret"
    assert options["socket_timeout"] == options["sentinel_kwargs"]["socket_timeout"] == 2
    first.aclose.side_effect = RuntimeError("test close failure")
    close = vault.close()
    with pytest.raises(RuntimeError, match="test close failure"):
        asyncio.run(close)
    primary.aclose.assert_awaited_once()
    first.aclose.assert_awaited_once()
    second.aclose.assert_awaited_once()


def test_memory_vault_is_not_distributed():
    assert not Vault(Settings(demo=True)).distributed


@pytest.mark.skipif(not os.getenv("SEIF_TEST_REDIS_SERVER"), reason="set SEIF_TEST_REDIS_SERVER for real failover test")
def test_real_sentinel_failover_retains_replicated_ciphertext(tmp_path):
    executable = os.environ["SEIF_TEST_REDIS_SERVER"]
    ports = []
    sockets = []
    for _ in range(6):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
        ports.append(sock.getsockname()[1])
    for sock in sockets:
        sock.close()
    processes = []
    password = secrets.token_urlsafe(24)
    primary_port = ports[0]
    for index, port in enumerate(ports[:3]):
        directory = tmp_path / f"redis-{index}"
        directory.mkdir()
        config = directory / "redis.conf"
        replica = f"replicaof 127.0.0.1 {primary_port}\n" if index else ""
        config.write_text(
            f"bind 127.0.0.1\nport {port}\nprotected-mode yes\ndir {directory}\n"
            f'save ""\nappendonly no\nrequirepass {password}\nmasterauth {password}\n{replica}'
        )
        config.chmod(0o600)
        processes.append(
            subprocess.Popen([executable, str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        )
    for index, port in enumerate(ports[3:]):
        directory = tmp_path / f"sentinel-{index}"
        directory.mkdir()
        config = directory / "sentinel.conf"
        config.write_text(
            f"bind 127.0.0.1\nport {port}\ndir {directory}\nrequirepass {password}\n"
            f"sentinel monitor seif-master 127.0.0.1 {primary_port} 2\n"
            f"sentinel auth-pass seif-master {password}\nsentinel sentinel-pass {password}\n"
            "sentinel down-after-milliseconds seif-master 500\n"
            "sentinel failover-timeout seif-master 5000\n"
            "sentinel parallel-syncs seif-master 1\n"
        )
        config.chmod(0o600)
        processes.append(
            subprocess.Popen(
                [executable, str(config), "--sentinel"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        )

    async def verify():
        import httpx
        from redis.asyncio import Redis
        from redis.exceptions import RedisError

        from seif.app import create_app

        settings = Settings(
            demo=True,
            encryption_key=secrets.token_bytes(32),
            sentinels=tuple(("127.0.0.1", port) for port in ports[3:]),
            redis_password=password,
            sentinel_password=password,
        )
        app = create_app(settings)
        vault = app.state.vault
        replicas = [
            Redis(host="127.0.0.1", port=port, password=password, socket_timeout=1, socket_connect_timeout=1)
            for port in ports[1:3]
        ]

        async def until(check, timeout=20):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    result = await check()
                    if result:
                        return result
                except (RedisError, OSError):
                    pass
                await asyncio.sleep(0.05)
            raise AssertionError("Isolated Sentinel cluster did not converge in time")

        try:
            await until(lambda: vault.redis.ping())
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://testserver"
                ) as client:
                    source = "Клиент Иванов Иван Иванович, email: synthetic@example.net."
                    payload_id = "isolated-sentinel-failover"
                    masked_response = await client.post("/process", json={"payload": source, "payload_id": payload_id})
                    assert masked_response.status_code == 200
                    masked = masked_response.json()["result"]
                    assert masked != source
                    key = vault.key("demo", payload_id)
                    record = await vault.get(key)
                    assert await vault.put_if_absent(key, {"original": "other"}) == record

                    async def replicated_and_discovered():
                        blobs = await asyncio.gather(*(replica.get(key) for replica in replicas))
                        if not all(blobs):
                            return False
                        # Replicated data and peer discovery are insufficient:
                        # Sentinel learns eligible replicas on its own INFO cycle.
                        # The old guard raced this cycle under concurrent test load,
                        # allowing a primary kill before any election candidate existed.
                        expected_ports = set(ports[1:3])
                        for sentinel_client in vault.sentinel.sentinels:
                            master, candidates = await asyncio.gather(
                                sentinel_client.sentinel_master("seif-master"),
                                sentinel_client.sentinel_slaves("seif-master"),
                            )
                            eligible = {
                                candidate["port"]
                                for candidate in candidates
                                if candidate.get("master-link-status") == "ok"
                                and candidate.get("slave-priority", 0) > 0
                                and candidate.get("role-reported") == "slave"
                                and not any(candidate.get(flag) for flag in ("is_sdown", "is_odown", "is_disconnected"))
                            }
                            if master["num-other-sentinels"] < 2 or not expected_ports <= eligible:
                                return False
                            await sentinel_client.execute_command("SENTINEL", "CKQUORUM", "seif-master")
                        return True

                    await until(replicated_and_discovered)
                    raw = await replicas[0].get(key)
                    assert b"synthetic@example.net" not in raw
                    failover_started = time.monotonic()
                    processes[0].kill()
                    processes[0].wait(timeout=3)

                    async def promoted():
                        address = await vault.sentinel.discover_master("seif-master")
                        return address if address[1] != primary_port else None

                    new_master = await until(promoted)
                    await until(lambda: vault.get(key))
                    assert await vault.get(key) == record
                    restored_response = await client.post(
                        "/process", json={"payload": masked, "payload_id": payload_id}
                    )
                    assert restored_response.status_code == 200
                    assert restored_response.json() == {"result": source}
                    recovered_seconds = time.monotonic() - failover_started
                    new_response = await client.post(
                        "/process", json={"payload": source, "payload_id": "after-failure"}
                    )
                    assert new_response.status_code == 200
                    assert new_response.json() == {"result": masked}
                    assert await vault.get(vault.key("demo", "after-failure")) is not None
                    report = {
                        "scenario": "Isolated 3 Redis + 3 authenticated Sentinel processes; quorum 2; primary killed",
                        "redis_nodes": 3,
                        "sentinel_nodes": 3,
                        "original_primary_port": primary_port,
                        "promoted_primary_port": new_master[1],
                        "recovery_seconds": round(recovered_seconds, 3),
                        "same_vault_and_client_after_failover": True,
                        "record_verified_on_both_replicas_before_failure": True,
                        "all_sentinels_confirmed_both_eligible_replicas_and_quorum_before_failure": True,
                        "redis_payload_ciphertext_only": True,
                        "http_process_roundtrip_exact_after_failover": True,
                        "new_http_writes_after_failover": True,
                        "first_writer_preserved": True,
                        "limitations": [
                            "Loopback processes, not Kubernetes or cross-host network failure.",
                            "HTTP contract exercised through ASGI transport, not an external load generator.",
                            "Replication was verified before primary termination; asynchronous replication can lose recent acknowledged writes.",
                        ],
                    }
                    report_path = os.getenv("SEIF_TEST_SENTINEL_REPORT")
                    if report_path:
                        from pathlib import Path

                        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        finally:
            await vault.close()
            await asyncio.gather(*(replica.aclose() for replica in replicas))

    try:
        asyncio.run(verify())
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
