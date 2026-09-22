"""Opt-in integration tests against an explicitly supplied isolated Redis.

Run with SEIF_TEST_REDIS_URL=redis://127.0.0.1:6385/15 pytest tests/test_redis.py.
Tests delete only their own UUID-scoped keys; they never FLUSHDB/FLUSHALL.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from redis import Redis

from seif.app import create_app
from seif.config import Policy, Settings
from seif.vault import Vault


@pytest.fixture
def redis_context():
    url = os.getenv("SEIF_TEST_REDIS_URL")
    if not url:
        pytest.skip("Set SEIF_TEST_REDIS_URL to opt into isolated Redis integration tests")
    tenant = "integration-" + uuid.uuid4().hex
    settings = Settings(redis_url=url, ttl_seconds=30, policies={tenant: Policy(api_key="integration-only", rps=10_000)})
    headers = {"X-System-ID": tenant, "X-API-Key": "integration-only"}
    sync_redis = Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
    sync_redis.ping()
    owned_keys = []
    # Compute cleanup keys locally without accessing or enumerating other data.
    key_vault = Vault(settings)
    owned_keys.append("seif:rate:" + key_vault.digest(tenant))
    try:
        yield settings, tenant, headers, owned_keys, sync_redis
    finally:
        if owned_keys:
            sync_redis.delete(*owned_keys)
        sync_redis.close()
        asyncio.run(key_vault.close())


def test_independent_workers_share_mapping_and_retry_state(redis_context):
    settings, tenant, headers, keys, sync_redis = redis_context
    app_a, app_b = create_app(settings), create_app(settings)
    payload_id = uuid.uuid4().hex
    key = app_a.state.vault.key(tenant, payload_id)
    keys.append(key)
    original = "Клиент Иванов Иван Иванович, email worker@example.net."
    with TestClient(app_a) as a, TestClient(app_b) as b:
        first = a.post("/process", json={"payload": original, "payload_id": payload_id}, headers=headers)
        assert first.status_code == 200
        masked = first.json()["result"]
        assert masked != original
        retry = b.post("/process", json={"payload": original, "payload_id": payload_id}, headers=headers)
        assert retry.json() == first.json()
        restored = b.post("/process", json={"payload": masked, "payload_id": payload_id}, headers=headers)
        assert restored.status_code == 200
        assert restored.json() == {"result": original}
        reverse_retry = a.post("/process", json={"payload": masked, "payload_id": payload_id}, headers=headers)
        assert reverse_retry.json() == restored.json()
        assert not app_a.state.vault.records and not app_b.state.vault.records
        ciphertext = sync_redis.get(key)
        assert ciphertext
        assert b"worker@example.net" not in ciphertext
        assert 0 < sync_redis.ttl(key) <= settings.ttl_seconds


def test_mapping_survives_http_process_restart(redis_context):
    settings, tenant, headers, keys, _ = redis_context
    first_app = create_app(settings)
    payload_id = uuid.uuid4().hex
    keys.append(first_app.state.vault.key(tenant, payload_id))
    original = "restart@example.net"
    with TestClient(first_app) as client:
        masked = client.post("/process", json={"payload": original, "payload_id": payload_id}, headers=headers).json()["result"]
    # The new app has no shared Python objects or in-memory mappings with the first.
    with TestClient(create_app(settings)) as client:
        result = client.post("/process", json={"payload": masked, "payload_id": payload_id}, headers=headers)
        assert result.status_code == 200
        assert result.json() == {"result": original}


def test_redis_atomic_setnx_has_one_winner_under_concurrency(redis_context):
    settings, tenant, _, keys, sync_redis = redis_context

    async def scenario():
        first, second = Vault(settings), Vault(settings)
        key = first.key(tenant, uuid.uuid4().hex)
        keys.append(key)
        try:
            values = await asyncio.gather(*(
                (first if index % 2 else second).put_if_absent(key, {"winner": index, "secret": "nx-private@example.net"})
                for index in range(64)
            ))
            assert len({value["winner"] for value in values}) == 1
            assert await first.get(key) == await second.get(key) == values[0]
            assert b"nx-private@example.net" not in sync_redis.get(key)
        finally:
            await asyncio.gather(first.close(), second.close())

    asyncio.run(scenario())


def test_redis_ttl_removes_mapping(redis_context):
    settings, tenant, _, keys, _ = redis_context
    settings.ttl_seconds = 1

    async def scenario():
        vault = Vault(settings)
        key = vault.key(tenant, uuid.uuid4().hex)
        keys.append(key)
        try:
            await vault.put_if_absent(key, {"secret": "ttl-private@example.net"})
            assert await vault.get(key) is not None
            await asyncio.sleep(1.1)
            assert await vault.get(key) is None
        finally:
            await vault.close()

    asyncio.run(scenario())
