"""Bounded TTL vault. Only authenticated ciphertext is stored, including in RAM.

Redis SET NX commits one winner atomically: no expiring-lock race can overwrite
another request's mapping. Both keys and payload digests use a domain-separated
HMAC. Tenant + payload ID are bound to AES-GCM as associated data.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from collections import OrderedDict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class VaultFull(Exception):
    pass


class Vault:
    def __init__(self, settings):
        self.settings = settings
        self.cipher = AESGCM(settings.encryption_key)
        self.hash_key = hmac.digest(settings.encryption_key, b"SEIF:hmac:v1", "sha256")
        self.records = OrderedDict()
        self.total_bytes = 0
        self.guard = asyncio.Lock()
        self.redis = None
        self.sentinel = None
        if settings.sentinels:
            from redis.asyncio.sentinel import Sentinel

            # Discovery authentication is independent from Redis authentication.
            # Every new master connection goes through Sentinel's managed pool,
            # which discards stale primary connections after a failover.
            self.sentinel = Sentinel(
                settings.sentinels,
                sentinel_kwargs={
                    "password": settings.sentinel_password or None,
                    "socket_timeout": 2,
                    "socket_connect_timeout": 2,
                    "max_connections": 32,
                    "decode_responses": False,
                },
                password=settings.redis_password or None,
                socket_timeout=2,
                socket_connect_timeout=2,
                max_connections=256,
                decode_responses=False,
            )
            self.redis = self.sentinel.master_for(settings.sentinel_master)
        elif settings.redis_url:
            from redis.asyncio import Redis

            self.redis = Redis.from_url(
                settings.redis_url,
                password=settings.redis_password or None,
                socket_timeout=2,
                socket_connect_timeout=2,
                max_connections=256,
                decode_responses=False,
            )

    @property
    def distributed(self) -> bool:
        """Whether correlation records are shared across API processes."""
        return self.redis is not None

    def digest(self, text: str) -> str:
        return hmac.new(self.hash_key, text.encode(), hashlib.sha256).hexdigest()

    def key(self, tenant: str, payload_id: str) -> str:
        return "seif:v1:" + self.digest(json.dumps([tenant, payload_id], ensure_ascii=False))

    def _seal(self, key: str, record: dict) -> bytes:
        nonce = os.urandom(12)
        plain = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
        return nonce + self.cipher.encrypt(nonce, plain, key.encode())

    def _open(self, key: str, blob: bytes) -> dict:
        return json.loads(self.cipher.decrypt(blob[:12], blob[12:], key.encode()))

    def _expire(self):
        now = time.monotonic()
        while self.records:
            _, (expiry, blob) = next(iter(self.records.items()))
            if expiry > now:
                break
            _, (_, removed) = self.records.popitem(last=False)
            self.total_bytes -= len(removed)

    async def get(self, key: str) -> dict | None:
        if self.redis:
            blob = await self.redis.get(key)
        else:
            async with self.guard:
                self._expire()
                entry = self.records.get(key)
                blob = entry[1] if entry else None
        return self._open(key, blob) if blob is not None else None

    async def put_if_absent(self, key: str, record: dict) -> dict:
        blob = self._seal(key, record)
        if self.redis:
            # Capacity is bounded by Redis maxmemory/noeviction (see compose).
            if await self.redis.set(key, blob, ex=self.settings.ttl_seconds, nx=True):
                return record
            winner = await self.get(key)
            if winner is None:
                raise VaultFull("Concurrent expiration; retry")
            return winner
        async with self.guard:
            self._expire()
            if key in self.records:
                return self._open(key, self.records[key][1])
            if (
                len(self.records) >= self.settings.max_records
                or self.total_bytes + len(blob) > self.settings.max_store_bytes
            ):
                raise VaultFull("Capacity reached")
            self.records[key] = (time.monotonic() + self.settings.ttl_seconds, blob)
            self.total_bytes += len(blob)
            return record

    async def ping(self):
        if self.redis:
            await self.redis.ping()

    async def close(self):
        # The master client owns its pool; Sentinel owns separate discovery pools.
        # Close both even if one fails, without exposing connection credentials.
        clients = [self.redis] if self.redis is not None else []
        if self.sentinel is not None:
            clients.extend(self.sentinel.sentinels)
        results = await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
