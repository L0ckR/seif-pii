"""Explicit tenant policy; secrets are supplied only through the environment."""

from __future__ import annotations

import base64
import ipaddress
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def parse_sentinels(value: str) -> tuple[tuple[str, int], ...]:
    """Parse DNS/IPv4 host:port and bracketed IPv6 endpoints without secrets.

    Error messages intentionally omit the supplied address, which may contain
    credentials accidentally pasted from a URL.
    """
    if not value:
        return ()
    entries = value.split(",")
    if len(entries) > 16 or len(value) > 4096:
        raise ValueError("SEIF_SENTINELS accepts at most 16 host:port endpoints")
    addresses = []
    for entry in entries:
        endpoint = entry.strip()
        match = re.fullmatch(r"(?:\[([^\]]+)\]|([^:]+)):(\d{1,5})", endpoint)
        if not match:
            raise ValueError("SEIF_SENTINELS requires host:port or [IPv6]:port endpoints")
        ipv6, hostname, port_text = match.groups()
        host = ipv6 or hostname
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise ValueError("SEIF_SENTINELS ports must be between 1 and 65535")
        if ipv6:
            try:
                ipaddress.IPv6Address(host)
            except ValueError:
                raise ValueError("SEIF_SENTINELS contains an invalid IPv6 address") from None
        else:
            labels = host.rstrip(".").split(".")
            if len(host) > 253 or not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in labels
            ):
                raise ValueError("SEIF_SENTINELS contains an invalid hostname")
            if re.fullmatch(r"[0-9.]+", host):
                try:
                    ipaddress.IPv4Address(host)
                except ValueError:
                    raise ValueError("SEIF_SENTINELS contains an invalid IPv4 address") from None
        address = (host, port)
        if address in addresses:
            raise ValueError("SEIF_SENTINELS endpoints must be unique")
        addresses.append(address)
    return tuple(addresses)


@dataclass(frozen=True)
class Policy:
    enabled: bool = True
    api_key: str = field(default="", repr=False)
    types: tuple[str, ...] = ()  # Empty means all types.
    mode: str = "mask"
    allow_unmask: bool = True
    min_types: int = 1
    required_types: tuple[str, ...] = ()
    allowed_modes: tuple[str, ...] = ("mask", "token", "synthetic")
    extra_rules: tuple[dict, ...] = ()
    rps: int = 1000


@dataclass
class Settings:
    demo: bool = False
    encryption_key: bytes = field(default_factory=lambda: os.urandom(32), repr=False)
    redis_url: str = field(default="", repr=False)
    sentinels: tuple[tuple[str, int], ...] = ()
    sentinel_master: str = "seif-master"
    redis_password: str = field(default="", repr=False)
    sentinel_password: str = field(default="", repr=False)
    ttl_seconds: int = 900
    max_records: int = 500_000
    max_store_bytes: int = 256 * 1024 * 1024
    max_payload_chars: int = 2_000_000
    max_body_bytes: int = 12_100_000
    max_inflight_body_bytes: int = 64 * 1024 * 1024
    max_inflight: int = 128
    cpu_workers: int = 4
    policies: dict[str, Policy] = field(default_factory=lambda: {"demo": Policy()})

    @classmethod
    def from_env(cls) -> "Settings":
        demo = os.getenv("SEIF_DEMO", "0") == "1"
        encoded = os.getenv("SEIF_MASTER_KEY", "")
        if not encoded and not demo:
            raise ValueError("SEIF_MASTER_KEY is required outside demo mode")
        key = base64.b64decode(encoded, validate=True) if encoded else os.urandom(32)
        if len(key) != 32:
            raise ValueError("SEIF_MASTER_KEY must encode exactly 32 bytes")
        redis_url = os.getenv("SEIF_REDIS_URL", "")
        sentinels = parse_sentinels(os.getenv("SEIF_SENTINELS", ""))
        if redis_url and sentinels:
            raise ValueError("Configure SEIF_REDIS_URL or SEIF_SENTINELS, not both")
        sentinel_master = os.getenv("SEIF_SENTINEL_MASTER", "seif-master")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sentinel_master):
            raise ValueError("SEIF_SENTINEL_MASTER must be a valid service name")
        redis_password = os.getenv("SEIF_REDIS_PASSWORD", "")
        sentinel_password = os.getenv("SEIF_SENTINEL_PASSWORD", redis_password)
        if (redis_url or sentinels) and not encoded:
            raise ValueError("Shared Redis requires a stable SEIF_MASTER_KEY")
        path = Path(os.getenv("SEIF_CONFIG", "config/policies.yaml"))
        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
        policies = {}
        for name, item in (raw or {}).get("systems", {}).items():
            item = dict(item)
            secret_name = item.pop("api_key_env", "")
            item["api_key"] = os.getenv(secret_name, "") if secret_name else ""
            for field_name in ("types", "required_types", "allowed_modes", "extra_rules"):
                if field_name in item:
                    item[field_name] = tuple(item[field_name])
            policy = Policy(**item)
            if policy.mode not in {"mask", "token", "synthetic"} or policy.min_types < 1 or policy.rps < 1:
                raise ValueError("Invalid system policy")
            if policy.enabled and not policy.api_key and not (demo and name == "demo"):
                raise ValueError(f"Missing API key for enabled system: {name}")
            policies[name] = policy
        if demo:
            policies.setdefault("demo", Policy())
        if not policies:
            raise ValueError("At least one system must be configured")
        return cls(
            demo=demo,
            encryption_key=key,
            redis_url=redis_url,
            sentinels=sentinels,
            sentinel_master=sentinel_master,
            redis_password=redis_password,
            sentinel_password=sentinel_password,
            ttl_seconds=int(os.getenv("SEIF_TTL_SECONDS", "900")),
            cpu_workers=int(os.getenv("SEIF_CPU_WORKERS", "4")),
            policies=policies,
        )
