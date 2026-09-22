"""Explicit tenant policy; secrets are supplied only through the environment."""

from __future__ import annotations

import base64
import ipaddress
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _validate_host(host: str, ipv6: bool) -> None:
    if ipv6:
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            raise ValueError("SEIF_SENTINELS contains an invalid IPv6 address") from None
        return
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


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?:\[([^\]]+)\]|([^:]+)):(\d{1,5})", endpoint)
    if not match:
        raise ValueError("SEIF_SENTINELS requires host:port or [IPv6]:port endpoints")
    ipv6, hostname, port_text = match.groups()
    host = ipv6 or hostname
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError("SEIF_SENTINELS ports must be between 1 and 65535")
    _validate_host(host, ipv6)
    return host, port


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
        host, port = _parse_endpoint(entry.strip())
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
    masking_enabled: bool = True

    def __post_init__(self) -> None:
        # YAML and programmatic policies share strict validation: a quoted
        # "false", number or null must never silently change protection policy.
        for name in ("enabled", "allow_unmask", "masking_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("min_types", "rps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


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
    request_body_timeout_seconds: float = 30.0
    cpu_workers: int = 4
    ner_url: str = field(default="", repr=False)
    ner_token: str = field(default="", repr=False)
    ner_timeout_seconds: float = 20.0
    policies: dict[str, Policy] = field(default_factory=lambda: {"demo": Policy()})

    def __post_init__(self) -> None:
        if type(self.demo) is not bool:
            raise ValueError("demo must be a boolean")
        for name in (
            "ttl_seconds", "max_records", "max_store_bytes", "max_payload_chars", "max_body_bytes",
            "max_inflight_body_bytes", "max_inflight", "cpu_workers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        timeout = self.request_body_timeout_seconds
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request_body_timeout_seconds must be a positive finite number")

    @staticmethod
    def _load_master_key(demo: bool) -> tuple[bytes, str]:
        encoded = os.getenv("SEIF_MASTER_KEY", "")
        if not encoded and not demo:
            raise ValueError("SEIF_MASTER_KEY is required outside demo mode")
        key = base64.b64decode(encoded, validate=True) if encoded else os.urandom(32)
        if len(key) != 32:
            raise ValueError("SEIF_MASTER_KEY must encode exactly 32 bytes")
        return key, encoded

    @staticmethod
    def _load_redis(encoded: str) -> tuple[str, tuple[tuple[str, int], ...], str, str, str]:
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
        return redis_url, sentinels, sentinel_master, redis_password, sentinel_password

    @staticmethod
    def _load_policies(demo: bool) -> dict[str, Policy]:
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
        return policies

    @classmethod
    def from_env(cls) -> "Settings":
        demo = os.getenv("SEIF_DEMO", "0") == "1"
        key, encoded = cls._load_master_key(demo)
        redis_url, sentinels, sentinel_master, redis_password, sentinel_password = cls._load_redis(encoded)
        policies = cls._load_policies(demo)
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
            request_body_timeout_seconds=float(os.getenv("SEIF_REQUEST_BODY_TIMEOUT_SECONDS", "30")),
            ner_url=os.getenv("SEIF_NER_URL", ""),
            ner_token=os.getenv("SEIF_NER_TOKEN", ""),
            ner_timeout_seconds=float(os.getenv("SEIF_NER_TIMEOUT_SECONDS", "20")),
            policies=policies,
        )
