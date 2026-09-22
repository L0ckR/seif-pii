#!/usr/bin/env python3
"""Open-loop HTTP pair benchmark. Only synthetic data; no hidden retries.

--rps is the offered HTTP request rate (two requests per correlation pair).
Pairs are scheduled at that rate independently of response times. When the
client concurrency cap is reached, the pair is dropped and reported rather
than silently turning this into a closed-loop benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from collections import Counter
from pathlib import Path

import aiohttp

SYNTHETIC_PAYLOADS = (
    "Клиент Иванов Иван Иванович, паспорт 4509 123456; email ivan@example.net.",
    "Телефон: +7 (999) 123-45-67. Дата рождения: 12 марта 1990 года.",
    "Карта: 4111 1111 1111 1111, CVV: 123, ПИН: 4321.",
    "Адрес проживания: г. Москва, ул. Тестовая, дом 17, квартира 8.",
    "Обычный запрос: объясни, почему небо голубое. Личных сведений здесь нет.",
)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def validate_options(args: argparse.Namespace) -> None:
    if not all(math.isfinite(value) and value > 0 for value in (args.rps, args.duration, args.timeout)):
        raise ValueError("rps, duration and timeout must be finite and positive")
    if (type(args.concurrency) is not int or not 1 <= args.concurrency <= 8192
            or args.rps * args.duration > 10_000_000 or args.duration > 3600 or args.timeout > 120):
        raise ValueError("Benchmark exceeds bounded rate, duration, concurrency or timeout settings")


async def benchmark(args: argparse.Namespace) -> dict:
    validate_options(args)
    latencies: list[float] = []
    errors: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    active: set[asyncio.Task] = set()
    headers = {}
    if args.system:
        headers["X-System-ID"] = args.system
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    run_id = uuid.uuid4().hex
    connector = aiohttp.TCPConnector(limit=args.concurrency, keepalive_timeout=30)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout), headers=headers,
        connector=connector, trust_env=False,
    ) as client:
        async def request(payload: str, payload_id: str) -> str | None:
            started = time.perf_counter()
            counters["requests_attempted"] += 1
            try:
                async with client.post(args.url.rstrip("/") + "/process",
                                       json={"payload": payload, "payload_id": payload_id},
                                       allow_redirects=False) as response:
                    statuses[str(response.status)] += 1
                    if response.status != 200:
                        errors[f"http_{response.status}"] += 1
                        return None
                    try:
                        body = await response.json()
                    except (ValueError, aiohttp.ContentTypeError):
                        errors["invalid_json"] += 1
                        return None
                if not isinstance(body, dict) or set(body) != {"result"} or not isinstance(body["result"], str):
                    errors["invalid_contract"] += 1
                    return None
                counters["requests_successful"] += 1
                return body["result"]
            except asyncio.TimeoutError:
                errors["timeout"] += 1
                return None
            except aiohttp.ClientError:
                errors["transport"] += 1
                return None
            finally:
                latencies.append((time.perf_counter() - started) * 1000)

        async def pair(index: int) -> None:
            source = SYNTHETIC_PAYLOADS[index % len(SYNTHETIC_PAYLOADS)]
            correlation = f"bench-{run_id}-{index}"
            masked = await request(source, correlation)
            if masked is None:
                counters["pairs_mask_failed"] += 1
                return
            restored = await request(masked, correlation)
            if restored is None:
                counters["pairs_unmask_failed"] += 1
            elif restored != source:
                errors["roundtrip_mismatch"] += 1
            else:
                counters["pairs_verified"] += 1

        started = time.perf_counter()
        index = 0
        scheduling_lag_max = 0.0
        interval = 2 / args.rps
        # A bounded set prevents the load generator itself becoming an unbounded queue.
        while index * interval < args.duration:
            scheduled = started + index * interval
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            scheduling_lag_max = max(scheduling_lag_max, time.perf_counter() - scheduled)
            counters["pairs_offered"] += 1
            if len(active) >= args.concurrency:
                counters["pairs_dropped_client_capacity"] += 1
            else:
                task = asyncio.create_task(pair(index))
                active.add(task)
                task.add_done_callback(active.discard)
                counters["pairs_dispatched"] += 1
            index += 1
            # Give HTTP tasks time to progress even when scheduler catch-up is needed.
            if index % 32 == 0:
                await asyncio.sleep(0)
        remaining = started + args.duration - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
        await asyncio.gather(*active)
        elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "method": "open-loop correlation pairs; 2 offered HTTP requests/pair; no retries",
        "transport": "aiohttp",
        "event_loop": type(asyncio.get_running_loop()).__module__,
        "target_url": args.url,
        "synthetic_data_only": True,
        "configured_duration_seconds": args.duration,
        "observed_elapsed_seconds_including_drain": round(elapsed, 3),
        "offered_http_rps": args.rps,
        "max_concurrent_pairs": args.concurrency,
        "achieved_attempted_http_rps": round(counters["requests_attempted"] / elapsed, 2),
        "achieved_successful_http_rps": round(counters["requests_successful"] / elapsed, 2),
        "counts": dict(counters),
        "http_statuses": dict(statuses),
        "errors": dict(errors),
        "latency_ms_all_attempts": {
            "p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99), "max": round(max(latencies), 3) if latencies else None,
        },
        "client_scheduler_max_lag_ms": round(scheduling_lag_max * 1000, 3),
        "limitations": [
            "This is a local synthetic sample, not the organizer's closed evaluation dataset.",
            "Reported throughput includes request completion after the scheduling window.",
            "Client drops or significant scheduler lag mean the client could not offer the target load.",
            "Fast responses do not establish PII detection quality or production availability.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rps", type=float, default=1000)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--system")
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        validate_options(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        import uvloop
    except ImportError:
        report = asyncio.run(benchmark(args))
    else:
        report = uvloop.run(benchmark(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
