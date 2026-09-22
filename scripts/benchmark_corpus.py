#!/usr/bin/env python3
"""Isolated open-loop corpus benchmark: mask/unmask pairs, no hidden retries.

JSONL rows contain case_id, payload and optional positive weight. --prepare-only
validates inputs without contacting any service. Credentials come only from
SEIF_BENCH_SYSTEM / SEIF_BENCH_API_KEY. Reports never contain texts or case IDs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import uuid
from array import array
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

MAX_CORPUS_BYTES = 256 * 1024 * 1024
MAX_PAYLOAD_CHARS = 2_000_000
PROTECTED_PORTS = {8765, 8766, 8770, 6386, 6387, 6388, 26386, 26387, 26388}


class BenchmarkError(ValueError):
    """An input-free diagnostic safe for the console."""


@dataclass(frozen=True)
class Case:
    payload: str = field(repr=False)
    weight: float = 1.0


@dataclass(frozen=True)
class Corpus:
    cases: tuple[Case, ...] = field(repr=False)
    sha256: str
    byte_count: int

    def summary(self):
        lengths = [len(case.payload) for case in self.cases]
        return {"sha256": self.sha256, "bytes": self.byte_count, "cases": len(self.cases),
                "total_weight": math.fsum(case.weight for case in self.cases),
                "payload_characters": {"min": min(lengths), "max": max(lengths), "sum": sum(lengths)}}


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError("Duplicate JSON object key")
        result[key] = value
    return result


def reject_constant(_value):
    raise BenchmarkError("Non-finite JSON number")


def strict_json(content):
    return json.loads(content, object_pairs_hook=strict_object, parse_constant=reject_constant)


def _read_case(raw, seen, line_number):
    try:
        row = strict_json(raw.decode("utf-8"))
        if not isinstance(row, dict) or set(row) - {"case_id", "payload", "weight"}:
            raise ValueError
        identifier, payload = row["case_id"], row["payload"]
        weight = row.get("weight", 1)
        if (not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 256
                or identifier in seen or not isinstance(payload, str)
                or len(payload) > MAX_PAYLOAD_CHARS or type(weight) not in (int, float)
                or not math.isfinite(weight) or not 0.000001 <= weight <= 1_000_000):
            raise ValueError
        identifier.encode("utf-8")
        payload.encode("utf-8")
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise BenchmarkError(f"Invalid corpus row at line {line_number}") from None
    return identifier, Case(payload, float(weight))


def load_corpus(path: Path) -> Corpus:
    cases, seen = [], set()
    digest, byte_count = hashlib.sha256(), 0
    try:
        with path.open("rb") as stream:
            for line_number, raw in enumerate(iter(lambda: stream.readline(MAX_CORPUS_BYTES + 1), b""), 1):
                byte_count += len(raw)
                if byte_count > MAX_CORPUS_BYTES:
                    raise BenchmarkError("Corpus exceeds the 256 MiB limit")
                digest.update(raw)
                if not raw.strip():
                    continue
                identifier, case = _read_case(raw, seen, line_number)
                seen.add(identifier)
                cases.append(case)
                if len(cases) > 100_000:
                    raise BenchmarkError("Corpus exceeds the 100000-case limit")
    except OSError:
        raise BenchmarkError("Cannot read corpus file") from None
    if not cases:
        raise BenchmarkError("Corpus contains no cases")
    return Corpus(tuple(cases), digest.hexdigest(), byte_count)


class WeightedCycle:
    """Seeded weighted virtual cycles; equal weights visit every case per cycle.

    Case i has one visit in each interval [k/weight_i, (k+1)/weight_i).
    Independent seeded jitter shuffles visits without expanding large weights
    into a repeated list. Memory is O(cases), each offered pair O(log cases).
    """
    def __init__(self, cases, seed):
        self.cases, self.rng = cases, random.Random(seed)
        self.heap = [(self.rng.random() / case.weight, i, 0) for i, case in enumerate(cases)]
        heapq.heapify(self.heap)

    def __next__(self):
        _, index, cycle = heapq.heappop(self.heap)
        weight = self.cases[index].weight
        heapq.heappush(self.heap, ((cycle + 1 + self.rng.random()) / weight, index, cycle + 1))
        return index


@dataclass(frozen=True)
class Config:
    url: str
    rps: float = 1000
    duration: float = 30
    concurrency: int = 256
    timeout: float = 10
    seed: int = 20260922
    max_response_bytes: int = 32 * 1024 * 1024

    @property
    def offered_pairs(self):
        return math.ceil(self.rps * self.duration / 2)

    def validate(self):
        try:
            parsed = urlsplit(self.url)
            valid_url = (parsed.scheme in {"http", "https"}
                         and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                         and parsed.port is not None and 1 <= parsed.port <= 65535
                         and parsed.port not in PROTECTED_PORTS
                         and not parsed.username and not parsed.password
                         and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment)
        except ValueError:
            valid_url = False
        if not valid_url:
            raise BenchmarkError("Use a dedicated loopback HTTP endpoint; shared live service ports are forbidden")
        if not all(math.isfinite(value) and value > 0 for value in (self.rps, self.duration, self.timeout)):
            raise BenchmarkError("Rate, duration and timeout must be finite and positive")
        if (not math.isfinite(self.rps * self.duration) or self.rps * self.duration > 10_000_000
                or self.duration > 3600 or self.timeout > 120
                or not 1 <= self.concurrency <= 8192 or not 1024 <= self.max_response_bytes <= 64 * 1024 * 1024):
            raise BenchmarkError("Benchmark exceeds bounded rate/duration/concurrency/response settings")


@dataclass(frozen=True)
class Outcome:
    status: int | None = None
    result: str | None = field(default=None, repr=False)
    error: str | None = None


def parse_response(content: bytes) -> Outcome:
    try:
        body = strict_json(content.decode("utf-8"))
        if not isinstance(body, dict) or set(body) != {"result"} or not isinstance(body["result"], str):
            return Outcome(200, error="invalid_contract")
        body["result"].encode("utf-8")
        return Outcome(200, result=body["result"])
    except (ValueError, TypeError, RecursionError):
        return Outcome(200, error="invalid_json")


async def post_process(client, config, payload, correlation) -> Outcome:
    try:
        async with client.post(config.url.rstrip("/") + "/process",
                               json={"payload": payload, "payload_id": correlation},
                               allow_redirects=False) as response:
            if response.status != 200:
                return Outcome(response.status, error=f"http_{response.status}")
            content = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                if len(content) + len(chunk) > config.max_response_bytes:
                    return Outcome(200, error="response_too_large")
                content.extend(chunk)
            return parse_response(content)
    except asyncio.TimeoutError:
        return Outcome(error="timeout")
    except aiohttp.ClientError:
        return Outcome(error="transport")
    except Exception:
        return Outcome(error="request_failed")


def latency_summary(values):
    ordered = sorted(values)

    def percentile(percent):
        if not ordered:
            return None
        position = (len(ordered) - 1) * percent / 100
        lower, upper = math.floor(position), math.ceil(position)
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)

    return {"count": len(values), "p50": percentile(50), "p95": percentile(95), "p99": percentile(99),
            "max": round(ordered[-1], 3) if ordered else None,
            "over_500ms": sum(value > 500 for value in ordered),
            "over_1000ms": sum(value > 1000 for value in ordered)}


def _dispatch_pair(active, counters, finished, pair, limits):
    expired, concurrency = limits
    if expired:
        counters["pairs_dropped_client_deadline"] += 1
    elif len(active) >= concurrency:
        counters["pairs_dropped_client_capacity"] += 1
    else:
        task = asyncio.create_task(pair())
        active.add(task)
        task.add_done_callback(finished)
        counters["pairs_dispatched"] += 1


async def schedule(config, pair, counters, on_offer):
    active = set()
    task_failed = False

    def finished(task):
        nonlocal task_failed
        active.discard(task)
        # Retrieve every exception, including tasks completed before the drain.
        # Never let asyncio emit a task traceback containing source data.
        if task.cancelled() or task.exception() is not None:
            task_failed = True

    started = time.perf_counter()
    interval, max_lag = 2 / config.rps, 0.0
    try:
        for index in range(config.offered_pairs):
            scheduled = started + index * interval
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            now = time.perf_counter()
            max_lag = max(max_lag, now - scheduled)
            counters["pairs_offered"] += 1
            assignment = on_offer(index)
            _dispatch_pair(active, counters, finished, lambda index=index, assignment=assignment: pair(index, assignment),
                           (now >= started + config.duration, config.concurrency))
            if index % 32 == 31:
                await asyncio.sleep(0)
        remaining = started + config.duration - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
        results = await asyncio.gather(*active, return_exceptions=True)
        if task_failed or any(isinstance(result, BaseException) for result in results):
            raise BenchmarkError("Pair execution failed; sensitive exception details suppressed")
    except BaseException:
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        raise
    return time.perf_counter() - started, max_lag * 1000


class _CorpusRun:
    def __init__(self, config, corpus, requester):
        self.config, self.corpus, self.requester = config, corpus, requester
        self.counters = Counter(dict.fromkeys((
            "pairs_offered", "pairs_dispatched", "pairs_dropped_client_capacity", "pairs_dropped_client_deadline",
            "pairs_verified", "pairs_mask_failed", "pairs_unmask_failed", "pairs_roundtrip_mismatch",
            "pairs_mask_changed", "pairs_mask_unchanged", "requests_attempted", "requests_successful"), 0))
        self.errors, self.statuses, self.visited = Counter(), Counter(), Counter()
        self.latency = {phase: array("d") for phase in ("mask", "unmask")}
        self.source = WeightedCycle(self.corpus.cases, self.config.seed)
        self.sequence_hash, self.run_id = hashlib.sha256(), uuid.uuid4().hex

    def on_offer(self, _index):
        # Advance before deciding whether to dispatch: overload cannot shift
        # later sampling, and coroutine start order cannot change assignments.
        case_index = next(self.source)
        self.sequence_hash.update(case_index.to_bytes(8, "big"))
        return case_index

    async def request(self, payload, correlation, phase):
        self.counters["requests_attempted"] += 1
        started = time.perf_counter()
        try:
            try:
                outcome = await self.requester(payload, correlation)
            except Exception:
                outcome = Outcome(error="request_failed")
            if outcome.status is not None:
                self.statuses[str(outcome.status)] += 1
            if outcome.error:
                self.errors[outcome.error] += 1
                return None
            if outcome.status != 200 or not isinstance(outcome.result, str):
                self.errors["invalid_contract"] += 1
                return None
            self.counters["requests_successful"] += 1
            return outcome.result
        finally:
            self.latency[phase].append((time.perf_counter() - started) * 1000)

    async def pair(self, index, case_index):
        original = self.corpus.cases[case_index].payload
        self.visited[case_index] += 1
        correlation = f"corpus-{self.run_id}-{index}"
        masked = await self.request(original, correlation, "mask")
        if masked is None:
            self.counters["pairs_mask_failed"] += 1
        else:
            self.counters["pairs_mask_unchanged" if masked == original else "pairs_mask_changed"] += 1
            restored = await self.request(masked, correlation, "unmask")
            if restored is None:
                self.counters["pairs_unmask_failed"] += 1
            elif restored != original:
                self.counters["pairs_roundtrip_mismatch"] += 1
                self.errors["roundtrip_mismatch"] += 1
            else:
                self.counters["pairs_verified"] += 1

    def report(self, elapsed, lag):
        self.counters["http_requests_offered"] = 2 * self.counters["pairs_offered"]
        self.counters["http_requests_not_attempted"] = self.counters["http_requests_offered"] - self.counters["requests_attempted"]
        all_latencies = self.latency["mask"] + self.latency["unmask"]
        return {"schema_version": 1, "measurement": "completed",
                "method": "open-loop correlation pairs; 2 offered HTTP requests/pair; no retries",
                "transport": "aiohttp", "event_loop": type(asyncio.get_running_loop()).__module__,
                "target_url": self.config.url, "corpus": self.corpus.summary(),
                "sampling": {"algorithm": "weighted virtual cycles with seeded per-visit jitter v1",
                             "seed": self.config.seed, "offered_sequence_sha256": self.sequence_hash.hexdigest(),
                             "unique_cases_dispatched": len(self.visited),
                             "min_visits_dispatched": min(self.visited.values()) if self.visited else 0,
                             "max_visits_dispatched": max(self.visited.values()) if self.visited else 0},
                "configured_duration_seconds": self.config.duration,
                "observed_elapsed_seconds_including_drain": round(elapsed, 6),
                "offered_http_rps": self.config.rps, "max_concurrent_pairs": self.config.concurrency,
                "request_timeout_seconds": self.config.timeout, "max_response_bytes": self.config.max_response_bytes,
                "achieved_attempted_http_rps": round(self.counters["requests_attempted"] / elapsed, 3),
                "achieved_successful_http_rps": round(self.counters["requests_successful"] / elapsed, 3),
                "counts": dict(self.counters), "http_statuses": dict(self.statuses), "errors": dict(self.errors),
                "latency_ms_all_attempts": latency_summary(all_latencies),
                "latency_ms_mask": latency_summary(self.latency["mask"]),
                "latency_ms_unmask": latency_summary(self.latency["unmask"]),
                "client_scheduler_max_lag_ms": round(lag, 3),
                "all_offered_pairs_verified": self.counters["pairs_verified"] == self.config.offered_pairs,
                "limitations": ["Throughput includes draining requests after the scheduling window.",
                                "Pair arrival times are open-loop; the second request follows its first response.",
                                "A failed mask skips its unmask; client drops and unsent second requests remain counted.",
                                "Roundtrip equality is not a PII recall measurement; annotation is evaluated separately.",
                                "No original text, case identifier, request identifier or token is retained in this report."]}


async def run_with_requester(config, corpus, requester):
    """Pure transport boundary makes scheduling and failure tests network-free."""
    config.validate()
    run = _CorpusRun(config, corpus, requester)
    elapsed, lag = await schedule(config, run.pair, run.counters, run.on_offer)
    return run.report(elapsed, lag)


async def benchmark(config, corpus):
    system, key = os.getenv("SEIF_BENCH_SYSTEM", ""), os.getenv("SEIF_BENCH_API_KEY", "")
    if bool(system) != bool(key):
        raise BenchmarkError("Benchmark system and API key must be configured together")
    headers = {"X-System-ID": system, "X-API-Key": key} if system else {}
    connector = aiohttp.TCPConnector(limit=config.concurrency, keepalive_timeout=30)
    async with aiohttp.ClientSession(headers=headers, trust_env=False, connector=connector,
                                     timeout=aiohttp.ClientTimeout(total=config.timeout)) as client:
        return await run_with_requester(config, corpus,
                                        lambda payload, correlation: post_process(client, config, payload, correlation))


def file_hash(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(65536):
                digest.update(chunk)
    except OSError:
        raise BenchmarkError("Cannot read provenance file") from None
    return digest.hexdigest()


def metadata(protocol):
    root = Path(__file__).resolve().parents[1]
    versions = {}
    for package in ("aiohttp", "uvloop", "seif-pii"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = ["scripts/benchmark_corpus.py", "seif/app.py", "seif/config.py", "seif/detector.py",
               "seif/ner.py", "scripts/ner_service.py", "requirements.lock", "deploy/ner/requirements-ner.txt"]
    return {"recorded_at_utc": datetime.now(UTC).isoformat(), "python": sys.version.split()[0],
            "platform": platform.platform(), "cpu_count": os.cpu_count(), "packages": versions,
            "source_sha256": {name: file_hash(root / name) for name in sources},
            "source_sha256_scope": "Local checkout at run start; match deployed service revisions separately.",
            "protocol_sha256": file_hash(protocol) if protocol else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, help="Hash a separately frozen protocol without printing its content")
    parser.add_argument("--url", required=True)
    parser.add_argument("--rps", type=float, default=1000)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-response-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = Config(args.url, args.rps, args.duration, args.concurrency,
                        args.timeout, args.seed, args.max_response_bytes)
        config.validate()
        corpus = load_corpus(args.corpus)
        provenance = metadata(args.protocol)
        if args.output.exists():
            raise BenchmarkError("Output already exists; preserve earlier evidence with a new output path")
        if args.prepare_only:
            report = {"schema_version": 1, "measurement": "not_run", "corpus": corpus.summary(),
                      "target_url": config.url, "planned_pairs": config.offered_pairs, "seed": config.seed}
        else:
            try:
                import uvloop
            except ImportError:
                report = asyncio.run(benchmark(config, corpus))
            else:
                report = uvloop.run(benchmark(config, corpus))
        report["provenance"] = provenance
        encoded = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
        print(json.dumps({"report": str(args.output.resolve()), "sha256": hashlib.sha256(encoded).hexdigest(),
                          "measurement": report["measurement"]}, ensure_ascii=False))
    except BenchmarkError as exc:
        parser.exit(2, str(exc) + "\n")
    except KeyboardInterrupt:
        parser.exit(130, "Benchmark interrupted; no completed result claimed\n")
    except Exception:
        parser.exit(2, "Benchmark failed; sensitive exception details suppressed\n")


if __name__ == "__main__":
    main()
