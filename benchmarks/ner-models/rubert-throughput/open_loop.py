"""Fixed-arrival organizer HTTP load with explicit loss and scheduling latency.

Mask mode offers target_rps NEW mask requests each second. Mixed mode offers
target_rps / 2 mask+restore pairs: the dependent restore follows its mask,
so its HTTP arrival time is not an independent fixed-rate event.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import secrets
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
scaling = importlib.import_module("benchmarks.ner-models.rubert-throughput.http_scaling")
common, smoke, require = scaling.common, scaling.smoke, scaling.require


def provenance(args):
    result = scaling.provenance(args)
    result["source_sha256"][str(Path(__file__).relative_to(ROOT))] = scaling.legacy.file_digest(Path(__file__))
    return result


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": len(ordered), "p50": statistics.median(ordered),
            "p95": ordered[math.ceil(len(ordered) * .95) - 1],
            "p99": ordered[math.ceil(len(ordered) * .99) - 1], "max": ordered[-1],
            "over_500ms": sum(value > 500 for value in ordered)}


def attempt(pairs, cases, index, run_id, scheduled_at, mode):  # noqa: PLR0913
    """Carry fixed scheduled-request fields explicitly across the executor boundary."""
    dispatched_at = time.perf_counter()
    case, replica = cases[index % len(cases)], index % len(pairs)
    row = scaling.request(pairs, cases, index, "mask", run_id)
    mask_finished_at = time.perf_counter()
    row.update(scheduling_delay_ms=(dispatched_at - scheduled_at) * 1000,
               scheduled_mask_latency_ms=(mask_finished_at - scheduled_at) * 1000,
               finished_at=mask_finished_at)
    if mode == "mixed" and "error" not in row:
        started = time.perf_counter()
        restore = {"status": None}
        try:
            status, body = pairs[replica]["api"].call("/v1/unmask", {
                "payload": row["masked"], "payload_id": row["payload_id"]})
            restore["status"] = status
            if status != 200:
                restore["error"] = f"http_{status}"
            else:
                restore["exact"] = isinstance(body, dict) and body.get("result") == case["text"]
        except Exception as exc:
            restore["error"] = "timeout" if isinstance(exc, TimeoutError) else "invalid_response_or_transport"
        finished_at = time.perf_counter()
        restore.update(latency_ms=(finished_at - started) * 1000, finished_at=finished_at,
                       scheduled_pair_latency_ms=(finished_at - scheduled_at) * 1000)
        row["restore"] = restore
    # Keep only eight possible examples for the out-of-band restoration check.
    # Raw text and generated payload IDs are never written into the report.
    if mode != "mask" or index >= 8:
        row.pop("masked", None)
        row.pop("payload_id", None)
    return row


def wait_for_window_and_drain(pending, lock, window_end, timeout):
    remaining = window_end - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    with lock:
        draining = tuple(pending)
    _, unfinished = wait(draining, timeout=timeout)
    return len(unfinished)


def fixed_arrivals(args, pairs, cases):
    pair_rate = args.target_rps / (2 if args.mode == "mixed" else 1)
    offered = math.floor(pair_rate * args.duration)
    run_id = secrets.token_hex(16)
    outcomes, pending, lock = [], set(), threading.Lock()
    drops = Counter(capacity=0, arrival_deadline=0)
    state = {"active": 0, "peak_active": 0, "callback_failures": 0}
    started = time.perf_counter()
    window_end = started + args.duration

    def completed(future):
        try:
            outcome = future.result()
        except Exception:
            # The worker normally converts transport/validation failures into
            # an outcome. An unexpected harness failure invalidates the run.
            outcome = None
        with lock:
            if outcome is not None:
                outcomes.append(outcome)
            else:
                state["callback_failures"] += 1
            state["active"] -= 1
            pending.discard(future)

    pool = ThreadPoolExecutor(max_workers=args.max_inflight)
    try:
        for index in range(offered):
            scheduled_at = started + index / pair_rate
            remaining = scheduled_at - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            now = time.perf_counter()
            if now > window_end or (now - scheduled_at) * 1000 > args.arrival_deadline_ms:
                drops["arrival_deadline"] += 1
                continue
            with lock:
                full = state["active"] >= args.max_inflight
                if not full:
                    state["active"] += 1
                    state["peak_active"] = max(state["peak_active"], state["active"])
            if full:
                drops["capacity"] += 1
                continue
            future = pool.submit(attempt, pairs, cases, index, run_id, scheduled_at, args.mode)
            with lock:
                pending.add(future)
            future.add_done_callback(completed)
        outstanding_at_drain_deadline = wait_for_window_and_drain(pending, lock, window_end, args.drain_timeout)
    finally:
        # Timed HTTP calls are bounded by --timeout. Never abandon processes
        # while their workers are still writing outcomes/counters.
        pool.shutdown(wait=True, cancel_futures=False)
    finished = time.perf_counter()
    return outcomes, {"offered_mask_requests": offered,
                      "nominal_offered_http_requests": offered * (2 if args.mode == "mixed" else 1),
                      "scheduled_arrival_rate": pair_rate, "window_seconds": args.duration,
                      "measurement_seconds_including_drain": finished - started,
                      "drain_seconds": max(0, finished - window_end), "window_end": window_end,
                      "unsent_capacity_drops": drops["capacity"],
                      "unsent_arrival_deadline_drops": drops["arrival_deadline"],
                      "outstanding_at_drain_deadline": outstanding_at_drain_deadline, **state}


def drain_ner(args, pairs):
    deadline = time.monotonic() + args.timeout
    after = scaling.snapshot_pairs(pairs)
    while any(row["ner"]["http_active"] or row["ner"]["model_active"] for row in after):
        if time.monotonic() >= deadline:
            break
        time.sleep(.1)
        after = scaling.snapshot_pairs(pairs)
    return after


def report_phase(args, outcomes, traffic, before, after):
    elapsed, window_end = traffic["measurement_seconds_including_drain"], traffic["window_end"]
    dispatched = len(outcomes)
    successful = [row for row in outcomes if "error" not in row]
    matches = [row for row in successful if row["mask_match"] and row["entities_match"] and row["types_match"]]
    completed_in_window = sum(row["finished_at"] <= window_end for row in successful)
    losses = traffic["unsent_capacity_drops"] + traffic["unsent_arrival_deadline_drops"]
    summary = scaling.summarize(outcomes, elapsed, "mask", args.max_inflight, before, after) if outcomes else {
        "valid_successful_measurement": False, "every_request_completed_real_ner": False}
    summary.update({key: value for key, value in traffic.items() if key not in {"window_end", "active"}})
    summary.update(dispatched_mask_requests=dispatched,
                   dispatched_mask_rps=dispatched / args.duration,
                   successful_mask_requests=len(successful),
                   completed_mask_rps_with_drain=len(successful) / elapsed,
                   completed_mask_rps_within_window=completed_in_window / args.duration,
                   completed_mask_requests_within_window=completed_in_window,
                   successful_mask_requests_after_window=len(successful) - completed_in_window,
                   reference_match_ratio_of_offered=len(matches) / traffic["offered_mask_requests"],
                   scheduling_delay_ms=distribution([row["scheduling_delay_ms"] for row in outcomes]),
                   scheduled_arrival_mask_latency_ms=distribution([row["scheduled_mask_latency_ms"] for row in outcomes]),
                   response_latency_ms=distribution([row["latency_ms"] for row in outcomes]),
                   unsent_total=losses,
                   unsent_fraction=losses / traffic["offered_mask_requests"])
    valid_arrivals = (losses == 0 and traffic["callback_failures"] == 0
                      and traffic["outstanding_at_drain_deadline"] == 0
                      and dispatched == traffic["offered_mask_requests"])
    summary["valid_successful_measurement"] &= valid_arrivals
    if args.mode == "mixed":
        restores = [row["restore"] for row in outcomes if "restore" in row]
        good_restores = [row for row in restores if "error" not in row and row.get("exact")]
        summary["mixed"] = {
            "target_total_http_rps": args.target_rps, "target_mask_pair_rate": args.target_rps / 2,
            "restore_requests": len(restores), "successful_exact_restores": len(good_restores),
            "restore_errors": dict(Counter(row["error"] for row in restores if "error" in row)),
            "inexact_restores": sum("error" not in row and not row.get("exact") for row in restores),
            "restores_not_dispatched_due_to_mask_failure": traffic["offered_mask_requests"] - len(restores),
            "completed_total_http_rps_with_drain": (len(successful) + len(good_restores)) / elapsed,
            "completed_total_http_rps_within_window": (completed_in_window + sum(
                row["finished_at"] <= window_end for row in good_restores)) / args.duration,
            "restore_response_latency_ms": distribution([row["latency_ms"] for row in restores]),
            "scheduled_pair_completion_latency_ms": distribution([row["scheduled_pair_latency_ms"] for row in restores]),
            "sticky_restore": True,
            "arrival_note": "Fixed-rate mask+restore pairs; restores follow mask completion, not independent HTTP arrivals."}
        summary["valid_successful_measurement"] &= len(good_restores) == traffic["offered_mask_requests"]
    return summary


def run(args, cases):
    children, pairs = [], []
    report = {"status": "FAIL"}
    with tempfile.TemporaryDirectory(prefix="seif-rubert-open-loop-") as folder, ExitStack() as stack:
        temporary = Path(folder)
        try:
            for replica in range(args.replicas):
                logs = [stack.enter_context((temporary / f"{label}-{replica}.log").open("wb"))
                        for label in ("ner", "api")]
                pair = scaling.launch_pair(args, replica, temporary, children, logs)
                pairs.append(pair)
                for _ in range(5):
                    status, _body = pair["api"].call("/v1/mask", {
                        "payload": "Иван Иванов приехал в Москву.", "payload_id": "warmup-" + secrets.token_hex(12)})
                    require(status == 200, "warmup_success")
            report["health"] = [pair["health"] for pair in pairs]
            report["models"] = [common.ner_snapshot(pair["ner"]) for pair in pairs]
            before = scaling.snapshot_pairs(pairs)
            print(json.dumps({"phase": "started", "mode": args.mode, "target_rps": args.target_rps,
                              "seconds": args.duration, "replicas": args.replicas}), flush=True)
            outcomes, traffic = fixed_arrivals(args, pairs, cases)
            after = drain_ner(args, pairs)
            report["phase"] = report_phase(args, outcomes, traffic, before, after)
            if args.mode == "mask":
                restored = 0
                for row in outcomes:
                    if "masked" not in row or "error" in row:
                        continue
                    case = next(case for case in cases if case["case_id"] == row["case_id"])
                    status, body = pairs[row["replica"]]["api"].call("/v1/unmask", {
                        "payload": row["masked"], "payload_id": row["payload_id"]})
                    require(status == 200 and body.get("result") == case["text"], "sample_sticky_restore_exact")
                    restored += 1
                report["restore_samples"] = {"checked": restored, "exact": restored, "excluded_from_mask_timing": True}
                require(scaling.snapshot_pairs(pairs) == after, "restore_did_not_invoke_ner")
            smoke.terminate_children([("ner-0", pairs[0]["ner_process"])])
            status, _body = pairs[0]["api"].call("/v1/mask", {
                "payload": "Иван Иванов", "payload_id": "failure-" + secrets.token_hex(16)})
            require(status == 503, "missing_ner_fails_closed")
            report["missing_ner_returns_503"] = True
            report["status"] = "PASS" if report["phase"]["valid_successful_measurement"] else "OVERLOAD"
            print(json.dumps({"phase": "completed", "status": report["status"],
                              "mask_rps_with_drain": report["phase"]["completed_mask_rps_with_drain"],
                              "unsent": report["phase"]["unsent_total"],
                              "scheduled_p95_ms": report["phase"]["scheduled_arrival_mask_latency_ms"]["p95"]}), flush=True)
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            for pair in pairs:
                pair["api"].close()
                pair["ner"].close()
            report["cleanup"] = smoke.terminate_children(children)
            if report["status"] != "PASS":
                report["failure_logs"] = scaling.preserve_logs(args, args.replicas, temporary)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--ner-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--api-python", type=Path, default=ROOT.parent / "seif-pii/.venv/bin/python")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    parser.add_argument("--replicas", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--mode", choices=("mask", "mixed"), default="mask")
    parser.add_argument("--target-rps", type=float, default=2000)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--max-inflight", type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument("--arrival-deadline-ms", type=float, default=250)
    parser.add_argument("--drain-timeout", type=float, default=60)
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--api-cpu-workers", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--ner-http-backend", choices=("httpx", "aiohttp"), default="httpx")
    parser.add_argument("--api-log-level", choices=("INFO", "WARNING"), default="WARNING")
    parser.add_argument("--api-ner-concurrency", type=int, default=64)
    parser.add_argument("--model-batch-size", "--batch-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=2)
    parser.add_argument("--ner-max-jobs", type=int, default=128)
    parser.add_argument("--ner-max-http", type=int, default=256)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bounded = {"target_rps": (1, 10000), "duration": (1, 600), "arrival_deadline_ms": (1, 1000),
               "drain_timeout": (1, 120), "batch_wait_ms": (0, 20), "startup_timeout": (1, 1800)}
    for name, (minimum, maximum) in bounded.items():
        if not math.isfinite(getattr(args, name)) or not minimum <= getattr(args, name) <= maximum:
            parser.error(f"{name} must be finite and between {minimum} and {maximum}")
    if not 20 < args.timeout <= 120 or not 1 <= args.api_ner_concurrency <= 128:
        parser.error("Require 20 < timeout <= 120 and 1 <= api-ner-concurrency <= 128")
    if not args.model_batch_size <= args.ner_max_jobs <= args.ner_max_http <= 512:
        parser.error("Require batch-size <= ner-max-jobs <= ner-max-http <= 512")
    if math.floor(args.target_rps * args.duration / (2 if args.mode == "mixed" else 1)) < 1:
        parser.error("At least one request must be scheduled")
    for name in ("model_path", "reference_cache", "ner_python", "api_python", "dataset", "output"):
        path = getattr(args, name).expanduser()
        setattr(args, name, path.absolute() if name in {"ner_python", "api_python"} else path.resolve())
    args.logs_dir = ROOT / "local-data/rubert-throughput" / (args.output.stem + "-" + secrets.token_hex(8))
    return args


def main():
    args = parse_args()
    config_keys = ("replicas", "mode", "target_rps", "duration", "max_inflight", "arrival_deadline_ms",
                   "drain_timeout", "cpu_threads", "api_cpu_workers", "api_ner_concurrency", "model_batch_size",
                   "ner_http_backend", "api_log_level",
                   "batch_wait_ms", "ner_max_jobs", "ner_max_http", "timeout")
    report = {"schema_version": 1, "status": "FAIL", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": {key: getattr(args, key) for key in config_keys},
              "limitations": ["Local loopback, one physical GPU; independent API memory vaults and sticky restore.",
                              "No shared Redis, tunnel, request capture, prediction cache or request retries.",
                              "Scheduled-arrival latency includes generator delay; unsent arrivals are losses, not omitted samples.",
                              "Dropped arrivals have no response latency; their count and fraction are reported separately.",
                              "Mixed mode offers fixed-rate mask+restore pairs, not independent equally spaced HTTP arrivals.",
                              "Exact reference parity is required, but is not perfect organizer ground truth."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        try:
            report["provenance"] = provenance(args)
            cases = scaling.prepare_cases(args)
            report["driver_environment"] = scaling.legacy.driver_environment()
            report["service"] = run(args, cases)
            require(report["provenance"] == provenance(args), "immutable_sources_assets_models")
            report["status"] = report["service"]["status"]
        except Exception as exc:
            report["error_type"] = type(exc).__name__
            if isinstance(exc, smoke.SmokeFailure):
                report["failed_check"] = str(exc)
        finally:
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
