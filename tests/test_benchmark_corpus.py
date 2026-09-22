"""No sockets, models or organizer texts: exercise the benchmark's measurement boundary."""

import asyncio
import hashlib
import json
from collections import Counter

import pytest

from scripts import benchmark_corpus as bench


def write_rows(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_corpus_hash_defaults_and_safe_summary(tmp_path):
    path = write_rows(tmp_path, [{"case_id": "private-id", "payload": "private-text"},
                                 {"case_id": "empty", "payload": "", "weight": 3}])
    corpus = bench.load_corpus(path)
    assert corpus.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert corpus.summary()["total_weight"] == 4
    assert corpus.summary()["payload_characters"] == {"min": 0, "max": 12, "sum": 12}
    assert "private" not in json.dumps(corpus.summary())
    assert "private" not in repr(corpus)
    assert "private" not in repr(corpus.cases[0])


@pytest.mark.parametrize("row", [
    {}, [], {"case_id": "x", "payload": 42}, {"case_id": 42, "payload": "private-text"},
    {"case_id": " ", "payload": "private-text"},
    {"case_id": "x", "payload": "private-text", "unknown": True},
    *[{"case_id": "x", "payload": "private-text", "weight": value}
      for value in (0, -1, True, "1", None, float("inf"), float("nan"), 10**9)],
    {"case_id": "x", "payload": "\ud800"},
])
def test_invalid_corpus_is_rejected_without_disclosing_data(tmp_path, row):
    path = write_rows(tmp_path, [row])
    with pytest.raises(bench.BenchmarkError, match="Invalid corpus row at line 1") as error:
        bench.load_corpus(path)
    assert "private" not in str(error.value)


def test_duplicate_case_ids_and_duplicate_json_keys_rejected(tmp_path):
    row = {"case_id": "private-id", "payload": "private-text"}
    path = write_rows(tmp_path, [row, row])
    with pytest.raises(bench.BenchmarkError, match="line 2"):
        bench.load_corpus(path)
    path = tmp_path / "duplicates.jsonl"
    path.write_text('{"case_id":"x","payload":"a","payload":"private-text"}\n')
    with pytest.raises(bench.BenchmarkError, match="line 1"):
        bench.load_corpus(path)


def test_empty_corpus_and_size_limits(tmp_path, monkeypatch):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"\n")
    with pytest.raises(bench.BenchmarkError, match="no cases"):
        bench.load_corpus(path)
    monkeypatch.setattr(bench, "MAX_CORPUS_BYTES", 16)
    path.write_bytes(b"x" * 17)
    with pytest.raises(bench.BenchmarkError, match="limit"):
        bench.load_corpus(path)


def test_weighted_cycles_reproducible_without_expanding_weights():
    cases = (bench.Case("a", 1), bench.Case("b", 3))
    first, second = bench.WeightedCycle(cases, 123), bench.WeightedCycle(cases, 123)
    a, b = [next(first) for _ in range(400)], [next(second) for _ in range(400)]
    assert a == b
    assert Counter(a) == {0: 100, 1: 300}
    assert len(first.heap) == 2


def test_equal_weights_shuffle_each_complete_cycle():
    source = bench.WeightedCycle(tuple(bench.Case(str(i)) for i in range(7)), 123)
    cycles = [[next(source) for _ in range(7)] for _ in range(8)]
    assert all(sorted(cycle) == list(range(7)) for cycle in cycles)
    assert len({tuple(cycle) for cycle in cycles}) > 1


@pytest.mark.parametrize("url", [
    "http://example.com:8966", "http://127.0.0.1", "http://user:secret@127.0.0.1:8966",
    "http://127.0.0.1:8966/?secret=value", "http://127.0.0.1:8966/path",
    *[f"http://127.0.0.1:{port}" for port in sorted(bench.PROTECTED_PORTS)],
])
def test_live_or_nonisolated_endpoints_forbidden(url):
    config = bench.Config(url)
    with pytest.raises(bench.BenchmarkError, match="dedicated loopback"):
        config.validate()


@pytest.mark.parametrize("settings", [
    {"rps": float("nan")}, {"duration": float("inf")}, {"timeout": 0},
    {"rps": 1e308, "duration": 1e308}, {"concurrency": 0}, {"max_response_bytes": 1},
])
def test_numeric_settings_are_bounded(settings):
    config = bench.Config("http://127.0.0.1:8966", **settings)
    with pytest.raises(bench.BenchmarkError):
        config.validate()


@pytest.mark.parametrize("content,error", [
    (b'{"result":null}', "invalid_contract"), (b'{"result":false}', "invalid_contract"),
    (b'{"result":[]}', "invalid_contract"), (b'{"result":"a","extra":1}', "invalid_contract"),
    (b'{"result":"a","result":"b"}', "invalid_json"), (b'{"result":NaN}', "invalid_json"),
    (b'{"result":"\\ud800"}', "invalid_json"), (b'not-json-private-text', "invalid_json"),
])
def test_success_requires_exact_string_result_contract(content, error):
    result = bench.parse_response(content)
    assert result.error == error
    assert result.result is None
    assert "private" not in repr(result)


def test_empty_string_is_valid_and_thresholds_use_unrounded_values():
    assert bench.parse_response(b'{"result":""}').result == ""
    summary = bench.latency_summary([0, 500, 500.0001, 1000, 1000.0001])
    assert summary["count"] == 5
    assert summary["over_500ms"] == 3
    assert summary["over_1000ms"] == 1
    assert summary["p50"] == 500
    assert bench.latency_summary([])["max"] is None


class FakeClock:
    def __init__(self, monkeypatch):
        self.now = 0.0
        self.real_sleep = asyncio.sleep
        self.on_sleep = lambda: None
        monkeypatch.setattr(bench.time, "perf_counter", lambda: self.now)
        monkeypatch.setattr(bench.asyncio, "sleep", self.sleep)

    async def sleep(self, delay):
        self.now += delay
        self.on_sleep()
        # Run scheduled work and its completion callbacks without real waiting.
        await self.real_sleep(0)
        await self.real_sleep(0)


def tiny_config(**kwargs):
    return bench.Config("http://127.0.0.1:8966", rps=200, duration=0.03, **kwargs)


def test_roundtrip_empty_negative_and_unicode_unique_ids_no_report_leaks(monkeypatch):
    FakeClock(monkeypatch)
    payloads = ("", "Нейтральный текст без личных сведений.", "private-text: 😀 Ирина")
    corpus = bench.Corpus(tuple(bench.Case(text) for text in payloads), "a" * 64, 100)
    pairs, calls = {}, []

    async def request(payload, correlation):
        calls.append((payload, correlation))
        if correlation in pairs:
            return bench.Outcome(200, result=pairs[correlation])
        pairs[correlation] = payload
        return bench.Outcome(200, result="private-token" if payload == payloads[2] else payload)

    report = asyncio.run(bench.run_with_requester(tiny_config(), corpus, request))
    assert report["all_offered_pairs_verified"]
    assert report["counts"]["pairs_verified"] == 3
    assert report["counts"]["requests_successful"] == 6
    assert report["counts"]["pairs_mask_unchanged"] == 2
    assert report["counts"]["pairs_mask_changed"] == 1
    assert len(pairs) == 3
    assert set(Counter(correlation for _, correlation in calls).values()) == {2}
    assert report["latency_ms_all_attempts"]["count"] == 6
    rendered = json.dumps(report, ensure_ascii=False)
    assert "private-text" not in rendered
    assert "private-token" not in rendered
    assert not any(correlation in rendered for correlation in pairs)


@pytest.mark.parametrize("failure", ["mask", "unmask", "mismatch", "exception"])
def test_failed_pairs_are_not_retried_or_counted_verified(monkeypatch, failure):
    FakeClock(monkeypatch)
    corpus = bench.Corpus((bench.Case("private-text"),), "b" * 64, 30)
    seen, calls = set(), []

    async def request(payload, correlation):
        calls.append(correlation)
        if failure == "exception":
            raise RuntimeError("private-text from unsafe upstream exception")
        if correlation not in seen:
            seen.add(correlation)
            return bench.Outcome(503, error="http_503") if failure == "mask" else bench.Outcome(200, "token")
        if failure == "unmask":
            return bench.Outcome(200, error="invalid_contract")
        return bench.Outcome(200, "wrong-text")

    report = asyncio.run(bench.run_with_requester(tiny_config(), corpus, request))
    assert not report["all_offered_pairs_verified"]
    assert report["counts"]["pairs_verified"] == 0
    assert report["counts"]["pairs_dispatched"] == 3
    assert len(calls) == (3 if failure in {"mask", "exception"} else 6)
    assert report["latency_ms_all_attempts"]["count"] == len(calls)
    assert report["counts"]["http_requests_not_attempted"] == 6 - len(calls)
    assert "private-text" not in json.dumps(report)


def test_capacity_drops_counted_and_do_not_shift_offered_sampling(monkeypatch):
    clock = FakeClock(monkeypatch)
    corpus = bench.Corpus((bench.Case("x"), bench.Case("y", 3)), "c" * 64, 20)
    config = tiny_config(concurrency=1)

    async def run():
        gate = asyncio.Event()
        clock.on_sleep = lambda: gate.set() if clock.now >= config.duration else None

        async def request(payload, _correlation):
            await gate.wait()
            return bench.Outcome(200, payload)

        return await bench.run_with_requester(config, corpus, request)

    report = asyncio.run(run())
    source, expected_hash = bench.WeightedCycle(corpus.cases, config.seed), hashlib.sha256()
    for _ in range(config.offered_pairs):
        expected_hash.update(next(source).to_bytes(8, "big"))
    assert report["counts"]["pairs_dispatched"] == 1
    assert report["counts"]["pairs_dropped_client_capacity"] == 2
    assert report["counts"]["http_requests_not_attempted"] == 4
    assert report["sampling"]["offered_sequence_sha256"] == expected_hash.hexdigest()
    assert not report["all_offered_pairs_verified"]


def test_scheduler_deadline_drops_are_distinct_from_capacity(monkeypatch):
    clock = FakeClock(monkeypatch)
    config, counters = tiny_config(), Counter()
    offers = []

    def on_offer(index):
        offers.append(index)
        return index

    async def pair(_index, _assignment):
        clock.now = 10

    asyncio.run(bench.schedule(config, pair, counters, on_offer))
    assert offers == [0, 1, 2]
    assert counters["pairs_dispatched"] == 1
    assert counters["pairs_dropped_client_deadline"] == 2
    assert counters["pairs_dropped_client_capacity"] == 0


def test_completed_task_failure_is_retrieved_and_sanitized(monkeypatch):
    FakeClock(monkeypatch)

    async def pair(_index, _assignment):
        raise RuntimeError("private-text")

    config = tiny_config()
    scheduled = bench.schedule(config, pair, Counter(), lambda index: index)
    with pytest.raises(bench.BenchmarkError, match="Pair execution failed") as error:
        asyncio.run(scheduled)
    assert "private-text" not in str(error.value)
