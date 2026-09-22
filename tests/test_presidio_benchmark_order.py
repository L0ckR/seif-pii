"""Benchmark scheduling is reproducible, complete and outside timed calls."""

from collections import Counter
from types import SimpleNamespace

from scripts import compare_presidio


def test_ranked_order_version_one_has_a_stable_schedule():
    items = list(range(8))
    ranked = compare_presidio._ranked_order(items, 0, domain="rows")
    assert ranked == [5, 1, 4, 2, 0, 6, 7, 3]
    assert items == list(range(8))
    assert ranked == compare_presidio._ranked_order(items, 0, domain="rows")


def test_ranked_order_preserves_occurrences_and_separates_passes_domains_and_seeds():
    items = list(range(64)) + [3, 3, 7]
    baseline = compare_presidio._ranked_order(items, 0, domain="rows")
    variants = [
        compare_presidio._ranked_order(items, 1, domain="rows"),
        compare_presidio._ranked_order(items, 0, domain="systems:0"),
        compare_presidio._ranked_order(items, 0, domain="rows", seed=20260923),
    ]
    assert Counter(baseline) == Counter(items)
    for variant in variants:
        assert variant != baseline
        assert Counter(variant) == Counter(items)


def test_ranked_order_digest_ties_retain_input_positions_without_hashing_payloads(monkeypatch):
    items = [{"payload": object()}, {"payload": object()}]
    monkeypatch.setattr(compare_presidio.hashlib, "sha256", lambda _: SimpleNamespace(digest=lambda: bytes(32)))
    ranked = compare_presidio._ranked_order(items, 1, domain="rows")
    assert ranked == items
    assert ranked[0] is items[0]
    assert ranked[1] is items[1]


def run_stub_benchmark(monkeypatch):
    events = []
    clock = SimpleNamespace(now=0)

    def tick():
        clock.now += 1_000_000
        events.append(("clock", clock.now))
        return clock.now

    def detect(text):
        events.append(("call", "seif", text))

    def analyze(*, text, language, score_threshold):
        assert language == "ru"
        assert score_threshold == 0.0
        events.append(("call", "presidio", text))

    monkeypatch.setattr(compare_presidio.time, "perf_counter_ns", tick)
    corpora = {"alpha": {"a0": "alpha zero", "a1": "alpha one"},
               "beta": {"b0": "beta zero", "b1": "beta one"}}
    result = compare_presidio.benchmark(corpora, detect, SimpleNamespace(analyze=analyze), repeats=3, warmup=2)
    return corpora, result, events


def test_benchmark_warmup_and_every_timed_pass_visit_all_cases_on_both_systems(monkeypatch):
    corpora, result, events = run_stub_benchmark(monkeypatch)
    texts = [text for corpus in corpora.values() for text in corpus.values()]
    warmup = [("call", system, text) for _ in range(2) for text in texts for system in ("seif", "presidio")]
    assert events[:len(warmup)] == warmup
    measured = events[len(warmup):]
    assert len(measured) == 3 * len(texts) * 2 * 3
    row_texts = []
    first_systems = set()
    for start in range(0, len(measured), 6):
        before_first, first, after_first, before_second, second, after_second = measured[start:start + 6]
        assert before_first[0] == "clock"
        assert after_first[0] == "clock"
        assert before_second[0] == "clock"
        assert after_second[0] == "clock"
        assert before_first[1] < after_first[1] < before_second[1] < after_second[1]
        assert first[0] == "call"
        assert second[0] == "call"
        assert first[2] == second[2]
        assert {first[1], second[1]} == {"seif", "presidio"}
        row_texts.append(first[2])
        first_systems.add(first[1])
    for start in range(0, len(row_texts), len(texts)):
        assert Counter(row_texts[start:start + len(texts)]) == Counter(texts)
    assert first_systems == {"seif", "presidio"}
    for systems in result["corpora"].values():
        for stats in systems.values():
            assert stats["samples"] == 6
            assert stats["mean_ms"] == 1.0
    for systems in result["raw_latency_ms"].values():
        for samples in systems.values():
            assert samples == [1.0] * 6


def test_benchmark_repeats_schedule_and_reports_protocol_change(monkeypatch):
    _, first, first_events = run_stub_benchmark(monkeypatch)
    _, second, second_events = run_stub_benchmark(monkeypatch)
    assert first == second
    assert first_events == second_events
    assert first["order_protocol"]["algorithm"] == "sha256-rank-v1"
    assert first["order_protocol"]["seed"] == 20260922
    assert "differs" in first["order_protocol"]["historical_compatibility"]
