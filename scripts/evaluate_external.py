#!/usr/bin/env python3
"""Evaluate a frozen Russian PII-Bench revision, without distributing the corpus.

Run in the isolated Python 3.13 Presidio environment, with pyarrow==25.0.1.
Downloads are checksum-verified and stored under ignored output/external-bench.
The report contains aggregates, identifiers and offsets, never example text.
No thresholds or rules are fitted, and this script never edits annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_presidio import build_presidio, percentile  # noqa: E402

DATASET = "hivetrace/pii-bench"
REVISION = "cd6a18ace16daf23e79247ccf1e2b245d4054654"
FILES = {
    "README.md": "e8f203645e78f712be38a58b795a23f41c18655b76ec67796478135f6bb60e34",
    "domain-00000-of-00001.parquet": "7ef3574273c0fe3a987981e38e32cd2764031a82766ca5b235bfccd1f27058eb",
    "entity-00000-of-00001.parquet": "6e0c77d566c2f04e7213917b26f1038336e28fc006551fac7a7f44f7627d5f96",
}
SEIF_MAP = {"PERSON": "NAME", "CARDHOLDER": "NAME", "PHONE": "PHONE_NUMBER", "EMAIL": "EMAIL", "CARD": "BANK_CARD_NUMBER",
            "ADDRESS": "ADDRESS", "CVV": "CVC", "INN": "INN", "PASSPORT": "PASSPORT_NUMBER"}
PRESIDIO_MAP = {"PERSON": "NAME", "PHONE_NUMBER": "PHONE_NUMBER", "EMAIL_ADDRESS": "EMAIL", "CREDIT_CARD": "BANK_CARD_NUMBER"}
COMMON = frozenset(PRESIDIO_MAP.values())
SUPPORTED = frozenset(SEIF_MAP.values())
ALL_TYPES = frozenset((*SUPPORTED, "KPP", "OGRN", "OGRNIP", "SNILS", "TOKEN"))
ADDRESS_COMPONENTS = frozenset(("ADDRESS", "CITY", "STREET", "HOUSE", "APARTMENT", "POSTAL_CODE", "COUNTRY"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download_and_read(folder):
    import pyarrow.parquet as pq
    import requests

    folder.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES.items():
        path = folder / filename
        source_path = filename if filename == "README.md" else "data/" + filename
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{source_path}"
        if not path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if digest(response.content) != expected:
                raise RuntimeError("Downloaded dataset checksum mismatch.")
            path.write_bytes(response.content)
        if digest(path.read_bytes()) != expected:
            raise RuntimeError("Cached dataset checksum mismatch.")
    corpora = {}
    for split in ("domain", "entity"):
        rows = pq.read_table(folder / f"{split}-00000-of-00001.parquet").to_pylist()
        if len(rows) != {"domain": 900, "entity": 910}[split] or len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError("Unexpected dataset size or duplicate IDs.")
        _validate_rows(rows)
        corpora[split] = rows
    return corpora


def _validate_rows(rows):
    for row in rows:
        for entity in row["entities"]:
            if (entity["type"] not in ALL_TYPES or not 0 <= entity["start"] < entity["end"] <= len(row["text"])
                    or row["text"][entity["start"]:entity["end"]] != entity["text"]):
                raise RuntimeError("Invalid original annotation; evaluation aborted without editing labels.")


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(2 * tp / (2 * tp + fp + fn), 6) if 2 * tp + fp + fn else 0.0,
            "f2": round(5 * tp / (5 * tp + fp + 4 * fn), 6) if 5 * tp + fp + 4 * fn else 0.0,
            "true_positive": tp, "false_positive": fp, "false_negative": fn}


def measure(truth, predicted, include_details=True):
    counts, types = Counter(), {}
    exact_cases = negative_cases = negative_fp = 0
    examples = []
    for case_id, gold in truth.items():
        actual = predicted[case_id]
        groups = {"tp": gold & actual, "fp": actual - gold, "fn": gold - actual}
        exact_cases += gold == actual
        negative_cases += not gold
        negative_fp += not gold and bool(actual)
        for name, spans in groups.items():
            counts[name] += len(spans)
            for kind, *_offsets in spans:
                types.setdefault(kind, Counter())[name] += 1
        if gold != actual and len(examples) < 5:
            examples.append({"id": case_id, "false_positive": sorted(groups["fp"]), "false_negative": sorted(groups["fn"])})
    result = {**scores(counts["tp"], counts["fp"], counts["fn"]), "cases": len(truth), "exact_cases": exact_cases,
              "negative_cases": negative_cases, "negative_cases_with_fp": negative_fp}
    if include_details:
        result["by_type"] = {kind: scores(c["tp"], c["fp"], c["fn"]) for kind, c in sorted(types.items())}
        result["first_five_error_offsets"] = examples
    return result


def typed_characters(rows):
    return {key: {(kind, offset) for kind, start, end in spans for offset in range(start, end)} for key, spans in rows.items()}


def map_predictions(rows, mapping):
    return {key: {(mapping[kind], start, end) for kind, start, end in spans if kind in mapping} for key, spans in rows.items()}


def subset_metric(truth, predictions, kinds):
    ids = [case_id for case_id, spans in truth.items() if all(kind in kinds for kind, *_ in spans)]
    selected_truth = {key: truth[key] for key in ids}
    systems = {}
    for name, values in predictions.items():
        selected = {key: {(kind, start, end) for kind, start, end in values[key] if kind in kinds} for key in ids}
        systems[name] = {
            "exact_span": measure(selected_truth, selected),
            "typed_character": measure(typed_characters(selected_truth), typed_characters(selected), include_details=False),
        }
    return {"selection": "Whole cases with gold labels contained in this type set, plus all empty-gold cases; selection never uses predictions",
            "types": sorted(kinds), "case_ids": ids, "systems": systems}


def paired_bootstrap(truth, predictions, by_domain, kinds):
    """Paired case resampling; fixed domain mix and no detector/model fitting."""
    import numpy as np

    rng = np.random.default_rng(20260922)
    draws = 2000
    sums = {system: np.zeros((draws, 3), dtype=np.int64) for system in predictions}
    for domain_ids in by_domain.values():
        ids = [key for key in domain_ids if all(kind in kinds for kind, *_ in truth[key])]
        if not ids:
            continue
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
        for system, values in predictions.items():
            counts = []
            for key in ids:
                actual = {span for span in values[key] if span[0] in kinds}
                gold = truth[key]
                counts.append((len(gold & actual), len(actual - gold), len(gold - actual)))
            sums[system] += weights @ np.array(counts, dtype=np.int64)
    f1 = {}
    for system, totals in sums.items():
        numerator = 2 * totals[:, 0]
        denominator = numerator + totals[:, 1] + totals[:, 2]
        f1[system] = np.divide(numerator, denominator, out=np.zeros(draws), where=denominator != 0)
    return {
        "method": "2000 paired bootstrap resamples, stratified by dataset domain; fixed seed20260922; percentile95 intervals",
        "assumption": "Cases are treated as independent within domain; synthetic-template correlations are not modelled, so intervals are conditional diagnostics",
        "f1_intervals": {system: [round(float(x), 6) for x in np.percentile(values, [2.5, 97.5])] for system, values in f1.items()},
        "f1_difference_vs_presidio": {system: {
            "interval95": [round(float(x), 6) for x in np.percentile(values - f1["presidio_ru"], [2.5, 97.5])],
            "bootstrap_fraction_greater_than_zero": round(float(np.mean(values > f1["presidio_ru"])), 6),
        } for system, values in f1.items() if system != "presidio_ru"},
    }


def dataset_metadata(corpora):
    return {
        "repository": DATASET, "revision": REVISION, "license_in_pinned_card": "Apache-2.0",
        "dataset_card": f"https://huggingface.co/datasets/{DATASET}/blob/{REVISION}/README.md",
        "official_repository": "https://github.com/HiveTrace/gliner-guard",
        "paper": "https://arxiv.org/abs/2605.05277",
        "file_sha256": FILES, "annotation_integrity": "Every original substring matches its Unicode character offsets; no edits",
        "split_description": {name: {"cases": len(rows), "negative_cases": sum(not row["entities"] for row in rows),
                                     "gold_type_counts": dict(sorted(Counter(item["type"] for row in rows for item in row["entities"]).items())),
                                     "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
                                     "offered_case_ids": [row["id"] for row in rows]} for name, rows in corpora.items()},
    }


def _infer_corpus(name, rows, analyzer, raw_file):
    from seif.detector import Span, detect, merge_person_candidates

    truth, by_domain = {}, {}
    predictions = {key: {} for key in ("seif_fast", "presidio_ru", "seif_hybrid")}
    samples = {key: [] for key in predictions}
    for row in rows:
        case_id, text = row["id"], row["text"]
        truth[case_id] = {(item["type"], item["start"], item["end"]) for item in row["entities"]}
        by_domain.setdefault(row["domain"], []).append(case_id)
        start = time.perf_counter_ns()
        base = detect(text)
        samples["seif_fast"].append((time.perf_counter_ns() - start) / 1e6)
        start = time.perf_counter_ns()
        upstream = analyzer.analyze(text=text, language="ru", score_threshold=0.0)
        samples["presidio_ru"].append((time.perf_counter_ns() - start) / 1e6)
        candidates = [Span(item.start, item.end, "PERSON", item.score, "ner-person")
                      for item in upstream if item.entity_type == "PERSON"]
        start = time.perf_counter_ns()
        merged = merge_person_candidates(text, base, candidates)
        samples["seif_hybrid"].append((time.perf_counter_ns() - start) / 1e6)
        predictions["seif_fast"][case_id] = {(item.type, item.start, item.end) for item in base}
        predictions["seif_hybrid"][case_id] = {(item.type, item.start, item.end) for item in merged}
        predictions["presidio_ru"][case_id] = {(item.entity_type, item.start, item.end) for item in upstream}
        raw_file.write(json.dumps({"split": name, "id": case_id, "expected": sorted(truth[case_id]),
                                   "predictions": {system: sorted(data[case_id]) for system, data in predictions.items()}}, ensure_ascii=False) + "\n")
    mapped = {system: map_predictions(values, PRESIDIO_MAP if system == "presidio_ru" else SEIF_MAP)
              for system, values in predictions.items()}
    common = subset_metric(truth, mapped, COMMON)
    supported = subset_metric(truth, mapped, SUPPORTED)
    address_gold = {key: {span for span in spans if span[0] == "ADDRESS"} for key, spans in truth.items()}
    address_proxy = {}
    for system, values in predictions.items():
        allowed = {"LOCATION"} if system == "presidio_ru" else ADDRESS_COMPONENTS
        proxy = {key: {("ADDRESS", start, end) for kind, start, end in spans if kind in allowed} for key, spans in values.items()}
        address_proxy[system] = {"raw_exact_span": measure(address_gold, proxy),
                                 "typed_character": measure(typed_characters(address_gold), typed_characters(proxy), include_details=False)}
    report = {
        "common4_primary": common, "supported8_requirement_coverage": supported,
        "common4_uncertainty": paired_bootstrap(truth, mapped, by_domain, COMMON),
        "all13_coverage_diagnostic": {system: measure(truth, values) for system, values in mapped.items()},
        "name_all_cases": {system: measure({key: {span for span in spans if span[0] == "NAME"} for key, spans in truth.items()},
                                           {key: {span for span in spans if span[0] == "NAME"} for key, spans in values.items()})
                           for system, values in mapped.items()},
        "unsupported_by_assignment": sorted(ALL_TYPES - SUPPORTED),
        "address_location_component_proxy": address_proxy,
        "per_domain_common4": {domain: subset_metric({key: truth[key] for key in ids},
                                  {system: {key: values[key] for key in ids} for system, values in mapped.items()}, COMMON)
                               for domain, ids in sorted(by_domain.items())},
    }
    latency = {system: {"samples": len(values), "mean_ms": round(sum(values) / len(values), 6),
                              "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95), "p99_ms": percentile(values, 99)}
                     for system, values in samples.items()}
    return report, latency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "output/external-bench")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/pii-bench-comparison.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--repeat", action="store_true", help="Allow a reproducibility/development run after first inference")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    corpora = download_and_read(args.data_dir)
    metadata = dataset_metadata(corpora)
    protocol = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": metadata,
        "seif_type_mapping": SEIF_MAP, "presidio_strict_mapping": PRESIDIO_MAP,
        "primary": "common4, exact spans, whole-case subset including negatives; domain split is primary",
        "secondary": "supported8 requirement coverage; all13 support gaps; NAME and ADDRESS separately; typed-character diagnostics",
        "uncertainty": "2000 paired bootstrap resamples of common4 case counts, stratified by dataset domain, seed20260922; 95 percentile F1 intervals",
        "address_policy": "No post-result aggregation or boundary editing; strict ADDRESS and separate LOCATION/component proxy diagnostics",
        "presidio_configuration_source": "Unchanged build_presidio from original synthetic comparison; RU small model, threshold0, generic recognizers",
    }
    protocol_path = args.data_dir / "prepared-protocol.json"
    if not protocol_path.exists() or (args.prepare_only and not (args.data_dir / "first-inference.json").exists()):
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stored_protocol = json.loads(protocol_path.read_text())
    for key in ("seif_type_mapping", "presidio_strict_mapping", "primary", "secondary", "address_policy", "uncertainty"):
        if stored_protocol[key] != protocol[key]:
            raise RuntimeError("Protocol differs from the prepared version; do not silently change evaluation policy.")
    if args.prepare_only:
        print(json.dumps({"prepared": True, "revision": REVISION, "sha256": FILES,
                          "case_counts": {name: len(rows) for name, rows in corpora.items()}}))
        return
    marker = args.data_dir / "first-inference.json"
    if (marker.exists() or args.output.exists()) and not args.repeat:
        parser.error("A prior evaluation exists; use --repeat and report it as reproduction/development, never a new blind result.")
    detector_hash = digest((ROOT / "seif/detector.py").read_bytes())
    first_run = not marker.exists()
    if first_run:
        with marker.open("x", encoding="utf-8") as file:
            json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "detector_sha256": detector_hash,
                       "protocol_sha256": digest(protocol_path.read_bytes())}, file)
    analyzer, configuration = build_presidio()
    reports, latency = {}, {}
    predictions_path = args.data_dir / ("predictions-first.jsonl" if first_run else "predictions-repeat.jsonl")
    with predictions_path.open("w", encoding="utf-8") as raw_file:
        for name, rows in corpora.items():
            reports[name], latency[name] = _infer_corpus(name, rows, analyzer, raw_file)
    if detector_hash != digest((ROOT / "seif/detector.py").read_bytes()):
        raise RuntimeError("Detector changed during inference; report not published.")
    report = {
        "schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "first frozen-implementation evaluation in this task" if first_run else "repeated evaluation; reproduction or post-result development",
        "dataset": metadata, "protocol_sha256": digest(protocol_path.read_bytes()),
        "source_hashes": {"detector": detector_hash, "evaluator": digest(Path(__file__).read_bytes()),
                          "presidio_configuration_script": digest((ROOT / "scripts/compare_presidio.py").read_bytes())},
        "type_mapping": {"seif_strict": SEIF_MAP, "presidio_strict": PRESIDIO_MAP,
                         "address_proxy": "LOCATION and individual address components map to ADDRESS only in explicitly labelled diagnostics; boundaries stay unchanged"},
        "presidio_configuration": configuration, "splits": reports,
        "latency_observation": {"scope": "one sequential pass, same ordinary Python; not an HTTP benchmark; seif_hybrid timings are MERGE ONLY using cached NER outputs, never complete hybrid latency",
                                "results": latency},
        "raw_prediction_sha256": digest(predictions_path.read_bytes()),
        "environment": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))},
        "limitations": [
            "Externally authored but synthetic Russian examples; not real banking traffic and not a complete CIS benchmark.",
            "No threshold, model, regex, annotation, span aggregation or exception was tuned on these examples before the first evaluation.",
            "Common4 is the like-for-like primary comparison; broader coverage scores include capabilities missing from stock Presidio and must not be advertised as universal NER superiority.",
            "Five dataset types KPP/OGRN/OGRNIP/SNILS/TOKEN are outside the assignment and are reported as unsupported, not hidden.",
            "Original dataset labels remain unchanged even when public organizations or locations conflict with the assignment's private-PII policy.",
            "ADDRESS gold is consolidated; raw component/LOCATION spans can fail exact matching despite partial coverage. Typed-character diagnostics expose this; no post-result aggregation is applied.",
            "Hybrid reuses the identical Presidio PERSON outputs measured on each example plus our existing policy/rules; this is a detector quality ablation, not service availability or latency validation.",
            "Any later changes motivated by these results turn this corpus into development evidence and require a new unseen test set for further generalization claims.",
            "Pinned dataset card specifies Apache-2.0 and requests evaluation-only hygiene, explicitly not a license restriction; the GitHub README has older inconsistent wording. We only evaluate and do not redistribute examples.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: {system: result["exact_span"] | {"by_type": None, "first_five_error_offsets": None}
                             for system, result in values["common4_primary"]["systems"].items()}
                      for split, values in reports.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
