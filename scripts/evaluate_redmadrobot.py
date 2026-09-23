#!/usr/bin/env python3
"""Pinned independent Russian BIO benchmark, evaluated without training/tuning.

Run --prepare-only before inference to freeze token alignment exclusions and
the shared coarse mapping. Corpus bytes stay in ignored output/. Original
sentences are never printed, changed or copied into the aggregate report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_presidio import build_presidio  # noqa: E402
from scripts.evaluate_external import measure, paired_bootstrap, typed_characters  # noqa: E402

REPOSITORY = "redmadrobot-rnd/pii_benchmark"
REVISION = "f77ea831274daf980cc45c61a93c226be9d978d6"
FILES = {
    "test.csv": ("redmadrobot-test.csv", "6bf544a380a3ee5bec94b946124bea3afaecce49e734679ad0f0c0e7c12977bb"),
    "README.md": ("redmadrobot-README.md", "a346c4c012bff1cce330d15ef6271db32a8cf83696a9bd7237f180e45c23ac50"),
}
NAME_TYPES = frozenset(("FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"))
LOCATION_TYPES = frozenset(("COUNTRY", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE"))
GOLD_MAP = {
    **dict.fromkeys(NAME_TYPES, "PERSON"),
    **dict.fromkeys(LOCATION_TYPES, "LOCATION"),
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "CREDIT_CARD": "CARD",
    "PASSPORT": "PASSPORT",
    "INN": "INN",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
}
SEIF_MAP = {
    "PERSON": "PERSON",
    "CARDHOLDER": "PERSON",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "CARD": "CARD",
    "PASSPORT": "PASSPORT",
    "INN": "INN",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    **dict.fromkeys(
        ("ADDRESS", "COUNTRY", "CITY", "STREET", "HOUSE", "APARTMENT", "POSTAL_CODE", "BIRTH_PLACE"), "LOCATION"
    ),
}
PRESIDIO_MAP = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
}
COMMON = frozenset(("PERSON", "LOCATION", "EMAIL", "PHONE", "CARD"))
STRUCTURED = frozenset(("PASSPORT", "INN", "DRIVER_LICENSE"))
UNMAPPED = frozenset(("URL", "IP_ADDRESS", "SNILS", "OMS", "MILITARY_ID", "BIRTH_CERTIFICATE"))
ALL_FINE = frozenset(GOLD_MAP) | UNMAPPED


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(folder):
    import requests

    folder.mkdir(parents=True, exist_ok=True)
    for remote, (filename, checksum) in FILES.items():
        path = folder / filename
        if not path.exists():
            response = requests.get(
                f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{remote}", timeout=60
            )
            response.raise_for_status()
            if sha(response.content) != checksum:
                raise RuntimeError("Dataset download checksum mismatch.")
            path.write_bytes(response.content)
        if sha(path.read_bytes()) != checksum:
            raise RuntimeError("Cached dataset checksum mismatch.")


def parse_bio(row):
    """Exact, monotonic token alignment, no quote/case/whitespace rewriting."""
    try:
        tokens, labels = json.loads(row["tokens"]), json.loads(row["ner_tags"])
    except (ValueError, KeyError):
        return None, "invalid_token_json"
    if not isinstance(tokens, list) or not isinstance(labels, list) or len(tokens) != len(labels):
        return None, "token_label_length_mismatch"
    return _align_bio(row["text"], tokens, labels)


def _append_bio(spans, label, previous, offsets):
    start, end = offsets
    if label.startswith("B-") and label[2:] in ALL_FINE:
        spans.append((label[2:], start, end))
    elif label.startswith("I-") and label[2:] in ALL_FINE and previous in {"B-" + label[2:], label}:
        kind, begin, _ = spans[-1]
        spans[-1] = (kind, begin, end)
    elif label != "O":
        return False
    return True


def _align_bio(text, tokens, labels):
    cursor, previous, spans = 0, "O", []
    for token, label in zip(tokens, labels, strict=True):
        # An empty O token marks no characters/entities; preserve its BIO
        # boundary without changing the sentence or inventing a text offset.
        if token == "" and label == "O":
            previous = "O"
            continue
        if not isinstance(token, str) or not token or not isinstance(label, str):
            return None, "invalid_token_or_label"
        start = text.find(token, cursor)
        if start < 0 or text[cursor:start].strip():
            return None, "token_not_exactly_alignable"
        end = start + len(token)
        if not _append_bio(spans, label, previous, (start, end)):
            return None, "invalid_bio_transition"
        previous, cursor = label, end
    if text[cursor:].strip():
        return None, "unaligned_text_suffix"
    return set(spans), None


def read_rows(folder):
    with (folder / "redmadrobot-test.csv").open(newline="", encoding="utf-8-sig") as file:
        original = list(csv.DictReader(file))
    if len(original) != 2841:
        raise RuntimeError("Unexpected source row count.")
    valid, excluded = [], []
    for index, row in enumerate(original):
        row_id = f"row_{index:04d}"
        spans, reason = parse_bio(row)
        if reason:
            excluded.append({"id": row_id, "source_row_zero_based": index, "reason": reason})
        else:
            valid.append({"id": row_id, "text": row["text"], "fine_gold": spans})
    return valid, excluded


def coarsen(spans, mapping, keep_unknown=False):
    return {
        (mapping.get(kind, "UNMAPPED:" + kind), start, end)
        for kind, start, end in spans
        if keep_unknown or kind in mapping
    }


def merge_adjacent(text, spans):
    """Same category + overlap or whitespace-only gap, identical for all sides."""
    merged = []
    for kind in sorted({span[0] for span in spans}):
        merged.extend(_merge_kind(text, spans, kind))
    return set(merged)


def _merge_kind(text, spans, kind):
    merged = []
    current = None
    for _, start, end in sorted(span for span in spans if span[0] == kind):
        if current is not None and (start <= current[2] or not text[current[2] : start].strip()):
            current = (kind, current[1], max(end, current[2]))
        else:
            if current is not None:
                merged.append(current)
            current = (kind, start, end)
    if current is not None:
        merged.append(current)
    return merged


def score_scope(truth, predictions, allowed, whole_cases=True):
    ids = [key for key, gold in truth.items() if not whole_cases or all(kind in allowed for kind, *_ in gold)]
    gold = {key: {span for span in truth[key] if span[0] in allowed} for key in ids}
    results = {}
    for system, values in predictions.items():
        pred = {key: {span for span in values[key] if span[0] in allowed} for key in ids}
        results[system] = {
            "typed_character_primary": measure(typed_characters(gold), typed_characters(pred)),
            "merged_exact_span_secondary": measure(gold, pred),
        }
        # Character-level error lists can contain thousands of individual points.
        results[system]["typed_character_primary"].pop("first_five_error_offsets", None)
    return {
        "case_ids": ids,
        "types": sorted(allowed),
        "systems": results,
        "selection": "whole cases: every gold type is allowed; all gold-empty rows retained"
        if whole_cases
        else "all aligned cases, filter only types (diagnostic)",
    }


def _infer_rows(rows, analyzer, original_outputs):
    from seif.detector import Span, detect, merge_person_candidates

    truth, raw_truth = {}, {}
    predictions = {key: {} for key in ("seif_fast", "presidio_ru", "seif_hybrid")}
    raw_coarse = {key: {} for key in predictions}
    with original_outputs.open("w", encoding="utf-8") as stream:
        for row in rows:
            key, text = row["id"], row["text"]
            base = detect(text)
            upstream = analyzer.analyze(text=text, language="ru", score_threshold=0.0)
            candidates = [
                Span(item.start, item.end, "PERSON", item.score, "ner-person")
                for item in upstream
                if item.entity_type == "PERSON"
            ]
            hybrid = merge_person_candidates(text, base, candidates)
            original = {
                "seif_fast": {(item.type, item.start, item.end) for item in base},
                "presidio_ru": {(item.entity_type, item.start, item.end) for item in upstream},
                "seif_hybrid": {(item.type, item.start, item.end) for item in hybrid},
            }
            raw_truth[key] = coarsen(row["fine_gold"], GOLD_MAP, keep_unknown=True)
            truth[key] = merge_adjacent(text, raw_truth[key])
            for system, spans in original.items():
                raw_coarse[system][key] = coarsen(spans, PRESIDIO_MAP if system == "presidio_ru" else SEIF_MAP)
                predictions[system][key] = merge_adjacent(text, raw_coarse[system][key])
            stream.write(
                json.dumps(
                    {
                        "id": key,
                        "fine_gold": sorted(row["fine_gold"]),
                        "raw_coarse_gold": sorted(raw_truth[key]),
                        "merged_gold": sorted(truth[key]),
                        "original_predictions": {name: sorted(spans) for name, spans in original.items()},
                        "merged_predictions": {name: sorted(values[key]) for name, values in predictions.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return truth, raw_truth, predictions, raw_coarse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "output/external-bench-next")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/redmadrobot-comparison.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    fetch(args.data_dir)
    rows, excluded = read_rows(args.data_dir)
    metadata = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "license_declared_in_pinned_card": "MIT",
        "card_url": f"https://huggingface.co/datasets/{REPOSITORY}/blob/{REVISION}/README.md",
        "original_split": "test.csv",
        "declared_language": "ru",
        "offered_rows": 2841,
        "aligned_rows": len(rows),
        "excluded_before_inference": excluded,
        "file_sha256": {key: value[1] for key, value in FILES.items()},
        "offered_row_ids": [f"row_{index:04d}" for index in range(2841)],
        "aligned_gold_type_counts": dict(
            sorted(Counter(kind for row in rows for kind, *_ in row["fine_gold"]).items())
        ),
        "original_text_policy": "Original text unchanged; exact monotonic token offsets, whitespace-only gaps; an empty O token marks zero characters and preserves the BIO boundary; invalid rows excluded before detector predictions",
    }
    protocol = {
        "dataset": metadata,
        "mapping": {"gold": GOLD_MAP, "seif": SEIF_MAP, "presidio": PRESIDIO_MAP},
        "common_scope": sorted(COMMON),
        "primary": "typed-character precision/recall/F1 on whole cases in common5 plus all negatives",
        "secondary": "exact spans after identical merge; PERSON/LOCATION on all aligned cases; structured requirement coverage separately",
        "merge_rule": "Same coarse category; merge overlapping spans or spans separated only by Unicode whitespace, on BOTH gold and EVERY system; no punctuation merging",
        "uncertainty": "2000 paired case bootstrap resamples, seed20260922; percentile95 differences vs fixed PresidioRU; character and exact units separately",
        "pdf_overlap": {
            "13_fine_labels_with_direct_counterpart": sorted(
                NAME_TYPES
                | {
                    "COUNTRY",
                    "CITY",
                    "STREET",
                    "HOUSE",
                    "EMAIL",
                    "PHONE",
                    "PASSPORT",
                    "INN",
                    "CREDIT_CARD",
                    "DRIVER_LICENSE",
                }
            ),
            "2_address_components_only_coarse": ["REGION", "DISTRICT"],
            "6_without_unique_assignment_counterpart": sorted(UNMAPPED),
            "missing_distinctions": [
                "birth date/purpose",
                "birth place/purpose",
                "passport issuer",
                "passport issue date",
                "department code",
                "citizenship",
                "postal code",
                "apartment",
                "CVV",
                "PIN",
                "cardholder/purpose",
            ],
        },
        "policy_caveat": "Coarse LOCATION is a spatial NER category, not proof that the full personal ADDRESS or public/private exception was identified",
    }
    protocol_path = args.data_dir / "redmadrobot-protocol.json"
    marker = args.data_dir / "redmadrobot-first-inference.json"
    encoded_protocol = json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if not protocol_path.exists() or (args.prepare_only and not marker.exists()):
        protocol_path.write_text(encoded_protocol, encoding="utf-8")
    if protocol_path.read_text() != encoded_protocol:
        raise RuntimeError("Prepared protocol differs; evaluation mapping/exclusions cannot change silently.")
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared": True,
                    "rows": len(rows),
                    "excluded": excluded,
                    "protocol_sha256": sha(protocol_path.read_bytes()),
                    "pdf_overlap": protocol["pdf_overlap"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if (marker.exists() or args.output.exists()) and not args.repeat:
        parser.error("Prior inference/report exists; repeated runs require --repeat and are not a fresh holdout.")
    source_hash = sha((ROOT / "seif/detector.py").read_bytes())
    first = not marker.exists()
    if first:
        with marker.open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "detector_sha256": source_hash,
                    "protocol_sha256": sha(protocol_path.read_bytes()),
                },
                file,
            )
    analyzer, configuration = build_presidio()
    original_outputs = args.data_dir / (
        "redmadrobot-predictions-first.jsonl" if first else "redmadrobot-predictions-repeat.jsonl"
    )
    truth, raw_truth, predictions, raw_coarse = _infer_rows(rows, analyzer, original_outputs)
    if sha((ROOT / "seif/detector.py").read_bytes()) != source_hash:
        raise RuntimeError("Detector changed during inference; report not published.")
    primary = score_scope(truth, predictions, COMMON)
    ids = primary["case_ids"]
    selected_truth = {key: truth[key] for key in ids}
    selected_predictions = {
        system: {key: {span for span in values[key] if span[0] in COMMON} for key in ids}
        for system, values in predictions.items()
    }
    report = {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "first frozen-implementation evaluation on this previously unused corpus in this task"
        if first
        else "repeat: reproduction or post-result development",
        "dataset": metadata,
        "protocol": protocol,
        "protocol_sha256": sha(protocol_path.read_bytes()),
        "source_hashes": {
            "detector": source_hash,
            "evaluator": sha(Path(__file__).read_bytes()),
            "presidio_configuration_script": sha((ROOT / "scripts/compare_presidio.py").read_bytes()),
        },
        "presidio_configuration": configuration,
        "common5": primary,
        "common5_character_uncertainty": paired_bootstrap(
            typed_characters(selected_truth),
            {system: typed_characters(values) for system, values in selected_predictions.items()},
            {"all": ids},
            COMMON,
        ),
        "common5_exact_uncertainty": paired_bootstrap(selected_truth, selected_predictions, {"all": ids}, COMMON),
        "person_all_cases": score_scope(truth, predictions, {"PERSON"}, whole_cases=False),
        "location_all_cases": score_scope(truth, predictions, {"LOCATION"}, whole_cases=False),
        "structured_document_coverage": score_scope(truth, predictions, STRUCTURED, whole_cases=False),
        "all_mapped8_requirement_coverage": score_scope(truth, predictions, COMMON | STRUCTURED),
        "raw_unmerged_common5_secondary": score_scope(raw_truth, raw_coarse, COMMON),
        "raw_prediction_sha256": sha(original_outputs.read_bytes()),
        "environment": {
            "python": sys.version,
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        },
        "limitations": [
            "Russian corpus only; this result does not establish Ukrainian, Kazakh or general CIS language quality.",
            "Source has pseudonymized production examples, synthetic documents and hard negatives; no representative banking-traffic guarantee.",
            "Rows failing the strict token/BIO validator are excluded before inference, with fixed IDs/reasons; all systems see the identical remaining rows.",
            "Published leaderboard uses overlap and tuned thresholds; its numbers are not comparable to this fixed-threshold typed-character/merged-exact evaluation.",
            "Names and address components use an announced shared coarse mapping; LOCATION quality is not full-address exact extraction or private/public policy accuracy.",
            "Original gold labels are untouched, including cases that may conflict with the assignment's public-information exceptions.",
            "Broader Russian-document coverage and extra types are separate from the common5 comparison; unsupported categories are not silently omitted from provenance.",
            "Hybrid uses the same fixed Russian small-model PERSON outputs plus the frozen local merge policy; no Ukrainian model or stronger Presidio tuning was evaluated.",
            "Any improvements based on these results require a new holdout for independent claims; the first report remains unchanged.",
            "Corpora and raw source sentences remain outside the source archive. No private input was uploaded to a service.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                system: {
                    metric: {
                        key: value for key, value in data.items() if key not in ("by_type", "first_five_error_offsets")
                    }
                    for metric, data in results.items()
                }
                for system, results in primary["systems"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
