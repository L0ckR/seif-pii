#!/usr/bin/env python3
"""Final frozen mixed RU/UK Scanpatch test; no inferred language labels.

Uses the same fixed Russian Presidio baseline. This is not an optimal Ukrainian
baseline, and does not prove Ukrainian-language or complete CIS coverage.
"""
from __future__ import annotations

import argparse
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
from scripts.evaluate_external import paired_bootstrap, typed_characters  # noqa: E402
from scripts.evaluate_redmadrobot import (  # noqa: E402
    COMMON,
    PRESIDIO_MAP,
    SEIF_MAP,
    coarsen,
    merge_adjacent,
    score_scope,
)

REPOSITORY = "scanpatch/pii-ner-corpus-synthetic-controlled"
REVISION = "6af0587dde068d90d8a1080285159c374b46a09b"
FILES = {
    "data/test-00000-of-00001.parquet": ("scanpatch-test.parquet", "5d44f419692282285ae811bd54b69e57bb44121e31f3628408e203f9416371ba"),
    "README.md": ("scanpatch-README.md", "6a32382e9dbc27e1a0553f5d1dbf8bf57847f81ca3814d98c51b8403c8924d18"),
}
PERSON_LABELS = frozenset(("name", "first_name", "middle_name", "last_name", "name_initials"))
ADDRESS_LABELS = frozenset(("address", "address_country", "address_region", "address_city", "address_district",
                           "address_street", "address_house", "address_building", "address_apartment", "address_postal_code", "address_geolocation"))
GOLD_MAP = {**dict.fromkeys(PERSON_LABELS, "PERSON"), **dict.fromkeys(ADDRESS_LABELS, "LOCATION"),
            "email": "EMAIL", "mobile_phone": "PHONE", "tin": "INN", "document_number": "DOCUMENT"}
LOCAL_MAP = {**SEIF_MAP, "LOCATION": "LOCATION", "PASSPORT": "DOCUMENT", "DRIVER_LICENSE": "DOCUMENT", "FOREIGN_DOCUMENT": "DOCUMENT"}
OTHER_LABELS = frozenset(("nickname", "ip", "snils", "vehicle_number", "military_individual_number", "organization", "date"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_rows(folder):
    import pyarrow.parquet as pq
    import requests

    folder.mkdir(parents=True, exist_ok=True)
    for remote, (filename, checksum) in FILES.items():
        path = folder / filename
        if not path.exists():
            response = requests.get(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{remote}", timeout=60)
            response.raise_for_status()
            if sha(response.content) != checksum:
                raise RuntimeError("Downloaded dataset checksum mismatch.")
            path.write_bytes(response.content)
        if sha(path.read_bytes()) != checksum:
            raise RuntimeError("Cached dataset checksum mismatch.")
    rows = pq.read_table(folder / "scanpatch-test.parquet").to_pylist()
    if len(rows) != 532:
        raise RuntimeError("Unexpected test size.")
    for index, row in enumerate(rows):
        row["id"] = f"test_{index:04d}"
        fields = [row[key] for key in ("entity_starts", "entity_ends", "entity_labels", "entity_texts")]
        if len({len(values) for values in fields}) != 1:
            raise RuntimeError("Annotation lengths differ; do not repair using predictions.")
        for start, end, kind, value in zip(*fields):
            if (kind not in set(GOLD_MAP) | OTHER_LABELS or not 0 <= start < end <= len(row["text"])
                    or row["text"][start:end] != value):
                raise RuntimeError("Original span validation failed; no report published.")
        row["fine_gold"] = set(zip(row["entity_labels"], row["entity_starts"], row["entity_ends"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "output/external-bench-next")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/scanpatch-comparison.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    rows = load_rows(args.data_dir)
    truth = {row["id"]: merge_adjacent(row["text"], coarsen(row["fine_gold"], GOLD_MAP, keep_unknown=True)) for row in rows}
    ids = [key for key, spans in truth.items() if all(kind in COMMON for kind, *_ in spans)]
    metadata = {
        "repository": REPOSITORY, "revision": REVISION, "license_in_pinned_card": "MIT",
        "card_url": f"https://huggingface.co/datasets/{REPOSITORY}/blob/{REVISION}/README.md",
        "original_splits": {"train": 4786, "test": 532}, "evaluated_split": "entire test, no train download",
        "language_scope": "Mixed Russian/Ukrainian; source has no row-level language field; no fabricated per-language results",
        "offered_rows": len(rows), "gold_empty_rows": sum(not row["fine_gold"] for row in rows),
        "common_whole_cases": len(ids), "common_case_ids": ids,
        "offered_case_ids": [row["id"] for row in rows],
        "file_sha256": {remote: checksum for remote, (_, checksum) in FILES.items()},
        "fine_label_counts": dict(sorted(Counter(kind for row in rows for kind, *_ in row["fine_gold"]).items())),
        "annotation_validation": "All532 rows passed exact original substring/offset checks; no repairs/exclusions",
    }
    protocol = {
        "dataset": metadata, "mapping": {"gold": GOLD_MAP, "seif": LOCAL_MAP, "presidio": PRESIDIO_MAP},
        "primary": "Whole cases containing only PERSON/LOCATION/EMAIL/PHONE/CARD after coarse mapping, plus all empty-gold cases; typed-character P/R/F1",
        "card_coverage": "CARD is absent in this corpus; common5 schema has only four positive type categories, not a five-type validation",
        "merge": "Same coarse category, overlapping spans or whitespace-only gaps; identical gold/prediction transform, including nested name/address annotations; no punctuation normalization",
        "secondary": "Merged exact spans; raw unmerged diagnostics; PERSON/LOCATION over all532; generic document/tax ID coverage separately",
        "uncertainty": "2000 paired case bootstrap resamples, fixed seed20260922, one group because no language/domain gold split exists",
        "model": "Unchanged Presidio2.2.364 + ru_core_news_sm3.8.0, threshold0; fixed Russian configuration, not an optimized Ukrainian baseline",
        "hybrid": "Rules plus identical Presidio PERSON and LOCATION candidates through the frozen merge_ner_candidates policy",
        "pdf_overlap": {
            "names": sorted(PERSON_LABELS), "contacts": ["email", "mobile_phone"], "tax_identifier": ["tin"],
            "7_address_labels_with_direct_counterpart": ["address", "address_country", "address_city", "address_street", "address_house", "address_apartment", "address_postal_code"],
            "address_components_only_coarse": ["address_region", "address_district", "address_building"],
            "geolocation_not_explicit_in_assignment": ["address_geolocation"],
            "document_number_is_ambiguous": "Cannot distinguish passport, driving licence, residence document or issue authority from this gold type alone",
            "no_direct_mapping": sorted(OTHER_LABELS),
            "not_tested": ["CARD", "CVV", "PIN", "passport issue date/purpose", "passport issuer", "department code", "citizenship", "birth date/purpose", "cardholder/purpose"],
        },
    }
    protocol_file = args.data_dir / "scanpatch-protocol.json"
    marker = args.data_dir / "scanpatch-first-inference.json"
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if not protocol_file.exists() or (args.prepare_only and not marker.exists()):
        protocol_file.write_text(encoded, encoding="utf-8")
    if protocol_file.read_text() != encoded:
        raise RuntimeError("Frozen protocol changed; do not silently change mapping or case selection.")
    if args.prepare_only:
        print(json.dumps({"prepared": True, "cases": len(rows), "common_cases": len(ids), "negatives": metadata["gold_empty_rows"],
                          "protocol_sha256": sha(protocol_file.read_bytes())}))
        return
    if (marker.exists() or args.output.exists()) and not args.repeat:
        parser.error("Prior inference/report exists; use --repeat only for explicitly labelled subsequent runs.")
    from seif.detector import Span, detect, merge_ner_candidates

    source_hash = sha((ROOT / "seif/detector.py").read_bytes())
    first = not marker.exists()
    if first:
        with marker.open("x", encoding="utf-8") as file:
            json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "detector_sha256": source_hash,
                       "protocol_sha256": sha(protocol_file.read_bytes())}, file)
    analyzer, configuration = build_presidio()
    predictions = {key: {} for key in ("seif_fast", "presidio_ru", "seif_hybrid")}
    unmerged, raw_truth = {key: {} for key in predictions}, {}
    raw_path = args.data_dir / ("scanpatch-predictions-first.jsonl" if first else "scanpatch-predictions-repeat.jsonl")
    with raw_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            key, text = row["id"], row["text"]
            base = detect(text)
            upstream = analyzer.analyze(text=text, language="ru", score_threshold=0.0)
            candidates = [Span(item.start, item.end, item.entity_type, item.score, "ner")
                          for item in upstream if item.entity_type in {"PERSON", "LOCATION"}]
            merged = merge_ner_candidates(text, base, candidates)
            original = {"seif_fast": {(item.type, item.start, item.end) for item in base},
                        "presidio_ru": {(item.entity_type, item.start, item.end) for item in upstream},
                        "seif_hybrid": {(item.type, item.start, item.end) for item in merged}}
            raw_truth[key] = coarsen(row["fine_gold"], GOLD_MAP, keep_unknown=True)
            for system, spans in original.items():
                unmerged[system][key] = coarsen(spans, PRESIDIO_MAP if system == "presidio_ru" else LOCAL_MAP)
                predictions[system][key] = merge_adjacent(text, unmerged[system][key])
            stream.write(json.dumps({"id": key, "fine_gold": sorted(row["fine_gold"]), "merged_gold": sorted(truth[key]),
                                     "original_predictions": {system: sorted(spans) for system, spans in original.items()},
                                     "merged_predictions": {system: sorted(values[key]) for system, values in predictions.items()}}, ensure_ascii=False) + "\n")
    if sha((ROOT / "seif/detector.py").read_bytes()) != source_hash:
        raise RuntimeError("Detector changed during inference; report not published.")
    primary = score_scope(truth, predictions, COMMON)
    selected_gold = {key: truth[key] for key in ids}
    selected_pred = {system: {key: {span for span in values[key] if span[0] in COMMON} for key in ids} for system, values in predictions.items()}
    report = {
        "schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "first frozen-implementation evaluation of this previously unused mixed-language test" if first else "repeat: reproduction or post-result development",
        "dataset": metadata, "protocol": protocol, "protocol_sha256": sha(protocol_file.read_bytes()),
        "source_hashes": {"detector": source_hash, "evaluator": sha(Path(__file__).read_bytes()),
                          "presidio_configuration_script": sha((ROOT / "scripts/compare_presidio.py").read_bytes())},
        "presidio_configuration": configuration, "common_schema_primary": primary,
        "character_uncertainty": paired_bootstrap(typed_characters(selected_gold),
                                      {system: typed_characters(values) for system, values in selected_pred.items()}, {"all": ids}, COMMON),
        "exact_uncertainty": paired_bootstrap(selected_gold, selected_pred, {"all": ids}, COMMON),
        "person_all_cases": score_scope(truth, predictions, {"PERSON"}, whole_cases=False),
        "location_all_cases": score_scope(truth, predictions, {"LOCATION"}, whole_cases=False),
        "document_tax_diagnostic": score_scope(truth, predictions, {"DOCUMENT", "INN"}, whole_cases=False),
        "unmerged_common_diagnostic": score_scope(raw_truth, unmerged, COMMON),
        "raw_prediction_sha256": sha(raw_path.read_bytes()),
        "environment": {"python": sys.version, "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))},
        "limitations": [
            "External synthetic documents annotated by an LLM under human-verified guidelines; not independent manual gold or representative real traffic.",
            "RU/UK rows have no language labels. All language claims are mixed-corpus only; ru_core_news_sm is not an optimized Ukrainian model.",
            "CARD has no gold examples. This cannot validate all five requested common types or all required banking fields.",
            "Generic LOCATION includes components and coordinates; quality here is not complete personal postal-address extraction or public/private policy compliance.",
            "Nested name/address gold spans are coarsened and merged symmetrically using a rule frozen before inference; raw granularity diagnostics are separate.",
            "No threshold, model, pattern, label or merge policy was adjusted after seeing this corpus's predictions. Stop iteration after this final evaluation.",
            "The original HiveTrace and RedMadRobot first reports remain unchanged; the latter motivated the LOCATION development iteration.",
            "This is offline detector quality, not HTTP latency, storage, availability or throughput. Corpus text is excluded from the source archive.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({system: {metric: {key: value for key, value in data.items() if key not in ("by_type", "first_five_error_offsets")}
                               for metric, data in results.items()}
                      for system, results in primary["systems"].items()}, indent=2))


if __name__ == "__main__":
    main()
