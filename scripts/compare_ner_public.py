"""Frozen external transfer check for the selected GLiNER service adapter.

Prepare before inference; cache each backend in its own installed environment;
score both caches offline with the same detector and immutable dataset labels.
Source texts stay in the ignored prepared-input copy; reports contain no text.
This runner never downloads or tunes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_external as pii  # noqa: E402
from scripts import evaluate_redmadrobot as redmad  # noqa: E402
from scripts.cache_gliner import MODEL_ID, MODEL_REVISION, fingerprint_model, percentile  # noqa: E402
from scripts.evaluate_golden import save_json, sha256  # noqa: E402
from scripts.ner_service import build_analyzer, infer  # noqa: E402
from seif.detector import Span, detect, merge_ner_candidates, merge_person_candidates  # noqa: E402
from seif.gliner_ner import SCHEMAS, GlinerAnalyzer  # noqa: E402
from seif.ner import _entity_values, chunks  # noqa: E402
from seif.transform import mask, restore_exact  # noqa: E402

SELECTED = ROOT / "benchmarks/ner-models/gliner25-multi-v1/selected-service.meta.json"
SPACY_VERSIONS = {"spacy": "3.8.16", "ru-core-news-sm": "3.8.0", "presidio-analyzer": "2.2.364"}
SYSTEMS = ("rules", "spacy_hybrid", "gliner_hybrid", "spacy_person_only", "gliner_person_only")
SETTINGS = {
    "schema": "described-names",
    "threshold": 0.8,
    "device": "cuda",
    "dtype": "torch.float32",
    "cpu_threads": 4,
    "seed": 20260922,
    "compile": False,
    "batch_size": 1,
    "overlap_policy": "flat",
    "service_word_window": 384,
    "service_word_overlap": 64,
    "gateway_character_window": 16000,
    "gateway_character_overlap": 256,
}
SOURCE_FILES = (
    "scripts/compare_ner_public.py",
    "scripts/evaluate_external.py",
    "scripts/evaluate_redmadrobot.py",
    "scripts/evaluate_annotations.py",
    "scripts/evaluate_golden.py",
    "scripts/cache_gliner.py",
    "scripts/ner_service.py",
    "scripts/compare_presidio.py",
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze_source():
    paths = sorted((ROOT / "seif").glob("*.py")) + [ROOT / name for name in SOURCE_FILES]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def verify_files(folder, expected):
    for filename, checksum in expected.items():
        if not (folder / filename).is_file() or sha256(folder / filename) != checksum:
            raise ValueError(f"Pinned dataset file missing or changed: {filename}")


def load_pii(folder):
    import pyarrow.parquet as pq

    verify_files(folder, pii.FILES)
    stored = json.loads((folder / "prepared-protocol.json").read_text())
    if stored["seif_type_mapping"] != pii.SEIF_MAP or stored["presidio_strict_mapping"] != pii.PRESIDIO_MAP:
        raise ValueError("Historical PII taxonomy differs")
    result = []
    for split, count in (("domain", 900), ("entity", 910)):
        rows = pq.read_table(folder / f"{split}-00000-of-00001.parquet").to_pylist()
        if len(rows) != count or len({row["id"] for row in rows}) != count:
            raise ValueError("PII split size or IDs changed")
        pii._validate_rows(rows)
        result.extend(
            {
                "key": f"pii/{split}/{row['id']}",
                "id": row["id"],
                "dataset": "pii",
                "split": split,
                "domain": row["domain"],
                "text": row["text"],
                "gold": {(e["type"], e["start"], e["end"]) for e in row["entities"]},
            }
            for row in rows
        )
    return result


def load_redmad(folder):
    verify_files(folder, dict(redmad.FILES.values()))
    stored = json.loads((folder / "redmadrobot-protocol.json").read_text())
    rows, excluded = redmad.read_rows(folder)
    expected = [
        {"id": f"row_{index:04d}", "source_row_zero_based": index, "reason": "token_not_exactly_alignable"}
        for index in (0, 1628)
    ]
    if len(rows) != 2839 or excluded != expected or excluded != stored["dataset"]["excluded_before_inference"]:
        raise ValueError("Frozen redmadrobot alignment exclusions differ")
    if stored["mapping"]["gold"] != redmad.GOLD_MAP or stored["mapping"]["seif"] != redmad.SEIF_MAP:
        raise ValueError("Historical redmadrobot taxonomy differs")
    return [
        {
            "key": f"redmadrobot/test/{row['id']}",
            "id": row["id"],
            "dataset": "redmadrobot",
            "split": "test",
            "domain": "test",
            "text": row["text"],
            "gold": row["fine_gold"],
        }
        for row in rows
    ], excluded


def load_corpora(args):
    first = load_pii(args.pii_data)
    second, excluded = load_redmad(args.redmad_data)
    return first + second, excluded


def save_prepared_inputs(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({**row, "gold": sorted(row["gold"])}, ensure_ascii=False) + "\n")


def load_prepared_inputs(path, protocol):
    if sha256(path) != protocol["prepared_inputs_sha256"]:
        raise ValueError("Prepared input copy changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != protocol["cases"] or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Prepared input coverage differs")
    return [{**row, "gold": {tuple(span) for span in row["gold"]}} for row in rows]


def input_hashes(args):
    groups = (
        ("pii", args.pii_data, [*pii.FILES, "prepared-protocol.json"]),
        ("redmadrobot", args.redmad_data, [*(item[0] for item in redmad.FILES.values()), "redmadrobot-protocol.json"]),
    )
    return {f"{name}/{file}": sha256(folder / file) for name, folder, files in groups for file in files}


def make_protocol(args, rows, excluded):
    selected = json.loads(SELECTED.read_text())
    model_hashes = fingerprint_model(args.model_path)
    if model_hashes != selected["model_file_sha256"]:
        raise ValueError("Model snapshot differs from selected organizer configuration")
    configuration = selected["configuration"]
    if (
        configuration["schema"] != SETTINGS["schema"]
        or configuration["threshold"] != SETTINGS["threshold"]
        or configuration["labels"] != SCHEMAS[SETTINGS["schema"]]
    ):
        raise ValueError("Selected schema or threshold differs")
    return {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "External transfer check after organizer-only model selection; no external-result tuning.",
        "source_sha256": freeze_source(),
        "input_sha256": input_hashes(args),
        "prepared_inputs_sha256": sha256(args.run_dir / "prepared-inputs.jsonl"),
        "datasets": {
            "pii": {"repository": pii.DATASET, "revision": pii.REVISION},
            "redmadrobot": {
                "repository": redmad.REPOSITORY,
                "revision": redmad.REVISION,
                "excluded_before_inference": excluded,
            },
        },
        "cases": len(rows),
        "split_counts": dict(Counter(f"{r['dataset']}/{r['split']}" for r in rows)),
        "ordered_membership_sha256": digest_json([r["key"] for r in rows]),
        "text_sha256": digest_json({r["key"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in rows}),
        "gold_sha256": digest_json({r["key"]: sorted(r["gold"]) for r in rows}),
        "input_lengths": {
            "maximum_characters": max(len(r["text"]) for r in rows),
            "maximum_whitespace_words": max(len(r["text"].split()) for r in rows),
        },
        "selected_organizer_metadata_sha256": sha256(SELECTED),
        "gliner": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "file_sha256": model_hashes,
            "versions": selected["versions"],
            "settings": SETTINGS,
            "labels": SCHEMAS[SETTINGS["schema"]],
        },
        "spacy": {"model": "ru_core_news_sm", "versions": SPACY_VERSIONS, "score_threshold": 0.0},
        "mapping": {"pii": pii.SEIF_MAP, "redmadrobot": {**redmad.SEIF_MAP, "LOCATION": "LOCATION"}},
        "policy": {
            "labels": "Original labels and fixed BIO exclusions unchanged.",
            "pii": "Historical common4 and supported8 whole-case subsets; exact spans and typed characters.",
            "redmadrobot": "Historical common5 and mapped8 subsets; identical adjacent/overlap merge on every side.",
            "unfiltered_protection": "All rows and all original gold types, including unsupported types; actual maskable alphanumeric characters.",
            "typed_unfiltered": "All original gold types and every prediction retained, unmapped predictions penalized explicitly.",
            "prepared_inputs": "Text copy remains under ignored local-data; reports and model caches contain no raw text.",
            "historical_person_only": "Same current detector and merge, retaining only each model's PERSON candidates.",
            "timing": "Sequential local NER calls only; not HTTP throughput or service RPS.",
        },
    }


def verify_frozen(args, rows, protocol):
    checks = (
        (protocol["source_sha256"], freeze_source()),
        (protocol["input_sha256"], input_hashes(args)),
        (protocol["ordered_membership_sha256"], digest_json([r["key"] for r in rows])),
        (
            protocol["text_sha256"],
            digest_json({r["key"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in rows}),
        ),
        (protocol["gold_sha256"], digest_json({r["key"]: sorted(r["gold"]) for r in rows})),
    )
    if any(before != after for before, after in checks):
        raise ValueError("Sources, inputs, order or annotations differ from the pre-inference protocol")


def disable_network():
    def denied(*_args, **_kwargs):
        raise RuntimeError("Network is disabled during local external evaluation")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def backend_versions(backend, protocol):
    expected = protocol[backend]["versions"]
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise ValueError(f"{backend} dependencies differ from frozen configuration")
    return actual


def create_backend(args, protocol):
    versions = backend_versions(args.backend, protocol)
    if args.backend == "spacy":
        os.environ["SEIF_NER_BACKEND"] = "presidio"
        return build_analyzer(), versions
    import torch

    if fingerprint_model(args.model_path) != protocol["gliner"]["file_sha256"]:
        raise ValueError("Checkpoint changed after protocol preparation")
    torch.set_num_threads(SETTINGS["cpu_threads"])
    torch.manual_seed(SETTINGS["seed"])
    analyzer = GlinerAnalyzer.from_local(
        args.model_path, device=SETTINGS["device"], schema=SETTINGS["schema"], threshold=SETTINGS["threshold"]
    )
    if str(next(analyzer.extractor.parameters()).dtype) != SETTINGS["dtype"]:
        raise ValueError("Model precision differs from the frozen FP32 protocol")
    return analyzer, versions


def infer_document(analyzer, text):
    """Apply the gateway's existing character windows and output validation."""
    result = []
    for offset, part in chunks(text):
        output = infer(analyzer, part)
        for entity in output["entities"]:
            start, end, score, kind = _entity_values(entity, len(part))
            if (offset and start == 0) or (offset + len(part) < len(text) and end == len(part)):
                continue
            result.append({"start": offset + start, "end": offset + end, "entity_type": kind, "score": score})
    return result


def synchronize(backend):
    if backend == "gliner":
        import torch

        torch.cuda.synchronize()


def cache_backend(args, rows, protocol):
    target = args.run_dir / f"{args.backend}.jsonl"
    metadata_path = target.with_suffix(".meta.json")
    if target.exists() or metadata_path.exists():
        raise FileExistsError("Model caches are immutable; use a new run directory")
    analyzer, versions = create_backend(args, protocol)
    for _ in range(5):
        infer_document(analyzer, "Иван Иванов приехал в Москву.")
    synchronize(args.backend)
    timings, counts = [], Counter()
    started = time.perf_counter()
    with target.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            before = time.perf_counter()
            entities = infer_document(analyzer, row["text"])
            synchronize(args.backend)
            timings.append((time.perf_counter() - before) * 1000)
            counts.update(e["entity_type"] for e in entities)
            record = {
                "case_id": row["key"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "entities": entities,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            if index % 100 == 0:
                stream.flush()
                print(json.dumps({"backend": args.backend, "completed": index, "total": len(rows)}), flush=True)
    elapsed = time.perf_counter() - started
    verify_frozen(args, rows, protocol)
    metadata = {
        "backend": args.backend,
        "cases": len(rows),
        "versions": versions,
        "protocol_sha256": sha256(args.run_dir / "protocol.json"),
        "cache_sha256": sha256(target),
        "candidate_counts": dict(counts),
        "source_sha256": protocol["source_sha256"],
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "timing": {
            "elapsed_seconds": elapsed,
            "sequential_documents_per_second": len(rows) / elapsed,
            "mean_ms": statistics.mean(timings),
            "p50_ms": statistics.median(timings),
            "p95_ms": percentile(timings, 0.95),
            "p99_ms": percentile(timings, 0.99),
            "max_ms": max(timings),
            "scope": protocol["policy"]["timing"],
        },
    }
    save_json(metadata_path, metadata)
    return metadata


def load_cache(path, rows, protocol_hash):
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if metadata["cache_sha256"] != sha256(path) or metadata["protocol_sha256"] != protocol_hash:
        raise ValueError("Cache checksum or protocol provenance differs")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if [r["case_id"] for r in records] != [r["key"] for r in rows]:
        raise ValueError("Cache coverage or order differs")
    for record, row in zip(records, rows, strict=True):
        if record["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("Cache belongs to different text")
        for item in record["entities"]:
            _entity_values(item, len(row["text"]))
    return {r["case_id"]: r["entities"] for r in records}, metadata


def predictions_for(text, candidates):
    base = detect(text)
    result = {"rules": base}
    for backend, entities in candidates.items():
        spans = [Span(e["start"], e["end"], e["entity_type"], e["score"], "frozen-model") for e in entities]
        result[backend + "_hybrid"] = merge_ner_candidates(text, base, spans)
        result[backend + "_person_only"] = merge_person_candidates(text, base, [s for s in spans if s.type == "PERSON"])
    return result


def mask_positions(text, spans):
    masked, replacements = mask(text, spans, "mask")
    if len(masked) != len(text) or restore_exact({"masked": masked, "replacements": replacements}) != text:
        raise ValueError("Mask restoration invariant failed")
    return {("PII", i) for i, char in enumerate(text) if char.isalnum() and masked[i] == "*"}


def untyped_gold(text, spans):
    return {("PII", i) for _, start, end in spans for i in range(start, end) if text[i].isalnum()}


def compact_measure(gold, predicted):
    result = pii.measure(gold, predicted)
    result.pop("first_five_error_offsets", None)
    return result


def scope_metrics(gold, predictions, allowed=None, whole_cases=False):
    ids = [key for key, spans in gold.items() if not whole_cases or all(kind in allowed for kind, *_ in spans)]
    truth = {key: {span for span in gold[key] if allowed is None or span[0] in allowed} for key in ids}
    systems = {}
    for name, values in predictions.items():
        actual = {key: {span for span in values[key] if allowed is None or span[0] in allowed} for key in ids}
        systems[name] = {
            "exact_span": compact_measure(truth, actual),
            "typed_character": compact_measure(pii.typed_characters(truth), pii.typed_characters(actual)),
        }
    return {
        "cases": len(ids),
        "case_ids_sha256": digest_json(ids),
        "whole_cases": whole_cases,
        "types": sorted(allowed) if allowed is not None else "all including unmapped",
        "systems": systems,
    }


def map_split(rows, predictions):
    is_redmad = rows[0]["dataset"] == "redmadrobot"
    mapping = {**redmad.SEIF_MAP, "LOCATION": "LOCATION"} if is_redmad else pii.SEIF_MAP
    truth, mapped, raw_truth, raw_pred = {}, {s: {} for s in SYSTEMS}, {}, {s: {} for s in SYSTEMS}
    for row in rows:
        key, text = row["key"], row["text"]
        raw_truth[key] = redmad.coarsen(row["gold"], redmad.GOLD_MAP, keep_unknown=True) if is_redmad else row["gold"]
        truth[key] = redmad.merge_adjacent(text, raw_truth[key]) if is_redmad else raw_truth[key]
        for name, values in predictions.items():
            raw_pred[name][key] = {
                (mapping.get(s.type, "UNMAPPED_PRED:" + s.type), s.start, s.end) for s in values[key]
            }
            mapped[name][key] = redmad.merge_adjacent(text, raw_pred[name][key]) if is_redmad else raw_pred[name][key]
    return truth, mapped, raw_truth, raw_pred


def protection_metrics(rows, predictions):
    truth = {r["key"]: untyped_gold(r["text"], r["gold"]) for r in rows}
    systems = {
        name: compact_measure(truth, {r["key"]: mask_positions(r["text"], values[r["key"]]) for r in rows})
        for name, values in predictions.items()
    }
    return {
        "cases": len(rows),
        "gold_scope": "All original gold categories, including unsupported categories.",
        "prediction_scope": "Actual service masks; every model/rule type retained; alphanumeric characters only.",
        "systems": systems,
    }


def raw_person_metrics(rows, caches):
    is_redmad = rows[0]["dataset"] == "redmadrobot"
    allowed = redmad.NAME_TYPES if is_redmad else {"NAME"}
    truth = {r["key"]: {("PERSON", start, end) for kind, start, end in r["gold"] if kind in allowed} for r in rows}
    predictions = {
        name: {
            r["key"]: {("PERSON", e["start"], e["end"]) for e in values[r["key"]] if e["entity_type"] == "PERSON"}
            for r in rows
        }
        for name, values in caches.items()
    }
    if is_redmad:
        truth = {r["key"]: redmad.merge_adjacent(r["text"], truth[r["key"]]) for r in rows}
        predictions = {
            name: {r["key"]: redmad.merge_adjacent(r["text"], values[r["key"]]) for r in rows}
            for name, values in predictions.items()
        }
    return scope_metrics(truth, predictions)


def score_split(rows, predictions, caches):
    truth, mapped, raw_truth, raw_pred = map_split(rows, predictions)
    is_redmad = rows[0]["dataset"] == "redmadrobot"
    common = redmad.COMMON if is_redmad else pii.COMMON
    supported = common | redmad.STRUCTURED if is_redmad else pii.SUPPORTED
    result = {
        "cases": len(rows),
        "common5" if is_redmad else "common4": scope_metrics(truth, mapped, common, True),
        "supported8": scope_metrics(truth, mapped, supported, True),
        "supported_types_all_rows": scope_metrics(truth, mapped, supported),
        "all_types_unfiltered": scope_metrics(truth, mapped),
        "full_masking_all_gold_types": protection_metrics(rows, predictions),
        "raw_model_person": raw_person_metrics(rows, caches),
        "raw_unmerged_common": scope_metrics(raw_truth, raw_pred, common, True),
        "span_policy": "Identical coarse-category adjacency merge on gold and all predictions."
        if is_redmad
        else "Original exact boundaries, no aggregation.",
    }
    if is_redmad:
        result["person_all_rows"] = scope_metrics(truth, mapped, {"PERSON"})
        result["location_all_rows"] = scope_metrics(truth, mapped, {"LOCATION"})
    return result


def evaluate_caches(args, rows, protocol):
    protocol_hash = sha256(args.run_dir / "protocol.json")
    caches, metadata = {}, {}
    for name in ("spacy", "gliner"):
        caches[name], metadata[name] = load_cache(args.run_dir / f"{name}.jsonl", rows, protocol_hash)
    predictions = {name: {} for name in SYSTEMS}
    for row in rows:
        values = predictions_for(row["text"], {name: cache[row["key"]] for name, cache in caches.items()})
        for name, spans in values.items():
            predictions[name][row["key"]] = spans
    splits = {}
    for dataset, split in (("pii", "domain"), ("pii", "entity"), ("redmadrobot", "test")):
        selected = [r for r in rows if r["dataset"] == dataset and r["split"] == split]
        splits[f"{dataset}/{split}"] = score_split(selected, predictions, caches)
    splits["pii/all"] = score_split([r for r in rows if r["dataset"] == "pii"], predictions, caches)
    verify_frozen(args, rows, protocol)
    return {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "protocol": protocol,
        "caches": metadata,
        "splits": splits,
        "limitations": [
            "External datasets were already used for rule development; this is a fixed-model transfer check, not a wholly unseen pipeline holdout.",
            "The single schema and threshold were selected on organizer development data and frozen before external inference; no external-result tuning.",
            "PII common4/supported8 maps exclude LOCATION because generic geography is not equivalent to a full ADDRESS; full masking/unfiltered scopes expose these extra outputs.",
            "Redmadrobot uses historical coarse name/location mapping and identical adjacency merge; LOCATION is not exact private-address identification.",
            "Unsupported gold categories remain in provenance and unfiltered/full masking scores, which therefore include known assignment coverage gaps.",
            "PERSON-only systems are historical-protocol sensitivity checks with the current detector, not published historical numerical baselines.",
            "Typed-character scores count all characters inside spans; full masking scores count alphanumeric characters actually replaced by the service.",
            "Sequential inference observations are not HTTP RPS; no deployment or threshold changes follow automatically from this evaluation.",
            "Original source text is never included in the report; inference is local with network connections disabled.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cache", "evaluate"))
    parser.add_argument("--pii-data", type=Path, required=True)
    parser.add_argument("--redmad-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--backend", choices=("spacy", "gliner"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.mode == "prepare" or args.backend == "gliner") and args.model_path is None:
        parser.error("--model-path is required for protocol preparation and GLiNER inference")
    if args.mode == "cache" and args.backend is None:
        parser.error("--backend is required for inference")
    if args.mode == "evaluate" and args.output is None:
        parser.error("--output is required for evaluation")
    return args


def main():
    args = parse_args()
    if not args.run_dir.resolve().is_relative_to((ROOT / "local-data").resolve()):
        raise ValueError("Prepared texts and caches must remain under the ignored local-data directory")
    disable_network()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "protocol.json"
    if args.mode == "prepare":
        rows, excluded = load_corpora(args)
        save_prepared_inputs(args.run_dir / "prepared-inputs.jsonl", rows)
        protocol = make_protocol(args, rows, excluded)
        save_json(path, protocol)
        print(
            json.dumps(
                {
                    "prepared": True,
                    "cases": len(rows),
                    "splits": protocol["split_counts"],
                    "protocol_sha256": sha256(path),
                    "input_lengths": protocol["input_lengths"],
                }
            )
        )
        return
    protocol = json.loads(path.read_text())
    rows = load_prepared_inputs(args.run_dir / "prepared-inputs.jsonl", protocol)
    verify_frozen(args, rows, protocol)
    if args.mode == "cache":
        print(json.dumps(cache_backend(args, rows, protocol), ensure_ascii=False))
        return
    report = evaluate_caches(args, rows, protocol)
    save_json(args.output, report)
    print(json.dumps({"report": str(args.output), "cases": len(rows), "splits": list(report["splits"])}))


if __name__ == "__main__":
    main()
