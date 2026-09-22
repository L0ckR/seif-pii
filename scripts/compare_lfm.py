"""Freeze, cache once, and evaluate LFM native and gateway PII profiles locally.

No tuning or model downloads occur here. Native all-type masks retain every
model label; PERSON isolation and ADDRESS-to-LOCATION are separate profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts import evaluate_annotations as annotations  # noqa: E402
from scripts import evaluate_golden as golden  # noqa: E402
from scripts.compare_ner_models import compact_evaluation  # noqa: E402
from seif.detector import Span, detect, merge_ner_candidates, merge_person_candidates  # noqa: E402
from seif.transform import mask, restore_exact  # noqa: E402

MERGE_CONFIDENCE = 0.85
DECODERS = ("raw", "hybrid")
PERSON_LABEL = "identity.person_name"
ADDRESS_LABEL = "contact.address"
SETTINGS = {"device": "cuda", "dtype": "torch.float32", "cpu_threads": 4,
            "seed": 20260922, "merge_confidence": MERGE_CONFIDENCE, "maximum_tokens": 2048,
            "threshold_tuning": False, "batch_size": 1}
MODEL_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors")
SOURCE_FILES = ("scripts/compare_lfm.py", "scripts/compare_ner_public.py", "scripts/compare_ner_models.py",
                "scripts/evaluate_golden.py", "scripts/evaluate_annotations.py", "scripts/evaluate_external.py",
                "scripts/evaluate_redmadrobot.py", "scripts/ner_service.py")
BASELINE_REPORTS = {
    "organizer": ROOT / "benchmarks/ner-models/gliner25-multi-v1/comparison-selected-service.json",
    "public": ROOT / "benchmarks/ner-models/gliner25-multi-v1/public-transfer.json",
}


def source_hashes():
    files = sorted((ROOT / "seif").glob("*.py")) + [ROOT / name for name in SOURCE_FILES]
    return {str(path.relative_to(ROOT)): golden.sha256(path) for path in files}


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_hashes(path):
    from seif.lfm_pii import fingerprint_checkpoint

    # Only the seven publisher artifacts, never generated helper bytecode.
    return fingerprint_checkpoint(path)


def assets(args):
    paths = {"organizer_cases": args.organizer,
             "organizer_manifest": args.organizer.with_name("manifest.json"),
             "organizer_spacy": args.organizer_spacy,
             "organizer_spacy_metadata": args.organizer_spacy.with_suffix(".meta.json"),
             "organizer_gliner": args.organizer_gliner,
             "organizer_gliner_metadata": args.organizer_gliner.with_suffix(".meta.json")}
    for name in ("protocol.json", "prepared-inputs.jsonl", "spacy.jsonl", "spacy.meta.json",
                 "gliner.jsonl", "gliner.meta.json"):
        paths["public/" + name] = args.public_run_dir / name
    paths.update({"baseline_report/" + name: path for name, path in BASELINE_REPORTS.items()})
    return {name: golden.sha256(path) for name, path in paths.items()}


def load_inputs(args):
    cases = golden.load_cases(args.organizer)
    external_protocol = json.loads((args.public_run_dir / "protocol.json").read_text())
    external = public.load_prepared_inputs(args.public_run_dir / "prepared-inputs.jsonl", external_protocol)
    organizer = [{"key": "organizer/" + key, "id": key, "dataset": "organizer", "split": "v1",
                  "text": case["text"], "gold": {(e["type"], e["start"], e["end"]) for e in case["entities"]},
                  "uncertain": case["uncertain"], "traffic_weight": case["traffic_weight"]}
                 for key, case in cases.items()]
    if len(organizer) != 446 or len(external) != 4649:
        raise ValueError("Pinned organizer/public coverage differs")
    return organizer + external, cases, external_protocol


def load_references(args, rows, organizer, public_protocol):
    for name, expected in public_protocol["source_sha256"].items():
        if golden.sha256(ROOT / name) != expected:
            raise ValueError("Historical reference inference/evaluation source changed")
    caches, metadata = {}, {}
    external = [row for row in rows if row["dataset"] != "organizer"]
    for name in ("spacy", "gliner"):
        original_path = getattr(args, "organizer_" + name)
        original = golden.load_ner_cache(original_path, organizer)
        selected, info = public.load_cache(args.public_run_dir / f"{name}.jsonl", external,
                                           golden.sha256(args.public_run_dir / "protocol.json"))
        caches[name] = {**selected, **{"organizer/" + key: row["entities"] for key, row in original.items()}}
        metadata[name] = {"public": info,
                          "organizer": json.loads(original_path.with_suffix(".meta.json").read_text())}
    if public_protocol["cases"] != len(external):
        raise ValueError("Public protocol coverage differs")
    return caches, metadata


def prepare(args):
    from seif.lfm_pii import MODEL_ID, MODEL_REVISION, NATIVE_TYPES

    rows, organizer, external_protocol = load_inputs(args)
    load_references(args, rows, organizer, external_protocol)
    path = args.run_dir / "protocol.json"
    if path.exists():
        raise FileExistsError("Experiment protocol already exists")
    public.save_prepared_inputs(args.run_dir / "prepared-inputs.jsonl", rows)
    protocol = {
        "schema_version": 1, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID, "model_revision": MODEL_REVISION, "native_types": sorted(NATIVE_TYPES),
        "settings": SETTINGS, "model_file_sha256": model_hashes(args.model_path),
        "model_package_versions": {name: importlib.metadata.version(name) for name in MODEL_PACKAGES},
        "source_sha256": source_hashes(), "asset_sha256": assets(args), "cases": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "prepared_inputs_sha256": golden.sha256(args.run_dir / "prepared-inputs.jsonl"),
        "ordered_membership_sha256": public.digest_json([row["key"] for row in rows]),
        "gold_sha256": public.digest_json({row["key"]: sorted(row["gold"]) for row in rows}),
        "policy": {
            "model_isolation": "Only native identity.person_name maps to PERSON; compare all models under identical PERSON-only gateway merge.",
            "address_proxy": "contact.address maps to LOCATION only in gateway_address_proxy; ADDRESS is not generic geographic LOCATION.",
            "native_masking": "Union of every native span from each decoder against all original gold types; no unsupported types removed.",
            "merge_confidence": "Fixed 0.85 compatibility weight, matching spaCy NER; not a calibrated model probability or fitted threshold.",
            "decoding": "One model forward extraction per text, then the official hybrid decoder over the same raw spans.",
            "selection": "No schema or threshold fitting on any of the three corpora.",
            "overlong_gateway_candidates": "If any candidate exceeds the existing 200-character contract, the affected gateway profile is unavailable on that corpus; native scores retain it.",
        },
    }
    golden.save_json(path, protocol)
    return protocol


def verify_frozen(args, protocol):
    if protocol["source_sha256"] != source_hashes() or protocol["asset_sha256"] != assets(args):
        raise ValueError("Inference/evaluation sources or fixed assets changed after preparation")
    rows = public.load_prepared_inputs(args.run_dir / "prepared-inputs.jsonl", protocol)
    if (protocol["ordered_membership_sha256"] != public.digest_json([row["key"] for row in rows])
            or protocol["gold_sha256"] != public.digest_json({row["key"]: sorted(row["gold"]) for row in rows})):
        raise ValueError("Prepared corpus order or annotations changed")
    return rows


def validate_native(output, text, native_types):
    if not isinstance(output, dict) or set(output) != set(DECODERS):
        raise ValueError("LFM result needs exactly both native decoders")
    for values in output.values():
        if not isinstance(values, list):
            raise ValueError("LFM native spans must be a list")
        for item in values:
            if (not isinstance(item, dict) or set(item) != {"start", "end", "type"}
                    or type(item["start"]) is not int or type(item["end"]) is not int
                    or not 0 <= item["start"] < item["end"] <= len(text)
                    or not isinstance(item["type"], str) or item["type"] not in native_types):
                raise ValueError("LFM native span has invalid offsets or an unknown type")


def synchronize(device):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def timing_summary(values):
    elapsed = sum(values) / 1000
    return {"cases": len(values), "summed_call_seconds": elapsed,
            "sequential_documents_per_second": len(values) / elapsed if elapsed else None,
            "p50_ms": statistics.median(values), "p95_ms": public.percentile(values, .95),
            "p99_ms": public.percentile(values, .99), "mean_ms": statistics.mean(values),
            "scope": "Synchronized predict_both calls on this corpus only; includes both decoders, excludes cache serialization; not HTTP RPS."}


def cache_model(args, rows, protocol):
    import torch

    from seif.lfm_pii import LfmPiiAnalyzer

    path = args.run_dir / "lfm.jsonl"
    if path.exists() or path.with_suffix(".meta.json").exists():
        raise FileExistsError("LFM cache is immutable")
    if model_hashes(args.model_path) != protocol["model_file_sha256"]:
        raise ValueError("Model snapshot changed after preparation")
    if {name: importlib.metadata.version(name) for name in MODEL_PACKAGES} != protocol["model_package_versions"]:
        raise ValueError("LFM package versions changed after protocol preparation")
    torch.set_num_threads(SETTINGS["cpu_threads"])
    torch.manual_seed(SETTINGS["seed"])
    analyzer = LfmPiiAnalyzer.from_local(args.model_path, device=SETTINGS["device"])
    if analyzer.metadata()["dtype"] != SETTINGS["dtype"]:
        raise ValueError("LFM precision differs from the frozen protocol")
    lengths = [analyzer.token_count(row["text"]) for row in rows]
    if max(lengths) > SETTINGS["maximum_tokens"]:
        raise ValueError("Corpus exceeds the no-truncation model token budget")
    for _ in range(5):
        analyzer.predict_both("Иван Иванов приехал в Москву.")
    synchronize(SETTINGS["device"])
    if SETTINGS["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings, by_dataset = [], {}
    started = time.perf_counter()
    with path.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            before = time.perf_counter()
            output = analyzer.predict_both(row["text"])
            synchronize(SETTINGS["device"])
            timings.append((time.perf_counter() - before) * 1000)
            by_dataset.setdefault(row["dataset"], []).append(timings[-1])
            validate_native(output, row["text"], protocol["native_types"])
            stream.write(json.dumps({"case_id": row["key"], "text_sha256": text_digest(row["text"]),
                                     "decoders": output}, sort_keys=True, separators=(",", ":")) + "\n")
            if index % 100 == 0:
                stream.flush()
                print(json.dumps({"completed": index, "total": len(rows)}), flush=True)
    elapsed = time.perf_counter() - started
    verify_frozen(args, protocol)
    metadata = {
        "cases": len(rows), "protocol_sha256": golden.sha256(args.run_dir / "protocol.json"),
        "cache_sha256": golden.sha256(path), "source_sha256": protocol["source_sha256"],
        "model": analyzer.metadata(), "maximum_observed_tokens": max(lengths),
        "hardware": {"platform": platform.platform(),
                     "gpu": torch.cuda.get_device_name() if SETTINGS["device"] == "cuda" else None,
                     "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                     if SETTINGS["device"] == "cuda" else None},
        "timing_by_dataset": {name: timing_summary(values) for name, values in by_dataset.items()},
        "timing": {"elapsed_seconds": elapsed, "sequential_documents_per_second": len(rows) / elapsed,
                   "p50_ms": statistics.median(timings), "p95_ms": public.percentile(timings, .95),
                   "p99_ms": public.percentile(timings, .99), "mean_ms": statistics.mean(timings),
                   "scope": "Sequential model extraction plus both official decoders, no HTTP/masking/Redis; not service RPS."},
    }
    golden.save_json(path.with_suffix(".meta.json"), metadata)
    return metadata


def load_lfm_cache(path, rows, protocol, protocol_hash):
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if metadata["cache_sha256"] != golden.sha256(path) or metadata["protocol_sha256"] != protocol_hash:
        raise ValueError("LFM cache checksum or frozen protocol mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if [record["case_id"] for record in records] != [row["key"] for row in rows]:
        raise ValueError("LFM cache order or coverage mismatch")
    for record, row in zip(records, rows, strict=True):
        if record["text_sha256"] != text_digest(row["text"]):
            raise ValueError("LFM cache belongs to different input text")
        validate_native(record["decoders"], row["text"], protocol["native_types"])
    return {record["case_id"]: record["decoders"] for record in records}, metadata


def native_candidates(spans, *, address_proxy=False):
    mapping = {PERSON_LABEL: "PERSON"}
    if address_proxy:
        mapping[ADDRESS_LABEL] = "LOCATION"
    return [{"start": item["start"], "end": item["end"], "entity_type": mapping[item["type"]],
             "score": MERGE_CONFIDENCE} for item in spans if item["type"] in mapping]


def gateway_predictions(rows, caches):
    result = {"rules": {}}
    rejected = {}
    for row in rows:
        key, text = row["key"], row["text"]
        base = detect(text)
        result["rules"][key] = base
        for name, cache in caches.items():
            result.setdefault(name, {})
            entities = cache[key]
            if any(item["end"] - item["start"] > 200 for item in entities):
                rejected.setdefault(name, []).append(key)
                continue
            spans = [Span(e["start"], e["end"], e["entity_type"], e["score"], "frozen-ner") for e in entities]
            merger = merge_person_candidates if name.endswith("person_only") else merge_ner_candidates
            result[name][key] = merger(text, base, spans)
    # Do not publish a partial-corpus gateway score if the contract rejected any case.
    return {name: values for name, values in result.items() if name not in rejected}, rejected


def build_gateway_caches(rows, references, native):
    result = {}
    for name, values in references.items():
        result[name + "_person_only"] = {row["key"]: [e for e in values[row["key"]] if e["entity_type"] == "PERSON"]
                                         for row in rows}
        result[name + "_hybrid"] = {row["key"]: values[row["key"]] for row in rows}
    for decoder in DECODERS:
        for profile, address_proxy in (("person_only", False), ("gateway_address_proxy", True)):
            result[f"lfm_{decoder}_{profile}"] = {
                row["key"]: native_candidates(native[row["key"]][decoder], address_proxy=address_proxy)
                for row in rows}
    return result


def organizer_cases(rows):
    return {row["key"]: {
        "text": row["text"], "uncertain": row["uncertain"], "traffic_weight": row["traffic_weight"],
        "entities": [{"type": kind, "start": start, "end": end, "text": row["text"][start:end]}
                     for kind, start, end in sorted(row["gold"], key=lambda item: item[1:])],
        "decision": "positive" if row["gold"] else "negative",
    } for row in rows}


def score_organizer(rows, predictions):
    cases = organizer_cases(rows)
    weights = {row["key"]: row["traffic_weight"] for row in rows}
    results = {}
    for name, values in predictions.items():
        output = {}
        for row in rows:
            spans = values[row["key"]]
            masked, replacements = mask(row["text"], spans, "mask")
            if restore_exact({"masked": masked, "replacements": replacements}) != row["text"]:
                raise ValueError("Exact restoration failed")
            output[row["key"]] = {"masked": masked, "entities": [
                {"type": span.type, "start": span.start, "end": span.end} for span in spans]}
        results[name] = compact_evaluation(annotations.evaluate(cases, cases, output, weights))
    return results


def score_public(rows, predictions):
    redmad = rows[0]["dataset"] == "redmadrobot"
    mapping = {**public.redmad.SEIF_MAP, "LOCATION": "LOCATION"} if redmad else public.pii.SEIF_MAP
    truth, mapped = {}, {name: {} for name in predictions}
    for row in rows:
        gold = public.redmad.coarsen(row["gold"], public.redmad.GOLD_MAP, keep_unknown=True) if redmad else row["gold"]
        truth[row["key"]] = public.redmad.merge_adjacent(row["text"], gold) if redmad else gold
        for name, values in predictions.items():
            actual = {(mapping.get(span.type, "UNMAPPED_PRED:" + span.type), span.start, span.end)
                      for span in values[row["key"]]}
            mapped[name][row["key"]] = public.redmad.merge_adjacent(row["text"], actual) if redmad else actual
    common = public.redmad.COMMON if redmad else public.pii.COMMON
    supported = common | public.redmad.STRUCTURED if redmad else public.pii.SUPPORTED
    return {
        "all_types_unfiltered": public.scope_metrics(truth, mapped),
        "full_masking_all_gold_types": public.protection_metrics(rows, predictions),
        "common5" if redmad else "common4": public.scope_metrics(truth, mapped, common, True),
        "supported8": public.scope_metrics(truth, mapped, supported, True),
        "scope_notice": "Subset scores are secondary; all-type/unfiltered and full masking retain unsupported gold and predictions.",
    }


def raw_person_scores(rows, caches):
    if rows[0]["dataset"] != "organizer":
        return public.raw_person_metrics(rows, caches)
    truth = {row["key"]: {("PERSON", start, end) for kind, start, end in row["gold"]
                           if kind in {"PERSON", "CARDHOLDER"}} for row in rows}
    predicted = {name: {row["key"]: {("PERSON", item["start"], item["end"])
                                    for item in values[row["key"]] if item["entity_type"] == "PERSON"}
                       for row in rows} for name, values in caches.items()}
    return public.scope_metrics(truth, predicted)


def native_masking(rows, native):
    gold = {row["key"]: public.untyped_gold(row["text"], row["gold"]) for row in rows}
    result = {}
    for decoder in DECODERS:
        predicted = {row["key"]: public.untyped_gold(row["text"], {
            (item["type"], item["start"], item["end"]) for item in native[row["key"]][decoder]}) for row in rows}
        result[decoder] = public.compact_measure(gold, predicted)
        if rows[0]["dataset"] == "organizer":
            certain = [row["key"] for row in rows if not row["uncertain"]]
            result[decoder]["certain_cases_sensitivity"] = public.compact_measure(
                {key: gold[key] for key in certain}, {key: predicted[key] for key in certain})
    return {"scope": "All native labels and all original gold labels; union of alphanumeric character positions.",
            "systems": result}


def score_corpus(rows, references, native):
    gateway_caches = build_gateway_caches(rows, references, native)
    predictions, rejected = gateway_predictions(rows, gateway_caches)
    person_caches = {name: values for name, values in gateway_caches.items() if name.endswith("person_only")}
    return {
        "cases": len(rows), "gold_category_counts": dict(sorted(Counter(
            kind for row in rows for kind, _, _ in row["gold"]).items())),
        "native_prediction_category_counts": {decoder: dict(sorted(Counter(
            item["type"] for row in rows for item in native[row["key"]][decoder]).items())) for decoder in DECODERS},
        "raw_model_person_isolation": raw_person_scores(rows, person_caches),
        "gateway_profiles": score_organizer(rows, predictions) if rows[0]["dataset"] == "organizer"
                            else score_public(rows, predictions),
        "gateway_contract_rejections": {name: {"case_count": len(keys), "case_ids_sha256": public.digest_json(keys),
                                               "score_status": "unavailable for the complete corpus"}
                                        for name, keys in rejected.items()},
        "native_full_masking_all_types": native_masking(rows, native),
    }


def verify_baseline_metrics(corpora):
    organizer = json.loads(BASELINE_REPORTS["organizer"].read_text())
    external = json.loads(BASELINE_REPORTS["public"].read_text())
    checked = {}
    for name, role in (("spacy_hybrid", "reference"), ("gliner_hybrid", "candidate")):
        actual = corpora["organizer"]["gateway_profiles"][name]
        if actual != organizer["hybrid"][role]:
            raise ValueError("Organizer baseline aggregate replay differs")
        checked["organizer/" + name] = actual["unique_case_primary"]
        for corpus, split in (("pii", "pii/all"), ("redmadrobot", "redmadrobot/test")):
            actual = corpora[corpus]["gateway_profiles"]["full_masking_all_gold_types"]["systems"][name]
            expected = external["splits"][split]["full_masking_all_gold_types"]["systems"][name]
            if actual != expected:
                raise ValueError("Public baseline full-masking replay differs")
            checked[corpus + "/" + name] = actual
    return {"matched_all_compared_aggregates": True,
            "baseline_report_sha256": {name: golden.sha256(path) for name, path in BASELINE_REPORTS.items()},
            "replayed": checked}


def evaluate(args, rows, protocol):
    _, organizer, external_protocol = load_inputs(args)
    references, reference_metadata = load_references(args, rows, organizer, external_protocol)
    native, metadata = load_lfm_cache(args.run_dir / "lfm.jsonl", rows, protocol,
                                      golden.sha256(args.run_dir / "protocol.json"))
    corpora = {name: score_corpus([row for row in rows if row["dataset"] == name], references, native)
               for name in ("organizer", "pii", "redmadrobot")}
    verify_frozen(args, protocol)
    return {
        "schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": golden.sha256(args.run_dir / "protocol.json"), "protocol": protocol,
        "lfm_cache": metadata, "reference_caches": reference_metadata, "corpora": corpora,
        "baseline_replay": verify_baseline_metrics(corpora),
        "runtime": {"python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in ("regex",)}},
        "limitations": [
            "Organizer silver labels are provisional AI annotations, not official organizer ground truth.",
            "All corpora were previously inspected for system development; this is a fixed-model transfer experiment, not an untouched pipeline holdout.",
            "GLiNER schema/threshold selection used organizer data previously; LFM receives no tuning in this experiment.",
            "PERSON-only is a shared-label comparison; native all-type masking has a different task scope and is reported separately.",
            "Native contact.address is not generic LOCATION; its gateway mapping is an explicit proxy ablation.",
            "LFM native decoders provide no calibrated confidence; 0.85 is a fixed merge compatibility weight only.",
            "No unsupported original gold or native model category is discarded from full native masking or unfiltered gateway scopes.",
            "Typed character diagnostics count all span characters; full masking counts alphanumeric characters as the service does.",
            "Corpus metrics are not the organizer's private score. Local sequential timing is not HTTP RPS.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cache", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--public-run-dir", type=Path,
                        default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public")
    parser.add_argument("--organizer", type=Path, default=golden.DEFAULT_DATA)
    parser.add_argument("--organizer-spacy", type=Path, default=golden.DEFAULT_CACHE)
    parser.add_argument("--organizer-gliner", type=Path,
                        default=ROOT / "benchmarks/ner-models/gliner25-multi-v1/selected-service.jsonl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode in {"prepare", "cache"} and args.model_path is None:
        parser.error("--model-path is required before and during inference")
    if args.mode == "evaluate" and args.output is None:
        parser.error("--output is required for the offline report")
    return args


def main():
    args = parse_args()
    if not args.run_dir.resolve().is_relative_to((ROOT / "local-data").resolve()):
        raise ValueError("Prepared texts and native caches must stay under ignored local-data")
    public.disable_network()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        protocol = prepare(args)
        print(json.dumps({"prepared": True, "cases": protocol["cases"], "datasets": protocol["dataset_counts"]}))
        return
    protocol = json.loads((args.run_dir / "protocol.json").read_text())
    rows = verify_frozen(args, protocol)
    if args.mode == "cache":
        metadata = cache_model(args, rows, protocol)
        print(json.dumps({"cases": metadata["cases"], "timing": metadata["timing"]}))
        return
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing comparison report")
    report = evaluate(args, rows, protocol)
    golden.save_json(args.output, report)
    print(json.dumps({"report": str(args.output), "cases": len(rows), "corpora": list(report["corpora"])}))


if __name__ == "__main__":
    main()
