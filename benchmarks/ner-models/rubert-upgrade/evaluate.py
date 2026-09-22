"""Compare upgraded SEIF with immutable previous service and raw-text pii-guard.

Reference predictions use the committed pre-upgrade source in an isolated Python
process. Evaluation never edits corpus labels, filters rows, or omits unsupported
predictions. Prediction caches contain offsets, types and input hashes, not text.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "0ab6d876008af27ee015167ea50488be9d488fb7"
UPSTREAM = "24230abb72949a9f85499244dd4f15a0ad0cdd9e"
PUBLIC_SHA = "6058374163fc3978ec4a882d19328e96fad1032d61bbac168ad8a3c78df68a27"
ORGANIZER_SHA = "3bec44d5eafbffc799cd525bbe573f8b0adf84ec82e16271bd406fe92b521e46"
COUNTS = {"organizer": 446, "pii": 1810, "redmadrobot": 2839}
LOCAL = ROOT / "local-data/rubert-upgrade"
ALIASES = {"EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "CREDIT_CARD": "CARD"}
SERVICE_FAMILIES = {
    "PERSON": "PERSON", "CARDHOLDER": "PERSON", "LOCATION": "ADDRESS", "ADDRESS": "ADDRESS",
    "APARTMENT": "ADDRESS", "BIRTH_PLACE": "ADDRESS", "EMAIL": "EMAIL",
    "PHONE": "PHONE", "CARD": "CREDIT_CARD", "CREDIT_CARD": "CREDIT_CARD", "URL": "URL",
    "IP_ADDRESS": "IP_ADDRESS", "IP": "IP_ADDRESS", "PASSPORT": "PASSPORT", "INN": "INN",
    "SNILS": "SNILS", "OMS": "OMS", "DRIVER_LICENSE": "DRIVER_LICENSE",
    "MILITARY_ID": "MILITARY_ID", "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE",
    "POSTAL_CODE": "ADDRESS", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args):
    command = shutil.which("git")
    if command is None:
        raise RuntimeError("git executable required")
    return subprocess.check_output([command, "-C", str(ROOT), *args])  # noqa: S603


def snapshot():
    target = LOCAL / "baseline-source"
    archive = zipfile.ZipFile(io.BytesIO(git("archive", "--format=zip", BASELINE, "seif", "scripts")))
    for item in archive.infolist():
        if item.is_dir():
            continue
        path = target / item.filename
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError("Unsafe archive member")
        content = archive.read(item)
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Immutable baseline snapshot changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
    return target


def sources(source):
    return {str(p.relative_to(source)): sha(p) for folder in ("seif", "scripts")
            for p in sorted((source / folder).glob("*.py"))}


def load_modules(source):
    sys.path.insert(0, str(source))
    from scripts import compare_ner_public, evaluate_golden
    from seif import detector, transform
    return compare_ner_public, evaluate_golden, detector, transform


def load_inputs(args, public, golden):
    path = ROOT / "datasets/golden/organizer-v1/cases.jsonl"
    if sha(path) != ORGANIZER_SHA:
        raise ValueError("Organizer frozen corpus changed")
    cases = golden.load_cases(path)
    rows = [{"key": "organizer/" + key, "id": key, "dataset": "organizer", "split": "all",
             "text": row["text"], "gold": [(e["type"], e["start"], e["end"]) for e in row["entities"]]}
            for key, row in cases.items()]
    protocol_path = args.public_run_dir / "protocol.json"
    if sha(protocol_path) != PUBLIC_SHA:
        raise ValueError("Public frozen protocol changed")
    protocol = json.loads(protocol_path.read_text())
    # Historical source is verified against its committed snapshot, never against
    # intentionally upgraded source. This permits changes without dropping checks.
    baseline_source = snapshot()
    for name, checksum in protocol["source_sha256"].items():
        if sha(baseline_source / name) != checksum:
            raise ValueError("Historical source provenance mismatch: " + name)
    rows.extend(public.load_prepared_inputs(args.public_run_dir / "prepared-inputs.jsonl", protocol))
    frozen = json.loads((ROOT / "local-data/rubert-tensorrt/run-v1/protocol.json").read_text())
    check = {"ordered_membership_sha256": digest([r["key"] for r in rows]),
             "text_sha256": digest({r["key"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in rows}),
             "gold_sha256": digest({r["key"]: sorted(r["gold"]) for r in rows})}
    if any(frozen[key] != value for key, value in check.items()) or dict(Counter(r["dataset"] for r in rows)) != COUNTS:
        raise ValueError("Full frozen corpus membership, text or annotation mismatch")
    for row in rows:
        for label, start, end in row["gold"]:
            if not isinstance(label, str) or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row["text"]):
                raise ValueError("Invalid gold span")
    return rows, cases, check


def load_cache(path, rows, *, native=True):
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if sha(path) != meta["cache_sha256"]:
        raise ValueError("Prediction cache checksum mismatch")
    if native and meta.get("model_revision") != "73be581047bf123dac6505e7b3900ec292942296":
        raise ValueError("Prediction model lineage mismatch")
    data = [json.loads(line) for line in path.read_text().splitlines()]
    if [r["case_id"] for r in data] != [r["key"] for r in rows]:
        raise ValueError("Prediction membership/order mismatch")
    for row, record in zip(rows, data, strict=True):
        if record.get("inference_error") or record.get("gateway_error"):
            raise ValueError("Incomplete prediction cache cannot be scored")
        if record["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("Prediction text mismatch")
        if native:
            validate_spans(record["native"], row["text"], "label")
    if any(meta.get("failures_by_corpus", {}).values()):
        raise ValueError("Prediction metadata reports inference failures")
    return {r["case_id"]: r for r in data}, meta


def validate_spans(spans, text, label_key):
    previous = 0
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        start, end, score = span["start"], span["end"], span.get("score", 1.0)
        if (type(start) is not int or type(end) is not int or not previous <= start < end <= len(text)
                or not isinstance(span[label_key], str) or not span[label_key]
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid, overlapping or nonfinite prediction span")
        previous = end


def encode(spans):
    return [{"start": s.start, "end": s.end, "type": s.type, "score": s.confidence, "reason": s.reason} for s in sorted(spans, key=lambda item: (item.start, item.end, item.type))]


def decode(spans, detector):
    return [detector.Span(s["start"], s["end"], s["type"], s["score"], s["reason"]) for s in spans]


def make_references(args, modules, rows):  # noqa: PLR0915
    public, _, detector, _ = modules
    raw, _ = load_cache(args.old_cache, rows)
    word, _ = load_cache(args.word_cache, rows)
    actual = subprocess.check_output([shutil.which("git"), "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()  # noqa: S603
    dirty = subprocess.check_output([shutil.which("git"), "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()  # noqa: S603
    if actual != UPSTREAM or dirty:
        raise ValueError("Upstream source pin/cleanliness mismatch")
    sys.path.insert(0, str(args.upstream / "src"))
    from pii_guard.config import Config
    from pii_guard.detect import resolve_conflicts, resolve_ml_vs_rules_conflicts
    from pii_guard.engine import NER_ENTITY_MAPPING, Engine
    from pii_guard.entities._email_shape import filter_ner_email_spans
    from presidio_analyzer import RecognizerResult

    engine = Engine(Config(spacy_model=str(args.spacy_model), enable_translit=False,
                           enable_en_numbers=False, enable_base64=False), ner=False)
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    with args.reference.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            key, text = row["key"], row["text"]
            old = public.predictions_for(text, {"rubert": raw[key]["entities"]})
            rules = engine._rules_analyzer.analyze(text=text, language="ru")
            neural = [RecognizerResult(entity_type=NER_ENTITY_MAPPING[e["label"]], start=e["start"],
                                       end=e["end"], score=.7) for e in word[key]["native"]]
            neural = filter_ner_email_spans(neural, text)
            merged = resolve_ml_vs_rules_conflicts(rules, neural)
            guard = resolve_conflicts(merged, text, registry=engine._numeric_recognizer.registry)
            profiles = {"previous_rules": old["rules"], "previous_service": old["rubert_hybrid"],
                        "pii_guard_raw_word": [detector.Span(e.start, e.end, ALIASES.get(e.entity_type, e.entity_type), e.score, "pii-guard") for e in guard],
                        "solo_word": [detector.Span(e["start"], e["end"], e["label"], e["score"], "native") for e in word[key]["native"]]}
            record = {"case_id": key, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "profiles": {name: encode(spans) for name, spans in profiles.items()}}
            for spans in record["profiles"].values():
                validate_spans(spans, text, "type")
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            if index % 500 == 0:
                print(json.dumps({"references_completed": index}), flush=True)
    save(args.reference.with_suffix(".meta.json"), {"cache_sha256": sha(args.reference), "baseline_commit": BASELINE,
         "baseline_source_sha256": sources(LOCAL / "baseline-source"), "upstream_revision": UPSTREAM,
         "word_cache_sha256": sha(args.word_cache), "old_cache_sha256": sha(args.old_cache), "cases": COUNTS,
         "limitations": "Raw original texts; upstream preprocessing excluded. Same frozen word-decoder predictions, fixed priority0.70."})


def load_paper(args):
    path = args.upstream / "tests/quality/paper_score.py"
    committed = subprocess.check_output([shutil.which("git"), "-C", str(args.upstream), "show", f"{UPSTREAM}:tests/quality/paper_score.py"])  # noqa: S603
    if path.read_bytes() != committed:
        raise ValueError("Upstream paper scorer differs from pin")
    spec = importlib.util.spec_from_file_location("upgrade_paper_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paper_metrics(paper, rows, predictions):
    mapping = {**paper.NER_LABEL_TO_FAMILY, **SERVICE_FAMILIES}
    result = {}
    for name, values in predictions.items():
        records = []
        for row in rows:
            gold = [{"type": paper.GOLD_TO_FAMILY["rmr"][kind], "start": start, "end": end} for kind, start, end in row["gold"]]
            pred = [{"type": mapping.get(s.type, "UNSUPPORTED:" + s.type), "start": s.start, "end": s.end} for s in values[row["key"]]]
            records.append(paper.Record(id=row["key"], text=row["text"], gold=gold, pred=pred))
        result[name] = {"strict_miss_decomposition": strict_miss_decomposition(paper, records),
                        "raw_strict_common14_prediction_scope": common14_result(paper, records)}
        for merge in (False, True):
            for strict in (True, False):
                score = paper.score(records, strict=strict, pipeline=merge)
                result[name][("prediction_merge_" if merge else "raw_") + ("strict" if strict else "overlap")] = {
                    "precision": score.micro[0], "recall": score.micro[1], "f1": score.micro[2],
                    "tp": score.tp, "fp": score.fp, "fn": score.fn, "per_type_tp_fp_fn": score.per_type}
    return {"policy": "Gold labels mapped to14 families, original boundaries unchanged; all predictions retained, unsupported types count as FP. Exact/overlap one-to-one upstream scorer. Prediction-only adjacency merge reported separately.",
            "prediction_family_mapping": mapping, "systems": result}


def shared_typed_metrics(paper, rows, predictions, public):
    mapping = {**paper.NER_LABEL_TO_FAMILY, **SERVICE_FAMILIES,
               "NAME": "PERSON", "BANK_CARD_NUMBER": "CREDIT_CARD", "PASSPORT_NUMBER": "PASSPORT",
               "CVV": "CVC", "BIRTH_DATE": "DATE_TIME", "PASSPORT_DATE": "DATE_TIME"}
    raw_gold, merged_gold = {}, {}
    raw_pred = {name: {} for name in predictions}
    merged_pred = {name: {} for name in predictions}
    outside = Counter()
    gold_taxonomy = {mapping.get(kind, kind) for row in rows for kind, *_ in row["gold"]}
    for row in rows:
        key, text = row["key"], row["text"]
        raw_gold[key] = {(mapping.get(kind, kind), start, end) for kind, start, end in row["gold"]}
        merged_gold[key] = public.redmad.merge_adjacent(text, raw_gold[key])
        for name, values in predictions.items():
            raw_pred[name][key] = {(mapping.get(span.type, span.type), span.start, span.end) for span in values[key]}
            merged_pred[name][key] = public.redmad.merge_adjacent(text, raw_pred[name][key])
            outside[name] += sum(kind not in gold_taxonomy for kind, *_ in raw_pred[name][key])
    return {"policy": "Identical explicit family mapping for gold/all systems; unknown types retained. Raw boundaries and symmetric same-family overlap/whitespace-only adjacency merge are separate views. Typed-character scores count all Unicode code points, including punctuation/whitespace, unlike alphanumeric full-mask metrics. No row/type exclusion.",
            "family_mapping": mapping, "gold_families": sorted(gold_taxonomy),
            "predictions_outside_gold_taxonomy_retained_as_fp": outside,
            "original_boundaries": public.scope_metrics(raw_gold, raw_pred),
            "symmetric_whitespace_merge": public.scope_metrics(merged_gold, merged_pred)}


def common14_result(paper, records):
    allowed = set(paper.GOLD_TO_FAMILY["rmr"].values())
    excluded = 0
    scoped = []
    for record in records:
        kept = [span for span in record.pred if span["type"] in allowed]
        excluded += len(record.pred) - len(kept)
        scoped.append(paper.Record(id=record.id, text=record.text, gold=record.gold, pred=kept))
    score = paper.score(scoped, strict=True, pipeline=False)
    return {"policy": "Secondary scope-restricted view: drop predictions outside14 RMR gold families; same unchanged gold, rows and exact scorer. Main all-prediction and full-mask scores retain these predictions.",
            "excluded_predictions": excluded, "precision": score.micro[0], "recall": score.micro[1], "f1": score.micro[2],
            "tp": score.tp, "fp": score.fp, "fn": score.fn, "per_type_tp_fp_fn": score.per_type}


def strict_miss_decomposition(paper, records):
    """Separate masked-but-differently-grouped gold entities from actual exposure."""
    counts, by_type = Counter(), defaultdict(Counter)
    for record in records:
        _, missed, _ = paper.match(record.gold, record.pred, True)
        for gold in missed:
            target = {i for i in range(gold["start"], gold["end"]) if record.text[i].isalnum()}
            same_type, all_types = set(), set()
            for span in record.pred:
                protected = {i for i in range(max(span["start"], gold["start"]), min(span["end"], gold["end"]))
                             if record.text[i].isalnum()}
                all_types.update(protected)
                if span["type"] == gold["type"]:
                    same_type.update(protected)
            if target and target <= same_type:
                reason = "fully_masked_same_type_boundary_difference"
            elif target and target <= all_types:
                reason = "fully_masked_wrong_type"
            else:
                reason = "partly_masked" if target & all_types else "unmasked"
            counts[reason] += 1
            by_type[gold["type"]][reason] += 1
    return {"policy": "Categorize unmatched strict gold entities by actual alphanumeric coverage; diagnostic only, never changes strict score or gold boundaries.",
            "counts": counts, "by_type": by_type}


def error_deltas(rows, predictions, public):
    output = {}
    for reference in ("previous_service", "pii_guard_raw_word"):
        total, by_gold, fp_types, cases = Counter(), defaultdict(Counter), Counter(), []
        for row in rows:
            key, text = row["key"], row["text"]
            gold = public.untyped_gold(text, row["gold"])
            before = public.mask_positions(text, predictions[reference][key])
            after = public.mask_positions(text, predictions["upgraded_service"][key])
            counts = {"new_fp": len((after - before) - gold), "removed_fp": len((before - after) - gold),
                      "new_fn": len((before - after) & gold), "recovered_fn": len((after - before) & gold),
                      "gained_exact": int(after == gold and before != gold), "lost_exact": int(before == gold and after != gold)}
            total.update(counts)
            if any(counts.values()):
                cases.append({"case_id": key, **counts})
            for kind, start, end in row["gold"]:
                positions = {("PII", i) for i in range(start, end) if text[i].isalnum()}
                by_gold[kind].update(new_fn=len(positions & (before - after)), recovered_fn=len(positions & (after - before)))
            for span in predictions["upgraded_service"][key]:
                positions = {("PII", i) for i in range(span.start, span.end) if text[i].isalnum()}
                fp_types[span.type] += len(positions & ((after - before) - gold))
        output[reference] = {"counts": total, "changed_cases": cases, "by_gold_type": by_gold, "new_fp_by_predicted_type": fp_types}
    return output


def build_predictions(args, modules, rows):
    public, _, detector, _ = modules
    refs, meta = load_cache(args.reference, rows, native=False)
    if meta["baseline_commit"] != BASELINE or meta["word_cache_sha256"] != sha(args.word_cache):
        raise ValueError("Reference provenance differs")
    new, _ = load_cache(args.new_cache, rows)
    predictions = defaultdict(dict)
    for row in rows:
        key, text = row["key"], row["text"]
        for name, spans in refs[key]["profiles"].items():
            validate_spans(spans, text, "type")
            predictions[name][key] = decode(spans, detector)
        entities = new[key]["entities"]
        validate_spans(entities, text, "entity_type")
        current = public.predictions_for(text, {"new": entities})
        predictions["upgraded_native_only"][key] = [detector.Span(e["start"], e["end"], e["label"], e["score"], "native")
                                                      for e in new[key]["native"]]
        predictions["upgraded_rules"][key] = current["rules"]
        predictions["upgraded_service"][key] = current["new_hybrid"]
    return predictions


def score(args, modules, rows, cases):
    public = modules[0]
    predictions = build_predictions(args, modules, rows)
    paper = load_paper(args)
    result = {}
    historical = json.loads((ROOT / "benchmarks/ner-models/rubert-tensorrt/comparison.json").read_text())
    guard_historical = json.loads((ROOT / "benchmarks/ner-models/pii-guard-review/guard-ablation.json").read_text())
    for dataset in COUNTS:
        part = [r for r in rows if r["dataset"] == dataset]
        full = public.protection_metrics(part, predictions)
        if full["systems"]["previous_service"] != historical["corpora"][dataset]["full_masking_all_gold_types"]["systems"]["rubert_hybrid"]:
            raise ValueError("Previous service baseline did not reproduce")
        if full["systems"]["pii_guard_raw_word"] != guard_historical["corpora"][dataset]["full_masking_all_gold_types"]["systems"]["guard_word_decoder_cache"]:
            raise ValueError("Previous guard ablation did not reproduce")
        result[dataset] = {"cases": len(part), "full_masking_all_gold_types": full,
                           "error_deltas": error_deltas(part, predictions, public), "restoration_checks_per_profile": len(part),
                           "typed_family_all_types_unfiltered": shared_typed_metrics(paper, part, predictions, public)}
        if dataset == "redmadrobot":
            result[dataset]["paper14"] = paper_metrics(paper, part, predictions)
        if dataset == "organizer":
            for uncertain in (False, True):
                subset = [r for r in part if cases[r["id"]]["uncertain"] == uncertain]
                result[dataset]["uncertain" if uncertain else "certain"] = public.protection_metrics(subset, predictions)
    # Save final upgraded spans for targeted forensic inspection without text.
    if args.predictions_output:
        with args.predictions_output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"case_id": row["key"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                                         "entities": encode(predictions["upgraded_service"][row["key"]])}, ensure_ascii=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("references", "evaluate"), default="evaluate")
    parser.add_argument("--public-run-dir", type=Path, default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public")
    parser.add_argument("--upstream", type=Path, default=ROOT / "local-data/pii-guard-review/upstream")
    parser.add_argument("--spacy-model", type=Path)
    parser.add_argument("--old-cache", type=Path, default=ROOT / "local-data/rubert-tensorrt/run-v1/rubert.jsonl")
    parser.add_argument("--word-cache", type=Path, default=ROOT / "local-data/pii-guard-review/word-decoder/native.jsonl")
    parser.add_argument("--reference", type=Path, default=LOCAL / "reference.jsonl")
    parser.add_argument("--new-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args()
    source = snapshot() if args.mode == "references" else ROOT
    before = sources(source)
    modules = load_modules(source)
    modules[0].disable_network()
    rows, cases, membership = load_inputs(args, modules[0], modules[1])
    if args.mode == "references":
        if not args.spacy_model:
            parser.error("--spacy-model is required for references")
        make_references(args, modules, rows)
        if before != sources(source):
            raise ValueError("Baseline source changed during reference replay")
        print(json.dumps({"reference": str(args.reference), "cases": COUNTS}), flush=True)
        return
    if not args.new_cache or not args.output:
        parser.error("--new-cache and --output are required for evaluation")
    files = {"new_cache": args.new_cache, "new_cache_metadata": args.new_cache.with_suffix(".meta.json"),
             "reference": args.reference, "reference_metadata": args.reference.with_suffix(".meta.json"),
             "runner": Path(__file__), "organizer_manifest": ROOT / "datasets/golden/organizer-v1/manifest.json"}
    file_hashes = {k: sha(p) for k, p in files.items()}
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "baseline_commit": BASELINE,
              "source_sha256": before, "input_sha256": file_hashes, "corpus_identity": membership,
              "limitations": ["Previously examined development corpora; improvements are not independent holdout evidence.",
                              "Organizer labels are provisional AI silver, not official organizer ground truth.",
                              "pii_guard_raw_word is an upstream raw-text rules/merge ablation, not its complete preprocessing pipeline.",
                              "Upstream14 exact scores use identical rows/gold and retain unsupported predictions as FP; not a reproduction of card numbers.",
                              "Cached neural predictions isolate integration/rules; this is not HTTP RPS."],
              "corpora": score(args, modules, rows, cases)}
    if before != sources(source) or file_hashes != {k: sha(p) for k, p in files.items()}:
        raise ValueError("Sources or input caches changed during evaluation; rerun after edits finish")
    report["status"] = "PASS"
    save(args.output, report)
    print(json.dumps({"status": "PASS", "output": str(args.output), "fullmask_f1": {
        dataset: {name: values["f1"] for name, values in result["full_masking_all_gold_types"]["systems"].items()}
        for dataset, result in report["corpora"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
