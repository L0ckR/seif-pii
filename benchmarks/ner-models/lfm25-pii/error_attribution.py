"""Attribute frozen LFM decoder differences without exporting corpus text.

Run from any directory with the repository environment. No model inference,
downloads, threshold changes, or annotation edits occur here.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import compare_lfm as lfm  # noqa: E402


def positions(text, spans):
    result = defaultdict(set)
    for kind, start, end in spans:
        result[kind].update(index for index in range(start, end) if text[index].isalnum())
    return dict(result)


def union(values):
    return set().union(*values.values()) if values else set()


def native_spans(spans):
    return [(item["type"], item["start"], item["end"]) for item in spans]


def exclusive_labels(typed, selected):
    """Each changed character gets one label or an explicit overlap combination."""
    return Counter(" & ".join(sorted(kind for kind, points in typed.items() if index in points))
                   for index in selected)


def comparison(rows, before, after):
    counts = Counter()
    gold_types = defaultdict(Counter)
    labels = {key: Counter() for key in ("gained_tp", "lost_tp", "added_fp", "removed_fp")}
    cases = {key: [] for key in ("changed", "became_exact", "lost_exact")}
    for row in rows:
        key, text = row["key"], row["text"]
        gold_by_type = positions(text, row["gold"])
        old_by_type, new_by_type = positions(text, before[key]), positions(text, after[key])
        gold, old, new = union(gold_by_type), union(old_by_type), union(new_by_type)
        gained, lost = new - old, old - new
        counts.update(cases=1, gold_characters=len(gold), reference_tp=len(old & gold),
                      candidate_tp=len(new & gold), reference_fp=len(old - gold), candidate_fp=len(new - gold),
                      reference_fn=len(gold - old), candidate_fn=len(gold - new),
                      gained_tp=len(gained & gold), lost_tp=len(lost & gold),
                      added_fp=len(gained - gold), removed_fp=len(lost - gold),
                      overlapping_gold_type_characters=sum(map(len, gold_by_type.values())) - len(gold),
                      negative_cases=int(not gold), reference_negative_fp_cases=int(not gold and bool(old)),
                      candidate_negative_fp_cases=int(not gold and bool(new)),
                      reference_exact_cases=int(old == gold), candidate_exact_cases=int(new == gold))
        for kind, points in gold_by_type.items():
            gold_types[kind].update(gold_characters=len(points), reference_tp=len(points & old),
                                    candidate_tp=len(points & new), gained_tp=len(points & gained),
                                    lost_tp=len(points & lost), tp_delta=len(points & new) - len(points & old))
        for field, typed, selected in (("gained_tp", new_by_type, gained & gold),
                                       ("lost_tp", old_by_type, lost & gold),
                                       ("added_fp", new_by_type, gained - gold),
                                       ("removed_fp", old_by_type, lost - gold)):
            labels[field].update(exclusive_labels(typed, selected))
        if old != new:
            cases["changed"].append(key)
        if old != gold and new == gold:
            cases["became_exact"].append(key)
        if old == gold and new != gold:
            cases["lost_exact"].append(key)
    for short in ("tp", "fp", "fn"):
        counts[short + "_delta"] = counts["candidate_" + short] - counts["reference_" + short]
    for field, attributed in labels.items():
        if sum(attributed.values()) != counts[field]:
            raise ValueError("Prediction attribution did not partition all changed positions")
    return {"counts": counts, "gold_types": gold_types, "prediction_label_attribution": labels,
            "case_ids": cases}


def verify_counts(counts, before, after):
    for prefix, expected in (("reference", before), ("candidate", after)):
        for key, short in (("true_positive", "tp"), ("false_positive", "fp"), ("false_negative", "fn")):
            if counts[prefix + "_" + short] != expected[key]:
                raise ValueError("Audit differs from frozen comparison counts")


def name_script(value):
    scripts = {"cyrillic" if "CYRILLIC" in unicodedata.name(char, "") else
               "latin" if "LATIN" in unicodedata.name(char, "") else "other"
               for char in value if char.isalpha()}
    return "+".join(sorted(scripts)) or "no_letters"


def person_diagnostics(rows, caches):
    output = {}
    for model, cache in caches.items():
        by_script = defaultdict(Counter)
        for row in rows:
            spans = [e for e in cache[row["key"]] if e["entity_type"] == "PERSON"]
            predicted = {(e["start"], e["end"]) for e in spans}
            all_positions = {i for e in spans for i in range(e["start"], e["end"])}
            for kind, start, end in row["gold"]:
                if kind not in {"PERSON", "CARDHOLDER"}:
                    continue
                script = name_script(row["text"][start:end])
                gold = set(range(start, end))
                covered = gold & all_positions
                by_script[script].update(gold_spans=1, gold_characters=len(gold), protected_characters=len(covered),
                                         exact_spans=int((start, end) in predicted),
                                         wholly_missed_spans=int(not covered),
                                         partially_protected_spans=int(bool(covered) and covered != gold),
                                         fully_protected_spans=int(covered == gold))
        output[model] = dict(by_script)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "local-data/lfm25")
    parser.add_argument("--public-run-dir", type=Path, default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("error-attribution.json"))
    options = parser.parse_args()
    lfm.public.disable_network()
    args = SimpleNamespace(run_dir=options.run_dir, public_run_dir=options.public_run_dir,
                           organizer=lfm.golden.DEFAULT_DATA, organizer_spacy=lfm.golden.DEFAULT_CACHE,
                           organizer_gliner=ROOT / "benchmarks/ner-models/gliner25-multi-v1/selected-service.jsonl")
    protocol_path = args.run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    rows = lfm.verify_frozen(args, protocol)
    report_path = Path(__file__).with_name("comparison.json")
    report = json.loads(report_path.read_text())
    protocol_hash = lfm.golden.sha256(protocol_path)
    if report["protocol_sha256"] != protocol_hash or report["protocol"] != protocol:
        raise ValueError("Report belongs to a different frozen protocol")
    native, metadata = lfm.load_lfm_cache(args.run_dir / "lfm.jsonl", rows, protocol, protocol_hash)
    if metadata != report["lfm_cache"]:
        raise ValueError("Report belongs to a different native cache")
    decoders = {decoder: {key: native_spans(value[decoder]) for key, value in native.items()}
                for decoder in lfm.DECODERS}
    attributed = {}
    for name in ("organizer", "pii", "redmadrobot"):
        subset = [row for row in rows if row["dataset"] == name]
        attributed[name] = comparison(subset, decoders["raw"], decoders["hybrid"])
        expected = report["corpora"][name]["native_full_masking_all_types"]["systems"]
        verify_counts(attributed[name]["counts"], expected["raw"], expected["hybrid"])
    _, organizer, public_protocol = lfm.load_inputs(args)
    references, reference_metadata = lfm.load_references(args, rows, organizer, public_protocol)
    if reference_metadata != report["reference_caches"]:
        raise ValueError("Reference cache metadata changed")
    organizer_rows = [row for row in rows if row["dataset"] == "organizer"]
    caches = lfm.build_gateway_caches(organizer_rows, references, native)
    predictions, rejected = lfm.gateway_predictions(organizer_rows, caches)
    if rejected:
        raise ValueError("Organizer gateway contract changed")
    gateway_spans = {name: {key: [(span.type, span.start, span.end) for span in spans]
                             for key, spans in values.items()} for name, values in predictions.items()}
    gateway = {}
    for reference in ("spacy_hybrid", "gliner_hybrid"):
        candidate = "lfm_hybrid_gateway_address_proxy"
        result = comparison(organizer_rows, gateway_spans[reference], gateway_spans[candidate])
        expected = report["corpora"]["organizer"]["gateway_profiles"]
        verify_counts(result["counts"], expected[reference]["unique_case_primary"]["character_metrics"],
                      expected[candidate]["unique_case_primary"]["character_metrics"])
        gateway[reference + "_to_" + candidate] = result
    output = {
        "schema_version": 1, "script_sha256": lfm.golden.sha256(Path(__file__)),
        "comparison_sha256": lfm.golden.sha256(report_path), "protocol_sha256": protocol_hash,
        "cache_sha256": metadata["cache_sha256"],
        "scope": "All native labels versus all original gold; union of alphanumeric positions. No text output or tuning.",
        "prediction_attribution": "Each changed character assigned to one native/gateway label or explicit overlap combination.",
        "native_raw_to_official_hybrid": attributed,
        "organizer_gateway": gateway,
        "organizer_person_by_script": person_diagnostics(
            organizer_rows, {name: cache for name, cache in caches.items() if name.endswith("person_only")}),
        "person_scope": "PERSON and CARDHOLDER gold. Diagnostic character counts include whitespace, like typed_character in comparison.json.",
        "limitations": ["Organizer labels are provisional AI annotations, not the private organizer score.",
                        "These previously inspected corpora are transfer checks, not untouched holdouts.",
                        "Native all-type masking and gateway PERSON/address-proxy profiles have different task scopes.",
                        "An original gold type gains protection even if the native predicted type is semantically incorrect."],
    }
    lfm.golden.save_json(options.output, output)
    print(json.dumps({"audit": str(options.output), "verified_cases": len(rows),
                      "native_deltas": {key: value["counts"] for key, value in attributed.items()}}))


if __name__ == "__main__":
    main()
