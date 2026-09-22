"""Read-only attribution of frozen external model deltas; no raw-text output."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts.evaluate_golden import save_json, sha256  # noqa: E402


def point_sets(text, spans):
    result = defaultdict(set)
    for kind, start, end in spans:
        result[kind].update(i for i in range(start, end) if text[i].isalnum())
    return dict(result)


def candidate_contract(rows, caches):
    return {name: {"documents": len(rows), "entities": sum(len(v) for v in cache.values()),
                   "maximum_entities_per_document": max(len(v) for v in cache.values()),
                   "documents_exceeding_2048": sum(len(v) > 2048 for v in cache.values()),
                   "maximum_span_characters": max((e["end"] - e["start"] for v in cache.values() for e in v), default=0),
                   "candidate_types": dict(Counter(e["entity_type"] for v in cache.values() for e in v))}
            for name, cache in caches.items()}


def new_group():
    return {"counts": Counter(), "gold_types": defaultdict(Counter), "gain_prediction_types": Counter(),
            "loss_prediction_types": Counter(), "added_false_positive_types": Counter(),
            "removed_false_positive_types": Counter(), "gained_gold_by_prediction_type": defaultdict(Counter),
            "lost_gold_by_prediction_type": defaultdict(Counter), "changed_case_ids": [],
            "new_negative_false_positive_case_ids": [], "recovered_negative_case_ids": []}


def gold_attribution(group, gold_types, before, after):
    covered_more, covered_less = after - before, before - after
    for kind, values in gold_types.items():
        counts = group["gold_types"][kind]
        counts.update(gold_characters=len(values), reference_tp=len(values & before), candidate_tp=len(values & after),
                      gained_tp=len(values & covered_more), lost_tp=len(values & covered_less))
        counts["tp_delta"] += len(values & after) - len(values & before)
        counts["fn_delta"] += len(values & before) - len(values & after)


def prediction_attribution(group, gold_types, prediction_types, changed, added):
    gold = set().union(*gold_types.values()) if gold_types else set()
    for kind, values in prediction_types.items():
        selected = values & changed
        group["gain_prediction_types" if added else "loss_prediction_types"][kind] += len(selected & gold)
        group["added_false_positive_types" if added else "removed_false_positive_types"][kind] += len(selected - gold)
        table = group["gained_gold_by_prediction_type" if added else "lost_gold_by_prediction_type"]
        for gold_kind, gold_values in gold_types.items():
            if count := len(selected & gold_values):
                table[gold_kind][kind] += count


def record_case(group, row, outputs):
    text, key = row["text"], row["key"]
    gold_types = point_sets(text, row["gold"])
    gold = set().union(*gold_types.values()) if gold_types else set()
    prediction_types = {name: point_sets(text, [(s.type, s.start, s.end) for s in spans])
                        for name, spans in outputs.items()}
    masks = {name: {index for _, index in public.mask_positions(text, spans)} for name, spans in outputs.items()}
    before, after = masks["spacy_hybrid"], masks["gliner_hybrid"]
    gained, lost = after - before, before - after
    counts = group["counts"]
    counts.update(cases=1, gained_tp=len(gained & gold), lost_tp=len(lost & gold),
                  added_fp=len(gained - gold), removed_fp=len(lost - gold),
                  reference_tp=len(before & gold), candidate_tp=len(after & gold),
                  reference_fp=len(before - gold), candidate_fp=len(after - gold),
                  reference_fn=len(gold - before), candidate_fn=len(gold - after),
                  overlapping_gold_type_characters=sum(len(v) for v in gold_types.values()) - len(gold),
                  negative_cases=int(not gold), reference_negative_fp_cases=int(not gold and bool(before)),
                  candidate_negative_fp_cases=int(not gold and bool(after)),
                  became_exact=int(before != gold and after == gold), lost_exact=int(before == gold and after != gold))
    if before != after:
        group["changed_case_ids"].append(key)
    if not gold and not before and after:
        group["new_negative_false_positive_case_ids"].append(key)
    if not gold and before and not after:
        group["recovered_negative_case_ids"].append(key)
    gold_attribution(group, gold_types, before, after)
    prediction_attribution(group, gold_types, prediction_types["gliner_hybrid"], gained, True)
    prediction_attribution(group, gold_types, prediction_types["spacy_hybrid"], lost, False)


def verify_metrics(groups, report):
    for name, group in groups.items():
        counts = group["counts"]
        for label, system in (("reference", "spacy_hybrid"), ("candidate", "gliner_hybrid")):
            expected = report["splits"][name]["full_masking_all_gold_types"]["systems"][system]
            actual = {name: counts[f"{label}_{short}"] for name, short in
                      (("true_positive", "tp"), ("false_positive", "fp"), ("false_negative", "fn"))}
            if any(expected[key] != value for key, value in actual.items()):
                raise ValueError("Attribution differs from frozen full-mask metrics")


def summarize_group(group, dataset):
    counts = group["counts"]
    counts["tp_delta"] = counts["candidate_tp"] - counts["reference_tp"]
    counts["fp_delta"] = counts["candidate_fp"] - counts["reference_fp"]
    counts["fn_delta"] = counts["candidate_fn"] - counts["reference_fn"]
    unsupported = public.redmad.UNMAPPED if dataset == "redmadrobot" else public.pii.ALL_TYPES - public.pii.SUPPORTED
    group["unsupported_gold_types"] = sorted(unsupported)
    group["unsupported_gold_delta"] = {key: sum(value[key] for name, value in group["gold_types"].items() if name in unsupported)
                                       for key in ("gained_tp", "lost_tp", "tp_delta", "fn_delta")}
    group["negative_case_transitions"] = {
        "new_false_positive_cases": len(group["new_negative_false_positive_case_ids"]),
        "recovered_cases": len(group["recovered_negative_case_ids"]),
    }
    return group


def run(args):
    public.disable_network()
    report = json.loads(args.report.read_text())
    protocol = json.loads((args.run_dir / "protocol.json").read_text())
    if protocol != report["protocol"] or sha256(args.run_dir / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("Report protocol differs from frozen inputs")
    rows = public.load_prepared_inputs(args.run_dir / "prepared-inputs.jsonl", protocol)
    public.verify_frozen(args, rows, protocol)
    caches = {name: public.load_cache(args.run_dir / f"{name}.jsonl", rows, report["protocol_sha256"])[0]
              for name in ("spacy", "gliner")}
    groups = {name: new_group() for name in report["splits"]}
    for row in rows:
        outputs = public.predictions_for(row["text"], {name: cache[row["key"]] for name, cache in caches.items()})
        selected = {name: outputs[name] for name in ("spacy_hybrid", "gliner_hybrid")}
        record_case(groups[f"{row['dataset']}/{row['split']}"], row, selected)
        if row["dataset"] == "pii":
            record_case(groups["pii/all"], row, selected)
    verify_metrics(groups, report)
    public.verify_frozen(args, rows, protocol)
    return {"schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_sha256": sha256(args.report), "protocol_sha256": report["protocol_sha256"],
            "audit_source_sha256": sha256(Path(__file__)), "candidate_contract": candidate_contract(rows, caches),
            "groups": {name: summarize_group(group, name.split("/")[0]) for name, group in groups.items()},
            "method": "Replay frozen model outputs through identical current rules/merge/mask; compare only changed alphanumeric mask positions. Gold attribution uses original fine types, including unsupported labels. Attribution does not edit labels, settings or predictions.",
            "limitations": ["False positives are errors against frozen annotations, not independent semantic adjudication.",
                            "Candidate type attribution reports the final merged span type covering changed characters.",
                            "A model can protect an identifier under the wrong entity type; untyped masking gains do not imply correctly typed extraction.",
                            "Historical datasets were used in rule development. This fixed-model transfer check is not a wholly blind pipeline holdout."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pii-data", type=Path, required=True)
    parser.add_argument("--redmad-data", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    save_json(args.output, result)
    print(json.dumps({name: group["counts"] for name, group in result["groups"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
