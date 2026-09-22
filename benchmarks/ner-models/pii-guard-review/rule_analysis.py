"""Replay rule/native/gateway ablations without inference or raw-text reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts import evaluate_golden as golden  # noqa: E402
from seif import context_filters as context  # noqa: E402
from seif import detector  # noqa: E402
from seif.rubert_ner import LOCATION_TYPES, PERSON_TYPES, gateway_entities  # noqa: E402

STAGES = ("native_and_rules_union", "gateway_and_rules_union", "normalized_gateway_and_rules",
          "refined_gateway_and_rules", "service_hybrid")
TRANSITIONS = ("excluded_native_types", "ner_context_or_role_normalization", "shared_context_refinement",
               "overlap_priority_resolution")
EXAMPLES = (
    ("organizer/org_0078", "historical biography still masked by core rules",
     "[ФИО исторического деятеля], родился в [городе] [дате]",
     "Gold excludes the biography. Core name rules still select the name, and the birthplace rule also includes the trailing date."),
    ("organizer/org_0037", "phone-like prefix inside a decimal number",
     "79991110000.5",
     "The phone recognizer can select an eleven-digit prefix without excluding the decimal tail; organizer gold marks this case uncertain."),
    ("redmadrobot/test/row_0230", "short structured rule overrides complete model street",
     "Адрес склада: пр-т Примерный, стр. 8Б.",
     "The rule STREET covers only the name; its higher overlap priority discards a model LOCATION covering the whole street. The prefix loses protection."),
    ("redmadrobot/test/row_0578", "personal 'by the name of' phrase misclassified as a public reference",
     "Здесь жил человек по имени Алёна Никитична.",
     "The public-name predicate matches 'имени', although this construction identifies a person. The native name is removed before overlap resolution."),
    ("redmadrobot/test/row_1646", "institution geography policy differs from benchmark annotation scope",
     "Руководитель МВД [страны] по [федеральному округу] назначен на должность.",
     "The benchmark labels institutional COUNTRY/DISTRICT mentions. Shared context filtering exempts institution geography without personal context."),
    ("redmadrobot/test/row_0124", "tokenized email missed by rules and excluded from model gateway",
     "Почта: a . petrov@example . net",
     "Native EMAIL recognizes spaced benchmark tokenization. Core email syntax misses it, while the gateway projects away EMAIL and other non-name/address types."),
    ("redmadrobot/test/row_0004", "noncanonical card format and checksum versus annotated native type",
     "Номер карты платёжной системы: 1234 - 5678 - 9012 - 3456.",
     "Gold labels a card with spaced separators and an invalid Luhn checksum. Rules miss it; native CREDIT_CARD finds it but the gateway excludes that type. These combined traits do not prove a single regex/checksum gate is the sole cause."),
    ("redmadrobot/test/row_0218", "address rule covers more than annotated components",
     "Адрес: второй этаж, торговый центр, дом 8, [улица], [город], 123456",
     "One broad ADDRESS span covers commercial/scaffolding words and a postcode omitted by the benchmark's component annotations, producing gold-relative false positives."),
)


def geometry(text, spans):
    """Union geometry, safe even when hypothetical ablations have overlaps."""
    return {i for span in spans for i in range(span.start, span.end) if text[i].isalnum()}


def measure(gold, predicted):
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    return {"tp": tp, "fp": fp, "fn": fn, "exact": int(gold == predicted)}


def normalization_reason(text, span, starts, ends):
    if span.type == "PERSON":
        if detector._is_public_name(text, span.start, span.end):
            return "public_name_context"
        changed = detector._trim_ner_person_role(text, span, starts, ends)
        if changed != span:
            return "leading_client_role_trim"
    if span.type == "LOCATION":
        prefix = detector._local_record_prefix(text, span.start, limit=150)
        if detector._is_public_address(prefix, len(prefix)):
            return "public_address_context"
    return "unchanged_or_duplicate"


def refinement_reason(text, span, personal):
    if span.type == "INN":
        if context._corporate_inn(text, span):
            return "corporate_inn_context"
        if context._decimal_inn_fragment(text, span):
            return "decimal_inn_fragment"
    if span.type in {"CITY", "LOCATION"}:
        if context._office_geography(text, span, personal):
            return "institution_geography_without_personal_context"
        if context._geographic_scaffolding(text, span):
            return "geographic_or_corporate_scaffolding"
    return {"PERSON": "name_prose_boundary", "CARDHOLDER": "cardholder_missing_or_prose_boundary",
            "PASSPORT": "document_field_scaffolding", "DRIVER_LICENSE": "document_field_scaffolding",
            "ADDRESS": "address_field_or_prose_boundary", "BIRTH_PLACE": "place_field_or_prose_boundary",
            "PASSPORT_ISSUER": "issuer_boundary"}.get(span.type, "other_refinement")


def replay(text, native, gateway):
    rules = detector.detect(text)
    candidates = [detector.Span(e["start"], e["end"], e["entity_type"], e["score"], "frozen-model") for e in gateway]
    raw_native = [detector.Span(e["start"], e["end"], e["label"], e["score"], "native21") for e in native]
    starts, ends = [], []
    for start, end in sorted((s.start, s.end) for s in rules if s.type == "PERSON"):
        starts.append(start)
        ends.append(max(end, ends[-1] if ends else end))
    existing = {(s.type, s.start, s.end) for s in rules if s.type in {"PERSON", "LOCATION"}}
    accepted = list(detector._accept_ner_candidates(text, candidates, existing, starts, ends).values())
    before_refinement = [*rules, *accepted]
    refined = detector.refine_candidates(text, before_refinement)
    resolved = detector._resolve(refined)
    if resolved != detector.merge_ner_candidates(text, rules, candidates):
        raise ValueError("Diagnostic stage replay differs from actual service merge")
    masked = {index for _, index in public.mask_positions(text, resolved)}
    if masked != geometry(text, resolved):
        raise ValueError("Actual service mask differs from geometric coverage")
    positions = {"rules": geometry(text, rules), "native21": geometry(text, raw_native),
                 "gateway_model_only": geometry(text, candidates),
                 "native_and_rules_union": geometry(text, [*rules, *raw_native]),
                 "gateway_and_rules_union": geometry(text, [*rules, *candidates]),
                 "normalized_gateway_and_rules": geometry(text, before_refinement),
                 "refined_gateway_and_rules": geometry(text, refined), "service_hybrid": masked}
    for before, after in zip(STAGES, STAGES[1:], strict=False):
        if not positions[after] <= positions[before]:
            raise ValueError("Ablation stages unexpectedly add protected alphanumeric positions")
    reasons = defaultdict(set)
    for span in candidates:
        reason = normalization_reason(text, span, starts, ends)
        reasons[reason].update(geometry(text, [span]))
    personal = bool(context._PERSONAL_CONTEXT.search(text)) if any(s.type in {"CITY", "LOCATION"}
                                                                               for s in before_refinement) else False
    for span in before_refinement:
        reasons[refinement_reason(text, span, personal)].update(geometry(text, [span]))
    removed_by_priority = geometry(text, refined) - masked
    for span in refined:
        lost = geometry(text, [span]) & removed_by_priority
        if lost:
            winners = sorted({s.type + ":" + s.reason for s in resolved if s.start < span.end and s.end > span.start})
            reasons["priority:" + span.type + ":" + span.reason + " -> " + ",".join(winners)].update(lost)
    return rules, raw_native, positions, reasons


def traits(text, kind, start, end):
    """Observed policy predicates; co-occurrence is not proof of root cause."""
    value = text[start:end]
    result = []
    if kind in PERSON_TYPES | {"PERSON", "CARDHOLDER"} and detector._is_public_name(text, start, end):
        result.append("public_name_context_predicate")
    if kind in LOCATION_TYPES | {"ADDRESS", "BIRTH_PLACE", "CITIZENSHIP"} and detector._is_public_address(text, start):
        result.append("public_address_context_predicate")
    if kind == "INN":
        digits = re.sub(r"[^0-9]", "", value)
        if len(digits) in {10, 12}:
            result.append("inn_checksum_valid" if detector._valid_inn(digits) else "inn_checksum_invalid")
        if detector._is_public_inn(text, start, value):
            result.append("public_or_corporate_inn_predicate")
    if kind in {"CARD", "CREDIT_CARD"}:
        digits = re.sub(r"[^0-9]", "", value)
        if 12 <= len(digits) <= 19:
            result.append("card_luhn_valid" if detector._luhn(value) else "card_luhn_invalid")
    if kind in {"URL", "IP_ADDRESS", "SNILS", "OMS", "MILITARY_ID", "BIRTH_CERTIFICATE"}:
        result.append("outside_current_rule_type_inventory")
    return result


def add_stat(target, name, positions, gold, case_id):
    if not positions:
        return
    bucket = target.setdefault(name, {"characters": 0, "gold_characters": 0, "non_gold_characters": 0, "case_ids": []})
    if positions:
        bucket["characters"] += len(positions)
        bucket["gold_characters"] += len(positions & gold)
        bucket["non_gold_characters"] += len(positions - gold)
        if case_id not in bucket["case_ids"]:
            bucket["case_ids"].append(case_id)


def attribute_transitions(case, native_spans, positions, stage_reasons, buckets):
    text, gold, key = case["text"], case["gold_positions"], case["key"]
    gold_types = case["gold_types"]
    losses, loss_types, reasons = buckets
    case_losses = {}
    for before, after, reason in zip(STAGES, STAGES[1:], TRANSITIONS, strict=False):
        removed = positions[before] - positions[after]
        add_stat(losses, reason, removed, gold, key)
        case_losses[reason] = {"gold": len(removed & gold), "non_gold": len(removed - gold)}
        for kind, wanted in gold_types.items():
            add_stat(loss_types, reason + " / " + kind, removed & wanted, gold, key)
        if reason == "excluded_native_types":
            for span in native_spans:
                add_stat(reasons, reason + " / " + span.type, removed & geometry(text, [span]), gold, key)
        elif reason in {"ner_context_or_role_normalization", "shared_context_refinement"}:
            allowed = ({"public_name_context", "leading_client_role_trim", "public_address_context"}
                       if reason == "ner_context_or_role_normalization" else
                       {k for k in stage_reasons if not k.startswith("priority:")} -
                       {"public_name_context", "leading_client_role_trim", "public_address_context",
                        "unchanged_or_duplicate"})
            for label in allowed:
                add_stat(reasons, reason + " / " + label, removed & stage_reasons[label], gold, key)
        else:
            for label, covered in stage_reasons.items():
                if label.startswith("priority:"):
                    add_stat(reasons, reason + " / " + label, removed & covered, gold, key)
    return case_losses


def compact_case_lists(result):
    for value in result.values():
        if isinstance(value, dict):
            for bucket in value.values():
                if isinstance(bucket, dict) and "case_ids" in bucket:
                    ids = bucket.pop("case_ids")
                    bucket["cases"] = len(ids)
                    bucket["example_case_ids"] = ids[:8]
    return result


def add_metric_ratios(systems):
    for value in systems.values():
        tp, fp, fn = value["tp"], value["fp"], value["fn"]
        value["precision"] = tp / (tp + fp) if tp + fp else 0
        value["recall"] = tp / (tp + fn) if tp + fn else 0
        value["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0


def analyze(rows, cache):
    systems = defaultdict(Counter)
    losses, loss_types, reasons, rule_fp, rule_fn, gains, trait_stats, gains_by_gold = {}, {}, {}, {}, {}, {}, {}, {}
    losses = {reason: {"characters": 0, "gold_characters": 0, "non_gold_characters": 0, "case_ids": []}
              for reason in TRANSITIONS}
    changes = []
    for row in rows:
        key, text = row["key"], row["text"]
        native = [{**e, "text": text[e["start"]:e["end"]]} for e in cache[key]["native"]]
        gateway = gateway_entities(text, native)
        if gateway != cache[key]["entities"]:
            raise ValueError("Cached gateway differs from the frozen native mapping")
        rules, native_spans, positions, stage_reasons = replay(text, native, gateway)
        gold_types = defaultdict(set)
        for kind, start, end in row["gold"]:
            gold_types[kind].update(i for i in range(start, end) if text[i].isalnum())
        gold = set().union(*gold_types.values()) if gold_types else set()
        for name, predicted in positions.items():
            systems[name].update(measure(gold, predicted))
            systems[name]["cases"] += 1
        for span in rules:
            kind = span.type + " / " + span.reason
            add_stat(rule_fp, kind, geometry(text, [span]) - gold, gold, key)
            add_stat(gains, kind, geometry(text, [span]) & gold - positions["native21"], gold, key)
        for kind, wanted in gold_types.items():
            add_stat(rule_fn, kind, wanted - positions["rules"], gold, key)
            add_stat(gains_by_gold, kind, wanted & positions["rules"] - positions["native21"], gold, key)
        for kind, start, end in row["gold"]:
            missing = {i for i in range(start, end) if text[i].isalnum()} - positions["rules"]
            for trait in traits(text, kind, start, end):
                add_stat(trait_stats, kind + " / " + trait, missing, gold, key)
        case = {**row, "gold_positions": gold, "gold_types": gold_types}
        case_losses = attribute_transitions(case, native_spans, positions, stage_reasons,
                                             (losses, loss_types, reasons))
        relevant = key in {item[0] for item in EXAMPLES}
        if relevant:
            changes.append({"case_id": key, "uncertain": row.get("uncertain"),
                            "rules": measure(gold, positions["rules"]),
                            "native21": measure(gold, positions["native21"]),
                            "service_hybrid": measure(gold, positions["service_hybrid"]),
                            "losses": case_losses})
    add_metric_ratios(systems)
    result = {"systems": dict(systems), "pipeline_removed_characters_by_stage": losses,
            "pipeline_removed_gold_characters_by_stage_and_gold_type": loss_types,
            "pipeline_removal_reason_attribution": reasons,
            "rule_false_positive_characters_by_type_and_reason": rule_fp,
            "rule_false_negative_characters_by_gold_type": rule_fn,
            "rule_correct_characters_not_covered_by_native21_by_type_and_reason": gains,
            "rule_correct_characters_not_covered_by_native21_by_gold_type": gains_by_gold,
            "rule_missed_gold_with_policy_or_checksum_traits": trait_stats,
            "selected_case_diagnostics": changes,
            "rules_increment_over_native21": {
                "correct_gold_characters": systems["native_and_rules_union"]["tp"] - systems["native21"]["tp"],
                "new_false_positive_characters": systems["native_and_rules_union"]["fp"] - systems["native21"]["fp"],
            }}
    return compact_case_lists(result)


def interpretation():
    return {
        "general_implementation_defects": [
            {"kind": "public-name language collision", "case_ids": ["redmadrobot/test/row_0578"],
             "evidence": "The exact normalization stage drops a correctly detected personal name after 'по имени'. The predicate treats the word 'имени' as a public-reference cue."},
            {"kind": "whole-span overlap rejection loses uncovered tails", "case_ids": ["redmadrobot/test/row_0230", "redmadrobot/test/row_0556"],
             "evidence": "Shorter higher-priority rule spans discard an overlapping longer model span wholesale. Some losses are street prefixes; others are remaining house/apartment values. This stage loses 187 gold characters in RMR."},
            {"kind": "number-boundary ambiguity", "case_ids": ["organizer/org_0037"],
             "evidence": "An eleven-digit prefix inside a decimal-looking string is accepted as a phone. The annotation is uncertain, so the intended privacy policy remains debatable."},
        ],
        "policy_or_annotation_scope_differences": [
            {"kind": "PERSON/LOCATION-only model integration", "evidence": "Native structured/contact categories are intentionally omitted before merge. On RMR the rule fallback leaves 19749 otherwise correctly detected gold characters uncovered; on organizer this count is 24."},
            {"kind": "institution geography versus all location annotations", "case_ids": ["redmadrobot/test/row_1646", "redmadrobot/test/row_2080"],
             "evidence": "The service exempts public institution geography while RMR annotates these mentions. This is 44 gold characters removed by shared context refinement, not a model failure."},
            {"kind": "address component versus whole field", "case_ids": ["redmadrobot/test/row_0218"],
             "evidence": "A whole ADDRESS field may contain scaffolding, commercial words or an unannotated postcode beyond the benchmark's component spans; apparent FP counts alone do not establish that every extra masked value is public."},
            {"kind": "checksum policy and artificial formatting", "case_ids": ["redmadrobot/test/row_0004", "redmadrobot/test/row_0033"],
             "evidence": "Gold includes invalid-checksum identifiers and tokenized separators. Bare rules require checksums, while explicit context may bypass them; format matching also matters. Predicate counts are not a single-cause attribution."},
            {"kind": "organizer silver public-biography decision", "case_ids": ["organizer/org_0078"],
             "evidence": "The silver label excludes a historic biography, but rule context does not establish that the mention is public. Name and broad birthplace rules still mask it; authoritative organizer policy is unavailable."},
        ],
        "format_coverage_limitations": [
            {"case_ids": ["redmadrobot/test/row_0124", "redmadrobot/test/row_0033"],
             "evidence": "Spaced email tokens and slash-separated INN values evade the current strict formats. Their native model categories are then excluded by the gateway; these are two distinct stages, not one rule suppression."},
        ],
        "metric_notice": "Every F1 in this report is alphanumeric untyped masking F1. It must not be compared directly with published exact-span or BIO token F1.",
    }


def verify_frozen_scores(corpora):
    report_path = ROOT / "benchmarks/ner-models/rubert-tensorrt/comparison.json"
    reference = json.loads(report_path.read_text())
    for dataset, analysis in corpora.items():
        expected = reference["corpora"][dataset]["full_masking_all_gold_types"]["systems"]
        for actual_name, reference_name in (("rules", "rules"), ("native21", "rubert_native21"),
                                            ("service_hybrid", "rubert_hybrid")):
            actual = analysis["systems"][actual_name]
            for short, full in (("tp", "true_positive"), ("fp", "false_positive"),
                                ("fn", "false_negative"), ("exact", "exact_cases")):
                if actual[short] != expected[reference_name][full]:
                    raise ValueError("Forensic replay differs from the frozen aggregate evaluation")
    return {"matches": True, "reference_sha256": golden.sha256(report_path)}


def load_inputs(args):
    cases = golden.load_cases(ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    rows = [{"key": "organizer/" + key, "dataset": "organizer", "text": c["text"], "uncertain": c["uncertain"],
             "gold": [(e["type"], e["start"], e["end"]) for e in c["entities"]]} for key, c in cases.items()]
    public_rows = [json.loads(line) for line in args.public_inputs.read_text().splitlines()]
    rows.extend(row for row in public_rows if row["dataset"] == "redmadrobot")
    if Counter(r["dataset"] for r in rows) != {"organizer": 446, "redmadrobot": 2839}:
        raise ValueError("Unexpected frozen dataset coverage")
    metadata = json.loads(args.cache.with_suffix(".meta.json").read_text())
    if golden.sha256(args.cache) != metadata["cache_sha256"]:
        raise ValueError("Model cache checksum mismatch")
    protocol_path = args.cache.parent / "protocol.json"
    if golden.sha256(protocol_path) != metadata["protocol_sha256"]:
        raise ValueError("Frozen model protocol checksum mismatch")
    protocol = json.loads(protocol_path.read_text())
    if golden.sha256(args.public_inputs) != protocol["input_sha256"]["public_prepared-inputs.jsonl"]:
        raise ValueError("Frozen public texts or annotations changed")
    for name, expected in protocol["source_sha256"].items():
        if golden.sha256(ROOT / name) != expected:
            raise ValueError("Frozen inference/rule/evaluation source changed: " + name)
    cache = {r["case_id"]: r for r in map(json.loads, args.cache.read_text().splitlines())}
    for row in rows:
        record = cache[row["key"]]
        if record.get("inference_error") or record["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("Inference failure or mismatched cached text")
    return rows, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-inputs", type=Path,
                        default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public/prepared-inputs.jsonl")
    parser.add_argument("--cache", type=Path, default=ROOT / "local-data/rubert-tensorrt/run-v1/rubert.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("rules-analysis.json"))
    args = parser.parse_args()
    rows, cache = load_inputs(args)
    files = [Path(__file__), args.public_inputs, args.cache, args.cache.with_suffix(".meta.json"),
             ROOT / "datasets/golden/organizer-v1/cases.jsonl", *sorted((ROOT / "seif").glob("*.py"))]
    report = {"schema_version": 1, "method": "Sequential cached ablations; all gold types; alphanumeric untyped mask union. "
              "Stages are monotonically removing coverage from the hypothetical native21+rules union. "
              "Only final service uses actual merge/mask; union profiles are diagnostics, not proposed deployments.",
              "hashes": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): golden.sha256(p) for p in files},
              "corpora": {name: analyze([r for r in rows if r["dataset"] == name], cache)
                          for name in ("organizer", "redmadrobot")},
              "interpretation": interpretation(),
              "synthetic_illustrations": [{"case_id": key, "finding": finding, "synthetic_text": text,
                                           "explanation": explanation} for key, finding, text, explanation in EXAMPLES],
              "illustration_notice": "Sentences are synthetic or use redacted placeholders; they illustrate observed cached cases, not new model predictions. No original corpus texts are included.",
              "limitations": ["Organizer annotations are AI silver; 117 of 446 cases are uncertain.",
                              "RedMadRobot 2839 is the previously aligned subset, not the full 2841-row publication protocol.",
                              "Gold-type counts preserve overlapping gold categories; attribution categories may overlap.",
                              "Checksum/public-context traits on missed gold are predicates, not proof that this single gate caused each miss.",
                              "No new inference, tuning, gold edits or production changes; native21 outputs stay distinct from gateway types."]}
    report["frozen_aggregate_replay"] = verify_frozen_scores(report["corpora"])
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    for name, part in report["corpora"].items():
        print(json.dumps({"corpus": name, "systems": part["systems"],
                          "stage_loss": {k: {kk: vv for kk, vv in v.items() if kk != "case_ids"}
                                         for k, v in part["pipeline_removed_characters_by_stage"].items()}}))


if __name__ == "__main__":
    main()
