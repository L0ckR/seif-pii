"""One scoring contract for three corpora, two NER models and three wrappers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "local-data/wrapper-matrix-20260923"
OUTPUT = ROOT / "benchmarks/ner-models/wrapper-matrix-20260923"
MODELS = ("rubert", "spacy")
WRAPPERS = ("seif", "pii_guard", "presidio")
DATASETS = ("organizer", "pii", "redmadrobot")
GROUP_NAMES = {"organizer": "Golden", "pii": "PII-bench", "redmadrobot": "Redmadrobot"}
COARSE = {
    **dict.fromkeys(("FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "NAME", "CARDHOLDER"), "PERSON"),
    **dict.fromkeys(("LOCATION", "ADDRESS", "COUNTRY", "REGION", "DISTRICT", "CITY", "STREET",
                     "HOUSE", "APARTMENT", "POSTAL_CODE", "BIRTH_PLACE"), "ADDRESS"),
    **dict.fromkeys(("EMAIL_ADDRESS",), "EMAIL"),
    **dict.fromkeys(("PHONE_NUMBER",), "PHONE"),
    **dict.fromkeys(("CARD", "CREDIT_CARD", "BANK_CARD_NUMBER"), "CARD"),
    **dict.fromkeys(("BIRTH_DATE", "PASSPORT_DATE", "DATE_TIME"), "DATE_TIME"),
    **dict.fromkeys(("PASSPORT_NUMBER",), "PASSPORT"),
    **dict.fromkeys(("IP",), "IP_ADDRESS"),
}


def read(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def checked_spans(spans, text):
    result = []
    for span in spans:
        start, end, kind = span["start"], span["end"], span["type"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("Invalid span offsets")
        if not isinstance(kind, str) or not kind:
            raise ValueError("Invalid span type")
        result.append((COARSE.get(kind, kind), start, end))
    return set(result)


def _merge_adjacent(text, spans):
    from scripts.evaluate_redmadrobot import merge_adjacent

    return merge_adjacent(text, spans)


def alnum_positions(text, spans):
    return {index for _, start, end in spans for index in range(start, end) if text[index].isalnum()}


def update(counts, gold, predicted):
    counts.update(tp=len(gold & predicted), fp=len(predicted - gold), fn=len(gold - predicted),
                  exact_cases=int(gold == predicted), cases=1)


def metrics(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {**{key: counts[key] for key in ("tp", "fp", "fn", "exact_cases", "cases")},
            "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 6) if tp + fn else 0.0,
            "f1": round(2 * tp / (2 * tp + fp + fn), 6) if 2 * tp + fp + fn else 0.0}


def score():  # noqa: C901 - retain one validated case loop for all metrics
    from scripts.evaluate_golden import sha256

    row_meta = json.loads((LOCAL / "inputs.meta.json").read_text())
    wrapper_meta = json.loads((LOCAL / "wrappers.meta.json").read_text())
    if sha256(LOCAL / "inputs.jsonl") != row_meta["input_sha256"]:
        raise ValueError("Frozen input changed")
    if sha256(LOCAL / "wrappers.jsonl") != wrapper_meta["cache_sha256"]:
        raise ValueError("Wrapper prediction checksum mismatch")
    groups = {group: defaultdict(lambda: defaultdict(Counter))
              for group in (*DATASETS, "organizer_certain", "pii_domain", "pii_entity")}
    cases = Counter()
    source = [LOCAL / "inputs.jsonl", LOCAL / "seif.jsonl", LOCAL / "wrappers.jsonl"]
    streams = [read(path) for path in source]
    names = {f"{wrapper}_{model}" for wrapper in WRAPPERS for model in MODELS}
    for row, seif, others in zip(*streams, strict=True):
        text = row["text"]
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        if (row["key"] != seif["case_id"] or row["key"] != others["case_id"]
                or seif["text_sha256"] != fingerprint or others["text_sha256"] != fingerprint):
            raise ValueError("Evaluation case mismatch")
        predictions = seif["profiles"] | others["profiles"]
        if set(predictions) != names:
            raise ValueError("A wrapper/model combination is missing")
        dataset = row["dataset"]
        part = [dataset]
        if dataset == "organizer" and not row["uncertain"]:
            part.append("organizer_certain")
        if dataset == "pii":
            part.append("pii_" + row["split"])
        for group in part:
            cases[group] += 1
        gold = checked_spans(({"type": kind, "start": start, "end": end}
                              for kind, start, end in row["gold"]), text)
        gold_chars = alnum_positions(text, gold)
        gold_joined = _merge_adjacent(text, gold)
        for profile, spans in predictions.items():
            actual = checked_spans(spans, text)
            actual_chars = alnum_positions(text, actual)
            actual_joined = _merge_adjacent(text, actual)
            for group in part:
                update(groups[group][profile]["character"], gold_chars, actual_chars)
                update(groups[group][profile]["entity_exact"], gold, actual)
                update(groups[group][profile]["entity_joined"], gold_joined, actual_joined)
    if {dataset: cases[dataset] for dataset in DATASETS} != row_meta["cases"]:
        raise ValueError("Corpus counts differ")
    if cases["organizer_certain"] != row_meta["golden_certain"]:
        raise ValueError("Golden confidence membership differs")
    result = {
        "schema_version": 1,
        "corpora": {group: {profile: {metric: metrics(counts) for metric, counts in metrics_group.items()}
                            for profile, metrics_group in profiles.items()}
                    for group, profiles in groups.items()},
        "cases": dict(cases),
        "input_membership": row_meta["identity"],
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source},
        "runner_sha256": {path.name: sha256(path) for path in (
            OUTPUT / "prepare.py", OUTPUT / "seif_run.py", OUTPUT / "wrappers_run.py", OUTPUT / "score.py",
        )},
        "seif_source_sha256": {name: sha256(ROOT / "seif" / name) for name in (
            "detector.py", "ner_boundaries.py", "ner_contract.py", "rubert_ner.py", "transform.py",
        )},
        "wrapper_runtime": wrapper_meta,
        "model_cache": {
            "rubert_sha256": sha256(ROOT.parent / "seif-pii-rubert/local-data/rubert-upgrade/run-v2/native.jsonl"),
            "rubert_meta": json.loads((ROOT.parent / "seif-pii-rubert/local-data/rubert-upgrade/run-v2/native.meta.json").read_text()),
            "spacy_sha256": sha256(LOCAL / "spacy.jsonl"),
            "spacy_meta": json.loads((LOCAL / "spacy.meta.json").read_text()),
        },
        "metric_policy": {
            "character": "micro precision/recall/F1 over alphanumeric Unicode code points covered by any final entity; all original gold labels and all predicted labels counted",
            "entity_exact": "micro typed exact span after one fixed family map, original boundaries",
            "entity_joined": "same family map, symmetric same-type overlap/whitespace adjacency merge of both predictions and gold",
            "unmapped": "unknown prediction types remain their own family and count as false positives",
            "models": "RuBERT word/native no probability cutoff; ru_core_news_sm PERSON/LOCATION cache; no new NER inference",
            "wrappers": "raw-text detection layers; PII Guard preprocessing disabled to preserve offsets; Presidio stock generic rules adapted to ru; SEIF current detector",
        },
    }
    return result


def write_tables(report):
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for wrapper in WRAPPERS:
                metrics_group = report["corpora"][dataset][f"{wrapper}_{model}"]
                rows.append({"dataset": GROUP_NAMES[dataset], "cases": report["cases"][dataset],
                             "model": "RuBERT" if model == "rubert" else "ru_core_news_sm",
                             "wrapper": {"seif": "СЕЙФ", "pii_guard": "PII Guard", "presidio": "Presidio"}[wrapper],
                             **{prefix + "_" + key: value
                                for prefix, data in (("char", metrics_group["character"]),
                                                     ("exact", metrics_group["entity_exact"]),
                                                     ("joined", metrics_group["entity_joined"]))
                                for key, value in data.items()}})
    target = OUTPUT / "table.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    result = score()
    (OUTPUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    rows = write_tables(result)
    for row in rows:
        print(f"{row['dataset']:12} {row['model']:15} {row['wrapper']:10} "
              f"char={row['char_f1']:.4f} exact={row['exact_f1']:.4f} joined={row['joined_f1']:.4f}")


if __name__ == "__main__":
    main()
