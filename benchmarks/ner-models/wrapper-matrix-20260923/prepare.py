"""Verify and align frozen evaluation texts and the two existing NER caches.

Raw text stays in ignored local-data. Public reports contain only aggregates.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LOCAL = ROOT / "local-data/wrapper-matrix-20260923"
PUBLIC = ROOT.parent / "seif-pii-gliner/local-data/gliner25-public"
RUBERT = ROOT.parent / "seif-pii-rubert/local-data/rubert-upgrade/run-v2/native.jsonl"
SPACY_GOLDEN = ROOT / "benchmarks/ner-models/gliner25-multi-v1/spacy-fresh.jsonl"
SPACY_PUBLIC = PUBLIC / "spacy.jsonl"
EXPECTED = {"organizer": 446, "pii": 1810, "redmadrobot": 2839}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def checked_cache(path: Path, rows: list[dict], *, organizer: bool = False) -> list[dict]:
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    require(sha(path) == meta["cache_sha256"], f"Cache checksum mismatch: {path}")
    records = read(path)
    require(len(records) == len(rows), "Cache case count mismatch")
    require([record["case_id"] for record in records] == [row["id" if organizer else "key"] for row in rows],
            "Cache case IDs mismatch")
    for row, record in zip(rows, records, strict=True):
        require(record["text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest(),
                "Cache text fingerprint mismatch")
        require(not record.get("inference_error") and not record.get("gateway_error"), "Cached inference failed")
    return [{**record, "case_id": row["key"]} for row, record in zip(rows, records, strict=True)]


def main() -> None:
    from scripts import compare_ner_public as public
    from scripts import evaluate_golden as golden

    LOCAL.mkdir(parents=True, exist_ok=True)
    protocol_path = PUBLIC / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    require(sha(protocol_path) == "6058374163fc3978ec4a882d19328e96fad1032d61bbac168ad8a3c78df68a27",
            "Public corpus protocol changed")
    cases = golden.load_cases(ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    rows = [
        {"key": "organizer/" + key, "id": key, "dataset": "organizer", "split": "all",
         "uncertain": case["uncertain"], "text": case["text"],
         "gold": sorted((e["type"], e["start"], e["end"]) for e in case["entities"])}
        for key, case in cases.items()
    ]
    rows.extend({**row, "gold": sorted(row["gold"])} for row in public.load_prepared_inputs(
        PUBLIC / "prepared-inputs.jsonl", protocol,
    ))
    require(dict(Counter(row["dataset"] for row in rows)) == EXPECTED, "Corpus membership changed")
    identity = {
        "ordered_membership_sha256": public.digest_json([row["key"] for row in rows]),
        "text_sha256": public.digest_json({row["key"]: hashlib.sha256(row["text"].encode()).hexdigest()
                                               for row in rows}),
        "gold_sha256": public.digest_json({row["key"]: sorted(row["gold"]) for row in rows}),
    }
    old_protocol = json.loads(RUBERT.with_name("protocol.json").read_text())
    require(identity == old_protocol["input_membership"], "Historical RuBERT corpus identity mismatch")
    from seif.rubert_ner import MODEL_REVISION, PINNED_FILES, fingerprint_checkpoint

    rubert_meta = json.loads(RUBERT.with_suffix(".meta.json").read_text())
    require(rubert_meta["model_revision"] == MODEL_REVISION, "RuBERT model revision mismatch")
    require(rubert_meta["model_sha256"] == PINNED_FILES, "RuBERT model fingerprint mismatch")
    require(rubert_meta["decoder"] == "word" and rubert_meta["profile"] == "native"
            and rubert_meta["confidence_threshold"] is None, "RuBERT decoder/profile mismatch")
    require(not rubert_meta["failures_by_corpus"], "RuBERT cache has failed cases")
    model_files = ROOT.parent / "seif-pii-rubert/local-data/rubert-tensorrt/model"
    require(fingerprint_checkpoint(model_files) == PINNED_FILES, "Local RuBERT weights differ from cache")
    inputs = LOCAL / "inputs.jsonl"
    with inputs.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (LOCAL / "inputs.meta.json").write_text(json.dumps({
        "cases": EXPECTED, "identity": identity, "input_sha256": sha(inputs),
        "golden_certain": sum(not case["uncertain"] for case in cases.values()),
        "source_sha256": {str(path): sha(path) for path in (
            ROOT / "datasets/golden/organizer-v1/cases.jsonl",
            ROOT / "datasets/golden/organizer-v1/manifest.json",
            PUBLIC / "prepared-inputs.jsonl", PUBLIC / "protocol.json",
        )},
    }, ensure_ascii=False, indent=2) + "\n")
    rubert = checked_cache(RUBERT, rows)
    for record in rubert:
        require(len(record["native"]) >= len(record["entities"]), "Missing native RuBERT spans")
    public_rows = rows[len(cases):]
    spacy = [*checked_cache(SPACY_GOLDEN, rows[:len(cases)], organizer=True),
             *checked_cache(SPACY_PUBLIC, public_rows)]
    target = LOCAL / "spacy.jsonl"
    with target.open("w") as stream:
        for record in spacy:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    source = {str(path): sha(path) for path in (SPACY_GOLDEN, SPACY_PUBLIC, RUBERT)}
    (LOCAL / "spacy.meta.json").write_text(json.dumps({
        "cache_sha256": sha(target), "source_sha256": source, "cases": EXPECTED,
        "model": "ru_core_news_sm", "model_version": "3.8.0",
        "candidate_types": sorted({e["entity_type"] for record in spacy for e in record["entities"]}),
        "source_profile": "PERSON/LOCATION only, 0.0 threshold",
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": EXPECTED, "identity": identity, "inputs_sha256": sha(inputs),
                      "spacy_sha256": sha(target)}, indent=2))


if __name__ == "__main__":
    main()
