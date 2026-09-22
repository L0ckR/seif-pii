"""Read-only overlap audit of the frozen organizer and external corpora.

Requires the previously downloaded datasets and pyarrow. Does not download,
infer, edit annotations, or write input texts/identifiers into the report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_external import FILES as PII_FILES  # noqa: E402
from scripts.evaluate_golden import load_cases, save_json, sha256  # noqa: E402
from scripts.evaluate_redmadrobot import FILES as REDMAD_FILES  # noqa: E402
from scripts.evaluate_redmadrobot import read_rows  # noqa: E402


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def multiset_digest(counts: Counter) -> str:
    encoded = json.dumps(sorted(counts.items()), separators=(",", ":"))
    return digest(encoded)


def count_texts(texts: list[str], *, normalize: bool) -> Counter:
    # Casefold and whitespace collapse only; keep numbers, punctuation,
    # accents and all other characters. This is not a fuzzy/template match.
    return Counter(digest(" ".join(text.split()).casefold() if normalize else text) for text in texts)


def within(counts: Counter) -> dict:
    duplicate_sizes = [size for size in counts.values() if size > 1]
    return {
        "rows": sum(counts.values()), "unique_texts": len(counts),
        "duplicate_groups": len(duplicate_sizes),
        "extra_rows_beyond_first": sum(size - 1 for size in duplicate_sizes),
        "rows_in_duplicate_groups": sum(duplicate_sizes),
        "duplicate_row_pairs": sum(size * (size - 1) // 2 for size in duplicate_sizes),
        "maximum_multiplicity": max(counts.values(), default=0),
        "text_hash_multiset_sha256": multiset_digest(counts),
    }


def cross(first: Counter, second: Counter) -> dict:
    shared = first.keys() & second.keys()
    return {
        "shared_unique_texts": len(shared),
        "matching_rows_first": sum(first[value] for value in shared),
        "matching_rows_second": sum(second[value] for value in shared),
        "matching_row_pairs": sum(first[value] * second[value] for value in shared),
        "shared_text_hash_set_sha256": digest(json.dumps(sorted(shared), separators=(",", ":"))),
    }


def verified_file(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Pinned dataset checksum mismatch: {path.name}")
    return actual


def load_corpora(golden: Path, pii_dir: Path, redmad_dir: Path) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    cases = load_cases(golden)
    pii_texts = []
    source_hashes = {
        "organizer/cases.jsonl": sha256(golden),
        "organizer/manifest.json": sha256(golden.with_name("manifest.json")),
    }
    for split, expected_count in (("domain", 900), ("entity", 910)):
        name = f"{split}-00000-of-00001.parquet"
        path = pii_dir / name
        source_hashes[f"hivetrace/{name}"] = verified_file(path, PII_FILES[name])
        rows = pq.read_table(path, columns=["text"]).to_pylist()
        if len(rows) != expected_count or any(not isinstance(row["text"], str) for row in rows):
            raise ValueError("Unexpected HiveTrace rows")
        pii_texts.extend(row["text"] for row in rows)
    filename, checksum = REDMAD_FILES["test.csv"]
    source_hashes[f"redmadrobot/{filename}"] = verified_file(redmad_dir / filename, checksum)
    aligned, excluded = read_rows(redmad_dir)
    if len(cases) != 446 or len(aligned) != 2839 or len(excluded) != 2:
        raise ValueError("Unexpected pinned organizer or aligned RedMadRobot case count")
    return {
        "organizer446": [row["text"] for row in cases.values()],
        "hivetrace1810": pii_texts,
        "redmadrobot2839": [row["text"] for row in aligned],
    }, source_hashes


def audit(golden: Path, pii_dir: Path, redmad_dir: Path) -> dict:
    corpora, source_hashes = load_corpora(golden, pii_dir, redmad_dir)
    comparisons = {}
    for name, normalize in (("exact_raw", False), ("casefold_whitespace", True)):
        hashes = {key: count_texts(texts, normalize=normalize) for key, texts in corpora.items()}
        comparisons[name] = {
            "within": {key: within(counts) for key, counts in hashes.items()},
            "cross": [{"first": first, "second": second, **cross(hashes[first], hashes[second])}
                      for first, second in combinations(hashes, 2)],
        }
    return {
        "schema_version": 1, "dataset_file_sha256": source_hashes,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                Path(__file__), ROOT / "scripts/evaluate_golden.py", ROOT / "scripts/evaluate_redmadrobot.py",
                ROOT / "scripts/evaluate_external.py",
            )
        },
        "runtime": {"python": sys.version, "pyarrow": importlib.metadata.version("pyarrow")},
        "normalization": "Unicode casefold after joining str.split() tokens with one ASCII space; no other rewriting.",
        "redmadrobot_selection": {"source_rows": 2841, "aligned_rows": 2839, "excluded_rows": 2},
        "comparisons": comparisons,
        "limitations": [
            "No exact overlap does not rule out near-duplicate templates, paraphrases, or shared generated entities.",
            "External corpora were used previously in rule development; they are not a fully untouched system holdout.",
            "This text audit cannot establish which corpora the pretrained NER models saw during training.",
            "Organizer labels and repeated-case traffic weights are not used to compute text overlap.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=ROOT / "datasets/golden/organizer-v1/cases.jsonl")
    parser.add_argument("--pii-dir", type=Path, required=True)
    parser.add_argument("--redmad-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Overlap reports are immutable; choose a new output path")
    report = audit(args.golden, args.pii_dir, args.redmad_dir)
    save_json(args.output, report)
    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
