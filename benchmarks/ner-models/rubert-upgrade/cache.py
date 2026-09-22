"""Freeze and run the upgraded local TensorRT analyzer on all fixed 5095 texts."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
evaluation = importlib.import_module("benchmarks.ner-models.rubert-upgrade.evaluate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "local-data/rubert-tensorrt/model")
    parser.add_argument("--public-run-dir", type=Path, default=ROOT.parent / "seif-pii-gliner/local-data/gliner25-public")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    modules = evaluation.load_modules(ROOT)
    rows, _cases, membership = evaluation.load_inputs(args, modules[0], modules[1])
    from seif.ner_contract import validate_entity
    from seif.rubert_ner import MODEL_NAME, MODEL_REVISION, RubertAnalyzer, fingerprint_checkpoint

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    model_files = fingerprint_checkpoint(args.model_path)
    source_names = ("seif/rubert_decoder.py", "seif/rubert_ner.py", "seif/ner_contract.py", "seif/gliner_ner.py")
    source_hashes = {name: evaluation.sha(ROOT / name) for name in source_names}
    protocol = {"model": MODEL_NAME, "model_revision": MODEL_REVISION, "model_sha256": model_files,
                "source_sha256": source_hashes, "runner_sha256": evaluation.sha(__file__),
                "input_membership": membership, "corpus_counts": dict(Counter(r["dataset"] for r in rows)),
                "decoder": "word", "profile": "native", "confidence_threshold": None,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "tokenizers", "numpy", "tensorrt-cu12")}}
    evaluation.save(args.run_dir / "protocol.json", protocol)
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    analyzer = RubertAnalyzer.from_local(args.model_path, decoder="word", profile="native")
    for _ in range(5):
        analyzer.predict_both("Иван Иванов приехал в Москву.")
    target = args.run_dir / "native.jsonl"
    organizer = args.run_dir / "organizer.jsonl"
    started = time.perf_counter()
    changed = Counter()
    reference = {r["case_id"]: r for r in map(json.loads,
                 (ROOT / "local-data/pii-guard-review/word-decoder/native.jsonl").read_text().splitlines())}
    def signatures(spans):
        return [(s["start"], s["end"], s["label"]) for s in spans]

    with target.open("x", encoding="utf-8") as stream, organizer.open("x", encoding="utf-8") as org:
        for index, row in enumerate(rows, 1):
            output = analyzer.predict_both(row["text"])
            if output["gateway_error"]:
                raise RuntimeError("Native transport failed; refusing partial corpus")
            for entity in output["gateway"]:
                validate_entity(entity, len(row["text"]))
            record = {"case_id": row["key"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                      "native": [{k: v for k, v in span.items() if k != "text"} for span in output["native"]],
                      "entities": output["gateway"]}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if row["dataset"] == "organizer":
                org.write(json.dumps({**record, "case_id": row["id"]}, ensure_ascii=False) + "\n")
            if signatures(record["native"]) != signatures(reference[row["key"]]["native"]):
                changed[row["dataset"]] += 1
            if index % 1000 == 0:
                print(json.dumps({"completed": index, "total": len(rows)}), flush=True)
    elapsed = time.perf_counter() - started
    if source_hashes != {name: evaluation.sha(ROOT / name) for name in source_names}:
        raise ValueError("Inference source changed during run")
    metadata = {**protocol, "protocol_sha256": evaluation.sha(args.run_dir / "protocol.json"),
                "cases": len(rows), "cache_sha256": evaluation.sha(target), "failures_by_corpus": {},
                "upstream_span_differences": dict(changed), "elapsed_seconds_not_rps": elapsed,
                "analyzer": analyzer.metadata()}
    evaluation.save(target.with_suffix(".meta.json"), metadata)
    evaluation.save(organizer.with_suffix(".meta.json"), {**metadata, "cases": 446,
                    "cache_sha256": evaluation.sha(organizer),
                    "dataset_sha256": evaluation.sha(ROOT / "datasets/golden/organizer-v1/cases.jsonl")})
    print(json.dumps({"cases": len(rows), "upstream_span_differences": dict(changed), "cache": str(target)}))


if __name__ == "__main__":
    main()
