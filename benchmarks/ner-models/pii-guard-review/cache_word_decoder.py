"""Cache pinned upstream word decoding on the unchanged 5095-case experiment.

Uses upstream code unchanged with an identity mapping for all 21 native types.
Only the logits provider is replaced for the TensorRT run. No raw texts leave
the local process; no normalization, engine rules or corpus tuning is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import compare_ner_public as public  # noqa: E402
from scripts import compare_onnx as reference  # noqa: E402
from seif.rubert_ner import MODEL_REVISION, NATIVE_TYPES, RubertAnalyzer, fingerprint_checkpoint  # noqa: E402

UPSTREAM_COMMIT = "24230abb72949a9f85499244dd4f15a0ad0cdd9e"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def verify_upstream(path):
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git required to verify upstream source")
    head = subprocess.check_output([git, "-C", str(path), "rev-parse", "HEAD"], text=True).strip()  # noqa: S603
    changes = subprocess.check_output([git, "-C", str(path), "status", "--porcelain"], text=True)  # noqa: S603
    if head != UPSTREAM_COMMIT or changes:
        raise ValueError("Upstream must be the unchanged pinned checkout")
    return {str(p.relative_to(path)): sha(p) for p in sorted((path / "src").rglob("*.py"))}


class TensorRTLogits:
    """Keep upstream decoding intact, providing verified engine logits."""

    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, **encoding):
        import torch

        inputs = {name: value.detach().cpu().numpy() for name, value in encoding.items()}
        logits = self.runtime.backend.run(inputs)
        return SimpleNamespace(logits=torch.from_numpy(logits))


def recognizer(args):
    from pii_guard.ner import recognizer as upstream
    from presidio_analyzer import EntityRecognizer

    if Path(upstream.__file__).resolve() != (args.upstream / "src/pii_guard/ner/recognizer.py").resolve():
        raise ValueError("Wrong upstream package imported")
    mapping = {name: name for name in NATIVE_TYPES}
    if args.backend == "torch-fp32":
        return upstream.TransformerNERRecognizer(
            upstream.TransformerNERConfig(model_name=str(args.source_model), device="cuda"), mapping,
        )
    analyzer = RubertAnalyzer.from_local(args.model_path)
    result = upstream.TransformerNERRecognizer.__new__(upstream.TransformerNERRecognizer)
    EntityRecognizer.__init__(result, supported_entities=sorted(NATIVE_TYPES), supported_language="ru")
    result.config = upstream.TransformerNERConfig(model_name=str(args.model_path), device="cpu")
    result.entity_mapping = mapping
    result._device = "cpu"
    result._tokenizer = analyzer.runtime.tokenizer
    result._id2label = analyzer.runtime.id2label
    result._model = TensorRTLogits(analyzer.runtime)
    return result


def checked_output(model, text):
    spans = sorted(model.analyze(text=text, entities=sorted(NATIVE_TYPES)), key=lambda s: (s.start, s.end))
    last = 0
    for item in spans:
        if item.entity_type not in NATIVE_TYPES or not last <= item.start < item.end <= len(text) or item.score != .7:
            raise ValueError("Invalid upstream native span")
        last = item.end
    return [{"start": s.start, "end": s.end, "label": s.entity_type, "score": s.score} for s in spans]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-run-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--backend", choices=("trt-graph", "torch-fp32"), default="trt-graph")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = reference.load_inputs(args)
    model_files = fingerprint_checkpoint(args.model_path)
    if args.backend == "torch-fp32":
        if args.source_model is None:
            raise ValueError("--source-model required for Torch")
        model_files = {name: sha(args.source_model / name) for name in
                       ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")}
    protocol = {
        "upstream_revision": UPSTREAM_COMMIT, "upstream_sha256": verify_upstream(args.upstream),
        "runner_sha256": sha(__file__), "model_sha256": model_files, "model_revision": MODEL_REVISION,
        "source_model_revision": "c802e8cd26f85d1cf920973ea6f83965a0618d63",
        "backend": args.backend, "decoder": "unmodified pii-guard word-first-subword, center windows, punctuation bridge",
        "native_types": sorted(NATIVE_TYPES), "score": "constant 0.70 arbitration priority, NOT confidence threshold",
        "cpu_threads": 4, "tf32": False, "seed": 20260922, "max_tokens": 512, "stride": 128,
        "cases": len(rows), "corpus_counts": dict(Counter(r["dataset"] for r in rows)),
        "membership_sha256": public.digest_json([r["key"] for r in rows]),
        "text_sha256": public.digest_json({r["key"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in rows}),
        "gold_sha256": public.digest_json({r["key"]: sorted(r["gold"]) for r in rows}),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "tokenizers", "numpy", "presidio-analyzer", "spacy", "tensorrt-cu12")},
        "policy": "Fixed full original corpora. Native output before engine/rules/normalization. Timings are NOT RPS.",
    }
    save(args.run_dir / "protocol.json", protocol)
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = recognizer(args)
    for _ in range(5):
        checked_output(model, "Иван Иванов приехал в Москву.")
    if checked_output(model, ""):
        raise ValueError("Empty input returned entities")
    started = time.perf_counter()
    failures = {}
    cache = args.run_dir / "native.jsonl"
    with cache.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            try:
                native = checked_output(model, row["text"])
            except Exception as exc:
                failures.setdefault(row["dataset"], []).append({"case_id": row["key"], "error": type(exc).__name__})
                native = []
            record = {"case_id": row["key"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(), "native": native}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 500 == 0:
                print(json.dumps({"completed": index, "total": len(rows)}), flush=True)
    torch.cuda.synchronize()
    metadata = {**protocol, "protocol_sha256": sha(args.run_dir / "protocol.json"), "cache_sha256": sha(cache),
                "failures_by_corpus": failures, "elapsed_seconds": time.perf_counter() - started}
    if verify_upstream(args.upstream) != protocol["upstream_sha256"] or sha(__file__) != protocol["runner_sha256"]:
        raise ValueError("Sources changed during run")
    save(cache.with_suffix(".meta.json"), metadata)
    print(json.dumps({"cache": str(cache), "failures": failures, "elapsed_seconds": metadata["elapsed_seconds"]}))
    if failures:
        raise RuntimeError("Partial inference is not eligible for full-corpus scoring")


if __name__ == "__main__":
    main()
