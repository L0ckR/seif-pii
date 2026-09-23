"""Replay both frozen model outputs through the current SEIF detector."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from seif.detector import Span, detect, merge_ner_candidates  # noqa: E402
from seif.rubert_ner import gateway_entities  # noqa: E402
from seif.transform import mask, restore_exact  # noqa: E402

LOCAL = ROOT / "local-data/wrapper-matrix-20260923"
RUBERT = ROOT.parent / "seif-pii-rubert/local-data/rubert-upgrade/run-v2/native.jsonl"


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    inputs = LOCAL / "inputs.jsonl"
    spacy = LOCAL / "spacy.jsonl"
    output = LOCAL / "seif.jsonl"
    with output.open("w") as stream:
        for index, (row, rubert, core) in enumerate(zip(
            records(inputs), records(RUBERT), records(spacy), strict=True,
        ), 1):
            text = row["text"]
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            if not (rubert["case_id"] == core["case_id"] == row["key"]
                    and rubert["text_sha256"] == core["text_sha256"] == fingerprint):
                raise ValueError("Cache membership or text mismatch")
            native = [{**entity, "text": text[entity["start"]:entity["end"]]}
                      for entity in rubert["native"]]
            neural = {"rubert": gateway_entities(text, native, profile="native"), "spacy": core["entities"]}
            profiles = {}
            base = detect(text)
            for model, entities in neural.items():
                candidates = [Span(e["start"], e["end"], e["entity_type"], e["score"], "frozen-ner")
                              for e in entities]
                spans = merge_ner_candidates(text, base, candidates)
                masked, replacements = mask(text, spans, "mask")
                if restore_exact({"masked": masked, "replacements": replacements}) != text:
                    raise ValueError("SEIF mask round-trip failure")
                profiles["seif_" + model] = [
                    {"start": s.start, "end": s.end, "type": s.type, "score": s.confidence}
                    for s in spans
                ]
            stream.write(json.dumps({"case_id": row["key"], "text_sha256": fingerprint,
                                     "profiles": profiles}, ensure_ascii=False) + "\n")
            if index % 500 == 0:
                print(f"SEIF replay: {index}", flush=True)
    print(f"SEIF replay complete: {index} cases", flush=True)


if __name__ == "__main__":
    main()
