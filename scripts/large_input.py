"""Verify a real 100,000-cl100k_base-token synthetic request and exact roundtrip.

Optional test dependency: pip install tiktoken. Its first run fetches the public
tokenizer vocabulary; the service itself never needs network access or tiktoken.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

import httpx
import tiktoken

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8765")
parser.add_argument("--output", type=Path, default=Path("docs/large-input.json"))
args = parser.parse_args()
encoder = tiktoken.get_encoding("cl100k_base")
prefix = "Email: long@example.invalid. "
text = prefix + " neutral" * (100_000 - len(encoder.encode(prefix)))
# Boundary merges can change a token; correct construction without truncating PII.
while len(encoder.encode(text)) < 100_000:
    text += " neutral"
while len(encoder.encode(text)) > 100_000:
    text = text.rsplit(" neutral", 1)[0]
assert len(encoder.encode(text)) == 100_000
payload_id = "large-" + uuid.uuid4().hex
with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
    start = time.perf_counter()
    first = client.post("/process", json={"payload": text, "payload_id": payload_id})
    first.raise_for_status()
    mask_ms = (time.perf_counter() - start) * 1000
    masked = first.json()["result"]
    start = time.perf_counter()
    second = client.post("/process", json={"payload": masked, "payload_id": payload_id})
    second.raise_for_status()
    unmask_ms = (time.perf_counter() - start) * 1000
    result = {"tokenizer": "cl100k_base", "tokens": len(encoder.encode(text)), "characters": len(text),
              "mask_ms": round(mask_ms, 3), "unmask_ms": round(unmask_ms, 3),
              "sensitive_span_masked": "long@example.invalid" not in masked,
              "exact_roundtrip": second.json()["result"] == text,
              "scope": "One synthetic document, isolated HTTP request, not 1000 RPS of large documents"}
    assert result["sensitive_span_masked"] and result["exact_roundtrip"]
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
