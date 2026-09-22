# Golden quality regression

`baseline-75b52e8.json` freezes rule and hybrid quality before tuning against the
versioned private corpus in `datasets/golden/organizer-v1`. It contains aggregate
metrics and case identifiers, not another copy of the original texts.

`ner-cache.jsonl` is **model output, never an annotation source**. The adjacent
metadata pins spaCy, `ru_core_news_sm`, Presidio, input and service hashes. Each
row is bound to the exact original text by SHA-256. The cache allows offline CI
to evaluate changes to rules and merge policy without downloading a model or
calling a server. It does not test NER availability, its HTTP transport or speed.

Run from a private repository checkout:

```bash
python scripts/evaluate_golden.py \
  --baseline benchmarks/golden/baseline-75b52e8.json \
  --output output/golden-quality/current.json
```

The evaluator rejects changed dataset checksums, missing cases, mismatched cache
inputs and incorrect masks. Every prediction must restore exactly. Reports use
unique-case character precision/recall/F1, certain-case sensitivity, type coverage
and traffic-weighted secondary metrics. The baseline gate checks precision,
recall and F1 on the complete and certain-only sets and negative-case false
positives. Existing result files are never silently overwritten.

These are development scores after inspecting the corpus, not held-out evidence
or the overall hackathon score. Neither the data nor this cache is included in
the submission ZIP or container build context. The canonical labels remain fixed.
