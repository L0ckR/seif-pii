# RuBERT: full NER transport, word decoding and contextual rules

The service now retains all 21 model categories through a 14-type private NER
contract, rather than dropping everything except names and geography. The HTTP
contract of `/process` and exact restoration remain unchanged. The word decoder
uses the existing pinned TensorRT engine, without loading a second model or
requiring the pii-guard/Presidio runtime. This is an experimental branch, not a
deployment to the existing public service.

**The target of beating pii-guard on every dataset and exact entity metric is
not achieved.** Full masking improves on all three development corpora and
slightly exceeds our raw-text pii-guard reference on RedMadRobot. Exact entity
F1 on RedMadRobot and full masking on PII-Bench still favor the reference.

## Fixed comparison

Final quality: [`quality-v5.json`](quality-v5.json). Previous service is commit
`0ab6d876008af27ee015167ea50488be9d488fb7`. All systems see the same 5,095 texts:
446 organizer, 1,810 PII-Bench and 2,839 RedMadRobot. The two historical
RedMadRobot alignment exclusions remain unchanged. Labels were not edited.
Organizer labels are provisional AI annotations, **not official organizer
ground truth**. All three corpora have been inspected during development and
therefore are not an independent held-out test.

The reference is upstream pii-guard at
`24230abb72949a9f85499244dd4f15a0ad0cdd9e`, its rules, word-decoded model and
merge policy, applied to original text. Normalization, transliteration,
English-number conversion and base64 analysis are excluded to retain original
offsets. This is a **raw-text ablation, not the entire pii-guard pipeline**.
These results do not reproduce or replace the published model-card 88.9 score.
See [evaluation protocol](EVALUATION.md) and [remaining errors](QUALITY_NOTES.md).

### Full masking F1, percent

All gold types and all predictions count; scoring uses protected alphanumeric
character positions in the exact original text. This evaluates what gets hidden,
not whether its entity type and boundaries are correct.

| Corpus | Previous service | Upgraded service | pii-guard raw-text reference |
|---|---:|---:|---:|
| Organizer, 446 | 96.8791 | **97.1285** | 81.8104 |
| Organizer certain subset, 329 | 99.1096 | **99.3156** | See JSON |
| PII-Bench, 1,810 | 74.0731 | 77.8639 | **78.5479** |
| RedMadRobot, 2,839 | 69.7101 | **93.9041** | 93.6748 |

Exact masking cases on organizer: 412 → 417; certain subset: 320 → 324.
On RedMadRobot our precision is 97.2250% versus 94.6488%, but recall is lower:
90.8026% versus 92.7205%. Its exact masking cases are also lower than the
reference, 2,404 versus 2,466. A small F1 lead does not imply dominance on every
privacy or extraction measure.

### Exact entity F1, percent

Same typed families, original gold and predicted boundaries. All out-of-scope
predictions are retained as false positives in the first three rows.

| Corpus / scope | Previous service | Upgraded service | pii-guard raw-text reference |
|---|---:|---:|---:|
| Organizer, all types | **89.5616** | 55.8140 | 33.0563 |
| PII-Bench, all types | **67.0761** | 45.6751 | 45.9606 |
| RedMadRobot, all predictions | 42.7411 | 86.6113 | **87.8188** |
| RedMadRobot, predictions restricted to paper14 | 42.8622 | 86.7601 | **89.8458** |

The output now retains separate model name/address components. Organizer and
PII-Bench often label complete names/addresses, whereas RedMadRobot labels
components. This is a deliberate output-granularity change and a real exact-span
regression on the first two corpora, despite improved masking. Rule remainders
retain every protected letter/digit. Valid complete IP addresses remain atomic.

Token and synthetic modes also create more component replacements. In particular,
the existing synthetic generator can replace each name component with its own
complete dummy name; exact restoration is preserved, but synthetic presentation
is not identical to the previous complete-name output.

A separate diagnostic merges only overlapping/whitespace-adjacent entities of
the same family **symmetrically in gold and predictions for every system**. Its
exact F1 is organizer 89.5616 → 81.5553, PII-Bench 67.0761 → 69.6377 and
RedMadRobot 58.7458 → 87.5917. The remaining organizer decrease is predominantly
comma-separated address components. This secondary view never replaces the raw
scores above. Typed-character metrics and every mapping are included in JSON.

The corrected decoder alone scores 89.5165% strict paper14 on RedMadRobot versus
89.0696% for the upstream word decoder. Its masking F1 is 94.1150%; adding our
rules/context policy lowers that to 93.9041% while adding hackathon-specific
coverage. Therefore model quality and service quality are reported separately.

## Changes and boundaries

- One tokenization per text, first-subword labels, center-owned overlapping
  windows and original Unicode offsets. Entirely BERT-ignored control/PDF words
  retain their positions instead of shifting all subsequent predictions.
- Actual mean first-subword softmax scores replace the upstream constant merge
  priority. These scores are not calibrated entity-correctness probabilities;
  no tuned confidence cutoff was added.
- All types cross authenticated NER HTTP and API policy validation. Malformed,
  oversized, overlapping or incomplete model output fails closed. Chunk overlap
  preserves even a maximum-length URL at a shared boundary.
- Contextual numeric/document candidates accept model-supported components and
  tokenized punctuation without normalizing stored offsets. Local number fields
  still reject order/product identifiers and corporate public INNs.
- Explicit SNILS and validated IPv4/IPv6 rules; zero fractional phone suffixes
  from spreadsheet exports; client-role trimming; birth-certificate typing;
  address fields stop before a new sentence and preserve compatible model tails.
- A generic document-series rule no longer overrides a same-range model driver
  license with the default PASSPORT label. Explicit passport context and custom
  rules remain authoritative. This changes two labels in one uncertain organizer
  case, where its AI annotation explicitly guessed passport; no masking or certain
  organizer result changes, and no case-specific exception was added.
- Atomic model boundaries preserve rule type, provenance and unmatched protected
  remainder. Explicit custom rules remain authoritative; valid whole IPs are
  not fragmented into model parts.

The retained corporate-public-data exceptions are a task policy, not a claim
that every external benchmark agrees. Missing OGRN/KPP/OGRNIP and differing
annotation granularity are documented rather than hidden by case filtering.
No benchmark IDs, source texts or corpus-specific switches occur in runtime code.
Algorithm attribution and the upstream license are in `third_party/pii-guard/`.

## Actual HTTP throughput

[`http-baseline-control.json`](http-baseline-control.json) and
[`http-upgraded-final.json`](http-upgraded-final.json) use the same installed interpreters,
model files, GPU and workload. Each run makes **22,300 masking requests**, 50
complete cycles of the 446 organizer texts, with concurrency 8 and new payload
IDs. Every request completes real model inference; references only verify results.

| Measurement | Previous service | Upgraded service |
|---|---:|---:|
| Successful matching masking RPS | 528.86 | 533.74 |
| p50 / p95 / p99, ms | 14.30 / 20.01 / 23.14 | 14.15 / 19.58 / 22.41 |
| HTTP errors / output mismatches | 0 / 0 | 0 / 0 |
| Completed model calls | 22,300 | 22,300 |

Observed final throughput difference: +0.92%. The preceding implementation before
the document-type priority fix measured 523.21 RPS (−1.07%); its immutable
[`http-upgraded.json`](http-upgraded.json) is also retained. This variation does
not establish a statistically significant speed gain or equivalence. API:
one Python 3.14.7t worker with GIL disabled;
NER: one Python 3.12.11 worker, TensorRT CUDA graphs on RTX 4070 Ti SUPER, four
Torch CPU threads. Memory storage, capture off, local loopback, no prediction
cache. Restore checks and warmup are outside the RPS timing. Stopping NER returns
503 rather than a successful rules-only answer. This is a closed-loop local
measurement, not a measured maximum, Cloudflare throughput or production SLA.

## Reproduction and validation

See [the NER deployment instructions](../../../deploy/service/README.md#опциональный-rubert-tensorrt).
GPU weights are not included in the source ZIP; the normal Compose deployment
continues to provide the CPU spaCy profile. Runtime sources, local import closure,
Docker build inputs, license files and SHA256 manifest are checked by the packager.

```bash
.venv/bin/python benchmarks/ner-models/rubert-upgrade/cache.py \
  --run-dir local-data/rubert-upgrade/a-new-run

local-data/pii-guard-review/model-venv/bin/python \
  benchmarks/ner-models/rubert-upgrade/evaluate.py \
  --new-cache local-data/rubert-upgrade/a-new-run/native.jsonl \
  --output benchmarks/ner-models/rubert-upgrade/a-new-quality.json \
  --predictions-output local-data/rubert-upgrade/a-new-predictions.jsonl

/path/to/api-python benchmarks/ner-models/rubert-upgrade/http_benchmark.py \
  --model-path local-data/rubert-tensorrt/model \
  --reference-cache local-data/rubert-upgrade/a-new-run/organizer.jsonl \
  --ner-python .venv/bin/python --api-python /path/to/api-python \
  --concurrency 8 --repeats 50 \
  --output benchmarks/ner-models/rubert-upgrade/a-new-http.json
```

Outputs are exclusive-create; never overwrite a completed experiment. Source,
input and model fingerprints are retained. Stop editing runtime sources while
an experiment runs. The partial failed `run-v1` decoder cache was never scored;
the successful complete 5,095-row cache is `run-v2`.

Validation: 3,119 core tests passed, 35 skipped for optional dependencies/local
services; 161 model-environment and evaluator checks passed. Ruff passed; all 48
synthetic exact-span fixtures passed. The immutable organizer CI gate reported
no regressions for the existing rules/spaCy profiles. Actual upgraded HTTP
inference, exact response agreement, restore samples and NER failure behavior
passed. Docker image builds and a fresh Sonar run were not performed here.
The actual environment-selected `SEIF_NER_BACKEND=rubert` application was also
started with the pinned local model: readiness advertised all 14 types,
unauthenticated inference returned 401, and authenticated inference returned
both PERSON and EMAIL. This verifies the deployment factory independently of
the instrumentation used by the throughput benchmark.
