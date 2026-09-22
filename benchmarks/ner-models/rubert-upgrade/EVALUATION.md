# Evaluation protocol

The upgrade is evaluated on the same 5,095 previously inspected development
examples: organizer 446, PII-Bench 1,810, RedMadRobot 2,839. Organizer labels are
provisional independent AI annotations, not the organizer's official ground
truth. Two RedMadRobot rows excluded by the original token-alignment protocol
remain excluded; there are no new case exclusions.

`evaluate.py` verifies the immutable organizer manifest and case-file hash, the
public prepared-input hash, and the complete corpus order/text/annotation hashes
from the frozen original TensorRT experiment. Historical source files are checked
against an isolated snapshot of commit `0ab6d876008af27ee015167ea50488be9d488fb7`.
Current source is fingerprinted before and after each new evaluation. Intentional
code changes therefore do not disable the historical checks.

Reference generation runs in a fresh process that imports the committed baseline
snapshot. It creates four sets of predictions: previous rules, previous service,
solo upstream word decoder, and upstream pii-guard rules plus the word decoder.
The last profile uses pinned upstream commit
`24230abb72949a9f85499244dd4f15a0ad0cdd9e` on original texts. It preserves its
label mapping, email gate, NER priority 0.70 and conflict resolution, but excludes
normalization, English-number conversion, transliteration and base64 analysis.
It is a **raw-text ablation, not the entire upstream preprocessing pipeline**.

The report also scores the upgraded decoder alone, separately from the frozen
upstream decoder. This exposes inference changes independently of rule changes.

The upgraded service consumes the actual new model adapter's gateway prediction
cache, including real confidence scores, and replays the current `detect` plus
`merge_ner_candidates` integration. Cache order, text hashes, entity bounds,
non-overlap and finite confidence scores are validated. Failed model calls are
rejected, never scored as empty predictions. Every profile must preserve exact
mask restoration on every evaluated text.

Three separate metric families are reported:

* **Full masking:** all original gold types, all returned predictions, original
  text and alphanumeric character positions. This is closest to SEIF's masking
  behavior. Unsupported labels do not disappear. Organizer certain and uncertain
  subsets are reported separately.
* **Typed families, all three corpora:** the same published mapping is applied
  to gold and every system, with unknown categories retained. Exact spans and
  typed Unicode character positions are reported both with original boundaries
  and with the historical symmetric overlap/whitespace-only adjacency merge.
  These metrics count punctuation and whitespace. The merge applies equally to
  every system and gold; it is not the upstream paper scorer.
* **RedMadRobot paper14:** the pinned upstream scorer maps 21 labels into 14
  families while preserving original gold boundaries. Unknown prediction types
  remain false positives. Raw exact matching is primary; one-to-one overlap and
  prediction-only adjacency merge are separate views. Gold entities are never
  merged. A secondary common14 view explicitly drops out-of-scope predictions
  and reports the number dropped; the primary views retain them. This protocol
  is shared by every compared system but is not claimed
  to reproduce model-card numbers on a different corpus/protocol.

Before accepting new scores, the runner recomputes the stored previous-service
and pii-guard raw-text full-mask results and requires exact aggregate parity with
both earlier reports. Error deltas include new/recovered false negatives,
added/removed false positives and exact-case changes against each reference.
Unmatched exact gold entities are additionally classified by actual protection:
fully covered with the same family but different boundaries, fully covered with
a different family, partially covered, or unprotected. This diagnostic never
changes the original strict score or the gold boundaries.
Reports contain case identifiers and aggregate counts, not original text.

## Reproduction

Use the pinned experiment environment described in
`../pii-guard-review/requirements-word-decoder.txt`, with the local spaCy model.
Reference caches live in ignored `local-data/rubert-upgrade/`.

```bash
local-data/pii-guard-review/model-venv/bin/python \
  benchmarks/ner-models/rubert-upgrade/evaluate.py \
  --mode references \
  --spacy-model /path/to/ru_core_news_sm-3.8.0

local-data/pii-guard-review/model-venv/bin/python \
  benchmarks/ner-models/rubert-upgrade/evaluate.py \
  --new-cache local-data/rubert-upgrade/run-v2/native.jsonl \
  --output benchmarks/ner-models/rubert-upgrade/quality-v1.json \
  --predictions-output local-data/rubert-upgrade/service-v1.jsonl
```

Outputs are exclusive-create. Use new filenames for each iteration. Stop source
editing while an evaluation runs; concurrent changes invalidate the result.
Cached neural comparisons measure integration and rules quality, not HTTP RPS.


The initial `quality-v1.json` and `quality-v2.json` are retained development
iterations. Their secondary paper14 service mapping omitted CARDHOLDER→PERSON
and APARTMENT/BIRTH_PLACE→ADDRESS aliases. Later reports correct these aliases
and include the complete mapping. The primary full-mask scores were unaffected;
use the final report for strict cross-system comparisons rather than mixing
mapping revisions. The failed initial `run-v1` inference is not used for scoring.
The complete neural cache is `run-v2/native.jsonl`.
