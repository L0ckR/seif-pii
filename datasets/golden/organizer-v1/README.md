# Organizer v1: frozen golden regression corpus

This private dataset contains all **446 unique organizer texts** and the previously
locked, adjudicated annotations. Here **golden** means immutable expected results
for regression testing. Annotation quality is **provisional AI / silver quality**:
these are not the organizer's official labels, and they have not received a full
human review. A higher score on this corpus does not establish a higher overall
hackathon score.

## Contents and provenance

`cases.jsonl` contains 369 positive cases, 77 negative cases, 469 entity spans of
19 types, and 117 uncertain cases. Confidence is an annotator's self-assessment,
not a calibrated probability. All texts, entity spans, decisions, uncertainty,
confidence and notes are copied without revisions from the locked source files.
Only the annotated spans are used to compute `expected_masked`; detector
predictions and observed service outputs are not used as labels.

The primary AI annotator annotated all 446 cases. A second AI annotator reviewed
a seeded random sample of 80 independently. Eight disagreements about entities
or uncertainty were adjudicated by a third AI annotator. The primary annotator
and adjudicator did not inspect model predictions. The secondary reviewer had
prior familiarity with detector code from an earlier task; during annotation
the reviewer used only the input texts and annotation protocol. These limits
remain relevant even when all integrity checks pass.

The manifest pins the SHA-256 hashes of every source and of the canonical corpus.
Original capture files remain under ignored `local-data/`. Source IP metadata,
request identifiers, credentials, raw request logs and detector outputs are not
part of this dataset. The corpus itself contains the original personal-data-like
texts and must be treated as private.

On 2026-09-22 the user explicitly requested committing this annotated corpus to
their **private** `L0ckR/seif-pii` repository. That instruction supersedes the
earlier local-only/no-Git instruction for these selected texts and annotations.
It does not authorize public distribution or inclusion in deployment images.
`datasets/` is excluded from Docker, and the source submission ZIP allowlist
excludes this directory. No additional license or ownership claim is implied.

## Record schema

Each UTF-8 JSON line contains these required fields and, for the eight adjudicated
cases, the original optional `adjudication_reason` string:

| Field | Meaning |
| --- | --- |
| `case_id` | Stable identifier `org_0001` through `org_0446`. |
| `text` | Original text, preserved exactly including whitespace and Unicode. |
| `decision` | `positive` when entity spans exist; otherwise `negative`. |
| `entities` | Original ordered list of `{type, start, end, text}` spans. |
| `uncertain` | Explicit boolean ambiguity flag; uncertain cases remain included. |
| `confidence` | Original `high`, `medium` or `low` self-assessment. |
| `notes` | Original annotation rationale, including empty strings. |
| `adjudication_reason` | Original reason for the adjudicator's decision; present in eight cases only. |
| `traffic_weight` | Count of this text's original mask/restore pairs, total 35,845. |
| `expected_masked` | Original string with alphanumeric code points inside annotated spans replaced by `*`. |

Entity offsets are **Unicode code points**, not UTF-8 bytes or JavaScript UTF-16
code units. Intervals are zero-based `[start, end)`, with
`text[start:end] == entity.text` under Python string indexing. Spans do not overlap.
Spaces and punctuation, including those inside spans, are preserved in the
expected mask. Canonical serialization uses sorted object keys, compact JSON
separators, unescaped Unicode, one record per line and a final LF. Input order and
entity list order are preserved. `schema.json` documents the structural schema;
the integrity tests also verify spans, mask derivation, counts and hashes.

## Annotation policy

Annotate personal field values, excluding their labels, surrounding quotes,
explanatory words and terminal sentence punctuation. Include value-internal
formatting such as address abbreviations and separators. For passport series and
numbers separated by explanatory words, annotate the values separately. Every
repeated sensitive occurrence needs its own span. Implausible checksums alone do
not make an explicitly identified personal field negative.

Names of ordinary people and client data are positive. Generic job titles,
organizations, field names without values, public figures in public context,
business addresses and generic geography are negative unless context makes them
an individual's personal data. Obvious standalone email, phone and payment-card
values are positive. Birth dates and passport issue dates require supporting
context; generic scheduling or historical dates are negative. Preserve the
`CARDHOLDER` type in explicit cardholder context. Address components can be
individually typed when presented as separate fields. Record ambiguities in
`uncertain` and `notes`, without discarding difficult cases.

## Reproduction and integrity

From the repository root, with the privately retained original files available:

```bash
python datasets/golden/organizer-v1/build.py \
  --source local-data/organizer-evaluation/20260922T102451Z
python -m pytest tests/test_golden_dataset.py -q
```

The builder checks all pinned source hashes before emitting the two canonical
artifacts. Repeating the command accepts identical files and refuses to overwrite
different content. It prints only aggregate metadata. Tests skip if the entire
dataset directory is absent from the intentionally dataset-free submission ZIP;
a partially present or damaged dataset fails verification.

## Evaluation rules

Report unique-case alphanumeric character precision, recall and F1 as the primary
metrics, plus false positives on negative cases, complete protection of positive
cases, per-type coverage and the certain-case sensitivity analysis. Report
traffic-weighted metrics separately. All 446 cases and all 117 uncertainty flags
must remain represented; uncertainty is not permission to remove hard examples.

There is **no held-out split**. Once detector changes use these texts or labels,
scores are in-sample development/regression results. They do not demonstrate
generalization. An arbitrary later split of already inspected cases would not
create an independent holdout. The official organizer's span-based Levenshtein
calculation and official reference masks are unavailable; character F1 is neither
that metric nor the overall hackathon score.

Version 1 is frozen. Correct actual annotation mistakes in a separately versioned
corpus with reasons and a change log; never rewrite v1 labels merely to match
detector predictions.
