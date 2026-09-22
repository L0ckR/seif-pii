# Design: Code quality fixes and CI checks

Date: 2026-09-22

## Goal

1. Add code quality checks to CI/CD (GitHub Actions) using an extended Ruff rule set.
2. Fix all 119 issues identified by the local SonarQube (Sonar way) and GitLab Code Quality (Ruff) analysis of the golden snapshot `4c85827`, including in tests.

## Constraints

- Preserve exact golden predictions: after every refactor, `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json` must produce identical output to the accepted reference.
- Preserve cancellation, deadline, and privacy behavior (existing tests must pass).
- Do not disable or suppress analyzer rules to game the score.
- Do not modify organizer data, `.env`, or captured requests.

## Scope

### 1. CI/CD code quality checks

Add a `code-quality` job to `.github/workflows/ci.yml`:

- `ruff check` with the extended rule set as a gate (fails on errors).
- Run `scripts/code_quality.py` on the packaged source ZIP to emit GitLab Code Quality reports as artifacts.
- Run `git diff --check` for whitespace errors.

Extend `pyproject.toml` `[tool.ruff.lint]` `select` to include the maintainability and security profiles used by the local analyzer:
- Project: `E4,E7,E9,F,I,B` (existing)
- Maintainability: `C901,C4,SIM,PERF,PLR0911,PLR0912,PLR0913,PLR0915`
- Security: `S`

The extended set is applied to `seif/`, `scripts/`, `tests/`, `deploy/`.

### 2. Confirmed defects

- `scripts/local_redis.py:274` — `JSONDecodeError` is a `ValueError` subclass and is intercepted by the earlier `except (ValueError, RuntimeError)`. Handle `JSONDecodeError` before `ValueError`.
- `web/app.js:58` — replace `Math.random` fallback for `payload_id` with Web Crypto (`crypto.getRandomValues`), keeping `crypto.randomUUID` as primary.

### 3. Duplicate strings (S1192)

Extract repeated literals into module-level constants:
- `scripts/ner_service.py:94` "NER temporarily unavailable." (4x), `:103` "Invalid request." (3x)
- `scripts/large_input.py:21` " neutral" (3x)
- `scripts/package.py:12` ".yaml" (3x)
- `seif/detector.py:273` "паспорт" (5x)
- `seif/ner.py:76` "NER is unavailable" (4x), `:100` "Invalid NER response" (5x)

### 4. Regex issues (S5843, S8786, S6353)

- `seif/detector.py:70` (S5843, complexity 26)
- `seif/person_fields.py:27` (S5843, 25), `:52` (S8786 super-linear)
- `seif/structured_fields.py:211` (S5843, 24 + S6353 `[0-9]` → `\d`)
- `seif/context_filters.py:17,31` (S6353), `:53` (S8786)
- `seif/location_fields.py:51` (S6353)

Simplify without changing matching behavior; verify with existing adversarial/custom-regex tests.

### 5. Minor code smells

- S3358 nested conditionals: `scripts/evaluate.py:94`, `scripts/evaluate_annotations.py:45`, `seif/app.py:492`, `web/app.js:69,170`
- S1940 opposite operator: `scripts/evaluate_annotations.py:104`
- S7519 dict fromkeys: `scripts/benchmark_corpus.py:276`, `scripts/evaluate_annotations.py:59`
- S7498 constructor → literal: `scripts/evaluate_annotations.py:82`, `seif/request_capture.py:93`
- S1481 unused variable: `seif/vault.py:91`
- S108 empty block: `scripts/evaluate_redmadrobot.py:90`
- S5713 redundant exception: `scripts/benchmark_corpus.py:102,183`, `scripts/local_redis.py:110`, `seif/request_capture.py:172`
- S7513 TaskGroup with one task: `seif/ner.py:148`
- S7490 checkpoint: `seif/ner.py:136` (verify; likely false positive, add explicit checkpoint if safe)

### 6. Test issues (S5778, S9073)

- S5778 exception tests with multiple throwing invocations: `tests/test_api.py:327`, `tests/test_benchmark_corpus.py:40,47,88,97,248`, `tests/test_ner.py:111,119,124,130`, `tests/test_ner_location.py:93,107,112,114`, `tests/test_ner_review.py:65`, `tests/test_ner_service.py:60,67`, `tests/test_sentinel.py:151`, `tests/test_synthetic_benchmark.py:18,27`
- S9073 composite assertions: `tests/test_benchmark_corpus.py:109,166`, `tests/test_golden_dataset.py:56,68,71`, `tests/test_redis.py:63`

### 7. Web issues

- `web/app.js:237,251,252` (S2681 unreachable conditional), `:246` (S2486 empty catch), `:262` (S7785 top-level await), `:60` (S3776 complexity 18)
- `web/index.html:61` (S6819 use `<output>`)

### 8. Cognitive complexity (S3776) — 37 functions

Refactor the largest functions first, verifying golden equality after each:

| File | Function | Complexity |
|---|---|---|
| `seif/app.py:217` | `create_app` | 178 |
| `seif/detector.py:802` | `detect` | 124 |
| `seif/detector.py:678` | `_merge_ner_candidates` | 42 |
| `seif/detector.py:531` | `_structured_addresses` | 35 |
| `seif/person_fields.py:153` | `_initial_candidates` | 46 |
| `scripts/ner_service.py:198` | `create_app` | 39 |
| `seif/structured_fields.py:232` | `_document_candidates` | 34 |
| `seif/person_fields.py:91` | `_joined_name` | 32 |
| `seif/ner.py:87` | `_chunk` | 31 |
| `scripts/evaluate_annotations.py:56` | | 31 |
| `seif/config.py:16,125` | | 28, 29 |
| `scripts/benchmark.py:48` | | 29 |
| `scripts/benchmark_corpus.py:77,224,273` | | 17, 18, 22 |
| `scripts/capture_requests.py:55` | | 66 |
| `scripts/evaluate_external.py:49,184` | | 24, 24 |
| `scripts/evaluate_golden.py:81` | | 16 |
| `scripts/evaluate_redmadrobot.py:68,157` | | 21, 18 |
| `scripts/evaluate_scanpatch.py:52,83` | | 18, 18 |
| `scripts/failover_check.py:36` | | 20 |
| `scripts/local_redis.py:84,163` | | 20, 27 |
| `scripts/package.py:28` | | 19 |
| `seif/detector.py:590,637` | | 19, 17 |
| `seif/person_fields.py:66,119,189` | | 19, 27, 19 |
| `seif/request_capture.py:113,204` | | 24, 16 |
| `seif/structured_fields.py:106,146` | | 22, 29 |
| `web/app.js:60` | | 18 |

Refactor by extracting cohesive helper functions, not by suppressing the rule.

### 9. BaseException (S5754)

- `seif/app.py:395`, `scripts/ner_service.py:271` — completion callbacks that transfer exceptions to awaiting coroutines. Review and, if safe, narrow to a specific exception while preserving cancellation/privacy behavior. If narrowing would break behavior, document the rationale.

## Verification

After each change:
- `pytest -q` (all tests pass)
- `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json` (identical golden output)
- `ruff check seif tests scripts` with extended rules (no errors)
- `git diff --check`

Final: run `scripts/code_quality.py` on the packaged ZIP to confirm reduced findings.