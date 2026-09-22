# Code Quality Fixes and CI Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add code quality checks to CI/CD and fix all 119 Sonar/GitLab code quality issues (including cognitive complexity) while preserving exact golden predictions.

**Architecture:** Extend the Ruff rule set in `pyproject.toml` and add a `code-quality` CI job. Fix issues in increasing risk order: confirmed defects → duplicate strings → regex → minor smells → test issues → web issues → cognitive complexity → BaseException. Verify golden equality after every change.

**Tech Stack:** Python 3.12+, Ruff 0.16.8, pytest, GitHub Actions, FastAPI.

## Global Constraints

- Preserve exact golden predictions: after every change, `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json` must produce identical output to the accepted reference.
- Preserve cancellation, deadline, and privacy behavior (existing tests must pass).
- Do not disable or suppress analyzer rules to game the score.
- Do not modify organizer data, `.env`, or captured requests.
- Ruff extended rule set: `E4,E7,E9,F,I,B,C901,C4,SIM,PERF,PLR0911,PLR0912,PLR0913,PLR0915,S`.
- Line length 120, target-version py312.

---

### Task 1: Extend Ruff rule set and add CI code-quality job

**Files:**
- Modify: `pyproject.toml:26-27`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `pyproject.toml` `[tool.ruff.lint] select` extended with maintainability and security rules; a `code-quality` job in CI that runs `ruff check` and `scripts/code_quality.py`.

- [ ] **Step 1: Extend the Ruff select set**

In `pyproject.toml`, change:
```toml
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "B"]
```
to:
```toml
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "B", "C901", "C4", "SIM", "PERF", "PLR0911", "PLR0912", "PLR0913", "PLR0915", "S"]
```

- [ ] **Step 2: Run ruff to see current failures**

Run: `.venv/bin/ruff check seif tests scripts`
Expected: many failures (this is the baseline to fix in later tasks). Do not fix them here.

- [ ] **Step 3: Add a code-quality job to CI**

Append to `.github/workflows/ci.yml` a new job:
```yaml
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
          cache: pip
      - run: pip install -r requirements.lock && pip install -e '.[dev]'
      - run: ruff check seif tests scripts
      - run: git diff --check
      - run: python scripts/package.py
      - name: Generate GitLab Code Quality reports
        run: |
          python scripts/code_quality.py output/seif-pii-source.zip \
            --ruff .venv/bin/ruff --output local-data/code-analysis/ci
      - uses: actions/upload-artifact@v4
        with:
          name: code-quality-reports
          path: local-data/code-analysis/ci/*.json
```

- [ ] **Step 4: Verify CI config is valid YAML**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "ci: add extended ruff gate and code-quality reports job"
```

---

### Task 2: Fix confirmed defects

**Files:**
- Modify: `scripts/local_redis.py:271-276`
- Modify: `web/app.js:57-59`

**Interfaces:**
- Consumes: existing `main()` in `scripts/local_redis.py`, `makeId()` in `web/app.js`.
- Produces: corrected exception ordering in `local_redis.py`; Web Crypto fallback in `app.js`.

- [ ] **Step 1: Fix JSONDecodeError ordering in local_redis.py**

In `scripts/local_redis.py`, the `except (ValueError, RuntimeError)` at line 271 catches `JSONDecodeError` (a `ValueError` subclass) before the intended `except (OSError, KeyError, json.JSONDecodeError)` at line 274. Reorder so `JSONDecodeError` is handled before `ValueError`:

```python
    except json.JSONDecodeError:
        print("Local topology state is unavailable or invalid; no unrelated process was changed.", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, KeyError):
        print("Local topology state is unavailable or invalid; no unrelated process was changed.", file=sys.stderr)
        return 1
```

- [ ] **Step 2: Fix Math.random fallback in web/app.js**

In `web/app.js`, replace `makeId()`:
```javascript
function makeId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return `web-${Date.now()}-${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/pytest tests/test_redis.py tests/test_api.py -q`
Expected: PASS.

- [ ] **Step 4: Verify golden unchanged**

Run: `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output /tmp/golden-task2.json`
Expected: output matches reference (no regression).

- [ ] **Step 5: Commit**

```bash
git add scripts/local_redis.py web/app.js
git commit -m "fix: correct JSONDecodeError ordering and Web Crypto fallback"
```

---

### Task 3: Extract duplicate string constants (S1192)

**Files:**
- Modify: `scripts/ner_service.py:94,103`
- Modify: `scripts/large_input.py:21`
- Modify: `scripts/package.py:12`
- Modify: `seif/detector.py:273`
- Modify: `seif/ner.py:76,100`

**Interfaces:**
- Produces: module-level string constants replacing repeated literals.

- [ ] **Step 1: ner_service.py constants**

In `scripts/ner_service.py`, add module constants and replace the repeated literals:
- `"NER temporarily unavailable."` (4x) → `NER_UNAVAILABLE`
- `"Invalid request."` (3x) → `INVALID_REQUEST`

- [ ] **Step 2: large_input.py constant**

In `scripts/large_input.py`, replace `" neutral"` (3x) with a module constant `NEUTRAL_SUFFIX = " neutral"`.

- [ ] **Step 3: package.py constant**

In `scripts/package.py`, replace `".yaml"` (3x) with `YAML_SUFFIX = ".yaml"`.

- [ ] **Step 4: detector.py constant**

In `seif/detector.py`, replace the literal `"паспорт"` (5x) with a module constant `PASSPORT_WORD = "паспорт"`.

- [ ] **Step 5: ner.py constants**

In `seif/ner.py`, replace:
- `"NER is unavailable"` (4x) → `NER_UNAVAILABLE`
- `"Invalid NER response"` (5x) → `INVALID_NER_RESPONSE`

- [ ] **Step 6: Run tests and golden**

Run: `.venv/bin/pytest -q` then `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output /tmp/golden-task3.json`
Expected: all tests pass, golden unchanged.

- [ ] **Step 7: Commit**

```bash
git add scripts/ner_service.py scripts/large_input.py scripts/package.py seif/detector.py seif/ner.py
git commit -m "refactor: extract duplicate string constants"
```

---

### Task 4: Fix regex issues (S5843, S8786, S6353)

**Files:**
- Modify: `seif/detector.py:70`
- Modify: `seif/person_fields.py:27,52`
- Modify: `seif/structured_fields.py:211`
- Modify: `seif/context_filters.py:17,31,53`
- Modify: `seif/location_fields.py:51`

**Interfaces:**
- Produces: simplified regexes with identical matching behavior.

- [ ] **Step 1: Replace `[0-9]` with `\d` (S6353)**

In `seif/context_filters.py:17,31`, `seif/structured_fields.py:211`, `seif/location_fields.py:51`, replace `[0-9]` character classes with `\d` where the regex is compiled with `re.UNICODE` (or the class is ASCII-only). Verify each replacement does not change matching.

- [ ] **Step 2: Simplify super-linear regexes (S8786)**

In `seif/person_fields.py:52` and `seif/context_filters.py:53`, simplify the regex to remove super-linear performance risk (e.g., replace nested quantifiers with atomic groups or possessive quantifiers) while preserving the matched language. Verify with existing adversarial/custom-regex tests.

- [ ] **Step 3: Simplify high-complexity regexes (S5843)**

In `seif/detector.py:70`, `seif/person_fields.py:27`, `seif/structured_fields.py:211`, reduce regex complexity below 20 by extracting repeated sub-patterns into named groups or constants, without changing the matched language.

- [ ] **Step 4: Run adversarial and custom-regex tests**

Run: `.venv/bin/pytest tests/test_field_adversarial.py tests/test_detector.py tests/test_person_fields.py tests/test_structured_fields.py tests/test_context_filters.py tests/test_location_fields.py -q`
Expected: PASS.

- [ ] **Step 5: Verify golden unchanged**

Run: `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output /tmp/golden-task4.json`
Expected: golden unchanged.

- [ ] **Step 6: Commit**

```bash
git add seif/detector.py seif/person_fields.py seif/structured_fields.py seif/context_filters.py seif/location_fields.py
git commit -m "refactor: simplify regexes without changing matching behavior"
```

---

### Task 5: Fix minor code smells (S3358, S1940, S7519, S7498, S1481, S108, S5713, S7513, S7490)

**Files:**
- Modify: `scripts/evaluate.py:94`
- Modify: `scripts/evaluate_annotations.py:45,59,82,104`
- Modify: `seif/app.py:492`
- Modify: `scripts/benchmark_corpus.py:102,183,276`
- Modify: `scripts/local_redis.py:110`
- Modify: `seif/request_capture.py:93,172`
- Modify: `seif/vault.py:91`
- Modify: `scripts/evaluate_redmadrobot.py:90`
- Modify: `seif/ner.py:136,148`

**Interfaces:**
- Produces: cleaned-up minor code smells with identical behavior.

- [ ] **Step 1: Extract nested conditionals (S3358)**

In `scripts/evaluate.py:94`, `scripts/evaluate_annotations.py:45`, `seif/app.py:492`, extract nested conditional expressions into independent statements (assign to a named variable first).

- [ ] **Step 2: Use opposite operator (S1940)**

In `scripts/evaluate_annotations.py:104`, replace `not (a <= b)` with `a > b` (or equivalent).

- [ ] **Step 3: Use dict.fromkeys (S7519)**

In `scripts/benchmark_corpus.py:276` and `scripts/evaluate_annotations.py:59`, replace dict-comprehension-from-iterable with `dict.fromkeys(...)`.

- [ ] **Step 4: Replace constructor with literal (S7498)**

In `scripts/evaluate_annotations.py:82` and `seif/request_capture.py:93`, replace `list()`/`dict()` constructor calls with `[]`/`{}` literals.

- [ ] **Step 5: Rename unused variable (S1481)**

In `seif/vault.py:91`, rename unused local `blob` to `_`.

- [ ] **Step 6: Fill or remove empty block (S108)**

In `scripts/evaluate_redmadrobot.py:90`, either fill the empty block with a comment explaining intent or remove it.

- [ ] **Step 7: Remove redundant exception classes (S5713)**

In `scripts/benchmark_corpus.py:102,183`, `scripts/local_redis.py:110`, `seif/request_capture.py:172`, remove exception classes that derive from another already caught exception, or reorder handlers.

- [ ] **Step 8: Replace single-task TaskGroup (S7513)**

In `seif/ner.py:148`, if the `TaskGroup` only ever spawns one task, replace it with a direct `await` call.

- [ ] **Step 9: Add checkpoint (S7490)**

In `seif/ner.py:136`, verify the timeout checkpoint. If the `TaskGroup` exit already provides a checkpoint, add a comment documenting it; otherwise add an explicit `await asyncio.sleep(0)` checkpoint.

- [ ] **Step 10: Run tests and golden**

Run: `.venv/bin/pytest -q` then `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output /tmp/golden-task5.json`
Expected: all tests pass, golden unchanged.

- [ ] **Step 11: Commit**

```bash
git add scripts/evaluate.py scripts/evaluate_annotations.py seif/app.py scripts/benchmark_corpus.py scripts/local_redis.py seif/request_capture.py seif/vault.py scripts/evaluate_redmadrobot.py seif/ner.py
git commit -m "refactor: fix minor code smells"
```

---

### Task 6: Fix test issues (S5778, S9073)

**Files:**
- Modify: `tests/test_api.py:327`
- Modify: `tests/test_benchmark_corpus.py:40,47,88,97,109,166,248`
- Modify: `tests/test_ner.py:111,119,124,130`
- Modify: `tests/test_ner_location.py:93,107,112,114`
- Modify: `tests/test_ner_review.py:65`
- Modify: `tests/test_ner_service.py:60,67`
- Modify: `tests/test_sentinel.py:151`
- Modify: `tests/test_synthetic_benchmark.py:18,27`
- Modify: `tests/test_golden_dataset.py:56,68,71`
- Modify: `tests/test_redis.py:63`

**Interfaces:**
- Produces: tests with single-throwing-invocation exception assertions and split composite assertions.

- [ ] **Step 1: Fix S5778 exception tests**

For each S5778 finding, refactor `with pytest.raises(...)` blocks so only one invocation can throw. If multiple calls are inside the block, move the non-throwing setup out, or use `pytest.raises` around only the throwing call.

- [ ] **Step 2: Fix S9073 composite assertions**

For each S9073 finding, split `assert a and b` into separate `assert a` and `assert b` statements.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: split composite assertions and isolate exception tests"
```

---

### Task 7: Fix web issues (S2681, S2486, S7785, S6819, S3358, S3776)

**Files:**
- Modify: `web/app.js:60,69,170,237,246,251,252,262`
- Modify: `web/index.html:61`

**Interfaces:**
- Produces: cleaned-up web JS and HTML with identical behavior.

- [ ] **Step 1: Fix unreachable conditionals (S2681)**

In `web/app.js:237,251,252`, fix statements that are not executed conditionally (missing braces around multi-statement `if` blocks).

- [ ] **Step 2: Handle empty catch (S2486)**

In `web/app.js:246`, add a comment explaining why the exception is ignored, or handle it.

- [ ] **Step 3: Use top-level await (S7785)**

In `web/app.js:262`, replace the async `initialize` function call with top-level `await` if the module context supports it.

- [ ] **Step 4: Use `<output>` element (S6819)**

In `web/index.html:61`, replace the `role="status"` element with `<output>`.

- [ ] **Step 5: Extract nested ternaries (S3358)**

In `web/app.js:69,170`, extract nested ternary operations into independent statements.

- [ ] **Step 6: Reduce cognitive complexity (S3776)**

In `web/app.js:60`, refactor the `api` function (complexity 18) by extracting helper functions.

- [ ] **Step 7: Verify web behavior**

Run: `.venv/bin/pytest tests/test_api.py -q` (web is served by the API; verify no JS syntax errors via `node --check web/app.js` if node is available).
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add web/app.js web/index.html
git commit -m "refactor: fix web code quality issues"
```

---

### Task 8: Refactor cognitive complexity in seif/ (S3776)

**Files:**
- Modify: `seif/app.py:217` (create_app, 178)
- Modify: `seif/detector.py:531,590,637,678,802` (35, 19, 17, 42, 124)
- Modify: `seif/person_fields.py:66,91,119,153,189` (19, 32, 27, 46, 19)
- Modify: `seif/structured_fields.py:106,146,232` (22, 29, 34)
- Modify: `seif/ner.py:87` (_chunk, 31)
- Modify: `seif/config.py:16,125` (28, 29)
- Modify: `seif/request_capture.py:113,204` (24, 16)

**Interfaces:**
- Consumes: existing functions and their callers.
- Produces: refactored functions with cognitive complexity ≤ 15, identical behavior.

**Strategy:** For each function, extract cohesive helper functions. Verify golden equality after EACH function refactor. Do not combine multiple function refactors in one commit without running golden between them.

- [ ] **Step 1: Refactor seif/config.py functions**

Refactor `seif/config.py:16` (28) and `:125` (29) by extracting validation helpers. Run golden after each.

- [ ] **Step 2: Refactor seif/request_capture.py functions**

Refactor `seif/request_capture.py:113` (24) and `:204` (16). Run golden after each.

- [ ] **Step 3: Refactor seif/ner.py _chunk**

Refactor `seif/ner.py:87` (_chunk, 31) by extracting chunk-processing helpers. Run golden.

- [ ] **Step 4: Refactor seif/person_fields.py functions**

Refactor `seif/person_fields.py:66,91,119,153,189` (19, 32, 27, 46, 19). Run golden after each.

- [ ] **Step 5: Refactor seif/structured_fields.py functions**

Refactor `seif/structured_fields.py:106,146,232` (22, 29, 34). Run golden after each.

- [ ] **Step 6: Refactor seif/detector.py functions**

Refactor `seif/detector.py:531,590,637,678,802` (35, 19, 17, 42, 124). Run golden after each. The `detect` function (124) is the largest; extract the `add` closure and each recognizer group into helpers.

- [ ] **Step 7: Refactor seif/app.py create_app**

Refactor `seif/app.py:217` (create_app, 178) by extracting `authorize`, `execute`, `lifespan`, and endpoint handlers into module-level or nested helper functions. Run golden.

- [ ] **Step 8: Run full test suite and golden**

Run: `.venv/bin/pytest -q` then `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output /tmp/golden-task8.json`
Expected: all tests pass, golden unchanged.

- [ ] **Step 9: Commit**

```bash
git add seif/
git commit -m "refactor: reduce cognitive complexity in seif package"
```

---

### Task 9: Refactor cognitive complexity in scripts/ (S3776)

**Files:**
- Modify: `scripts/ner_service.py:198` (create_app, 39)
- Modify: `scripts/capture_requests.py:55` (66)
- Modify: `scripts/benchmark.py:48` (29)
- Modify: `scripts/benchmark_corpus.py:77,224,273` (17, 18, 22)
- Modify: `scripts/evaluate_annotations.py:56` (31)
- Modify: `scripts/evaluate_external.py:49,184` (24, 24)
- Modify: `scripts/evaluate_golden.py:81` (16)
- Modify: `scripts/evaluate_redmadrobot.py:68,157` (21, 18)
- Modify: `scripts/evaluate_scanpatch.py:52,83` (18, 18)
- Modify: `scripts/failover_check.py:36` (20)
- Modify: `scripts/local_redis.py:84,163` (20, 27)
- Modify: `scripts/package.py:28` (19)

**Interfaces:**
- Consumes: existing script functions.
- Produces: refactored functions with cognitive complexity ≤ 15, identical behavior.

**Strategy:** Extract cohesive helper functions. These are evaluation/benchmark scripts; verify with their tests and golden.

- [ ] **Step 1: Refactor scripts/ner_service.py create_app**

Refactor `scripts/ner_service.py:198` (create_app, 39) by extracting endpoint handlers and helpers. Run `pytest tests/test_ner_service.py`.

- [ ] **Step 2: Refactor scripts/capture_requests.py**

Refactor `scripts/capture_requests.py:55` (66) by extracting helpers. Run `pytest tests/test_capture_requests.py`.

- [ ] **Step 3: Refactor scripts/benchmark.py and benchmark_corpus.py**

Refactor `scripts/benchmark.py:48` (29) and `scripts/benchmark_corpus.py:77,224,273` (17, 18, 22). Run `pytest tests/test_benchmark_corpus.py tests/test_benchmark_protocol.py tests/test_synthetic_benchmark.py`.

- [ ] **Step 4: Refactor scripts/evaluate_*.py**

Refactor `scripts/evaluate_annotations.py:56` (31), `scripts/evaluate_external.py:49,184` (24, 24), `scripts/evaluate_golden.py:81` (16), `scripts/evaluate_redmadrobot.py:68,157` (21, 18), `scripts/evaluate_scanpatch.py:52,83` (18, 18). Run `pytest tests/test_annotation_evaluation.py tests/test_golden_evaluation.py`.

- [ ] **Step 5: Refactor scripts/failover_check.py, local_redis.py, package.py**

Refactor `scripts/failover_check.py:36` (20), `scripts/local_redis.py:84,163` (20, 27), `scripts/package.py:28` (19). Run `pytest tests/test_redis.py tests/test_sentinel.py tests/test_package.py`.

- [ ] **Step 6: Run full test suite and golden**

Run: `.venv/bin/pytest -q` then `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output /tmp/golden-task9.json`
Expected: all tests pass, golden unchanged.

- [ ] **Step 7: Commit**

```bash
git add scripts/
git commit -m "refactor: reduce cognitive complexity in scripts"
```

---

### Task 10: Review BaseException (S5754)

**Files:**
- Modify: `seif/app.py:395`
- Modify: `scripts/ner_service.py:271`

**Interfaces:**
- Consumes: existing completion callbacks.
- Produces: narrowed exception handling or documented rationale.

- [ ] **Step 1: Review seif/app.py:395**

In `seif/app.py:395`, the `except BaseException` in `complete_cpu_work` transfers the exception to the awaiting coroutine. Verify whether narrowing to `Exception` preserves cancellation behavior. If `CancelledError` must be propagated, keep `BaseException` and add a comment documenting why.

- [ ] **Step 2: Review scripts/ner_service.py:271**

In `scripts/ner_service.py:271`, apply the same review. If narrowing breaks behavior, document the rationale.

- [ ] **Step 3: Run cancellation/deadline tests**

Run: `.venv/bin/pytest tests/test_api_hardening.py tests/test_ner_service.py -q`
Expected: PASS.

- [ ] **Step 4: Verify golden unchanged**

Run: `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output /tmp/golden-task10.json`
Expected: golden unchanged.

- [ ] **Step 5: Commit**

```bash
git add seif/app.py scripts/ner_service.py
git commit -m "refactor: review BaseException handling in completion callbacks"
```

---

### Task 11: Final verification and code-quality report

**Files:**
- None (verification only).

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass, 0 failures.

- [ ] **Step 2: Run golden verification**

Run: `python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v1.json --output output/golden-quality/final.json`
Expected: golden unchanged.

- [ ] **Step 3: Run extended ruff check**

Run: `.venv/bin/ruff check seif tests scripts`
Expected: no errors (all extended rules pass).

- [ ] **Step 4: Run code_quality.py on packaged source**

Run: `python scripts/package.py && python scripts/code_quality.py output/seif-pii-source.zip --ruff .venv/bin/ruff --output local-data/code-analysis/final`
Expected: reduced findings vs. baseline.

- [ ] **Step 5: Run git diff --check**

Run: `git diff --check`
Expected: no whitespace errors.

- [ ] **Step 6: Update docs/code-analysis.md**

Update the analysis doc to reflect the fixes and the new CI gate. Commit.

```bash
git add docs/code-analysis.md
git commit -m "docs: record code quality fixes and CI gate"
```