# Audit #2: diff audit of FIXAUDIT (PR #5)

Scope: `git diff a809220..8842baf` (merge of PR #5, 100 files). Auditor is read-only on code; this file is the only change.

**Verdict: PASS** (no confirmed blocking finding). 0 blocking, 2 major, 6 minor.

## Scope

Covered:
- Code: `experiments/pipeline.py`, `observations.py`, `report.py`, `render_docs.py`, `review1.py`, `run_module2.py`,
  `figures.py`, `lexer/normalize.py`.
- Tests: `tests/test_audit_fixes.py` (mutation-tested the thread probe), other test diffs read.
- CI: `.github/workflows/ci.yml`. Notebook: `notebooks/make_notebook.py` and the committed notebook (cell sources).
- `requirements.txt` pins, on Python 3.10 / 3.11 / 3.12.
- Results: the run JSONs checked only by script (old vs new metrics, compared with the `docs/DECISIONS.md` table);
  a search for hand-typed numbers in the docs.

Not covered (budget):
- Committed notebook *outputs* (only the cell sources were compared), and PNG figures.
- Prose of `CHANGELOG.md`, `docs/HANDOFF*.md`, `docs/PIPELINE.md`, `docs/versions/v4-tokenizer-lineup.md`; only
  checked for stale metric values by script.
- No full-data rerun except go_emotions nltk LR. The QUICK grid and the notebook were not executed end to end.
- Did not run the CI workflow itself (YAML parsed locally, steps run by hand).

Token estimate: about 90k tokens.

## What passed

| check | command | result |
|---|---|---|
| Install pins (py3.11) | `python3.11 -m venv .venv && .venv/bin/pip install -q -r requirements.txt` | OK, no errors |
| Tests | `.venv/bin/python -m pytest -q` | `125 passed in 3.83s` |
| Generated docs | `.venv/bin/python -m experiments.render_docs && git status --porcelain` | empty (clean) |
| Notebook vs generator | `.venv/bin/python notebooks/make_notebook.py --check` | `notebook matches make_notebook.py` |
| Thread pin, full data | go_emotions nltk LR via `run_cell(force=True)`, `OMP_NUM_THREADS=1` and `=4` (pools reported 1 and 4 threads) | both: acc 0.552723311546841, F1 0.3386004128601327, metrics == committed JSON, fingerprint == committed |
| Pin is needed | same, `single_threaded` replaced by `nullcontext`, `OMP_NUM_THREADS=4` | acc 0.5531590413943355 (differs from committed): the pin is what makes the result reproducible |
| OpenMP covered | `threadpool_info()` lists openblas (x2, numpy and scipy) and libgomp (sklearn); all three are limited by `threadpool_limits(1)` | OK |
| Probe test is real (4 CPUs) | scratch copy with `with single_threaded():` in `run_one` replaced by `if True:`; `pytest -k single_threaded` | `1 failed` (mutation caught) |
| Cache invalidation | scratch copy, warm cache then one change each (`fp_driver2.sh`) | recomputed after edits to `lexer/normalize.py`, `lexer/token_spec.py`, `lexer/dfa/scanner.py`, `experiments/tokenizers.py`, `lexer/slang_table.csv`, `VECTORIZER_PARAMS`, seed, subsample size; reused after edits to `lexer/prototype/`, `experiments/observations.py` (not result-relevant: correct) |
| Changed metrics | script diff of every run JSON vs `a809220` | exactly the 6 LR JSONs / 12 values listed in `docs/DECISIONS.md`, values match |
| Stale numbers | every old value (as 2/1-decimal %, 3/4/6-decimal fraction) searched in tracked `.md/.py/.ipynb` | only coincidental substrings of other, current cells; no hand-typed metric outside generated blocks |
| `obs:attribution` | recomputed from JSON: ag_news +1.50 = +1.50 / +0.00; go_emotions +17.58 = +17.49 / +0.09; tweet_topic -0.26 = +1.32 / -1.59 | matches `results/REPORT_NOTES.md` text |
| QUICK writes only under `results/quick` | `review1.results_dir(True)` used by every section | OK |
| Appendix read-only | notebook appendix calls `render_all_tables` (writes nothing; test `test_render_all_tables_writes_nothing`) | OK |
| Narrowed `except` | `report.render_all_tables`, `observations.observation` now catch only `FileNotFoundError`; `test_render_does_not_hide_bugs` checks a `KeyError` is raised | OK |
| CI YAML | `yaml.safe_load` | jobs `tests`, `notebook`; tests job runs without `dot` (compiler skips PNG when `shutil.which("dot")` is None; tests pass here without graphviz) |
| Pins on py3.12 | `pip download --no-deps --only-binary=:all: --python-version 3.12` for every pin | all 15 resolve (cp312 wheels) |

## Findings

| id | severity | file:line | evidence | suggested fix | status |
|---|---|---|---|---|---|
| A2-M1 | major | `requirements.txt:5` (also :6, :7, :8, :11, :24) | `pip download --no-deps --only-binary=:all: --python-version 3.10 ...` fails for `numpy==2.4.6` (3.10 tops out at 2.2.6), `pandas==3.0.6` (2.3.3), `scipy==1.17.1`, `scikit-learn==1.9.1` (1.7.2), `matplotlib==3.11.2`, `ipykernel==7.4.0`. Python 3.11 and 3.12 resolve. So `pip install -r requirements.txt` fails on any Python 3.10 runtime (older Colab images, local 3.10). | Either state `Requires Python >= 3.11` in README and the notebook (and fail early in cell 1 with a clear message), or split: keep the exact pins as `requirements-lock.txt` (CI, reproducing committed full results) and make `requirements.txt` ranges, e.g. `numpy>=1.26,<3`, `pandas>=2.2,<4`, `scipy>=1.11,<2`, `scikit-learn>=1.5,<2`, `matplotlib>=3.8,<4`, `ipykernel>=6`, keeping `threadpoolctl>=3.1`, `nltk==3.10.3`, `ply==3.11` exact. The sklearn/nltk versions are already in the config fingerprint, so a different version recomputes rather than mixing. | CONFIRMED |
| A2-M2 | major | `notebooks/make_notebook.py:40` (`%pip install -q -r requirements.txt`) with `requirements.txt:5-11,24` | On Colab (3.12) the exact pins force reinstalling the preinstalled numpy, pandas, scipy, scikit-learn, matplotlib and ipykernel inside a running kernel. If any of them is already imported by the runtime, the new versions only take effect after a restart (mixed numpy/sklearn binaries until then), and `pandas==3.0.6` conflicts with Colab's own `google-colab` pandas pin (resolver error lines, possible breakage of Colab features). Not reproduced: no Colab here. | Same as A2-M1 (ranges in the file the notebook installs), or in cell 1 install only what Colab lacks (`threadpoolctl nltk datasets ply`) and print the versions in use. If exact pins stay, add a restart prompt after `%pip` when numpy's version changed. | PLAUSIBLE |
| A2-m1 | minor | `experiments/classifiers.py:13` | `make_vectorizer` hard-codes `TfidfVectorizer(tokenizer=..., token_pattern=None, lowercase=True, min_df=2)` while the fingerprint and the JSON record `VECTORIZER_PARAMS` (`classifiers.py:7`). Patching `make_vectorizer` to `min_df=1` at runtime: `make_vectorizer min_df=1 at runtime (dict unchanged): reused`. A source edit is still caught (the file is hashed), but the recorded params can drift from the real ones. | `TfidfVectorizer(tokenizer=tokenizer, **{k: tuple(v) if k == "ngram_range" else v for k, v in VECTORIZER_PARAMS.items()})`, so the dict is the single source. | CONFIRMED |
| A2-m2 | minor | `experiments/pipeline.py:61`, `experiments/data.py:134`, `:147` | The config fingerprint uses `ds.summary()` (names, sizes, labels, subsample), not the data itself; go_emotions and ag_news are loaded with `load_dataset(...)` without a `revision`. Toy check: same sizes, upper-cased texts: `same sizes, different text content: reused`. An upstream dataset edit that keeps the split sizes would reuse a stale JSON. | Add a hash of the train/test texts and labels (or the HF commit sha) to `Dataset.summary()`/the config; pin `revision=` for all three datasets. | CONFIRMED (mechanism) |
| A2-m3 | minor | `experiments/pipeline.py:61-67` | The config fingerprint has `sklearn` and `nltk` versions but not numpy, scipy or Python. LR uses scipy's lbfgs; the FIXAUDIT itself shows LR metrics move in the 4th decimal with float summation order, so a scipy/BLAS upgrade can change results while the cache is reused (the versions are only in `environment`, which `is_fresh` ignores). | Add `numpy`, `scipy`, `threadpoolctl` versions (or the whole `environment()` minus `platform`) to the config dict. | PLAUSIBLE |
| A2-m4 | minor | `tests/test_audit_fixes.py:196` | The probe only sees 1 thread if the pin works *or* the host has 1 CPU. Same mutation as above under `taskset -c 0 pytest -k single_threaded`: `1 passed` (mutation missed). It also does not probe `vectorize`. | Run the call inside an outer `threadpool_limits(limits=2)` so an unpinned run reports 2; probe `vectorize` too (e.g. a tokenizer stand-in that records `threadpool_info()`). | CONFIRMED |
| A2-m5 | minor | `experiments/review1.py:65` (and `:63` `build_report`, `build_outputs`) | With `QUICK_MODE=False` (so `FORCE=False`) run JSONs are protected by the fingerprint, but `run_intrinsic` has no cache check and always rewrites `results/module2/intrinsic__*.json` (new `timestamp_utc`), and `build_report`/`build_outputs`/`write_summary` rewrite DFA files, `results/tables`, figures and `datasets.json` in the committed `results/`. Not run (full data); from the code. | Document it next to `QUICK_MODE`, or give `run_intrinsic` the same fingerprint/skip logic as `run_cell`. | PLAUSIBLE |
| A2-m6 | minor | `.github/workflows/ci.yml:11-13` | `group: ci-${{ github.ref }}` with `cancel-in-progress: true` also applies to pushes to `main`: two merges in quick succession cancel the first merge's CI run, so that commit never gets a result. | `cancel-in-progress: ${{ github.event_name == 'pull_request' }}`. | PLAUSIBLE |
| A2-m7 | minor | `notebooks/make_notebook.py:34` | `git pull -q --ff-only` with `check=False`: on failure (diverged or dirty clone, no network) the notebook continues on old code with no message in the cell (a subprocess's stderr goes to the kernel log, not the cell, in Colab). A half-cloned `lexfront-dfa-tokenizer/` dir also takes this branch. | Capture the result and `print` a warning when `returncode != 0`; check `lexer/token_spec.py` exists after `chdir`. | PLAUSIBLE |

Not findings (checked, fine): `figures.py` skips missing cells instead of drawing 0 bars (tested);
`normalize.py` slang lookup now exact (tested both switches); `run_module2` rejects
`--intrinsic-max-train` with `--max-test` (tested); `make_notebook.py --check` compares only cell types and sources
(documented). `test_attribution_sentence_signs` uses synthetic inputs, but `test_attribution_on_committed_results`
checks the real JSON, so the attribution is covered.

## Verification pass

The top findings were rerun once: A2-M1 (`pip download` per pin for 3.10 and 3.12), A2-m1 and A2-m2 (scratch-copy
cache driver, warm check `reused` before each change), A2-m4 (mutation under `taskset -c 0` vs 4 CPUs). Thread-pin
result rerun twice (1 and 4 threads) plus the unpinned control.
