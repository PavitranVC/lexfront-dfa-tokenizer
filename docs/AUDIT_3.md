# AUDIT 3: diff audit of P0+P1 (PR #10)

Scope: `git diff 071ca33..af6203b` (merge of PR #10). The auditor did not change any code; this file is the only change.
Date 2026-10-16. Environment: Python 3.11 venv from `requirements-lock.txt`, NLTK data downloaded.
Token estimate: about 75k of the 150k budget.

**Verdict: PASS.** No blocking or major finding. 5 minor findings. The statistics were recomputed independently
and reproduce.

## Scope covered
1. Sidecar integrity for every run (main, QUICK, sensitivity): 90 runs.
2. Independent bootstrap (own code, does not import `experiments/stats.py`, different seed) for tweet_topic. Also
   checked pairing, macro-F1 against sklearn, and McNemar against `scipy.stats.binomtest` (all datasets, full and QUICK).
3. Every generated CI sentence, the CAVEAT, LEARN Q20/Q30, DECISIONS, HANDOFF, CHANGELOG and REPORT_NOTES text about CIs.
4. Sensitivity grid: that `lowercase=False` is really applied, the "change vs main" columns, and that the main grid is untouched.
5. Fingerprint/cache, notebook QUICK section, sidecar size, tests, CI workflow, docs in sync with the generator,
   dead code and exception handling in the diff.

## Not covered
- Re-running any training (full or QUICK grid, sensitivity grid); no Hugging Face download. Run JSONs were checked
  against their own sidecars and against `071ca33`, not refit.
- The notebook was not executed (only `make_notebook.py --check` and the stored outputs were read).
- Independent bootstrap on go_emotions and ag_news: only reproduced via the project code (exact) and McNemar (independent).
- Line-by-line review of `notebooks/LexFront_Review1.ipynb` outputs outside section 5. `docs/PIPELINE.md` was only
  checked for the P1 row.

## What passed

| # | Check | Command / method | Result |
|---|---|---|---|
| 1 | Sidecar metrics = run JSON metrics (exact `==`) | own script: sklearn `accuracy_score`, `f1_score(macro, zero_division=0)` on every `predictions/*.json.gz` | 90/90 runs exact, 0 mismatches, no run without a sidecar |
| 1b | Same `y_true` across runs of a dataset | same script, grouped by grid and dataset | identical in all 9 (grid, dataset) groups (11/11/8 runs; n = 3399 / 4590 / 7600, QUICK 500) |
| 2 | Independent percentile bootstrap, tweet_topic, 4 tokenizers x LR + NLTK+NB, B = 2000, seed 12345 | own loop: `rng.integers(0,n,(B,n))`, own per-class F1, `np.percentile([2.5,97.5])` | all 16 run-CI endpoints and all 12 paired-difference endpoints within **0.151 bootstrap SE** of `bootstrap.json` (tolerance set before running: 0.5 SE; the Monte-Carlo SE of a 2.5% quantile at B = 2000 is about 0.3 to 0.4 SE). "excludes 0" agrees on all 6 differences |
| 2b | Own macro-F1 = sklearn on resamples | 8 resamples (indices 0, 1, 999, 1999 x 2 runs) | equal within 1e-12 |
| 2c | Pairing | code: `run_stats.dataset_stats` draws `idx` once per dataset and passes it to every run and pair; own check: paired accuracy-diff CI width 1.59 pts vs 4.21 pts with shuffled (unpaired) resamples | pairing is real and matters |
| 2d | McNemar | discordant counts from sidecars + `scipy.stats.binomtest`, all 3 datasets, full and QUICK | counts identical (e.g. tweet_topic 124 / 70), p identical to 1e-12 (tweet_topic p = 0.000129) |
| 2e | `bootstrap.json` reproducible | `run_stats.compute('results')` and `('results/quick')` vs committed files | exactly equal (both) |
| 3 | Generated text vs JSON | `python -m experiments.render_docs` on a copy, `git status` | no diff: every doc block is what the generator gives from the committed JSON |
| 3b | CI sentences | read `obs:ci` (module2.md, REPORT_NOTES), `ci_table`, LEARN Q20/Q30 | every number and every "excludes/includes 0" matches `bootstrap.json`; "1 of 6 difference CIs exclude 0" is correct (tweet_topic accuracy, LexFront worse); CAVEAT and Q30 state that training randomness is not covered; no overclaim found |
| 4 | `lowercase=False` applied | sensitivity run JSONs; analyzer on `"Apple CEO Tim Cook SAYS iPhone sales UP #AAPL @Reuters https://t.co/X"` | all 24 runs `lowercase: false`, all 24 config fingerprints differ from the main ones, n_features differ in 22/24. Baselines keep case (`'Apple','CEO','SAYS'`), LexFront lowercases WORD tokens (`'apple','ceo','says'`) and keeps `#AAPL`; matches the docs' "WORD tokens" wording |
| 4b | "change vs main" columns | recomputed from both JSON sets | 12/12 rows, both columns, exact string match |
| 4c | Main grid untouched | all 66 run JSONs (full + QUICK) at `071ca33` vs now, every field except timestamp/timing/fingerprint/environment/predictions_file | 0 metric differences, 0 field differences, config fingerprint unchanged in 66/66 (default `vec_params` keeps the old hash), no new or missing run; pre-existing tables unchanged (`git diff --stat` on `results/tables` shows only the 3 new files) |
| 5 | Tests | `.venv/bin/python -m pytest -q` | 144 passed. `tests/test_stats.py` is real: sklearn comparison, batched vs one-by-one confusion, known-answer accuracy CI, McNemar known table (p = 0.125), deterministic gzip bytes, missing sidecar forces a rerun |
| 5b | Notebook | `python notebooks/make_notebook.py --check` | matches |
| 5c | Size | `du` on `predictions/*.json.gz` | 0.23 MB total (main full 118 KB, sensitivity 94 KB, QUICK 15 KB) + 43 KB `bootstrap.json` x2. DECISIONS "about 0.2 MB" is right |
| 5d | Exceptions / cache | grep of the diff | no new `except`; `is_fresh` now also requires the sidecar; sensitivity runs in a separate directory not read by `load_runs`/`find_runs` of the main grid |

## Findings

| id | severity | file:line | evidence | suggested fix | status |
|---|---|---|---|---|---|
| A3-1 | minor | `experiments/observations.py:12-16` (CAVEAT) | The CAVEAT is appended to every observation block. The old rule ("gaps of about one point or less should not be read as real differences") was replaced by "A gap whose CI includes 0 should not be read as a real difference". Most gaps in the other blocks have no paired CI, e.g. `docs/modules/module1.md:64-66` "tokenizer spread is 1.82 points under LR", LR vs NB. A reader may also compare overlapping per-run CIs (`ci_runs`), which is not a paired test. | Add one sentence: "Gaps not in `ci_table` (e.g. spreads between baselines) have no paired CI and are not tested; overlapping per-run CIs are not a test of a difference." | CONFIRMED (text read in the rendered docs) |
| A3-2 | minor | `notebooks/make_notebook.py:168-175` (section 5) | The section says the stats run "here on the QUICK grid" but not that n_test = 500 or that the intervals are about 2 to 3 times wider than the full grid. QUICK tweet_topic tokenizer-only accuracy is +0.40 [-1.40, +2.20], full grid -1.59 [-2.38, -0.82]: the sign flips (notebook stored output, cell 31 vs `results/stats/bootstrap.json`). | One sentence: "QUICK uses 500 test examples per dataset, so these CIs are wide and can differ in sign from the full grid; the headline CIs are `results/stats/bootstrap.json` (table in the summary section)." | CONFIRMED |
| A3-3 | minor | `experiments/observations.py:18` (SENSITIVITY_CAVEAT), `docs/modules/module2.md:208-210` | The sensitivity grid has no CIs ("no confidence intervals for this grid", stated honestly). `run_stats.compute('results/sensitivity_case')` runs unchanged on it and shows the main tokenizer-only result is not robust to case folding: tweet_topic -0.91 [-1.88, +0.09] (includes 0; main grid excludes 0), go_emotions +1.15 [+0.46, +1.85] (excludes 0, McNemar p = 0.0009; main grid +0.09, includes 0), ag_news -0.18 [-0.55, +0.20]. A headline built on "LexFront+LR is worse on tweet_topic (CI excludes 0)" should mention this. | Run `run_stats` on `results/sensitivity_case` (write `sensitivity_case/stats/bootstrap.json`) and add those CIs to the sensitivity table/observation; or say in the CI observation that the one excluding-0 result changes under `lowercase=False`. | CONFIRMED (command above) |
| A3-4 | minor | `.github/workflows/ci.yml:28-33` | CI checks the doc blocks against the JSON (`render_docs` + `git diff`) but nothing checks that `results/stats/bootstrap.json` is what `run_stats` gives from the committed sidecars. Today it is (exact, check 2e), but a later sidecar regeneration without rerunning `run_stats` would go unnoticed. | Add a CI step `python -m experiments.run_stats && python -m experiments.run_stats --results results/quick && git diff --exit-code results/stats results/quick/stats` (about 2 s each). | CONFIRMED (ci.yml read; no other step runs `run_stats`) |
| A3-5 | minor | `experiments/stats.py:57` | `n_classes_of` has no caller (`grep -rn n_classes_of experiments tests` finds only the definition); `run_stats` takes K from the run JSON's `n_labels`. | Delete it. | CONFIRMED |

Observations, not findings: all datasets use the same seed for their resampling generator (`default_rng(20261016)`
per dataset), which is documented and harmless because no statistic combines datasets. The QUICK ag_news
"proposed minus existing" CI has an upper end of exactly 0.00 and is reported as "includes 0", which is correct
(boundary counts as included).

## Verdict
PASS: blocking 0, major 0, minor 5.

Scripts used (scratch, not committed): sidecar recompute (check 1), independent bootstrap and McNemar (check 2),
reproducibility of `bootstrap.json` (2e), main grid vs `071ca33` (4c), sensitivity checks (4, 4b).
