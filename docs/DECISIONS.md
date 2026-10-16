# Decisions log

Newest first. One short entry each: decision, reason, source. Gaps filled on 2026-10-16 from the team's notes; items
already recorded elsewhere are only pointed to. Citations below are from abstracts only and were not re-checked here.

## 2026-10-16 (P2: per-rule normalization ablation; AUDIT_3 minor fixes)

### P2 protocol (decisions by the conductor)
- Decision: LexFront + LR only, 3 datasets, full data. Main setting (TF-IDF `lowercase=True`): full, no_url,
  no_mention, no_number, no_elongated, no_slang, all_off. Case-kept setting (TF-IDF `lowercase=False`): full_cs,
  no_lowercase, all_off_cs. Each variant minus the full variant of its setting: paired bootstrap (same code, B = 2000,
  seed 20261016 as P1) on accuracy and macro-F1, exact McNemar on accuracy, no multiplicity correction (said in the
  generated observation). Protocol: `docs/versions/v4-ablation.md`.
- Reason: the lowercase rule cannot act in the main setting (TF-IDF lowercases the text first), so it is only tested
  with case kept.

### P2 implementation
- Decision: variants are tokenizer names `lexfront:no_<rule>` and `lexfront:all_off` (`LEXFRONT_VARIANTS` in
  `experiments/tokenizers.py`, not in `TOKENIZERS`, so the main grid and intrinsic metrics do not pick them up). The
  config fingerprint and the run JSON of every LexFront run hold its `NormalizeConfig` (`normalize`); baselines get no
  extra key. In file names `:` becomes `-` (valid on Windows); main-grid file names do not change.
- Decision: the full variants are recomputed inside `results/ablation/` (not copied); `results/ablation/stats/
  bootstrap.json` records per setting and dataset whether their predictions are identical to the full-LexFront+LR run
  of the main or sensitivity grid (`reference_vs_grid`).
- Decision: the ablation table also shows the TF-IDF vocabulary size and how many test predictions changed, so a rule
  that never fires on a dataset is visible as 0 changed predictions.

### Safety check
- The code fingerprint changed (tokenizers, pipeline, normalize), so all main, QUICK and sensitivity runs were
  regenerated with `--force` and compared by script with `main` (every field except timestamp, timing, fingerprint,
  environment, predictions_file and the new `normalize`; predictions compared too): see the P2 CHANGELOG entry.

### AUDIT_3 minor findings
- A3-1: CAVEAT adds that gaps without a paired CI are untested and that overlapping per-run CIs are not a test.
- A3-2: notebook section 5 says QUICK has 500 test examples, so its CIs are wider and can change sign. The audit
  suggested "2 to 3 times wider"; the committed JSONs give 1.9 to 5.8 times for accuracy differences (ag_news is the
  widest), so the text says "about 2 to 6 times".
- A3-3: `run_stats` now also runs on `results/sensitivity_case` (`sensitivity_case/stats/bootstrap.json`); new
  generated table `sensitivity_ci` (case-kept CI next to the main-grid CI) and lines in `obs:sensitivity_case` and
  `obs:ci` stating on how many datasets the tokenizer-only "excludes 0" verdict differs between the two grids.
  A separate table rather than extra columns: `sensitivity_case` has one row per tokenizer, the CIs are per pair.
- A3-4: CI reruns `run_stats` (main, QUICK, sensitivity) and `run_ablation --stats-only` (full and QUICK) and fails
  on any diff under `results/`.
- A3-5: `stats.n_classes_of` (no caller) deleted.

## 2026-10-16 (P0+P1: case sensitivity, stored predictions, bootstrap CIs)

### P0: TF-IDF lowercasing vs LexFront's lowercase rule (decision by the conductor)
- Decision: keep the main grid exactly as it was (TF-IDF `lowercase=True` for every tokenizer; LexFront's own
  lowercase rule is inactive there by design, so every committed main number stays identical and comparable). Add a
  sensitivity grid with TF-IDF `lowercase=False` for all four tokenizers x {NB, LR} x 3 datasets in
  `results/sensitivity_case/{module1,module2}/` (`python -m experiments.run_sensitivity`).
- Reason: in the sensitivity grid LexFront's rule is active, so LexFront's WORD tokens are lowercased while the
  baselines keep case; it is a robustness check, not a like-for-like comparison (said in the docs next to its table).
- How: `VECTORIZER_PARAMS` stays the single source; the grid passes `vectorizer_params(CASE_SENSITIVE_OVERRIDES)`
  (`{"lowercase": False}`), which goes into the run JSON and the config fingerprint. Results: table and observation
  `sensitivity_case` (generated).

### P1a: per-example predictions stored for every run
- Decision: `predictions/<run>.json.gz` next to each run JSON holds `{y_true, y_pred}`; the run JSON names it in
  `predictions_file` (relative to its own directory). gzip header has mtime 0 and no file name, so identical
  predictions give identical bytes. A cached run without its sidecar is recomputed.
- Reason: test-set statistics without refitting; JSON stays readable; small files (whole full grid about 0.2 MB).
- Safety check: changing `pipeline.py` changed every code fingerprint; all 66 main run JSONs (full + QUICK) were
  regenerated with `--force` and compared by script with `main`, ignoring timestamp, timing, fingerprint and the new
  `predictions_file`: no difference in any metric or other field.

### P1b/P1c: bootstrap CIs and paired tests
- Decision: `experiments/stats.py` + `python -m experiments.run_stats` -> `results/stats/bootstrap.json`. Percentile
  bootstrap over test examples, B = 2000, `numpy.random.default_rng(20261016)` per dataset; all runs of a dataset share
  the resampled indices, so differences are paired. Accuracy and macro-F1 from bincount confusion matrices (macro-F1
  averages classes present in gold or predictions, = sklearn `zero_division=0`; the script checks the point estimate
  equals the stored metric). Pairs (LexFront+LR minus A): A = NLTK+LR (headline, plus exact McNemar), TweetTokenizer+LR,
  NLTK+NB. Tables `ci_table`, `ci_runs`, observation `ci`; the shared CAVEAT now describes these CIs.
- Reason: the CIs cover test-set sampling only; training randomness is not covered (one run per cell; NB/LR fits are
  deterministic; no subsampling in the full grid). McNemar only on the headline pair to limit the number of tests.

## 2026-10-16 (FIXAUDIT2, fixes for docs/AUDIT_2.md)

### A2-M1/A2-M2: two dependency files
- Decision: `requirements.txt` = compatible ranges (numpy>=1.26,<3, pandas>=2.2,<4, scipy>=1.12,<2,
  scikit-learn>=1.5,<2, matplotlib>=3.8,<4, threadpoolctl>=3.1,<4, ipykernel>=6; `nltk==3.10.3` and `ply==3.11` stay
  exact); `requirements-lock.txt` = the exact pins behind the committed results, installed by CI. The notebook installs
  the ranges, prints the numpy/pandas/scikit-learn versions in use and warns when pip changed an already-imported
  package (`experiments/setup_check.py`).
- Reason: exact pins forced reinstalls inside a running Colab kernel. Verified: lowest allowed versions and the lock
  install and pass `pytest -q` on Python 3.11 and 3.12; newest resolved versions on 3.12. Python 3.10 not claimed.

### A2-m1: VECTORIZER_PARAMS is the single source
- Decision: `make_vectorizer` builds `TfidfVectorizer(**vectorizer_kwargs())` from the dict (`ngram_range` as tuple).

### A2-m2: data content in the fingerprint; no HF `revision=` pin
- Decision: config fingerprint includes `Dataset.content_hash()` (SHA-256 of train/test texts and labels), also stored as
  `data_hash` in each run JSON. HF revisions not pinned: the content hash already detects an upstream edit, and a pin
  would need commit SHAs looked up by hand. `Dataset` now stores plain lists (datasets 5 Columns were slow to hash).

### A2-m3: library versions in the fingerprint
- Decision: numpy, scipy and threadpoolctl versions join sklearn and nltk in the config fingerprint (`library_versions()`).

### A2-m4: thread-pin test works on a 1-CPU host
- Decision: the probe runs inside `threadpool_limits(2)` and also probes the tokenizer inside `vectorize`. Mutation check
  under `taskset -c 0`: removing either pin fails the test.

### A2-m5: what full mode rewrites is documented
- Decision: comment next to `QUICK_MODE` lists the files each mode rewrites; no skip logic for `run_intrinsic`.
- Reason: throughput is machine-dependent anyway; a cache for it would add code for little gain.

### A2-m6/A2-m7: CI concurrency and notebook setup
- Decision: `cancel-in-progress` only for pull_request events; the setup cell prints a warning when `git pull` fails and
  stops if `lexer/token_spec.py` is missing after `chdir`.

### Results regenerated
- All run JSONs (full and QUICK) regenerated with `--force` for the new fingerprints; every metric identical to main
  (checked by script). Intrinsic JSONs are not fingerprinted and were not rerun.

## 2026-10-16 (FIXAUDIT, fixes for docs/AUDIT.md)

### BLAS threads: every fit is single-threaded (decision by the conductor)
- Decision: `experiments/pipeline.py` wraps vectorize/fit/predict in `threadpoolctl.threadpool_limits(1)`; each run JSON
  records `environment.blas_threads = 1` and the config fingerprint includes it. All full-data and QUICK run JSONs were
  regenerated with `--force` and accepted as the committed results.
- Reason: the native thread count changes the float summation order in lbfgs. go_emotions nltk LR rerun with
  OMP/OPENBLAS/MKL_NUM_THREADS = 1 / 2 / 4 gave accuracy 0.552723 / 0.552941 / 0.553159 and macro-F1
  0.338600 / 0.338894 / 0.338969; the old committed value (0.552941 / 0.338894) is the 2-thread one. After the fix, all
  8 go_emotions and ag_news LR cells rerun with 4 threads were identical to the committed JSONs.
- Changed metrics, old (multi-threaded) -> new (single-threaded); all other run JSONs have identical metrics:

| run JSON | metric | old | new |
|---|---|---|---|
| results/module1/ag_news__tweet__lr.json | accuracy | 0.918289 | 0.918421 |
| results/module1/ag_news__tweet__lr.json | macro_f1 | 0.918149 | 0.918276 |
| results/module1/go_emotions__nltk__lr.json | accuracy | 0.552941 | 0.552723 |
| results/module1/go_emotions__nltk__lr.json | macro_f1 | 0.338894 | 0.338600 |
| results/module1/go_emotions__tweet__lr.json | accuracy | 0.554031 | 0.554684 |
| results/module1/go_emotions__tweet__lr.json | macro_f1 | 0.339879 | 0.340230 |
| results/module1/go_emotions__whitespace__lr.json | accuracy | 0.534423 | 0.535076 |
| results/module1/go_emotions__whitespace__lr.json | macro_f1 | 0.312263 | 0.314023 |
| results/module2/ag_news__lexfront__lr.json | accuracy | 0.918684 | 0.918816 |
| results/module2/ag_news__lexfront__lr.json | macro_f1 | 0.918504 | 0.918630 |
| results/module2/go_emotions__lexfront__lr.json | accuracy | 0.553159 | 0.553595 |
| results/module2/go_emotions__lexfront__lr.json | macro_f1 | 0.339532 | 0.339734 |

### M1: result cache keyed by a fingerprint
- Decision: every run JSON stores `fingerprint = {config, code}` (hash of data subset, tokenizer, vectorizer and
  classifier params, seed, sklearn/nltk versions; hash of `lexer/` and the experiment modules that affect a run).
  `run_cell` reuses a JSON only if both match. The notebook sets `FORCE = QUICK_MODE`, so the CI notebook job and the
  committed notebook recompute every QUICK run. All run JSONs were regenerated single-threaded (see "BLAS threads" below).
- Reason: the file name alone hid stale results (AUDIT M1). No test asserts that committed JSONs match the current
  code hash: that would force every later code change to rerun the full grid; a stale JSON is recomputed on next run.

### M2: proposed-vs-existing attribution is generated, per dataset
- Decision: new `obs:attribution` block (`experiments/observations.py`) replaces the hand-written sentence in
  `results/REPORT_NOTES.md` s.4.
- Reason: the old text said the classifier switch explains "almost all of the gain"; on TweetTopic proposed is below existing.

### m1: intrinsic subsample vs classification subsample
- Decision: `run_module2` rejects `--intrinsic-max-train` together with `--max-test`; `data.subsample` is unchanged.
- Reason: separate RNGs would change the committed QUICK subsets and every QUICK number; the bug is latent.

### m2: slang lookup and the lowercase switch
- Decision: the slang lookup is exact (`lexeme in table`); with `lowercase=False`, `OMG` stays `OMG`, `omg` expands.
- Reason: keeps the two rules independent for the P2 ablation. No result changes (TF-IDF lowercases first; P0 decides that).

### m3/m4: emoji sequences, NFD, `www.`, `heyyy'll`
- Decision: documented as limits in `docs/LEARN.md` Q26/Q28 and pinned by tests; no scanner change.
- Reason: changing the spec would change every LexFront number; out of scope for an audit-fix task.

### m6: dependencies
- Decision: `requirements.txt` pins the versions the results were produced with; pip `graphviz` dropped (unused);
  CI installs the apt package `graphviz` so `dfa_min.png` can be rendered by `dot`.

### m8/m9: errors and side effects in reporting
- Decision: tables and observations turn only `FileNotFoundError` into "_not available_"; other errors raise.
  `report.render_all_tables` builds tables without writing; `render_docs` and the notebook appendix use it.

### m10: CI
- Decision: `permissions: contents: read`, `concurrency`, a step that re-renders docs and checks `git diff --exit-code`,
  and a step that rebuilds the notebook from `make_notebook.py` and compares cell sources.

### m11: overnight mode and tokenizer line-up
- Decision: `docs/PIPELINE.md` gains an "Overnight mode" section that overrides its header; tokenizer plan 5 -> 8 in
  `docs/versions/v4-tokenizer-lineup.md`.

### m12/m13
- Decision: a test checks the DFA numbers and state ids hand-typed in `docs/LEARN.md`; the existing-vs-proposed figure
  skips a missing cell instead of drawing a 0 bar.

## 2026-10-16

### Proposal document (external)
- Decision: the full proposal lives outside the repo.
- Source: https://claude.ai/code/artifact/392bb277-7697-4b90-b7f1-7dd7a7656c00

### Next-phase order
- Decision: (1) bootstrap confidence intervals + paired test (NLTK+LR vs LexFront+LR) -> (2) normalization ablation
  -> (3) BPE and LexFront + BPE hybrid -> (4) WordPiece/Unigram -> (5) optional transformer baseline -> (6) a
  `docs/versions/v4-*.md` file for any design change.
- Reason: results are single-seed, so uncertainty comes first; the ablation depends on the item-6 decision below.

### Honest headline
- Decision: report LexFront as giving lower OOV rate and smaller vocabulary on all 3 datasets. Accuracy, per dataset:
  on GoEmotions and AG News the gain of proposed over existing comes almost entirely from NB -> LR; on TweetTopic
  proposed is 0.26 points below existing. Single runs, no CIs. (Corrected by FIXAUDIT, AUDIT M2.)
- Reason: see `results/REPORT_NOTES.md` section 4, block `obs:attribution` (generated from the run JSONs).

### Open: TF-IDF lowercasing vs LexFront's lowercase rule
- Decision: undecided; must be settled before the ablation (options: `lowercase=False` in TF-IDF, or drop the rule
  from the ablation).
- Reason: TF-IDF lowercases first, so the rule is inactive in all experiments. Source: `results/REPORT_NOTES.md` s.2.

### Credibility check (abstracts only)
- Decision: position the work as a combination, never claim "first" or "novel".
- Reason: regex tweet tokenizers exist (NLTK TweetTokenizer, CMU Twokenize, ekphrasis); automaton-based tokenization
  exists (Fast WordPiece, EMNLP 2021); normalization gave small downstream gains (MultiLexNorm, W-NUT 2021);
  pre-tokenization evidence is mixed (COLING 2025 vs Vemula et al. 2025). Our gap: compiler-built DFA pre-tokenizer
  compared on noisy classification, with a per-rule ablation. Self-assessed novelty ~3-4/10; fit for Compiler Design ~9/10.
- Source: team literature check (abstracts only, full papers not read).

### Scope cuts
- Decision: tokenizers 9 -> 5; datasets 4 -> 3; Logistic Regression only, with Naive Bayes as the existing method;
  transformer fine-tuning cut (optional baseline later); Docker dropped for Colab.
- Reason: 14-day side project. Source: team decision; datasets/Docker also in `docs/versions/v3-*.md`.

### Hinglish dropped
- Decision: English only.
- Reason: SentiMix dataset download unconfirmed; closest prior work (HingLM, NeurIPS 2026 workshop) is very recent;
  too much scope. (v3 and `docs/LEARN.md` Q27 give only the annotation/time reason.)
- Source: team decision.

### DFA engine covers 4 classes only
- Already recorded: `results/REPORT_NOTES.md` s.4 (ELONGATED needs a back-reference, not regular). No change.

### Origin: base paper
- Decision: start from a blog classification report (NLTK + PorterStemmer + CountVectorizer + Naive Bayes; 35% on
  40 labels, 54% after manually merging to 3).
- Reason: its weaknesses: no baseline, accuracy only, possible author leakage, misused terms, code that didn't run.
  We therefore add majority baseline, macro-F1, fixed classifier, and runnable code. (`v1-base-paper.md` has only the figures.)
- Source: team reading of the report; figures not re-run.
