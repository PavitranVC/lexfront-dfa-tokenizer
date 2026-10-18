# Changelog

Newest first. Dates are the day the change was made (IST workday).

## 2026-10-18 (next plan)
- `docs/NEXT_PLAN.md`: phases A-D to raise classifier accuracy and add large datasets (ground rules, baselines quoted from `results/large/REPORT.md`, compute
  reference from `results/large/runs.json`, a labelled dataset probe, cost guidance). Docs only; nothing run.

## 2026-10-18 (large-scale Sentiment140 results)
- The owner ran `notebooks/LexFront_Large.ipynb` on Colab (full mode, 100 fit sets, about 1.5 hours of fitting) and committed `results/large/`.
  Checks before merging: dataset commit, split sizes and the zero user overlap match an independent local run of the same code; 48 of 48 stored
  predictions reproduce the stored accuracies; `run_large --stats-only` in the CI environment (Python 3.11, lock file) regenerates `stats.json`
  and `REPORT.md` byte for byte. No existing result changed. Results and hypothesis verdicts: `results/large/REPORT.md`, summarised in
  `docs/FINAL_SUMMARY.md` with numbers copied from the report.
- The Colab runtime was Python 3.13, scikit-learn 1.6.1, numpy 2.1.3, scipy 1.16.3, NLTK 3.10.3 (recorded in `runs.json`): outside the
  tested 3.11 / 3.12 range, but the run completed and the statistics reproduce.

## 2026-10-17 (wording fix: NB alpha grid)
- Docs said the NB `alpha` grid was "probably too narrow" and the NB gains "understated". A spot check with `alpha` down to 0.003 on four of the
  eight datasets (not saved, so not a result) changed the selection on one dataset only, so the claim was stronger than the evidence. Reworded in
  DECISIONS, HANDOFF and FINAL_SUMMARY to "edge values were often chosen; a wider grid was not run in the committed results". No code or result changed.

## 2026-10-17 (large-scale Sentiment140 test: pipeline and Colab notebook; no results yet)
- Owner asked for one huge dataset, run on Colab only, finished in a day. Added `experiments/large.py` and `experiments/run_large.py`
  (user-grouped split, audit, seeded test-time noise, size ladder, resumable per-fit JSON, chunked paired bootstrap, practical-margin
  verdicts, generated `results/large/REPORT.md`, zip and push-to-new-branch helpers), the generated notebook
  `notebooks/LexFront_Large.ipynb` (`notebooks/make_large_notebook.py --check` runs in CI) and `docs/LARGE_PROTOCOL.md` (protocol and
  hypotheses, written before the run). 20 tests in `tests/test_large.py`; two mutations (row-level split, fingerprint ignoring the data)
  were confirmed to be caught. CI gets a `large-dry-run` job that executes the notebook in quick mode from an empty directory.
- No existing run, fingerprint or metric changed (`compare_results` 0 differences). `results/large/` does not exist yet: it is written by
  the owner's Colab run, and CI checks it with `run_large --stats-only` once it is committed.

## 2026-10-17 (dataset collection, cleaning and fair tuning)
- Owner asked for a proper collection of datasets, cleaning, and better accuracy where easy. Scope chosen with the owner: add English
  noisy-text sets, keep it additive, tune the classifier equally for every tokenizer.
- New (all in new files, so no existing run's fingerprint changed): `experiments/clean.py` (empty / conflicting / duplicate / test-in-train
  rules, texts never edited), `experiments/data_extra.py` (original three + tweet_eval sentiment, emotion, irony, hate, offensive),
  `experiments/tuning.py` (validation-selected TF-IDF 1- vs 1-2-grams and LR `C` / NB `alpha`, same grid for every tokenizer, test split
  scored once), `experiments/run_collection.py` (runner, bootstrap stats, generated `results/collection/REPORT.md`). 13 tests in
  `tests/test_collection.py`; CI regenerates the collection stats and report with `--stats-only` and fails on a diff.
- Results (`results/collection/REPORT.md`, full data, 4 word-level tokenizers x NB/LR x 8 datasets): tuning raised accuracy in 44 of 64
  cells, lowered it in 16, left 4 equal (mean +1.63 points); the large gains are Naive Bayes on go_emotions and tweet_eval_emotion.
  LexFront+LR minus NLTK+LR has 4 of 8 CIs excluding 0 with default settings (2 up, 2 down) and 0 of 8 with tuned settings.
- No existing metric changed (`compare_results` reports 0 differences against main).

## 2026-10-17 (finish 6: NLTK proxy opt-in made robust)
- The first fix (finish 5) decided whether to opt in by looking at four proxy variables, but NLTK's guard uses
  `urllib.request.getproxies()` (any `*_proxy` variable, system settings), so a runtime with e.g. only `ALL_PROXY` was still refused.
  Now the cell tries a normal download, and only if `punkt_tab` is still missing retries once with NLTK's documented opt-in
  (`NLTK_ALLOW_PROXIED_URLOPEN=1` and `nltk.pathsec.ALLOW_PROXIED_FETCH=True`, both restored afterwards). Verified with real NLTK
  (fresh HOME) for `HTTPS_PROXY` only and `ALL_PROXY` only; tests cover both and the no-proxy path.

## 2026-10-17 (finish 5: NLTK download behind a proxy)
- Owner's Colab error in setup cell: NLTK refused the `punkt_tab` download because a proxy is configured (its SSRF guard, CWE-918:
  the proxy, not NLTK, chooses the destination). Reproduced here (fresh HOME, proxy set): `nltk.download` returned False and the
  data stayed missing. The setup cell now sets `NLTK_ALLOW_PROXIED_URLOPEN=1` only while it downloads that one package and only if a
  proxy variable is set (previous value restored); without a proxy nothing changes. Tests in `tests/test_notebook_setup.py`.

## 2026-10-17 (finish 4: completion)
- Final checks on `main` after PRs 17-19: `pytest -q` (181 passed), `notebooks/make_notebook.py --check`, `render_docs` and the CI
  stats step leave `git diff` clean, and the notebook runs top to bottom from an empty directory (Python 3.12, `requirements.txt`
  ranges, first cell clones the repo) in about 2 minutes (QUICK_MODE).
- `docs/FINAL_SUMMARY.md`: one honest page for the review; README links it. `FIRST_PROMPT.md` deleted (stale, mentioned Docker).

## 2026-10-17 (P4)
- AUDIT_5 A5-1: `build_tokenizer` lists its training texts first (an iterator was consumed by the hash); test added.
- AUDIT_5 A5-2/A5-3: OOV wording corrected in the generated intrinsic lines, LEARN, module2 / REPORT_NOTES text, the
  `run_module2` docstring and the notebook (subword tokenizers split words into vocabulary pieces; only characters
  never seen in train and rare pieces can be OOV); "lowest OOV rate among the word-level tokenizers".
- AUDIT_5 A5-4: the hybrid's piece cache is cleared before each timed repetition of the intrinsic throughput.
- P4: WordPiece and Unigram (`subword/wordpiece.py`, `subword/unigram.py`), same factory and protocol as BPE (train
  split only, vocabulary 8000, `Whitespace`, `[UNK]`, `##` for WordPiece). Neither trainer was deterministic as shipped;
  fixed as in DECISIONS (sorted seeded alphabet for WordPiece; canonical Unigram form: restored characters get their
  scores back in code-point order, scores rounded to 3 decimals, pieces ordered). Version file
  `docs/versions/v6-wordpiece-unigram.md`.
- P4 grid `python -m experiments.run_subword2`: 3 datasets x {wordpiece, unigram} x {NB, LR} (`results/subword2/`, QUICK
  `results/quick/subword2/`); paired bootstrap CIs vs NLTK+LR and BPE+LR and unigram vs wordpiece, McNemar vs NLTK+LR.
  Generated tables `subword_runs`, `subword_table`, observation `subword`; intrinsic metrics for both; REPORT_NOTES,
  module2.md, LEARN Q37-Q38, notebook section 8. Pre-existing metrics unchanged (checked by script against `main`).

- P4 finished (`claude/finish-3-p4`): full-data runs for go_emotions and ag_news, stats (`results/subword2/stats/bootstrap.json`),
  generated tables and observation in REPORT_NOTES and module2.md, LEARN Q37-Q38, QUICK results and notebook re-executed,
  independent audit `docs/AUDIT_6.md` (leakage, determinism, CI recomputation: pass; Unigram rounding margin noted).
  `experiments/compare_results.py` ignores the code hash and has `--additive` (new keys allowed in existing files); all pre-existing
  metrics identical to `main` by script.

## 2026-10-17 (finish 1: notebook setup)
- Notebook setup cell, reproduced failure (the traceback from the owner's Colab was not available, so this is the
  failure found by simulation, not a confirmed match): a runtime that already holds a clone from an earlier session
  has 80+ tracked `results/quick/*` files rewritten by that run; `git pull --ff-only` then refuses ("local changes
  would be overwritten"), the cell only printed a warning and went on with the OLD code under the NEWER notebook, so
  the first cell using a newer signature (`review1.run_module1_section(QUICK_MODE, force=FORCE)` ->
  `TypeError: ... unexpected keyword argument 'force'`) failed. Now `update_checkout()` sets local changes aside
  (`git stash`, kept) and pulls again, and stops with a clear message if the clone still cannot be updated; the
  project modules are dropped from `sys.modules` so a re-run in a live kernel does not keep old imports.
- NLTK data: `punkt_tab` is checked after the download instead of ignoring a failed `nltk.download(quiet=True)`.
- Not reproduced (ran clean, Python 3.12, empty directory, `requirements.txt` ranges): newest releases of every
  dependency, and Colab-era older ones (numpy 2.0.2, pandas 2.2.2, scipy 1.15.3, scikit-learn 1.6.1, datasets 4.0.0,
  tokenizers 0.21.4 preinstalled, upgraded by the install).
- `tests/test_notebook_setup.py` (stale clone with local changes is updated; an un-updatable clone fails loudly),
  CI job `colab-simulation` (Python 3.12, ranges file, empty directory, first cell clones this checkout through
  `LEXFRONT_REPO_URL`), `experiments/compare_results.py` (+ tests): lists every non-timing difference between two
  results trees, used as the "existing metrics unchanged" check.

## 2026-10-17 (P3)
- AUDIT_4 A4-1: the ablation counts how often each removed rule fires (train / test documents and lexer tokens whose
  normalized form differs from full LexFront; `<out>/<setting>/firing__<dataset>.json`, `rule_fires` in
  `results/ablation/stats/bootstrap.json`, table column "rule fires"). Never-firing variants (no_url on tweet_topic and
  go_emotions, no_mention on ag_news) are listed apart as "not a test of the rule" and left out of the counts; LEARN
  Q33, `v4-ablation.md` and DECISIONS corrected (the "0 changed predictions" proxy is withdrawn).
- AUDIT_4 A4-2: the chance statement counts only compared variants (42 CIs, roughly 2.1) and says "if the intervals
  were independent"; "(TF-IDF still lowercases)" next to all_off of the main setting.
- P3: BPE and the LexFront + BPE hybrid (`subword/bpe.py`, Hugging Face `tokenizers` 0.23.3, new dependency).
  Tokenizers that need training are built by `build_tokenizer(name, train_texts)` from the train split only; the run
  fingerprint carries the BPE config, vocabulary size and training-text hash. Deterministic training (sorted alphabet
  seeded, see DECISIONS).
- P3 grid `python -m experiments.run_bpe`: 3 datasets x {bpe, lexfront_bpe} x {NB, LR}, vocabulary 8000
  (`results/bpe/`), sensitivity at 32000 (`results/bpe_v32000/`), QUICK in `results/quick/bpe/`; paired bootstrap
  CIs and McNemar against NLTK+LR / LexFront+LR (`stats/bootstrap.json`). Generated tables `bpe_runs`, `bpe_table`
  (and `_v32000`), observations `bpe`, `bpe_v32000`; intrinsic metrics for both new tokenizers; REPORT_NOTES,
  module2, LEARN Q34, notebook section 7, CI stats step. Protocol `docs/versions/v5-bpe-hybrid.md`.
- All main, sensitivity, ablation and QUICK runs regenerated (code fingerprint changed); every metric, CI and stored
  prediction identical to `main` (script compare).

## 2026-10-16 (P2)
- P2: per-rule normalization ablation (`python -m experiments.run_ablation`, `results/ablation/`): LexFront + LR with
  each of url, mention, number, elongated, slang switched off (TF-IDF lowercases, as the main grid), plus no_lowercase
  with case kept (TF-IDF `lowercase=False`), plus all rules off in both settings; paired bootstrap CIs and McNemar of
  each variant against full LexFront (`results/ablation/stats/bootstrap.json`). Generated table `ablation_table` and
  observation `ablation` in REPORT_NOTES, docs/modules/module2.md, LEARN.md (new Q33) and the notebook (new section 6,
  QUICK). Protocol in `docs/versions/v4-ablation.md`.
- Ablation variants `lexfront:no_<rule>` / `lexfront:all_off`; the run fingerprint and run JSON of every LexFront run
  record its `NormalizeConfig`.
- AUDIT_3 fixes: CAVEAT on untested gaps (A3-1); QUICK CI width note in the notebook (A3-2); bootstrap CIs for the
  case-sensitive grid, table `sensitivity_ci` and observation lines (A3-3); CI checks that every stats JSON matches the
  committed predictions (A3-4); unused `n_classes_of` removed (A3-5).
- All main, QUICK and sensitivity run JSONs regenerated (`--force`); every metric and every stored prediction identical
  to before (script compare).

## 2026-10-16 (P0+P1)
- P0: main grid unchanged (TF-IDF `lowercase=True`); new case-sensitivity grid with `lowercase=False` for all four
  tokenizers x {NB, LR} x 3 datasets (`python -m experiments.run_sensitivity`, `results/sensitivity_case/`), with a
  generated table and observation. Decision in `docs/DECISIONS.md`.
- P1: every run stores its test predictions (`predictions/<run>.json.gz`, named in the run JSON as `predictions_file`).
- P1: `experiments/stats.py` + `python -m experiments.run_stats`: percentile bootstrap CIs (accuracy, macro-F1) for every
  run, paired CIs for LexFront+LR vs NLTK+LR / TweetTokenizer+LR / NLTK+NB, exact McNemar for the headline pair
  (`results/stats/bootstrap.json`); generated tables `ci_table`, `ci_runs` and observation `ci` in REPORT_NOTES,
  docs/modules, LEARN.md (new viva Q30) and the notebook (new section 5 on the QUICK grid).
- The observations caveat now describes the CIs (test-set bootstrap, one training run per cell).
- All main run JSONs regenerated (full and QUICK, `--force`); every metric identical to before.

## 2026-10-16 (FIXAUDIT2)
- Fixed the findings of `docs/AUDIT_2.md` (A2-M1, A2-M2, A2-m1..m7); each decision is in `docs/DECISIONS.md`.
- Dependencies: `requirements.txt` now holds compatible ranges (Python 3.11 and 3.12, satisfied by a fresh Colab runtime);
  the exact pins moved to `requirements-lock.txt`, which CI installs.
- Notebook setup cell: prints the numpy/pandas/scikit-learn versions in use, warns when a restart is needed, warns when
  `git pull` fails and stops on an incomplete clone; the `QUICK_MODE` cell lists the files each mode rewrites.
- Result cache: the config fingerprint also covers a hash of the train/test texts and labels and the numpy, scipy and
  threadpoolctl versions; `make_vectorizer` is built from `VECTORIZER_PARAMS`.
- All run JSONs regenerated (full and QUICK, `--force`); every metric is identical to before.
- Thread-pin test now fails on an unpinned run even on a 1-CPU host and also probes vectorization.
- CI: concurrent runs are cancelled only for pull requests, not for pushes to `main`.

## 2026-10-16 (FIXAUDIT)
- Fixed the findings of `docs/AUDIT.md` (M1-M3, m1-m13); each decision is in `docs/DECISIONS.md`.
- Result cache: each run JSON now stores a config + code fingerprint and is recomputed when it differs.
- Reproducibility: every vectorize/fit/predict runs with BLAS/OpenMP pinned to 1 thread (`threadpoolctl`, recorded as
  `environment.blas_threads`); the thread count had changed LR (lbfgs) metrics. All run JSONs regenerated; QUICK and
  all TweetTopic/NB/majority metrics unchanged; changed LR metrics (old -> new, from the JSONs):
  - module1 ag_news tweet LR: accuracy 0.918289 -> 0.918421, macro-F1 0.918149 -> 0.918276
  - module1 go_emotions nltk LR: accuracy 0.552941 -> 0.552723, macro-F1 0.338894 -> 0.338600
  - module1 go_emotions tweet LR: accuracy 0.554031 -> 0.554684, macro-F1 0.339879 -> 0.340230
  - module1 go_emotions whitespace LR: accuracy 0.534423 -> 0.535076, macro-F1 0.312263 -> 0.314023
  - module2 ag_news lexfront LR: accuracy 0.918684 -> 0.918816, macro-F1 0.918504 -> 0.918630
  - module2 go_emotions lexfront LR: accuracy 0.553159 -> 0.553595, macro-F1 0.339532 -> 0.339734
- `obs:attribution` (generated) replaces the incorrect hand-written attribution sentence in `results/REPORT_NOTES.md`.
- Notebook: setup cell reuses an existing clone (Colab restart), dead dev-branch fallback removed, `FORCE` flag;
  re-executed in QUICK_MODE with every run recomputed.
- Pinned `requirements.txt`, dropped pip `graphviz`; CI installs apt `graphviz`, has read-only permissions and checks
  that rendered docs and the generated notebook match the committed files.
- Design: `docs/versions/v4-tokenizer-lineup.md` (tokenizers 5 -> 8); `docs/PIPELINE.md` "Overnight mode" section.

## 2026-10-16 (later)
- PR #1 merged into `main` by the author; Colab badge now resolves.
- Added GitHub Actions CI: pytest on every push/PR, plus the notebook executed in QUICK_MODE.

## 2026-10-16
- Module 2 core: token spec (11 classes), PLY longest-match lexer with UNKNOWN recovery, normalizer (6 switchable
  rules, 30-entry slang table), symbol table, `LexFront.tokenize()`, own DFA engine (regex parser, Thompson NFA,
  subset construction, Moore minimization, table-driven scanner), DFA state counts + `.dot`/`.png`.
- Module 1 / experiment code: dataset loaders, tokenizers, classifiers, metrics, evaluation loop, runners.
- Reports: tables, figures and observations built from JSON only; doc renderer; Colab notebook generator.
- Docs: `docs/LEARN.md`, `docs/modules/`, README with Colab badge, `results/REPORT_NOTES.md` skeleton.
- `huggingface.co` was blocked for about an hour; resolved. TweetTopic is read from the auto-converted parquet revision
  (its repo uses a loading script); COLING 2022 split pair confirmed by the dataset card.
- Ran Module 1 (27 runs) and Module 2 (6 classification runs, intrinsic metrics for 3 datasets, DFA report) on full data.
- Built tables, figures and DRAFT observations from the JSON; rendered `results/REPORT_NOTES.md` and module docs.
- Executed `notebooks/LexFront_Review1.ipynb` end to end (QUICK_MODE) and committed it with outputs; re-ran it from a fresh
  clone in an empty directory (Colab simulation): no errors, identical quick-run metrics.
- Phase 0: removed Docker files (`Dockerfile`, both compose files, `.dockerignore`).
- Phase 0: rewrote `requirements.txt` (CPU only, no torch, no transformers).
- Phase 0: added `docs/versions/` v1 (base paper), v2 (LexFront proposal),
  v3 (English only, 3 datasets, 5 tokenizers, Docker dropped, Colab).
