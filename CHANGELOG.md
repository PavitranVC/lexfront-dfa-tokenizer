# Changelog

Newest first. Dates are the day the change was made (IST workday).

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
