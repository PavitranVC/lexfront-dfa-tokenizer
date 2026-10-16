# Changelog

Newest first. Dates are the day the change was made (IST workday).

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
