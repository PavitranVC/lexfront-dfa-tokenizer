# HANDOFF (written by FIXAUDIT2, 2026-10-16, status ready)
DONE: fixed docs/AUDIT_2.md A2-M1, A2-M2, A2-m1..m7. requirements.txt = ranges (Colab-friendly), requirements-lock.txt = exact pins (CI);
notebook setup cell warns on git pull failure / incomplete clone and when pip changed an imported package (experiments/setup_check.py).
Fingerprint now covers data content hash and numpy/scipy/threadpoolctl versions; VECTORIZER_PARAMS drives make_vectorizer.
NUMBERS: all 66 run JSONs regenerated (--force, full + QUICK): every metric identical to main (results/module{1,2}/*__*__*.json,
results/quick/module{1,2}/*__*__*.json); only fingerprints and the new data_hash field changed. Intrinsic JSONs not rerun (full).
Tested installs: lock and lowest-range versions on Python 3.11 and 3.12, newest-range on 3.12; pytest passes in each.
OPEN: TF-IDF lowercasing vs LexFront lowercase rule (P0). HF dataset revisions not pinned (content hash detects drift).
DECISIONS: docs/DECISIONS.md "FIXAUDIT2" section (one entry per finding).
NEXT: P0 - decide TF-IDF lowercase=False vs dropping the rule; rerun affected results.
FILES TOUCHED: requirements.txt, requirements-lock.txt, experiments/{classifiers,data,pipeline,setup_check}.py, notebooks/*, tests/test_audit_fixes.py,
.github/workflows/ci.yml, README.md, results/**/ run JSONs + results/quick intrinsic/tables, docs/{DECISIONS,HANDOFF,HANDOFF_LOG}.md, CHANGELOG.md
CHECK: `pytest -q`, `python notebooks/make_notebook.py --check`, `python -m experiments.render_docs && git diff --exit-code`
