# HANDOFF (written by P0P1, 2026-10-16, status ready)
DONE: P0 main grid unchanged (TF-IDF lowercase=True); new case-sensitive grid (lowercase=False, 4 tokenizers x NB/LR x 3 datasets).
P1 every run stores test predictions (predictions/<run>.json.gz, `predictions_file` in run JSON); experiments/stats.py + run_stats:
percentile bootstrap (B, seed in JSON config), paired CIs for LexFront+LR vs NLTK+LR / Tweet+LR / NLTK+NB, exact McNemar on the first.
NUMBERS: results/stats/bootstrap.json (full), results/quick/stats/bootstrap.json (QUICK), results/sensitivity_case/module{1,2}/*.json;
tables ci_table, ci_runs, sensitivity_case + obs:ci, obs:sensitivity_case (REPORT_NOTES, docs/modules, LEARN Q30, notebook s.5).
All 66 main run JSONs (full + QUICK) regenerated with --force: every metric identical to main (script compare).
OPEN: CIs cover test-set sampling only (one training run per cell). No CIs for the sensitivity grid. HF revisions not pinned.
DECISIONS: docs/DECISIONS.md "P0+P1" section.
NEXT: P2 per-rule ablation (6 rules x 3 datasets, LR) with CIs via experiments/stats.py (pairs vs full LexFront+LR).
FILES TOUCHED: experiments/{classifiers,pipeline,report,observations,render_docs,review1,stats,run_stats,run_sensitivity}.py,
tests/test_stats.py, notebooks/*, results/** (run JSONs, predictions/, stats/, sensitivity_case/, tables), results/REPORT_NOTES.md,
docs/{modules/*,LEARN,DECISIONS,HANDOFF,HANDOFF_LOG}.md, CHANGELOG.md
CHECK: `pytest -q`, `python notebooks/make_notebook.py --check`, `python -m experiments.render_docs && git diff --exit-code`
