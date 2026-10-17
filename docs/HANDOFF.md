# HANDOFF (written by the finish session, 2026-10-17, status: P4 in review)
DONE: PR #17 notebook setup fix (stale Colab clone with local changes no longer runs old code; CI job colab-simulation), PR #18 claims pass
(stale prose, LEARN Q35-Q36), P4 branch claude/finish-3-p4: WordPiece + Unigram full-data runs, stats, tables, LEARN Q37-Q38, notebook, AUDIT_6.
NUMBERS: results/subword2/stats/bootstrap.json: 16 of 30 diff CIs exclude 0; no subword run is above NLTK+LR in accuracy with CI excl. 0;
Unigram macro-F1 vs NLTK+LR on go_emotions +1.26 [+0.10,+2.39] is the only one above (1 of 30, no multiplicity correction).
SAFETY: compare_results (python -m experiments.compare_results --additive <main>/results results ...) = 0 differences for all pre-existing results.
OPEN: one training run per cell; Unigram rounding margin ~5x (AUDIT_6); why differences appear is hypothesis; no transformers/Hinglish/more seeds.
CHECK: `pytest -q`, `python notebooks/make_notebook.py --check`, `python -m experiments.render_docs && git diff --exit-code`, CI stats step, Colab simulation.
