# HANDOFF (written by the finish session, 2026-10-17, status: complete)
DONE: PR #17 notebook setup fix (a Colab runtime that kept an older clone with local changes ran OLD code under the new notebook; now stashes + pulls,
fails loudly; CI job colab-simulation), PR #18 claims pass, PR #19 P4 (WordPiece + Unigram, train-only, deterministic, AUDIT_6), final PR (summary, cleanup).
NUMBERS: results/stats: NLTK->LexFront+LR -1.59 acc pts on tweet_topic [-2.38,-0.82], +0.09 / +0.00 (CIs include 0) on go_emotions / ag_news; NB->LR is the
whole "proposed over existing" gain. results/sensitivity_case + ablation: go_emotions +1.15 [+0.46,+1.85] with case kept = lowercasing alone.
results/bpe: hybrid below LexFront alone on all 3 datasets (17/24 CIs exclude 0, all down). results/subword2: no accuracy gain over NLTK+LR; 16/30 CIs exclude 0.
OPEN: one training run per cell, no multiplicity correction; Unigram rounding margin ~5x noise (AUDIT_6); why differences appear = hypotheses only.
OUT OF SCOPE (not done): transformers/DistilBERT (needs torch), Hinglish, more seeds.
NOT VERIFIED: the owner's Colab traceback was never supplied; the notebook failure fixed is the one reproduced by simulation (stale dirty clone). Other
simulations (3.12, newest and Colab-era older libraries, empty dir) ran clean.
CHECK: `pytest -q`, `python notebooks/make_notebook.py --check`, `python -m experiments.render_docs && git diff --exit-code`, CI stats step, `python -m experiments.compare_results`.
