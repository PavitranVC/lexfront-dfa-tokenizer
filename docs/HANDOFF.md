# HANDOFF (written by FIXAUDIT, 2026-10-16, status ready)
DONE: fixed docs/AUDIT.md M1-M3, m1-m13: fingerprinted result cache (experiments/pipeline.py), generated obs:attribution, notebook setup/FORCE, pinned deps, CI checks, overnight mode (docs/PIPELINE.md), docs/versions/v4-tokenizer-lineup.md.
REPRO: all fits single-threaded via threadpoolctl (environment.blas_threads=1, in fingerprint); thread count had changed LR metrics. All JSONs regenerated; LR cells rerun with 4 threads are identical.
NUMBERS: 6 LR runs changed by <= 0.18 pts (old -> new table in docs/DECISIONS.md "BLAS threads"); TweetTopic proposed - existing = -0.26 pts unchanged; go_emotions +17.58, ag_news +1.50 (results/REPORT_NOTES.md obs:attribution).
OPEN: TF-IDF lowercasing vs LexFront lowercase rule (P0). Committed JSONs go stale after any change to files in pipeline.CODE_GLOBS; rerun to refresh.
DECISIONS: BLAS threads = 1; M1 fingerprint {config, code}; m1 CLI guard; m2 exact slang lookup; m3/m4 documented (docs/DECISIONS.md).
NEXT: P0 - decide TF-IDF lowercase=False vs dropping the rule; rerun affected results (run_cell recomputes stale cells).
FILES TOUCHED: experiments/{pipeline,report,observations,render_docs,review1,figures,run_module2}.py, lexer/normalize.py, notebooks/*, tests/*, results/**, docs/*, requirements.txt, .github/workflows/ci.yml, CHANGELOG.md
CHECK: `pytest -q` (all pass) and `python -m experiments.render_docs && git diff --exit-code`
