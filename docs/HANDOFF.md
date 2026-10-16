# HANDOFF (written by FIXAUDIT, 2026-10-16)
DONE: fixed docs/AUDIT.md M1-M3, m1-m13: fingerprinted result cache (experiments/pipeline.py), generated obs:attribution, notebook setup/FORCE, pinned deps, CI checks, overnight mode (docs/PIPELINE.md), docs/versions/v4-tokenizer-lineup.md.
NUMBERS: none new; all 66 run JSONs regenerated with --force, metrics identical (results/module*/*.json, results/quick/module*/*.json). TweetTopic proposed - existing = -0.26 pts (results/module1/tweet_topic__nltk__nb.json, results/module2/tweet_topic__lexfront__lr.json).
OPEN: TF-IDF lowercasing vs LexFront lowercase rule (P0). Committed JSONs go stale after any change to lexer/ or the experiment modules in pipeline.CODE_GLOBS; rerun to refresh.
DECISIONS: M1 fingerprint {config, code}, no committed-freshness test; m1 CLI guard instead of new RNGs; m2 exact slang lookup; m3/m4 documented, no spec change (all in docs/DECISIONS.md).
NEXT: P0 - decide TF-IDF lowercase=False vs dropping the rule; rerun affected results (run_cell recomputes stale cells).
FILES TOUCHED: experiments/{pipeline,report,observations,render_docs,review1,figures,run_module2}.py, lexer/normalize.py, notebooks/*, tests/*, results/**, docs/*, requirements.txt, .github/workflows/ci.yml, CHANGELOG.md
CHECK: `pytest -q` (all pass) and `python -m experiments.render_docs && git diff --exit-code`
