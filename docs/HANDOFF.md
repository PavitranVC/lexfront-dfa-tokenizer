# HANDOFF (updated 2026-10-18, phase A session, status: STOPPED, PR not opened, nothing merged)
PHASE A (docs/BOOST_PROTOCOL.md, results/boost_sentiment140/, REPORT.md generated; branch claude/fix-lexfront-notebook-setup-brk26t, protocol commit precedes results commit).
STOP REASON: self-check (h) FAILED. Re-running the committed default setting reproduces Naive Bayes predictions exactly (4/4) but not Logistic Regression (0/4): 766-1457 of 78278 test
predictions differ, accuracy differs by <= 0.07 pt (e.g. LexFront 79.84 vs committed 79.85). Tried 2 environments (3.11 lock; 3.13 with the Colab library versions, LexFront LR 79.86): neither matches,
so the cause looks like hardware / BLAS-dependent lbfgs, but that is a hypothesis. The protocol required exact reproduction and was not changed after seeing results (rule: a mismatch = stop).
DECISION NEEDED FROM OWNER: accept a tolerance for LR anchors (then amend protocol in DECISIONS.md, re-verify, open PR) or keep it as a fail. All other checks (a, c-g, i) PASS; check b not run (no PR text written).
RESULT SUMMARY: see results/boost_sentiment140/REPORT.md (HA2 observed; HA1, HA3, HA5 not observed; HA4 opposite: char n-grams 2.7 pts below LexFront LR).
NOT DONE: D1-D3 (civil_comments, amazon_polarity, yelp_full): scaffolding (loader, D protocol, notebook generator) was drafted in a scratch worktree only and is NOT pushed; redo after the owner's decision.
CHECK: `pytest -q`, `python -m experiments.run_boost --stats-only --out results/boost_sentiment140`, `python -m experiments.verify_boost --out results/boost_sentiment140 --ref results/large --protocol docs/BOOST_PROTOCOL.md --base origin/main`.
OLDER (unchanged): results/large done (LexFront-NLTK within +-0.5 pt; H1 weak, H2 default-LR only, H3 not observed); transformers, Hinglish, more seeds out of scope.
CI: .github/workflows/ci.yml has a new stats-only step for results/boost_sentiment140 (unmerged).
