# HANDOFF (updated 2026-10-18, phase A merged with an owner-approved LR anchor tolerance of 0.10 pt; D1-D3 in progress)
PHASE A (docs/BOOST_PROTOCOL.md, results/boost_sentiment140/, REPORT.md generated; branch claude/fix-lexfront-notebook-setup-brk26t, protocol commit precedes results commit).
RESULT SUMMARY: see results/boost_sentiment140/REPORT.md (HA2 observed; HA1, HA3, HA5 not observed; HA4 opposite: char n-grams 2.7 pts below LexFront LR).
CHECK: `pytest -q`, `python -m experiments.run_boost --stats-only --out results/boost_sentiment140`, `python -m experiments.verify_boost --out results/boost_sentiment140 --ref results/large --protocol docs/BOOST_PROTOCOL.md --base origin/main`.
OLDER (unchanged): results/large done (LexFront-NLTK within +-0.5 pt; H1 weak, H2 default-LR only, H3 not observed); transformers, Hinglish, more seeds out of scope.
CI: .github/workflows/ci.yml has a new stats-only step for results/boost_sentiment140 (phase A).
