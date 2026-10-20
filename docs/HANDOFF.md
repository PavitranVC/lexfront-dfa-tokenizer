# HANDOFF (updated 2026-10-18; phase A merged (PR #28); D1 merged (PR #29); D2 merged (PR #30), D3 yelp_full in PR)
PHASE A (docs/BOOST_PROTOCOL.md, results/boost_sentiment140/, REPORT.md generated; branch feature/large-scale-experiments, protocol commit precedes results commit).
D1-D3: loader `experiments/datasets_big.py`, protocol `docs/D_PROTOCOL.md`, notebooks `notebooks/LexFront_Boost_{civil_comments,amazon_polarity,yelp_full}.ipynb` (owner runs on Colab, results -> results/boost_<dataset>/). Quick dry run done locally and in CI; no full-data result exists.
RESULT SUMMARY: see results/boost_sentiment140/REPORT.md (HA2 observed; HA1, HA3, HA5 not observed; HA4 opposite: char n-grams 2.7 pts below LexFront LR).
CHECK: `pytest -q`, `python -m experiments.run_boost --stats-only --out results/boost_sentiment140`, `python -m experiments.verify_boost --out results/boost_sentiment140 --ref results/large --protocol docs/BOOST_PROTOCOL.md --base origin/main`.
OLDER (unchanged): results/large done (LexFront-NLTK within +-0.5 pt; H1 weak, H2 default-LR only, H3 not observed); transformers, Hinglish, more seeds out of scope.
CI: .github/workflows/ci.yml has a new stats-only step for results/boost_sentiment140 (phase A).
YELP COST: yelp_full LR wide units took ~30 min each even on the 27k dry-run sample, so no CI dry run and a full run of a day or more; see docs/D_PROTOCOL.md cost note. Reduced grid needs a protocol amendment first.
