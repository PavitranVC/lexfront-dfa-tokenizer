# Phase A (accuracy levers on Sentiment140): protocol, written before any full-data result of this phase existed

Status: fixed 2026-10-18 in the commit that adds this file. Rules below are not changed after results are seen; if one must change, the change and
its reason go in `docs/DECISIONS.md`. Numbers do not appear here; they are generated into `results/boost_sentiment140/REPORT.md`.
Background and the baselines to compare with: `docs/NEXT_PLAN.md`, `docs/LARGE_PROTOCOL.md`, `results/large/REPORT.md`.

## Question
`results/large` tuned only a small grid (word 1-2-grams, `C` in 1/3/10, `alpha` in 0.1/0.3/1). With a wider search that is identical for every tokenizer,
(a) how much does accuracy rise, (b) does the tokenizer comparison change, and (c) how does a tokenizer-free character n-gram TF-IDF baseline compare?

## Data and split (unchanged from `docs/LARGE_PROTOCOL.md`)
Same cleaned user-grouped Sentiment140 split built by `large.build_dataset` (full train split, full test split, 0 users in both). Clean test text only
(no noise test in this phase). Labels: 0 negative, 1 positive. Majority-class accuracy is reported next to every table.

## Settings applied identically to every tokenizer (whitespace, NLTK, TweetTokenizer, LexFront)
* Selection data: a stratified random slice of the TRAIN split only (seed 0): 100 000 examples to fit candidates, a disjoint 20 000 to score them.
  The test split is never used to choose anything. Winner = highest validation accuracy, ties go to the earlier candidate in the grid order below.
  The winner is refit on the whole train split and scored once on the test split.
* Vectorizer grid (TF-IDF, lower-cased, same for all): word n-gram upper bound 1, 2, 3 (lower bound 1) x sublinear tf off / on x `min_df` 2 / 5 = 12 candidates
  (order: n-gram bound, then sublinear, then `min_df`; the first is the committed default unigram, `min_df` 2).
* Logistic Regression `C` in 0.3, 1, 3, 10, 30 (`max_iter` 1000); Naive Bayes `alpha` in 0.03, 0.1, 0.3, 1. Candidates = vectorizer x value, per classifier.
* Character baseline (no tokenizer): `TfidfVectorizer(analyzer="char_wb")`, lower-cased, sublinear tf, `min_df` 5, n-gram range (2,4) or (3,5), same `C` / `alpha` grids,
  same selection data and rule, 2 candidates for the vectorizer. If it does not fit in memory on the full train split it is reported as not run (no silent subsample).
* Anchor: the default setting of `results/large` (unigram, `min_df` 2, LR default, NB default) is re-run for every tokenizer in the same code path; it must reproduce the
  committed predictions exactly or the run is considered faulty.
* References for paired differences: the committed `tuned` and `default` full-data predictions of `results/large` (same test set).

## Statistics
Primary outcome: accuracy on the clean test set, Logistic Regression. Percentile bootstrap over test tweets (B = 2000, seed 20261017, paired, shared resamples);
verdicts relative to the practical margin of +-0.5 accuracy points, as in `docs/LARGE_PROTOCOL.md`: `within` (CI inside the margin), `beyond (up/down)`
(CI entirely beyond), otherwise `unclear`. Macro-F1 is stored too. One training run per cell, no multiplicity correction, selection noise on the 20 000 validation
examples is not covered by the CIs.

## Hypotheses (not claims)
* HA1. For each of the 4 tokenizers, wide-tuned LR accuracy minus the committed tuned LR accuracy is `beyond (up)`. Observed = all 4; opposite = any `beyond (down)`; otherwise not observed.
* HA2. Wide-tuned LR, LexFront minus NLTK and LexFront minus TweetTokenizer are both `within`. Observed = both within; opposite = either `beyond`; otherwise not observed.
* HA3. The spread (largest minus smallest accuracy over the 4 tokenizers, LR) is smaller wide-tuned than at the default anchor. Observed if the point values say so.
* HA4. The character baseline is not worse than wide-tuned LexFront (LR): char minus LexFront is `within` or `beyond (up)`. Opposite = `beyond (down)`; otherwise not observed.
* HA5. Wide-tuned NB with LexFront improves on the committed tuned NB with LexFront: `beyond (up)`.

## What this does not show
One dataset, one language, one seed per fit, labels derived from emoticons, no noise test, no classifier family beyond NB and LR, no transformer. A gain from wider
settings says nothing about why; differences between tokenizers are read against the margin, not against zero.
