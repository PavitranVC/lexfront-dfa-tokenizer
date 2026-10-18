# Next plan: raise classifier accuracy and widen the datasets

Status: written 2026-10-18, after PR #26 (results/large merged). Nothing here has been run. Numbers are quoted from stored results
(`results/large/REPORT.md`, `results/large/runs.json`) or labelled as a quick probe that is not a stored result. The first step of any
session that picks this up is to read `docs/HANDOFF.md`, `docs/FINAL_SUMMARY.md`, `docs/LARGE_PROTOCOL.md` and `results/large/REPORT.md`.

## What "the model" is
The classifier (TF-IDF + Naive Bayes or Logistic Regression) is a standard supervised ML model: it learns from labelled data, and accuracy
is a property of it. LexFront is a rule-based lexer: nothing is trained, and it has no accuracy of its own. The project is a rule-based front
end feeding a learned back end, so "raise accuracy" means improving the back end, and any such change must be applied equally to every
tokenizer or the tokenizer comparison stops being fair.

## Ground rules (unchanged)
Additive only: no existing metric under `results/` may change (`python -m experiments.compare_results [--additive] <main>/results results`
must report 0 differences). CPU only and no torch, unless the owner explicitly approves phase C. Every number in docs comes from a results JSON
written by a script that ran. No "better / first / novel". Hypotheses labelled. Never force-push. Notebooks are generated (`make_*notebook.py`,
`--check` in CI). Write the protocol and hypotheses BEFORE running, like `docs/LARGE_PROTOCOL.md`. Merge only when CI is green on the exact head SHA.
The owner runs everything heavy on Google Colab (CPU, about 12 GB RAM, a phone hotspot, privacy-first); sessions can disconnect, so runs must resume.

## Baselines to beat (full Sentiment140, accuracy %, from `results/large/REPORT.md`)
Default settings, Logistic Regression: whitespace 79.60, NLTK 79.88, TweetTokenizer 79.96, LexFront 79.85. Tuned (validation-selected 1-2-grams, `C`):
81.44, 82.37, 82.23, 82.54. Naive Bayes, default / tuned, LexFront: 77.33 / 80.49. LexFront minus NLTK stayed inside the +-0.5 point practical margin.

## Phase A: cheap CPU gains (one PR, run on Colab)
Scope, all applied to every tokenizer with the same search: wider `alpha` and `C` grids (the earlier NB spot check was informal and unsaved), word 1-3-grams,
`min_df` and sublinear tf, and a character n-gram TF-IDF baseline (the natural competitor for noisy text, currently missing).
Compute reference (stored in `results/large/runs.json`, Colab CPU, both classifiers, including the three noisy test variants): one full-training fit set took
81 to 263 s with default settings and 207 to 505 s tuned, depending on the tokenizer; the whole 100-fit grid took 1.47 hours; peak memory 3.26 GB.
Risks: character n-grams on 1.5 M tweets can exceed Colab RAM (use `HashingVectorizer`, a cap on features, or run at 100k and 1M and say so);
a wider search multiplies search cost but not the final fits. Deliverables: protocol + hypotheses doc, a `results/<name>/` folder, a generated report, tests,
a resumable Colab notebook (generated), CI dry run. Expected gain: modest and uncertain; for scale, tuning already moved LR by about 2 to 2.7 points (REPORT).

## Phase B: other classifier families (optional, CPU)
Linear SVM, NB-SVM, simple ensembles, SGD for speed. Same fairness rule. Only after A shows whether classifier changes move the tokenizer comparison at all.

## Phase C: transformer baseline (needs the owner's approval, GPU)
Fine-tune a small pretrained model (for example DistilBERT or a tweet-trained one) on a subset. Needs torch and a Colab T4, which the project rules currently exclude.
A transformer uses its own tokenizer, so this raises accuracy but does not test LexFront; testing LexFront inside it would be a separate experiment. Likely the
largest accuracy gain, not measured here. Scope and cost are the largest of all phases.

## Phase D: more large datasets (a loader per dataset, then runs)
Quick probe on 2026-10-18 (not a stored result; sizes from the Hub, noise rates on the first 3 000 rows streamed):
| Dataset | Rows | Notes |
|---|---|---|
| `google/civil_comments` | 2.0 M | loads; user comments, mean about 340 chars, 1.1% emoticons, 3.0% elongation; continuous toxicity score (threshold needed), toxic is the rare class |
| `fancyzhx/amazon_polarity` | 3.6 M | loads; clean reviews, a clean control |
| `Yelp/yelp_review_full`, `fancyzhx/yelp_polarity` | 650 k, 560 k | load; clean reviews |
| `fancyzhx/dbpedia_14` | 560 k | loads; clean topic labels |
| `TLeonidas/twitter-hate-speech-en-240ksamples` | 243 k | loads; very noisy tweets (58% hashtags, 74% mentions), label provenance unclear |
| `webis/tldr-17` | 1.3 M | script-based, refused by `datasets` 4+; a parquet copy might work, unverified |
Per dataset: a loader (no scripts), license check, label provenance, an audit like `audit.json`, the Hub commit SHA in the results, the cleaning pass, a grouped split
when author or thread ids exist, `clean.py` rules, and a size ladder only if it adds something. Mixing datasets is worth doing only for a reason: a cross-domain test
(train on one source or a mix, test on another, for example Sentiment140 to `tweet_eval` sentiment, with Amazon polarity as a clean control). Mix only datasets with the same
label meaning, report each source separately, and check cross-source duplicates. Do not mix for size alone. Do not scrape X/Twitter (terms and privacy).

## Cost guidance
Credits cannot be quoted from here. Relative order, cheapest first: A, B, one new dataset (D), C. Colab compute does not use the assistant's credits; the assistant's cost is
reading, writing, testing and the PR cycle. Measure usage before and after phase A and scale from that. One phase per conversation keeps context small.

## Open items
The NB `alpha` grid check is not saved. The Colab run used Python 3.13 and scikit-learn 1.6.1 (outside the tested 3.11 / 3.12). One run per cell, no multiplicity
correction. LexFront's emoticon class cannot be tested on data like Sentiment140, whose emoticons were largely stripped (1.93% of its training tweets contain one).
