# Phase D (three more large datasets): protocol, written before any full-data result of these datasets existed

Status: fixed 2026-10-18, before any run on these datasets. The only runs so far are small dry runs of the code (not results). Rules are not changed after
results are seen; a change and its reason go in `docs/DECISIONS.md`. Numbers do not appear here; they are generated into `results/boost_<dataset>/REPORT.md` by
the Colab notebook `notebooks/LexFront_Boost_<dataset>.ipynb`. The engine and settings are those of `docs/BOOST_PROTOCOL.md` (same grids, same selection rule,
same margin of +-0.5 points, same bootstrap), applied unchanged to every dataset and tokenizer; datasets are never mixed.

## Datasets
| name | Hub id | label meaning | train used | test used |
|---|---|---|---|---|
| `civil_comments` | `google/civil_comments` | 1 = toxicity score >= 0.5, else 0 | seeded random 500 000 of 1.8 M | official test split (about 97 k) |
| `amazon_polarity` | `fancyzhx/amazon_polarity` | 1 = positive review (4-5 stars), 0 = negative (1-2 stars); text = title + ' ' + content | seeded random 500 000 of 3.6 M | seeded random 100 000 of the 400 k official test |
| `yelp_full` | `Yelp/yelp_review_full` | 5 classes = 1 to 5 stars | seeded random 300 000 of 650 k | seeded random 50 000 of the official test |

* Threshold: 0.5 is the usual Civil Comments / Jigsaw convention (at least half of the raters called the comment toxic). It is fixed here, not tuned; no other threshold is run.
* Class balance: Amazon polarity and Yelp are balanced by construction; Civil Comments is imbalanced (toxic is the rare class), so its majority-class accuracy is high. For Civil
  Comments the primary outcome is macro-F1 (accuracy is still reported), for the others accuracy. Every report states the majority-class accuracy and the class counts.
* Split: no dataset has an author id, so the official Hub train / test split is used and cannot be author-grouped; exact duplicates, conflicting labels and test-in-train texts are
  removed by `experiments/clean.py` (after the cap). Possible author or near-duplicate overlap that the data does not expose is a stated limit.
* Selection (the engine of phase A): a stratified 100 000 / 20 000 slice of the TRAIN subsample only; accuracy on that slice picks the settings (also for Civil Comments).
* Character baseline: if it does not fit in memory on the Colab runtime it is reported as not run (as in `docs/BOOST_PROTOCOL.md`); the notebook runs it in its own process for that reason.
* Licenses to check before reuse: Civil Comments CC0 (Hub card); Amazon polarity Apache-2.0 (Hub card); Yelp: Hub card says 'other' (Yelp dataset terms, academic use).

## Hypotheses (per dataset, same rules as phase A; HA2-HA4 of the report)
* HA2. Wide-tuned LR, LexFront minus NLTK and LexFront minus TweetTokenizer on the primary metric are both `within` the margin.
* HA3. The spread over the 4 tokenizers (primary metric, LR) is smaller wide-tuned than at the default setting.
* HA4. The character baseline minus wide-tuned LexFront (LR, primary metric) is `within` or `beyond (up)`.
Verdicts, bootstrap and margin as in `docs/BOOST_PROTOCOL.md`; "not observed" is an equally reportable result. These datasets have no committed earlier results, so HA1 and HA5 do not apply
and the anchor row (default settings) is the only reference.

## What this does not show
Review and comment text, English, one run per cell; the noise features of tweets (emoticons, elongation, mentions) are rare here, so these datasets mostly test the clean-text case.
No transformer, no classifier family beyond NB and LR, no author-grouped split.
