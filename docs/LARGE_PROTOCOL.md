# Large-scale test on Sentiment140: protocol (written before the full run)

Status: protocol and hypotheses fixed on 2026-10-17, before any full-data result existed. The full run was completed on Colab (results: `results/large/`, no change to this protocol; see `docs/DECISIONS.md`). If the protocol changes after results
are seen, the change and its reason go in `docs/DECISIONS.md`. Numbers do not appear here; they are generated into
`results/large/REPORT.md` from the stored JSON.

## Question
Do the differences between pre-tokenizers seen on small datasets (`results/collection/REPORT.md`) persist, shrink or vanish on a
very large noisy dataset, and how do the tokenizers behave under extra noise and at scale (speed, memory)?

## Data
* `bdanko/sentiment140` (Hugging Face): 1.6 M tweets, binary polarity (labels 0 / 4 mapped to 0 / 1), raw text with `user` and
  `date` columns. The commit SHA of the dataset repo is recorded in `split.json`. A community re-upload, not the original host.
* Split by user, not by tweet: users are shuffled with a fixed seed and 5% of them (all their tweets) become the test set, so no
  author is in both splits (author leakage was a flaw of the base report this project started from).
* Cleaning after the split with the existing `experiments/clean.py` (empty, conflicting, duplicate, test-in-train); texts are not
  edited. `audit.json` records what the data looks like (emoticon, URL, mention, hashtag, elongation and HTML-entity rates per class);
  Sentiment140 labels were derived from emoticons by its authors, so how many emoticons survive in the text is checked, not assumed.

## Tokenizers and classifiers
Whitespace, NLTK, TweetTokenizer, LexFront, with TF-IDF and Naive Bayes / Logistic Regression exactly as in the main grid
(`default`), and the validation-selected setting of `experiments/tuning.py` (`tuned`, same grid for every tokenizer).

## Size ladder
Train sizes 10 000 and 100 000 (5 random subsets each), 1 000 000 (3 subsets) and the full cleaned train split (1 run). Every run is
scored on the same held-out test set. The `tuned` setting is run at 10 000, 100 000 and full only (cost). Subsets of one size overlap
(they come from the same pool), so the spread between repeats understates the variation across independent training sets.

## Noise test
The test text (never the training text) is also scored after seeded corruption of words that are at least 4 letters long and purely
alphabetic (mentions, hashtags and URLs are left alone): `light` = each such word is corrupted with probability 0.10, `heavy` = 0.30;
one corruption per word, chosen from swap two inner letters, drop an inner letter, double a letter, elongate a vowel (`good` ->
`goooood`). Case is not corrupted because TF-IDF lowercases for every tokenizer. Same noisy test text for every model.

## Statistics
Full-data runs: percentile bootstrap over test examples (B = 2000, paired, the same resamples for all runs), difference in accuracy
points, LexFront minus NLTK and LexFront minus TweetTokenizer, Logistic Regression. Ladder runs: mean and standard deviation over
subsets, and the same paired difference per subset. With this many test tweets even tiny differences have narrow CIs, so every
difference is also placed against a practical margin of +-0.5 accuracy points fixed here in advance: CI inside the margin =
"within the margin", CI outside it = "beyond the margin", otherwise "unclear". One training run per cell, no multiplicity correction.

## Hypotheses (motivated by the 8-dataset results; not claims)
* H1. The accuracy gap between LexFront and NLTK (Logistic Regression, default) shrinks as the training size grows.
* H2. LexFront loses less accuracy than NLTK from the clean to the heavy-noise test text.
* H3. With the tuned setting the differences between tokenizers are smaller than with the default setting.
The report states for each whether it was observed, with the numbers; "not observed" is an acceptable and equally reportable result.

## What this does not show
One dataset, one source (tweets), one language, labels with known noise, a synthetic noise model, no transformers, one seed per
training set. It does not show that any tokenizer is better in general, and it gives no reason for any difference.
