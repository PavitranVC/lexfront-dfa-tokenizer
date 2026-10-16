# Module 1 — data and existing methodology

## What it does
Compares three **existing tokenizers** with the classifier pipeline held fixed.

* Datasets (`experiments/data.py`): `cardiffnlp/tweet_topic_single`, `google-research-datasets/go_emotions`
  (`simplified`, single-label only), `fancyzhx/ag_news`. Each loader returns train/test texts, labels, label names and a
  `split_info` record saying which splits were used and why.
* Tokenizers (`experiments/tokenizers.py`), all `tokenize(text) -> list[str]`: whitespace split, NLTK `word_tokenize`,
  NLTK `TweetTokenizer`.
* Classifiers (`experiments/classifiers.py`): TF-IDF (`tokenizer=fn, token_pattern=None, lowercase=True, min_df=2`)
  feeding a majority baseline, Multinomial Naive Bayes (the existing method) and Logistic Regression
  (`max_iter=1000`, seed 0).
* Metrics: accuracy and macro-F1. Each run is written to `results/module1/<dataset>__<tokenizer>__<clf>.json` with
  its full configuration, timings and library versions.

Run: `python -m experiments.run_module1` (add `--max-train N --max-test N` to subsample with the fixed seed).

## Pseudocode: the evaluation loop
```text
(3) Evaluation loop

    for dataset in [tweet_topic, go_emotions, ag_news]:
        ds <- load(dataset)                         # train/test texts, labels, label names
        (optionally subsample train/test with a fixed seed)
        for tokenizer in [whitespace, nltk, tweet] (module 1) or [lexfront] (module 2):
            X_train <- TfidfVectorizer(tokenizer, token_pattern=None, lowercase=True, min_df=2).fit_transform(train)
            X_test  <- vectorizer.transform(test)
            for clf in [majority, naive_bayes, logistic_regression] (module 1) or [naive_bayes, logistic_regression] (module 2):
                clf.fit(X_train, y_train)
                record <- {config, accuracy(y_test, clf.predict(X_test)), macro_f1(...), timings, versions}
                write record to results/<module>/<dataset>__<tokenizer>__<clf>.json
    tables, figures, observations are built only from those JSON files
```

## Results
Generated from `results/module1/*.json` and `results/datasets.json`.

<!-- BEGIN table:datasets -->
| dataset | source | train split | test split | train docs | test docs | labels | largest test class % | subsample |
|---|---|---|---|---|---|---|---|---|
| tweet_topic | cardiffnlp/tweet_topic_single | train_coling2022 | test_coling2022 | 3598 | 3399 | 6 | 40.01 | none (full data) |
| go_emotions | google-research-datasets/go_emotions (simplified) | train | test | 36308 | 4590 | 28 | 34.99 | none (full data) |
| ag_news | fancyzhx/ag_news | train | test | 120000 | 7600 | 4 | 25.00 | none (full data) |
<!-- END table:datasets -->

<!-- BEGIN table:module1_results -->
| dataset | tokenizer | Majority acc / F1 | Naive Bayes acc / F1 | Logistic Reg. acc / F1 |
|---|---|---|---|---|
| tweet_topic | whitespace | 40.01 / 9.53 | 70.26 / 28.83 | 70.49 / 35.76 |
| tweet_topic | nltk | 40.01 / 9.53 | 70.99 / 28.56 | 72.32 / 36.95 |
| tweet_topic | tweet | 40.01 / 9.53 | 70.79 / 28.31 | 72.02 / 36.70 |
| go_emotions | whitespace | 34.99 / 1.85 | 37.73 / 4.93 | 53.51 / 31.40 |
| go_emotions | nltk | 34.99 / 1.85 | 37.78 / 4.88 | 55.27 / 33.86 |
| go_emotions | tweet | 34.99 / 1.85 | 37.80 / 4.86 | 55.47 / 34.02 |
| ag_news | whitespace | 25.00 / 10.00 | 90.14 / 90.12 | 91.43 / 91.42 |
| ag_news | nltk | 25.00 / 10.00 | 90.38 / 90.35 | 91.88 / 91.87 |
| ag_news | tweet | 25.00 / 10.00 | 90.22 / 90.20 | 91.84 / 91.83 |
<!-- END table:module1_results -->

<!-- BEGIN obs:module1 -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: best accuracy 72.32% (nltk + LR). Across the three tokenizers LR is +0.93 points vs NB on average; the tokenizer spread is 1.82 points under LR and 0.74 points under NB.
- **go_emotions**: best accuracy 55.47% (tweet + LR). Across the three tokenizers LR is +16.98 points vs NB on average; the tokenizer spread is 1.96 points under LR and 0.07 points under NB.
- **ag_news**: best accuracy 91.88% (nltk + LR). Across the three tokenizers LR is +1.47 points vs NB on average; the tokenizer spread is 0.45 points under LR and 0.24 points under NB.

_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file `stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score moves with the draw of the test set, not with training randomness, which one run per cell cannot show (the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed subsample). A gap whose CI includes 0 should not be read as a real difference._
<!-- END obs:module1 -->

Each run also stores its test predictions (`results/module1/predictions/<run>.json.gz`), so confidence intervals need
no refit. Per-run 95% bootstrap CIs over the test examples (`python -m experiments.run_stats`):

<!-- BEGIN table:ci_runs -->
| dataset | tokenizer | classifier | accuracy % [95% CI] | macro-F1 % [95% CI] |
|---|---|---|---|---|
| tweet_topic | whitespace | Naive Bayes | 70.26 [68.76, 71.76] | 28.83 [27.88, 29.74] |
| tweet_topic | whitespace | Logistic Reg. | 70.49 [68.87, 72.08] | 35.76 [34.24, 37.37] |
| tweet_topic | nltk | Naive Bayes | 70.99 [69.52, 72.49] | 28.56 [27.76, 29.44] |
| tweet_topic | nltk | Logistic Reg. | 72.32 [70.90, 73.87] | 36.95 [35.34, 38.58] |
| tweet_topic | tweet | Naive Bayes | 70.79 [69.29, 72.29] | 28.31 [27.53, 29.17] |
| tweet_topic | tweet | Logistic Reg. | 72.02 [70.58, 73.61] | 36.70 [35.13, 38.28] |
| tweet_topic | lexfront | Naive Bayes | 70.43 [68.90, 71.93] | 27.84 [27.12, 28.62] |
| tweet_topic | lexfront | Logistic Reg. | 70.73 [69.23, 72.29] | 36.18 [34.64, 37.84] |
| go_emotions | whitespace | Naive Bayes | 37.73 [36.34, 39.15] | 4.93 [4.47, 5.47] |
| go_emotions | whitespace | Logistic Reg. | 53.51 [52.11, 54.95] | 31.40 [29.73, 33.40] |
| go_emotions | nltk | Naive Bayes | 37.78 [36.38, 39.19] | 4.88 [4.42, 5.39] |
| go_emotions | nltk | Logistic Reg. | 55.27 [53.81, 56.71] | 33.86 [32.07, 35.92] |
| go_emotions | tweet | Naive Bayes | 37.80 [36.43, 39.24] | 4.86 [4.41, 5.37] |
| go_emotions | tweet | Logistic Reg. | 55.47 [54.03, 56.91] | 34.02 [32.24, 36.04] |
| go_emotions | lexfront | Naive Bayes | 37.78 [36.40, 39.19] | 4.81 [4.37, 5.32] |
| go_emotions | lexfront | Logistic Reg. | 55.36 [53.97, 56.80] | 33.97 [32.12, 36.02] |
| ag_news | whitespace | Naive Bayes | 90.14 [89.49, 90.79] | 90.12 [89.47, 90.75] |
| ag_news | whitespace | Logistic Reg. | 91.43 [90.80, 92.04] | 91.42 [90.79, 92.02] |
| ag_news | nltk | Naive Bayes | 90.38 [89.72, 91.01] | 90.35 [89.70, 90.98] |
| ag_news | nltk | Logistic Reg. | 91.88 [91.25, 92.46] | 91.87 [91.23, 92.44] |
| ag_news | tweet | Naive Bayes | 90.22 [89.54, 90.87] | 90.20 [89.53, 90.85] |
| ag_news | tweet | Logistic Reg. | 91.84 [91.22, 92.43] | 91.83 [91.20, 92.41] |
| ag_news | lexfront | Naive Bayes | 90.08 [89.39, 90.74] | 90.05 [89.37, 90.69] |
| ag_news | lexfront | Logistic Reg. | 91.88 [91.26, 92.47] | 91.86 [91.23, 92.44] |
<!-- END table:ci_runs -->

Figure: `results/figures/module1_accuracy.png`.
