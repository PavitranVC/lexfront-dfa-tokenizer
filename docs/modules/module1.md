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
            for clf in [majority, naive_bayes, logistic_regression]:
                clf.fit(X_train, y_train)
                record <- {config, accuracy(y_test, clf.predict(X_test)), macro_f1(...), timings, versions}
                write record to results/<module>/<dataset>__<tokenizer>__<clf>.json
    tables, figures, observations are built only from those JSON files
```

## Results
Generated from `results/module1/*.json` and `results/datasets.json`.

<!-- BEGIN table:datasets -->
_pending_
<!-- END table:datasets -->

<!-- BEGIN table:module1_results -->
_pending_
<!-- END table:module1_results -->

<!-- BEGIN obs:module1 -->
_pending_
<!-- END obs:module1 -->

Figure: `results/figures/module1_accuracy.png`.
