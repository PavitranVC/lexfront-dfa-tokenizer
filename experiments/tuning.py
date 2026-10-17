"""Validation-selected classifier settings, the same search for every tokenizer.

The main grid fixes TF-IDF and the classifier at their defaults so that only the tokenizer varies. This module asks
the follow-up question "how much accuracy do cheap, standard settings add, and does the tokenizer comparison survive
them?" without giving any tokenizer extra effort:

  * the search grid is identical for every (dataset, tokenizer, classifier): TF-IDF word 1-grams vs 1-2-grams with
    sublinear tf, and Logistic Regression C or Naive Bayes alpha;
  * candidates are scored on a validation slice carved out of the TRAIN split (fixed seed, stratified when every
    class has at least two examples); the test split is never used to choose anything;
  * the winner (highest validation accuracy, ties go to the earlier, simpler candidate) is refit on the whole train
    split and evaluated on the test split once.

Like `pipeline.py`, every fit runs single-threaded and each result JSON carries a fingerprint, so a cached run is reused
only while its configuration and code are unchanged.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from . import classifiers as clf_mod
from . import pipeline
from .metrics import classification_metrics
from .tokenizers import build_tokenizer

SEARCH = {
    "vectorizer": {"unigram": clf_mod.VECTORIZER_PARAMS,
                   "bigram_sublinear": {**clf_mod.VECTORIZER_PARAMS, "ngram_range": [1, 2], "sublinear_tf": True}},
    "lr_C": [1.0, 3.0, 10.0],
    "nb_alpha": [0.1, 0.3, 1.0],
}
VAL_FRACTION = 0.1
VAL_MAX = 5000            # cap on the validation slice
FIT_MAX = 20000           # cap on the examples a candidate is fit on while searching (the final refit uses all of train)
TUNED_CLASSIFIERS = ("nb", "lr")
CODE_FILES = ["experiments/tuning.py"]


def hyper_grid(classifier: str) -> list:
    return SEARCH["lr_C"] if classifier == "lr" else SEARCH["nb_alpha"]


def make_model(classifier: str, value: float, seed: int):
    if classifier == "lr":
        return LogisticRegression(C=value, max_iter=1000, random_state=seed)
    return MultinomialNB(alpha=value)


def split_indices(labels, seed: int) -> tuple[list, list]:
    """(fit indices, validation indices) of the train split; deterministic, stratified when possible."""
    idx = np.arange(len(labels))
    n_val = min(VAL_MAX, max(1, round(len(labels) * VAL_FRACTION)))
    counts = np.bincount(np.asarray(labels))
    stratify = labels if counts[counts > 0].min() >= 2 and n_val >= len(counts) else None
    fit, val = train_test_split(idx, test_size=n_val, random_state=seed, stratify=stratify)
    return sorted(fit[:FIT_MAX].tolist()), sorted(val.tolist())


def cached(tokenizer):
    """Memoize a tokenizer by input string: the same text is tokenized once across candidates (identical output)."""
    memo = {}

    def tokenize(text):
        if text not in memo:
            memo[text] = tuple(tokenizer(text))
        return list(memo[text])
    return tokenize


def code_fingerprint() -> str:
    h = hashlib.sha256(pipeline.code_fingerprint().encode())
    for rel in CODE_FILES:
        h.update((pipeline.REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def expected_fingerprint(ds, tokenizer_name: str, classifier: str, seed: int) -> dict:
    config = {"dataset": ds.summary(), "data_hash": ds.content_hash(), "tokenizer": tokenizer_name,
              "classifier": classifier, "search": SEARCH, "val_fraction": VAL_FRACTION, "val_max": VAL_MAX,
              "fit_max": FIT_MAX, "seed": seed, **pipeline.library_versions(), "blas_threads": pipeline.BLAS_THREADS}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {"config": digest, "code": code_fingerprint()}


def search(ds, tokenizer, classifier: str, seed: int) -> dict:
    """Score every candidate on the validation slice; returns the selection record (no test data touched)."""
    fit_idx, val_idx = split_indices(ds.train_labels, seed)
    fit_x = [ds.train_texts[i] for i in fit_idx]
    val_x = [ds.train_texts[i] for i in val_idx]
    fit_y = [ds.train_labels[i] for i in fit_idx]
    val_y = np.asarray([ds.train_labels[i] for i in val_idx])
    scores, best = [], None
    with pipeline.single_threaded():
        for vec_name, params in SEARCH["vectorizer"].items():
            vec = clf_mod.make_vectorizer(tokenizer, params)
            x_fit, x_val = vec.fit_transform(fit_x), vec.transform(val_x)
            for value in hyper_grid(classifier):
                accuracy = float((make_model(classifier, value, seed).fit(x_fit, fit_y).predict(x_val) == val_y).mean())
                scores.append({"vectorizer": vec_name, "value": value, "val_accuracy": accuracy})
                if best is None or accuracy > best["val_accuracy"]:
                    best = scores[-1]
    return {"n_fit": len(fit_idx), "n_val": len(val_idx), "scores": scores, "selected": best}


def run_cell(ds, tokenizer_name: str, out_dir, seed: int = 0, force: bool = False, log=print) -> list[dict]:
    """Tune and evaluate NB and LR for one (dataset, tokenizer); results go to `<out_dir>/<dataset>__<tok>__<clf>.json`."""
    paths = {c: pipeline.result_path(out_dir, ds.name, tokenizer_name, c) for c in TUNED_CLASSIFIERS}
    expected = {c: expected_fingerprint(ds, tokenizer_name, c, seed) for c in TUNED_CLASSIFIERS}
    todo = [c for c in TUNED_CLASSIFIERS if force or not pipeline.is_fresh(paths[c], expected[c])]
    tokenizer = cached(build_tokenizer(tokenizer_name, ds.train_texts))
    final_features = {}                    # vectorizer option -> (x_train, x_test, n_features, seconds)
    for classifier in todo:
        start = time.perf_counter()
        selection = search(ds, tokenizer, classifier, seed)
        chosen = selection["selected"]
        if chosen["vectorizer"] not in final_features:
            began = time.perf_counter()
            params = SEARCH["vectorizer"][chosen["vectorizer"]]
            vec = clf_mod.make_vectorizer(tokenizer, params)
            with pipeline.single_threaded():
                x_train, x_test = vec.fit_transform(ds.train_texts), vec.transform(ds.test_texts)
            final_features[chosen["vectorizer"]] = (x_train, x_test, len(vec.vocabulary_), time.perf_counter() - began)
        x_train, x_test, n_features, vec_seconds = final_features[chosen["vectorizer"]]
        model = make_model(classifier, chosen["value"], seed)
        with pipeline.single_threaded():
            fit_start = time.perf_counter()
            model.fit(x_train, ds.train_labels)
            fit_seconds = time.perf_counter() - fit_start
            predictions = model.predict(x_test)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "setting": "tuned", "dataset": ds.summary(), "data_hash": ds.content_hash(), "tokenizer": tokenizer_name,
            "vectorizer": {"type": "TfidfVectorizer", **SEARCH["vectorizer"][chosen["vectorizer"]],
                           "name": chosen["vectorizer"], "n_features": n_features},
            "classifier": {"name": classifier, "params": model.get_params()},
            "tuning": selection, "seed": seed, "fingerprint": expected[classifier],
            "metrics": classification_metrics(ds.test_labels, predictions),
            "timing_seconds": {"tokenize_and_vectorize": vec_seconds, "classifier_fit": fit_seconds,
                               "search_and_fit_total": time.perf_counter() - start},
            "environment": {**pipeline.environment(), "blas_threads": pipeline.BLAS_THREADS},
        }
        record["predictions_file"] = pipeline.predictions_relpath(paths[classifier])
        pipeline.save_predictions(paths[classifier].parent / record["predictions_file"], ds.test_labels, predictions)
        paths[classifier].parent.mkdir(parents=True, exist_ok=True)
        paths[classifier].write_text(json.dumps(record, indent=2))
        log(f"{ds.name:22s} {tokenizer_name:10s} {classifier:3s} tuned {chosen['vectorizer']}/{chosen['value']} "
            f"val={chosen['val_accuracy']:.4f} acc={record['metrics']['accuracy']:.4f}")
    return [json.loads(Path(paths[c]).read_text()) for c in TUNED_CLASSIFIERS]
