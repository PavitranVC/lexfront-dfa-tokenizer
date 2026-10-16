"""The evaluation loop: dataset x tokenizer x classifier -> metrics -> JSON.

Every run writes one JSON file holding its full configuration, so tables and figures
can be rebuilt from `results/` alone.
"""
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

from . import classifiers as clf_mod
from .metrics import classification_metrics
from .tokenizers import get_tokenizer


def environment() -> dict:
    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "machine": platform.machine(), "platform": platform.platform()}


def result_path(out_dir, dataset: str, tokenizer: str, classifier: str) -> Path:
    return Path(out_dir) / f"{dataset}__{tokenizer}__{classifier}.json"


def vectorize(ds, tokenizer_name: str):
    """Fit TF-IDF on train, transform test; returns (X_train, X_test, seconds)."""
    vec = clf_mod.make_vectorizer(get_tokenizer(tokenizer_name))
    start = time.perf_counter()
    x_train = vec.fit_transform(ds.train_texts)
    x_test = vec.transform(ds.test_texts)
    return x_train, x_test, time.perf_counter() - start, len(vec.vocabulary_)


def run_one(ds, tokenizer_name: str, classifier_name: str, features, seed: int = 0) -> dict:
    """Fit one classifier on prepared features and return the result record."""
    x_train, x_test, vec_seconds, n_features = features
    clf = clf_mod.make_classifier(classifier_name, seed)
    start = time.perf_counter()
    clf.fit(x_train, ds.train_labels)
    fit_seconds = time.perf_counter() - start
    metrics = classification_metrics(ds.test_labels, clf.predict(x_test))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.summary(),
        "tokenizer": tokenizer_name,
        "vectorizer": {"type": "TfidfVectorizer", **clf_mod.VECTORIZER_PARAMS, "n_features": n_features},
        "classifier": {"name": classifier_name, "params": clf_mod.classifier_params(classifier_name, seed)},
        "seed": seed,
        "metrics": metrics,
        "timing_seconds": {"tokenize_and_vectorize": vec_seconds, "classifier_fit": fit_seconds},
        "environment": environment(),
    }


def run_cell(ds, tokenizer_name: str, classifier_names, out_dir, seed: int = 0, force: bool = False,
             log=print) -> list[dict]:
    """All classifiers for one (dataset, tokenizer); features are computed once."""
    todo = [c for c in classifier_names
            if force or not result_path(out_dir, ds.name, tokenizer_name, c).exists()]
    if todo:
        features = vectorize(ds, tokenizer_name)
    records = []
    for name in classifier_names:
        path = result_path(out_dir, ds.name, tokenizer_name, name)
        if name in todo:
            record = run_one(ds, tokenizer_name, name, features, seed)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2))
            log(f"{ds.name:12s} {tokenizer_name:10s} {name:8s} "
                f"acc={record['metrics']['accuracy']:.4f} f1={record['metrics']['macro_f1']:.4f}")
        records.append(json.loads(path.read_text()))
    return records


def run_grid(dataset_names, tokenizer_names, classifier_names, out_dir, max_train=None, max_test=None,
             seed: int = 0, force: bool = False, log=print) -> list[dict]:
    """The evaluation loop over datasets x tokenizers x classifiers."""
    from . import data
    records = []
    for dname in dataset_names:
        ds = data.load(dname, max_train, max_test, seed)
        log(f"== {dname}: train={len(ds.train_texts)} test={len(ds.test_texts)} labels={len(ds.label_names)}")
        for tname in tokenizer_names:
            records += run_cell(ds, tname, classifier_names, out_dir, seed, force, log)
    return records
