"""The evaluation loop: dataset x tokenizer x classifier -> metrics -> JSON.

Every run writes one JSON file holding its full configuration, so tables and figures
can be rebuilt from `results/` alone. Each record also carries a *fingerprint* of its
configuration and of the code that produced it; a cached record is reused only while both
still match (compiler analogy: an incremental build that recompiles when a source hash changes).

All vectorize/fit/predict work runs with BLAS/OpenMP pinned to one thread: the thread count changes
the order of floating-point sums inside lbfgs, so a deterministic build must fix it (like a compiler
that must not depend on how many cores ran it).
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from . import classifiers as clf_mod
from .metrics import classification_metrics
from .tokenizers import get_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
# Source files whose change can change a run's result (tokenizers, normalizer, data, vectorizer, classifiers).
CODE_GLOBS = ["lexer/*.py", "lexer/dfa/*.py", "lexer/*.csv", "experiments/tokenizers.py", "experiments/classifiers.py",
              "experiments/pipeline.py", "experiments/metrics.py", "experiments/data.py"]
# Native thread count for every fit; part of the fingerprint because it changes LR metrics.
BLAS_THREADS = 1


def single_threaded():
    """Context manager pinning BLAS/OpenMP pools to BLAS_THREADS (reproducible float summation order)."""
    return threadpool_limits(limits=BLAS_THREADS)


def environment() -> dict:
    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "machine": platform.machine(), "platform": platform.platform()}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash of every source file that can change a result (line endings normalized)."""
    h = hashlib.sha256()
    for path in sorted({p for g in CODE_GLOBS for p in REPO_ROOT.glob(g)}):
        h.update(path.relative_to(REPO_ROOT).as_posix().encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def config_fingerprint(ds, tokenizer_name: str, classifier_name: str, seed: int) -> str:
    """Hash of everything a run is configured by: data subset, tokenizer, vectorizer, classifier, seed, versions."""
    config = {"dataset": ds.summary(), "tokenizer": tokenizer_name, "vectorizer": clf_mod.VECTORIZER_PARAMS,
              "classifier": classifier_name, "params": clf_mod.classifier_params(classifier_name, seed),
              "seed": seed, "sklearn": sklearn.__version__, "nltk": nltk.__version__,
              "blas_threads": BLAS_THREADS}
    return _sha256(json.dumps(config, sort_keys=True, default=str).encode())


def fingerprint(ds, tokenizer_name: str, classifier_name: str, seed: int) -> dict:
    return {"config": config_fingerprint(ds, tokenizer_name, classifier_name, seed), "code": code_fingerprint()}


def is_fresh(path: Path, expected: dict) -> bool:
    """True if a cached result exists and was made by the same configuration and code."""
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("fingerprint") == expected


def result_path(out_dir, dataset: str, tokenizer: str, classifier: str) -> Path:
    return Path(out_dir) / f"{dataset}__{tokenizer}__{classifier}.json"


def vectorize(ds, tokenizer_name: str):
    """Fit TF-IDF on train, transform test; returns (X_train, X_test, seconds)."""
    vec = clf_mod.make_vectorizer(get_tokenizer(tokenizer_name))
    start = time.perf_counter()
    with single_threaded():
        x_train = vec.fit_transform(ds.train_texts)
        x_test = vec.transform(ds.test_texts)
    return x_train, x_test, time.perf_counter() - start, len(vec.vocabulary_)


def run_one(ds, tokenizer_name: str, classifier_name: str, features, seed: int = 0) -> dict:
    """Fit one classifier on prepared features and return the result record."""
    x_train, x_test, vec_seconds, n_features = features
    clf = clf_mod.make_classifier(classifier_name, seed)
    with single_threaded():
        start = time.perf_counter()
        clf.fit(x_train, ds.train_labels)
        fit_seconds = time.perf_counter() - start
        predictions = clf.predict(x_test)
    metrics = classification_metrics(ds.test_labels, predictions)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.summary(),
        "tokenizer": tokenizer_name,
        "vectorizer": {"type": "TfidfVectorizer", **clf_mod.VECTORIZER_PARAMS, "n_features": n_features},
        "classifier": {"name": classifier_name, "params": clf_mod.classifier_params(classifier_name, seed)},
        "seed": seed,
        "fingerprint": fingerprint(ds, tokenizer_name, classifier_name, seed),
        "metrics": metrics,
        "timing_seconds": {"tokenize_and_vectorize": vec_seconds, "classifier_fit": fit_seconds},
        "environment": {**environment(), "blas_threads": BLAS_THREADS},
    }


def run_cell(ds, tokenizer_name: str, classifier_names, out_dir, seed: int = 0, force: bool = False,
             log=print) -> list[dict]:
    """All classifiers for one (dataset, tokenizer); features are computed once.

    A cached JSON is reused only if its fingerprint matches; `force=True` recomputes regardless.
    """
    todo = [c for c in classifier_names
            if force or not is_fresh(result_path(out_dir, ds.name, tokenizer_name, c),
                                     fingerprint(ds, tokenizer_name, c, seed))]
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
