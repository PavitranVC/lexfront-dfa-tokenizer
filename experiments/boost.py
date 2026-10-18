"""Phase A engine: wider validation-selected settings and a character n-gram baseline, identical for every tokenizer.

Protocol: docs/BOOST_PROTOCOL.md (written before any result). Selection uses a slice of the TRAIN split only; the test split
is scored once per refit winner. Generic over a `data.Dataset`, so later datasets reuse it. Every unit (tokenizer x stage) writes
its own JSON and prediction file under `<work>/`, so a dead session resumes; the layout matches `large.consolidate`.
"""
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from . import classifiers as clf_mod
from . import large, pipeline, tuning
from .metrics import classification_metrics
from .tokenizers import build_tokenizer

WORD_TOKENIZERS = large.TOKENIZERS
CHAR = "char"
CLASSIFIERS = ("lr", "nb")
LR_C = [0.3, 1.0, 3.0, 10.0, 30.0]
NB_ALPHA = [0.03, 0.1, 0.3, 1.0]
FIT_MAX, VAL_MAX, VAL_FRACTION = 100_000, 20_000, 0.1
SEARCH_SEED = 0


def _word_grid() -> dict:
    grid = {}
    for n in (1, 2, 3):
        for sub in (False, True):
            for df in (2, 5):
                grid[f"w{n}_{'sub' if sub else 'raw'}_df{df}"] = {**clf_mod.VECTORIZER_PARAMS, "ngram_range": [1, n],
                                                                   "sublinear_tf": sub, "min_df": df}
    return grid


WORD_GRID = _word_grid()
CHAR_GRID = {f"c{a}{b}": {"analyzer": "char_wb", "lowercase": True, "sublinear_tf": True, "min_df": 5, "ngram_range": [a, b]}
             for a, b in ((2, 4), (3, 5))}


def hyper_grid(classifier: str) -> list:
    return LR_C if classifier == "lr" else NB_ALPHA


def make_vec(params: dict, tokenizer=None) -> TfidfVectorizer:
    if "analyzer" in params:
        return TfidfVectorizer(**clf_mod.vectorizer_kwargs(params))
    return clf_mod.make_vectorizer(tokenizer, params)


def split_indices(labels, seed: int = SEARCH_SEED) -> tuple[list, list]:
    """(fit, validation) indices inside the train split; disjoint, stratified, deterministic."""
    idx = np.arange(len(labels))
    n_val = min(VAL_MAX, max(1, round(len(labels) * VAL_FRACTION)))
    n_fit = min(FIT_MAX, len(labels) - n_val)
    counts = np.bincount(np.asarray(labels))
    stratify = labels if counts[counts > 0].min() >= 2 and n_val >= len(counts) and n_fit >= len(counts) else None
    fit, val = train_test_split(idx, train_size=n_fit, test_size=n_val, random_state=seed, stratify=stratify)
    return sorted(fit.tolist()), sorted(val.tolist())


def pick_best(scores: list) -> dict:
    """Highest validation accuracy; ties go to the earlier candidate (grid order)."""
    best = scores[0]
    for s in scores[1:]:
        if s["val_accuracy"] > best["val_accuracy"]:
            best = s
    return best


def search(train_x, train_y, grid: dict, classifier: str, tokenizer, seed: int = SEARCH_SEED) -> dict:
    """Score every candidate on the validation slice of the train split. Takes no test data by construction."""
    fit_idx, val_idx = split_indices(train_y, seed)
    fit_x, val_x = [train_x[i] for i in fit_idx], [train_x[i] for i in val_idx]
    fit_y, val_y = [train_y[i] for i in fit_idx], np.asarray([train_y[i] for i in val_idx])
    scores = []
    with pipeline.single_threaded():
        for name, params in grid.items():
            vec = make_vec(params, tokenizer)
            x_fit, x_val = vec.fit_transform(fit_x), vec.transform(val_x)
            for value in hyper_grid(classifier):
                model = tuning.make_model(classifier, value, seed).fit(x_fit, fit_y)
                scores.append({"vectorizer": name, "value": value,
                               "val_accuracy": float((model.predict(x_val) == val_y).mean())})
    return {"n_fit": len(fit_idx), "n_val": len(val_idx), "scores": scores, "selected": pick_best(scores)}


def code_fingerprint() -> str:
    h = hashlib.sha256(pipeline.code_fingerprint().encode())
    for rel in ("experiments/boost.py", "experiments/large.py", "experiments/tuning.py", "experiments/clean.py"):
        h.update((pipeline.REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def unit_id(tokenizer: str, stage: str, classifier: str | None = None) -> str:
    return f"{tokenizer}__{stage}" + (f"__{classifier}" if classifier else "")


def unit_fingerprint(data_hash: str, uid: str, seed: int) -> dict:
    config = {"data_hash": data_hash, "unit": uid, "seed": seed, "word_grid": WORD_GRID, "char_grid": CHAR_GRID, "lr_C": LR_C,
              "nb_alpha": NB_ALPHA, "fit_max": FIT_MAX, "val_max": VAL_MAX, "val_fraction": VAL_FRACTION,
              **pipeline.library_versions(), "blas_threads": pipeline.BLAS_THREADS}
    return {"config": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16], "code": code_fingerprint()}


def run_wide(ds, tokenizer_name: str, classifier: str, test_tokenizer, seed: int) -> tuple[dict, dict]:
    """Wide search for one (tokenizer or char, classifier), refit the winner on all of train, score the test split once."""
    grid = CHAR_GRID if tokenizer_name == CHAR else WORD_GRID
    tokenizer = None if tokenizer_name == CHAR else tuning.cached(build_tokenizer(tokenizer_name, ds.train_texts))
    start = time.perf_counter()
    selection = search(ds.train_texts, ds.train_labels, grid, classifier, tokenizer, seed)
    search_seconds = time.perf_counter() - start
    chosen = selection["selected"]
    vec = make_vec(grid[chosen["vectorizer"]], build_tokenizer(tokenizer_name, ds.train_texts) if tokenizer else None)
    with pipeline.single_threaded():
        start = time.perf_counter()
        x_train = vec.fit_transform(ds.train_texts)
        vec_seconds = time.perf_counter() - start
        if tokenizer_name != CHAR:
            vec.tokenizer = test_tokenizer
        model = tuning.make_model(classifier, chosen["value"], seed)
        start = time.perf_counter()
        model.fit(x_train, ds.train_labels)
        fit_seconds = time.perf_counter() - start
        y_pred = model.predict(vec.transform(ds.test_texts))
    y_true = np.asarray(ds.test_labels)
    metrics = classification_metrics(ds.test_labels, y_pred)
    metrics.update(n=len(y_true), n_correct=int((y_pred == y_true).sum()))
    record = {"classifier": classifier, "setting": "wide", "vectorizer": {"name": chosen["vectorizer"], "n_features": len(vec.vocabulary_),
              **grid[chosen["vectorizer"]]}, "params": model.get_params(), "tuning": selection, "metrics": {"clean": metrics},
              "timing_seconds": {"search": search_seconds, "tokenize_and_vectorize_train": vec_seconds, "classifier_fit": fit_seconds}}
    return {classifier: record}, {classifier: {"clean": y_pred.astype(np.int8)}}


def run_units(ds, tokenizers, work_dir, seed: int = 0, force: bool = False, log=print) -> int:
    """Run (or resume) anchor + wide units for the word tokenizers and wide units for the character baseline."""
    work, computed = Path(work_dir), 0
    y_true, data_hash = np.asarray(ds.test_labels), ds.content_hash()
    for name in tokenizers:
        test_tokenizer = tuning.cached(build_tokenizer(name, [])) if name != CHAR else None
        units = [("anchor", None)] * (name != CHAR) + [("wide", c) for c in CLASSIFIERS]
        for stage, classifier in units:
            uid = unit_id(name, stage, classifier)
            path, fingerprint = work / "runs" / f"{uid}.json", unit_fingerprint(data_hash, uid, seed)
            if not force and large.is_fresh(path, fingerprint, True):
                log(f"cached {uid}")
                continue
            start = time.perf_counter()
            if stage == "anchor":
                records, preds = large.fit_and_evaluate(ds.train_texts, ds.train_labels, {"clean": ds.test_texts}, y_true, name,
                                                        "default", seed, test_tokenizer, True)
            else:
                records, preds = run_wide(ds, name, classifier, test_tokenizer, seed)
            record = {"run_id": uid, "tokenizer": name, "stage": stage, "n_train": len(ds.train_texts), "n_test": len(y_true),
                      "classifiers": records, "fingerprint": fingerprint, "wall_seconds": time.perf_counter() - start,
                      "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "environment": {**pipeline.environment(), "blas_threads": pipeline.BLAS_THREADS}}
            large.save_predictions(work / "predictions" / f"{uid}.json.gz", y_true, preds)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1))
            computed += 1
            acc = {c: round(r["metrics"]["clean"]["accuracy"], 4) for c, r in records.items()}
            log(f"done {uid}: accuracy {acc} in {record['wall_seconds']:.0f}s, peak {record['peak_rss_gb']:.1f} GB")
    return computed
