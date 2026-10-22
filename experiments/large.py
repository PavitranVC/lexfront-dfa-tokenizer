"""Large-scale Sentiment140 test: user-grouped split, audit, test-time noise, size ladder, resumable runs.

Protocol and hypotheses: docs/LARGE_PROTOCOL.md (written before the full run). This module holds the pieces; `run_large.py`
is the command line, the statistics and the generated report.

Design for a Colab session that can die at any time: every fit writes its own JSON (`<work>/runs/<run id>.json`) and a run
whose fingerprint still matches is skipped on the next start, so a restart resumes where it stopped. Nothing here is part of
the code fingerprint of the main grid, so the committed results of the main grid are untouched.
"""
import gzip
import hashlib
import io
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import classifiers as clf_mod
from . import clean, data, pipeline, tuning
from .metrics import classification_metrics
from .peak_memory import peak_rss_gb
from .tokenizers import build_tokenizer

SENTIMENT140 = "bdanko/sentiment140"
TEST_USER_FRACTION = 0.05
SPLIT_SEED = 0
TOKENIZERS = ("whitespace", "nltk", "tweet", "lexfront")
CLASSIFIERS = ("nb", "lr")
NOISE_LEVELS = {"clean": 0.0, "light": 0.10, "heavy": 0.30}      # probability that an eligible word is corrupted
NOISE_SEED = 0
# (label, train size or None for the whole cleaned train split, number of random subsets, settings)
LADDER = (("10000", 10_000, 5, ("default", "tuned")), ("100000", 100_000, 5, ("default", "tuned")),
          ("1000000", 1_000_000, 3, ("default",)), ("full", None, 1, ("default", "tuned")))
QUICK_LADDER = (("1000", 1_000, 2, ("default", "tuned")), ("5000", 5_000, 2, ("default", "tuned")),
                ("full", None, 1, ("default", "tuned")))


# ---- data ---------------------------------------------------------------------------------------------------------
@dataclass
class Raw:
    texts: list
    users: list
    labels: list
    label_names: list
    provenance: dict


def load_raw() -> Raw:
    """Sentiment140 as published on the Hub (needs the network). Labels 0 / 4 become 0 / 1."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    table = load_dataset(SENTIMENT140)["train"]
    raw_labels = [str(x) for x in table["sentiment"]]
    classes = sorted(set(raw_labels))
    if classes != ["0", "4"]:
        raise ValueError(f"expected polarity labels 0 and 4, found {classes}")
    sha = HfApi().dataset_info(SENTIMENT140).sha
    return Raw(list(table["text"]), list(table["user"]), [classes.index(x) for x in raw_labels],
               ["negative", "positive"], {"hf_id": SENTIMENT140, "repo_sha": sha, "n_rows": len(raw_labels)})


def user_test_mask(users, fraction: float = TEST_USER_FRACTION, seed: int = SPLIT_SEED) -> list[bool]:
    """True for rows of a test user. Whole users are held out, so no author is in both splits."""
    ids = sorted(set(users))
    random.Random(seed).shuffle(ids)
    test_users = set(ids[:max(1, round(len(ids) * fraction))])
    return [u in test_users for u in users]


def build_dataset(raw: Raw, max_train: int | None = None, max_test: int | None = None,
                  seed: int = 0) -> tuple[data.Dataset, dict]:
    """(cleaned Dataset, report). Split by user first, then clean with experiments/clean.py, then optionally subsample."""
    mask = user_test_mask(raw.users)
    pick = lambda flag: [i for i, m in enumerate(mask) if m == flag]          # noqa: E731
    tr, te = pick(False), pick(True)
    train_users, test_users = {raw.users[i] for i in tr}, {raw.users[i] for i in te}
    info = {"train": "users not held out", "test": f"{TEST_USER_FRACTION:.0%} of users, all their tweets",
            "reason": "split by user (seed %d) so no author is in both splits" % SPLIT_SEED, **raw.provenance}
    ds = data.Dataset("sentiment140", SENTIMENT140, None, [raw.texts[i] for i in tr], [raw.labels[i] for i in tr],
                      [raw.texts[i] for i in te], [raw.labels[i] for i in te], raw.label_names, info)
    cleaned, report = clean.clean_dataset(ds)
    split = {"n_users": len(train_users) + len(test_users), "n_test_users": len(test_users),
             "users_in_both_splits": len(train_users & test_users), "test_user_fraction": TEST_USER_FRACTION,
             "seed": SPLIT_SEED, "n_rows_before_split": len(raw.texts), **raw.provenance}
    return data.subsample(cleaned, max_train, max_test, seed), {"split": split, "cleaning": report}


# ---- audit ----------------------------------------------------------------------------------------------------------
PATTERNS = {"url": re.compile(r"https?://|www\."), "mention": re.compile(r"@\w+"), "hashtag": re.compile(r"#\w+"),
            "emoticon": re.compile(r"(?:^|\s)[:;=8xX][-o^']?[)(\]\[dDpPoO/\\|*]+(?=\s|$)"),
            "elongation": re.compile(r"([a-zA-Z])\1{2,}"), "html_entity": re.compile(r"&(?:quot|amp|lt|gt);")}


def audit(texts, labels, names=("negative", "positive")) -> dict:
    """Share of tweets containing each noise feature, per class and overall (simple regexes, not LexFront's token classes)."""
    out = {"n": len(texts), "features": list(PATTERNS)}
    for cls in [None, *range(len(names))]:
        rows = [t for t, y in zip(texts, labels) if cls is None or y == cls]
        key = "all" if cls is None else names[cls]
        out[key] = {"n": len(rows), "mean_chars": sum(map(len, rows)) / max(1, len(rows)),
                    **{f: sum(bool(p.search(t)) for t in rows) / max(1, len(rows)) for f, p in PATTERNS.items()}}
    return out


# ---- test-time noise ---------------------------------------------------------------------------------------------
def _corrupt_word(word: str, rng: random.Random) -> str:
    op = rng.choice(("swap", "drop", "double", "elongate"))
    k = rng.randrange(1, len(word) - 1)                                  # an inner position
    if op == "swap":
        return word[:k] + word[k + 1] + word[k] + word[k + 2:]
    if op == "drop":
        return word[:k] + word[k + 1:]
    if op == "double":
        return word[:k] + word[k] + word[k:]
    vowels = [j for j, c in enumerate(word) if c.lower() in "aeiou"]
    j = vowels[-1] if vowels else len(word) - 1
    return word[:j + 1] + word[j] * 3 + word[j + 1:]


def corrupt(text: str, p: float, index: int, seed: int = NOISE_SEED) -> str:
    """Corrupt each purely alphabetic word of 4+ letters with probability p (mentions, hashtags, URLs stay). Deterministic in
    (text, p, index, seed): the random stream depends on the tweet's position only, not on the order of the calls."""
    if p <= 0:
        return text
    rng = random.Random(f"{seed}-{index}")
    words = text.split(" ")
    for i, word in enumerate(words):
        if len(word) >= 4 and word.isalpha() and rng.random() < p:
            words[i] = _corrupt_word(word, rng)
    return " ".join(words)


def noisy_tests(texts) -> dict[str, list[str]]:
    return {level: [corrupt(t, p, i) for i, t in enumerate(texts)] for level, p in NOISE_LEVELS.items()}


# ---- ladder ---------------------------------------------------------------------------------------------------------
def subset_indices(n_train: int, size: int, repeat: int, seed: int = 0) -> list[int]:
    return sorted(random.Random(f"{seed}-{size}-{repeat}").sample(range(n_train), size))


def run_id(tokenizer: str, setting: str, label: str, repeat: int) -> str:
    return f"{tokenizer}__{setting}__{label}__r{repeat}"


def code_fingerprint() -> str:
    h = hashlib.sha256(pipeline.code_fingerprint().encode())
    for rel in ("experiments/large.py", "experiments/tuning.py", "experiments/clean.py"):
        h.update((pipeline.REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def run_fingerprint(data_hash: str, tokenizer: str, setting: str, label: str, repeat: int, seed: int) -> dict:
    config = {"data_hash": data_hash, "tokenizer": tokenizer, "setting": setting, "size": label, "repeat": repeat,
              "seed": seed, "search": tuning.SEARCH, "noise": NOISE_LEVELS, "noise_seed": NOISE_SEED,
              **pipeline.library_versions(), "blas_threads": pipeline.BLAS_THREADS}
    return {"config": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
            "code": code_fingerprint()}


# ---- one fit set ------------------------------------------------------------------------------------------------------
def fit_and_evaluate(train_x, train_y, tests: dict, y_true, tokenizer_name: str, setting: str, seed: int,
                     test_tokenizer, keep_predictions: bool = False) -> tuple[dict, dict]:
    """Fit NB and LR on the training texts, score every noisy test variant. Returns (records by classifier, predictions).

    `test_tokenizer` is a memoized copy of the (untrained, word-level) tokenizer used only for the test texts, so the 3 noise
    variants and all fits of one tokenizer tokenize each distinct test text once; the training texts are never memoized
    (a million token lists would not fit in a Colab runtime)."""
    tokenizer = build_tokenizer(tokenizer_name, train_x)
    sub = data.Dataset("subset", "", None, train_x, train_y, [], [], [], {})
    choices = {}
    for clf in CLASSIFIERS:
        if setting == "tuned":
            start = time.perf_counter()
            selection = tuning.search(sub, tuning.cached(tokenizer), clf, seed)
            choices[clf] = {"vectorizer": selection["selected"]["vectorizer"], "value": selection["selected"]["value"],
                            "tuning": selection, "search_seconds": time.perf_counter() - start}
        else:
            choices[clf] = {"vectorizer": "unigram", "value": None, "tuning": None, "search_seconds": 0.0}
    features, records, predictions = {}, {}, {}
    for clf in CLASSIFIERS:
        option = choices[clf]["vectorizer"]
        if option not in features:                                          # one vectorizer per distinct option
            vec = clf_mod.make_vectorizer(tokenizer, tuning.SEARCH["vectorizer"][option])
            start = time.perf_counter()
            with pipeline.single_threaded():
                x_train = vec.fit_transform(train_x)
            vec.tokenizer = test_tokenizer
            features[option] = {"vec": vec, "x_train": x_train, "seconds": time.perf_counter() - start, "x_test": {}}
        f = features[option]
        model = (clf_mod.make_classifier(clf, seed) if setting == "default"
                 else tuning.make_model(clf, choices[clf]["value"], seed))
        with pipeline.single_threaded():
            start = time.perf_counter()
            model.fit(f["x_train"], train_y)
            fit_seconds = time.perf_counter() - start
            metrics, predict_seconds, preds = {}, 0.0, {}
            for level, texts in tests.items():
                if level not in f["x_test"]:
                    f["x_test"][level] = f["vec"].transform(texts)
                start = time.perf_counter()
                y_pred = model.predict(f["x_test"][level])
                predict_seconds += time.perf_counter() - start
                m = classification_metrics(y_true, y_pred)
                m.update(n=len(y_true), n_correct=int((y_pred == np.asarray(y_true)).sum()))
                metrics[level], preds[level] = m, y_pred.astype(np.int8)
        records[clf] = {"classifier": clf, "setting": setting, "vectorizer": {"name": option,
                        "n_features": len(f["vec"].vocabulary_)}, "params": model.get_params(),
                        "tuning": choices[clf]["tuning"], "metrics": metrics,
                        "timing_seconds": {"tokenize_and_vectorize_train": f["seconds"], "search": choices[clf]["search_seconds"],
                                           "classifier_fit": fit_seconds, "predict_all_variants": predict_seconds}}
        if keep_predictions:
            predictions[clf] = preds
    return records, predictions


def measure_throughput(tokenizer, texts, repeats: int = 3) -> dict:
    """Documents and tokens per second on `texts` (lower-cased as the vectorizer does), best of `repeats`."""
    texts = [t.lower() for t in texts]
    best, n_tokens = float("inf"), 0
    for _ in range(repeats):
        start = time.perf_counter()
        n_tokens = sum(len(tokenizer(t)) for t in texts)
        best = min(best, time.perf_counter() - start)
    return {"docs": len(texts), "docs_per_s": len(texts) / best, "tokens_per_s": n_tokens / best}


def save_predictions(path: Path, y_true, predictions: dict) -> None:
    """Gzipped JSON {y_true, predictions: {classifier: {variant: [labels]}}}; deterministic bytes."""
    payload = {"y_true": [int(y) for y in y_true],
               "predictions": {c: {lv: [int(v) for v in p] for lv, p in by.items()} for c, by in predictions.items()}}
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as gz:
        gz.write(json.dumps(payload).encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def load_predictions(path) -> tuple[np.ndarray, dict]:
    payload = json.loads(gzip.decompress(Path(path).read_bytes()))
    preds = {c: {lv: np.asarray(p, dtype=np.int64) for lv, p in by.items()} for c, by in payload["predictions"].items()}
    return np.asarray(payload["y_true"], dtype=np.int64), preds


def is_fresh(path: Path, fingerprint: dict, needs_predictions: bool) -> bool:
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    sidecar = path.parent.parent / "predictions" / f"{path.stem}.json.gz"
    return record.get("fingerprint") == fingerprint and (sidecar.exists() or not needs_predictions)


def run_grid(ds: data.Dataset, tokenizers, ladder, work_dir, seed: int = 0, force: bool = False, log=print) -> int:
    """Run (or resume) the whole grid; returns the number of fit sets computed in this call."""
    work = Path(work_dir)
    y_true = np.asarray(ds.test_labels)
    tests = noisy_tests(ds.test_texts)
    data_hash = ds.content_hash()
    computed = 0
    for name in tokenizers:
        plain = build_tokenizer(name, [])
        test_tokenizer = tuning.cached(plain)
        throughput = measure_throughput(plain, ds.test_texts[:5000])
        for label, size, repeats, settings in ladder:
            if size is not None and size >= len(ds.train_texts):
                log(f"skip size {label}: only {len(ds.train_texts)} training texts")
                continue
            for rep in range(1 if size is None else repeats):
                idx = None if size is None else subset_indices(len(ds.train_texts), size, rep, seed)
                for setting in settings:
                    rid = run_id(name, setting, label, rep)
                    path = work / "runs" / f"{rid}.json"
                    fingerprint = run_fingerprint(data_hash, name, setting, label, rep, seed)
                    keep = size is None
                    if not force and is_fresh(path, fingerprint, keep):
                        log(f"cached {rid}")
                        continue
                    train_x = ds.train_texts if idx is None else [ds.train_texts[i] for i in idx]
                    train_y = ds.train_labels if idx is None else [ds.train_labels[i] for i in idx]
                    start = time.perf_counter()
                    records, predictions = fit_and_evaluate(train_x, train_y, tests, y_true, name, setting, seed,
                                                            test_tokenizer, keep)
                    record = {"run_id": rid, "tokenizer": name, "setting": setting, "size": label, "repeat": rep,
                              "n_train": len(train_x), "n_test": len(y_true), "classifiers": records,
                              "throughput_test_docs": throughput, "fingerprint": fingerprint,
                              "wall_seconds": time.perf_counter() - start,
                              "peak_rss_gb": peak_rss_gb(),
                              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "environment": {**pipeline.environment(), "blas_threads": pipeline.BLAS_THREADS}}
                    if keep:
                        save_predictions(work / "predictions" / f"{rid}.json.gz", y_true, predictions)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(record, indent=1))
                    computed += 1
                    acc = {c: round(r["metrics"]["clean"]["accuracy"], 4) for c, r in records.items()}
                    log(f"done {rid}: n_train={len(train_x)} clean accuracy {acc} in {record['wall_seconds']:.0f}s")
    return computed


def consolidate(work_dir, out_dir) -> Path:
    """Gather the per-run files into `<out>/runs.json` and copy the stored predictions next to it."""
    work, out = Path(work_dir), Path(out_dir)
    runs = [json.loads(p.read_text()) for p in sorted((work / "runs").glob("*.json"))]
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    for p in sorted((work / "predictions").glob("*.json.gz")):
        (out / "predictions" / p.name).write_bytes(p.read_bytes())
    path = out / "runs.json"
    path.write_text(json.dumps({"runs": runs}, indent=1) + "\n")
    return path
