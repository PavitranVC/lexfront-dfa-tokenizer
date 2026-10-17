"""The dataset collection: the three original datasets plus five tweet_eval tasks, optionally cleaned.

Kept apart from `data.py` on purpose: `data.py` is part of every run's code fingerprint, so editing it would mark
all committed results stale. Everything here is additive; `data.DATASET_NAMES` (the original grid) is unchanged.

tweet_eval (Barbieri et al., TweetEval, Findings of EMNLP 2020) is a set of tweet classification tasks with
official train/validation/test splits; we use train and test and carve our own validation out of train when tuning
(`tuning.py`). Notes that matter when reading results: `hate` has a known train/test distribution shift,
`irony` and `offensive` have small test sets (wide confidence intervals).
"""
from . import clean, data

TWEET_EVAL = "cardiffnlp/tweet_eval"
TWEET_EVAL_TASKS = ("sentiment", "emotion", "irony", "hate", "offensive")


def _tweet_eval_loader(task: str):
    def load() -> data.Dataset:
        from datasets import load_dataset
        raw = load_dataset(TWEET_EVAL, task)
        info = {"train": "train", "test": "test", "reason": "official splits (the validation split is not used)"}
        return data.Dataset(f"tweet_eval_{task}", TWEET_EVAL, task, raw["train"]["text"], raw["train"]["label"],
                            raw["test"]["text"], raw["test"]["label"], raw["train"].features["label"].names, info)
    return load


EXTRA_LOADERS = {f"tweet_eval_{task}": _tweet_eval_loader(task) for task in TWEET_EVAL_TASKS}
COLLECTION_NAMES = tuple(data.DATASET_NAMES) + tuple(EXTRA_LOADERS)


def load_raw(name: str) -> data.Dataset:
    """A collection dataset exactly as published (no cleaning, no subsample)."""
    loaders = {**data.LOADERS, **EXTRA_LOADERS}
    if name not in loaders:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(loaders)}")
    return loaders[name]()


def load(name: str, cleaned: bool = True, max_train: int | None = None, max_test: int | None = None,
         seed: int = 0) -> tuple[data.Dataset, dict | None]:
    """(dataset, cleaning report or None). Cleaning runs on the full data, then the optional fixed-seed subsample."""
    ds, report = load_raw(name), None
    if cleaned:
        ds, report = clean.clean_dataset(ds)
    return data.subsample(ds, max_train, max_test, seed), report
