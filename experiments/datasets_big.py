"""Loaders for the large review / comment datasets of phase D (no dataset scripts, Hub parquet only).

Protocol: docs/D_PROTOCOL.md. Each loader returns `(data.Dataset, report)` after the cleaning pass of `clean.py`. These datasets
carry no author id, so the split cannot be grouped by author: the official Hub train / test split is used, exact duplicates and
test-in-train texts are removed by `clean.py`, and the lack of an author-grouped split is stated as a limit. Large train sets are
capped by a seeded random subsample taken BEFORE cleaning (so a cap never needs the whole table in memory as Python strings).
`quick` streams a small shuffled sample instead of downloading the dataset (dry runs and CI).
"""
import random
from dataclasses import dataclass

from . import clean, data

TOXIC_THRESHOLD = 0.5


@dataclass(frozen=True)
class Spec:
    name: str
    hf_id: str
    train_split: str
    test_split: str
    label_names: tuple
    max_train: int | None
    max_test: int | None
    note: str


SPECS = {
    "civil_comments": Spec("civil_comments", "google/civil_comments", "train", "test", ("non-toxic", "toxic"), 500_000, None,
                           f"label = toxicity score >= {TOXIC_THRESHOLD} (the Jigsaw / Civil Comments convention: at least half of the raters "
                           "called the comment toxic); toxic is the rare class"),
}


# ---- pure helpers (unit-tested without network) ------------------------------------------------------------------
def extract(name: str, table) -> tuple[list, list]:
    """(texts, integer labels) from a table with the columns of the named dataset."""
    if name == "civil_comments":
        return [str(t) for t in table["text"]], [int(float(x) >= TOXIC_THRESHOLD) for x in table["toxicity"]]
    raise ValueError(f"unknown dataset {name!r}")


def cap_indices(n: int, limit: int | None, seed: int) -> list[int]:
    """Sorted random subset of range(n) of size `limit` (everything when limit is None or n <= limit); deterministic."""
    if limit is None or n <= limit:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), limit))


def build_from_tables(spec: Spec, train_table, test_table, repo_sha: str, seed: int = 0, extra: dict | None = None):
    """Cap, extract, clean. `extra` is recorded in the report (for example the quick-mode sample sizes)."""
    parts = {}
    for key, table, limit in (("train", train_table, spec.max_train), ("test", test_table, spec.max_test)):
        idx = cap_indices(len(table), limit, seed)
        parts[key] = extract(spec.name, table.select(idx) if len(idx) < len(table) else table)
    n_rows = {"train": len(train_table), "test": len(test_table)}
    ds = data.Dataset(spec.name, spec.hf_id, None, *parts["train"], *parts["test"], list(spec.label_names),
                      {"split": "official Hub train / test", "label_note": spec.note})
    cleaned, report = clean.clean_dataset(ds)
    split = {"hf_id": spec.hf_id, "repo_sha": repo_sha, "n_rows_before_cap": n_rows, "caps": {"train": spec.max_train, "test": spec.max_test},
             "seed": seed, "author_grouped": False, "label_note": spec.note, **(extra or {})}
    return cleaned, {"split": split, "cleaning": report}


# ---- network loaders -------------------------------------------------------------------------------------------------
def load(name: str, quick: bool = False, seed: int = 0, quick_train: int = 30_000, quick_test: int = 4_000):
    """Hub download (or, with `quick`, a streamed shuffled sample). Needs the network."""
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi
    spec = SPECS[name]
    sha = HfApi().dataset_info(spec.hf_id).sha
    if quick:
        tables = {}
        for key, split, n in (("train", spec.train_split, quick_train), ("test", spec.test_split, quick_test)):
            stream = load_dataset(spec.hf_id, split=split, streaming=True).shuffle(seed=seed, buffer_size=20_000)
            tables[key] = Dataset.from_list(list(stream.take(n)))
        spec = Spec(**{**spec.__dict__, "max_train": None, "max_test": None})
        return build_from_tables(spec, tables["train"], tables["test"], sha, seed, {"quick_sample": [quick_train, quick_test]})
    return build_from_tables(spec, load_dataset(spec.hf_id, split=spec.train_split), load_dataset(spec.hf_id, split=spec.test_split), sha, seed)
