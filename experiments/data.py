"""Dataset loaders: three Hugging Face datasets -> a common `Dataset` record.

Every loader returns train/test texts, integer labels and label names. Which split
pair was used (and why) is stored in `Dataset.split_info` and ends up in each result
JSON.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field

TWEET_TOPIC = "cardiffnlp/tweet_topic_single"
GO_EMOTIONS = "google-research-datasets/go_emotions"
AG_NEWS = "fancyzhx/ag_news"


@dataclass
class Dataset:
    name: str
    hf_id: str
    config: str | None
    train_texts: list
    train_labels: list
    test_texts: list
    test_labels: list
    label_names: list
    split_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Plain lists: `datasets` >= 4 returns lazy Column objects, which are slow to iterate repeatedly."""
        for name in ("train_texts", "train_labels", "test_texts", "test_labels"):
            setattr(self, name, list(getattr(self, name)))

    def summary(self) -> dict:
        return {"name": self.name, "hf_id": self.hf_id, "config": self.config,
                "n_train": len(self.train_texts), "n_test": len(self.test_texts),
                "n_labels": len(self.label_names), "label_names": self.label_names,
                "split_info": self.split_info}

    def content_hash(self) -> str:
        """SHA-256 of the train/test texts and labels, so the result cache notices changed data, not only sizes.

        Recomputed on every call (not cached): a Dataset's lists can be edited in place."""
        payload = [self.train_texts, self.train_labels, self.test_texts, self.test_labels]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[:16]


# ---- pure helpers (unit-tested without network) ---------------------------------------
def keep_single_label(texts, label_lists):
    """Keep only examples with exactly one label; return (texts, labels)."""
    kept = [(t, labels[0]) for t, labels in zip(texts, label_lists) if len(labels) == 1]
    return [t for t, _ in kept], [y for _, y in kept]


def choose_tweet_topic_splits(split_names, card_text: str) -> dict:
    """Pick the (train, test) split pair for TweetTopic and say why.

    Preferred: the COLING 2022 pair, if both splits exist *and* the dataset card
    mentions it. Otherwise fall back to the 2021 pair (recorded in `reason`).
    """
    names = set(split_names)
    coling = {"train_coling2022", "test_coling2022"}
    if coling <= names and "coling2022" in card_text.lower():
        return {"train": "train_coling2022", "test": "test_coling2022", "card_confirms_coling2022": True,
                "reason": "COLING 2022 temporal split confirmed by the dataset card and present in the repo"}
    fallback = {"train_2021", "test_2021"}
    if fallback <= names:
        return {"train": "train_2021", "test": "test_2021", "card_confirms_coling2022": False,
                "reason": "COLING 2022 pair not confirmed by the card; fell back to the 2021 pair"}
    raise ValueError(f"no usable split pair among {sorted(names)}")


def subsample(ds: Dataset, max_train: int | None, max_test: int | None, seed: int = 0) -> Dataset:
    """Random subset (fixed seed) of train and test; `None` keeps everything."""
    rng = random.Random(seed)

    def pick(texts, labels, limit):
        if limit is None or len(texts) <= limit:
            return texts, labels
        idx = sorted(rng.sample(range(len(texts)), limit))
        return [texts[i] for i in idx], [labels[i] for i in idx]

    tr_x, tr_y = pick(ds.train_texts, ds.train_labels, max_train)
    te_x, te_y = pick(ds.test_texts, ds.test_labels, max_test)
    info = dict(ds.split_info, subsample={"max_train": max_train, "max_test": max_test, "seed": seed})
    return Dataset(ds.name, ds.hf_id, ds.config, tr_x, tr_y, te_x, te_y, ds.label_names, info)


# ---- loaders (need the Hugging Face hub) ------------------------------------------------
def _card_quote(card_text: str) -> str:
    """The sentence of the dataset card that names the COLING 2022 split pair."""
    import re
    m = re.search(r"\*\*IMPORTANT NOTE:\*\*[^\n]*", card_text)
    return m.group(0) if m else ""


def _card_text(hf_id: str) -> str:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(hf_id, "README.md", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return f.read()


PARQUET_REVISION = "refs/convert/parquet"


def _parquet_split_files(hf_id: str, config: str) -> dict[str, list[str]]:
    """{split name: [parquet paths]} of the auto-converted parquet revision of a dataset repo."""
    from huggingface_hub import list_repo_files
    files = {}
    for path in sorted(list_repo_files(hf_id, repo_type="dataset", revision=PARQUET_REVISION)):
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == config and parts[2].endswith(".parquet"):
            files.setdefault(parts[1], []).append(path)
    return files


def _load_parquet_split(hf_id: str, paths: list[str]):
    from datasets import Dataset, concatenate_datasets
    from huggingface_hub import hf_hub_download
    parts = [Dataset.from_parquet(hf_hub_download(hf_id, p, repo_type="dataset", revision=PARQUET_REVISION))
             for p in paths]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def load_tweet_topic() -> Dataset:
    """TweetTopic (single-label): 6 topics; COLING 2022 split pair when the card confirms it.

    The repo ships a loading script, which `datasets` >= 4 refuses to run, so the same files
    are read from Hugging Face's auto-converted parquet revision (`refs/convert/parquet`).
    """
    files = _parquet_split_files(TWEET_TOPIC, "tweet_topic_single")
    splits = choose_tweet_topic_splits(list(files), _card_text(TWEET_TOPIC))
    train = _load_parquet_split(TWEET_TOPIC, files[splits["train"]])
    test = _load_parquet_split(TWEET_TOPIC, files[splits["test"]])
    splits = dict(splits, revision=PARQUET_REVISION, card_quote=_card_quote(_card_text(TWEET_TOPIC)),
                  loading_note="repo uses a loading script; read from the auto-converted parquet revision")
    return Dataset("tweet_topic", TWEET_TOPIC, None, train["text"], train["label"], test["text"],
                   test["label"], train.features["label"].names, splits)


def load_go_emotions() -> Dataset:
    """GoEmotions `simplified`, single-label examples only (original train/test splits)."""
    from datasets import load_dataset
    raw = load_dataset(GO_EMOTIONS, "simplified")
    tr_x, tr_y = keep_single_label(raw["train"]["text"], raw["train"]["labels"])
    te_x, te_y = keep_single_label(raw["test"]["text"], raw["test"]["labels"])
    info = {"train": "train", "test": "test", "reason": "official splits; examples with "
            "more than one label removed", "n_train_before_filter": len(raw["train"]),
            "n_test_before_filter": len(raw["test"])}
    return Dataset("go_emotions", GO_EMOTIONS, "simplified", tr_x, tr_y, te_x, te_y,
                   raw["train"].features["labels"].feature.names, info)


def load_ag_news() -> Dataset:
    """AG News: 4 topics, official train/test splits."""
    from datasets import load_dataset
    raw = load_dataset(AG_NEWS)
    info = {"train": "train", "test": "test", "reason": "official splits"}
    return Dataset("ag_news", AG_NEWS, None, raw["train"]["text"], raw["train"]["label"],
                   raw["test"]["text"], raw["test"]["label"],
                   raw["train"].features["label"].names, info)


LOADERS = {"tweet_topic": load_tweet_topic, "go_emotions": load_go_emotions, "ag_news": load_ag_news}
DATASET_NAMES = tuple(LOADERS)


def load(name: str, max_train: int | None = None, max_test: int | None = None, seed: int = 0) -> Dataset:
    """Load a dataset by short name, optionally subsampled with a fixed seed."""
    return subsample(LOADERS[name](), max_train, max_test, seed)


def write_summary(out_dir, max_train: int | None = None, max_test: int | None = None, seed: int = 0,
                  names=DATASET_NAMES, log=print) -> Path:
    """Load every dataset and save `datasets.json` (sizes, labels, splits, class counts)."""
    records = []
    for name in names:
        ds = load(name, max_train, max_test, seed)
        record = ds.summary()
        record["train_class_counts"] = dict(sorted(Counter(ds.train_labels).items()))
        record["test_class_counts"] = dict(sorted(Counter(ds.test_labels).items()))
        records.append(record)
        log(f"loaded {name}: train={record['n_train']} test={record['n_test']} labels={record['n_labels']}")
    path = Path(out_dir) / "datasets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"datasets": records}, indent=2))
    return path
