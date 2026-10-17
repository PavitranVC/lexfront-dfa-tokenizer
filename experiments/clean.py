"""Dataset cleaning: an additive, documented pass over a `data.Dataset` (nothing here edits a text).

Compiler analogy: a pre-pass that rejects malformed input before the scanner sees it. The texts themselves are
left alone on purpose: emoticons, hashtags, casing and misspellings are exactly the noise LexFront is tested on,
so "cleaning" here means removing rows that make the evaluation unreliable, not rewriting the text.

Rules, applied in this order (each count is reported):
  1. empty        a text that is empty or only whitespace is dropped (train and test).
  2. conflicting  texts that are equal after `text_key` but carry different labels are all dropped, per split
                  (the gold label is ambiguous).
  3. duplicate    among texts equal after `text_key` with one label, the first is kept, per split.
  4. leakage      a test text whose `text_key` also occurs in train (before rules 2-3, so a conflicting or
                  duplicated train text still counts) is dropped from test.

`text_key` is the lower-cased text with whitespace runs collapsed: it only decides which rows count as equal, the
kept text is the original string. The pass is deterministic (order preserved, no randomness).
"""
from collections import Counter, defaultdict

from .data import Dataset


def text_key(text: str) -> str:
    """What "the same text" means: case and whitespace differences do not count."""
    return " ".join(text.lower().split())


def _dedupe(texts, labels):
    """(kept indices, n_conflicting, n_duplicate) for one split, rules 2 and 3. Order of the kept rows is preserved."""
    groups = defaultdict(list)
    for i, text in enumerate(texts):
        groups[text_key(text)].append(i)
    kept, conflicting, duplicate = [], 0, 0
    for idx in groups.values():
        if len({labels[i] for i in idx}) > 1:
            conflicting += len(idx)
        else:
            kept.append(idx[0])
            duplicate += len(idx) - 1
    return sorted(kept), conflicting, duplicate


def _class_counts(labels):
    return {int(k): v for k, v in sorted(Counter(labels).items())}


def clean_dataset(ds: Dataset) -> tuple[Dataset, dict]:
    """(cleaned Dataset, full report). The cleaned Dataset keeps the name and records a compact report in
    `split_info["cleaning"]`, so a result file says which data it ran on; the full report is for `cleaning/<name>.json`."""
    n_train, n_test = len(ds.train_texts), len(ds.test_texts)
    train = [(t, y) for t, y in zip(ds.train_texts, ds.train_labels) if t.strip()]
    test = [(t, y) for t, y in zip(ds.test_texts, ds.test_labels) if t.strip()]
    empty = {"train": n_train - len(train), "test": n_test - len(test)}

    train_keys = {text_key(t) for t, _ in train}          # leakage is judged against every non-empty train text
    tr_keep, tr_conflict, tr_dup = _dedupe([t for t, _ in train], [y for _, y in train])
    te_keep, te_conflict, te_dup = _dedupe([t for t, _ in test], [y for _, y in test])
    test = [test[i] for i in te_keep]
    train = [train[i] for i in tr_keep]
    n_before_leak = len(test)
    test = [(t, y) for t, y in test if text_key(t) not in train_keys]

    cleaned = Dataset(ds.name, ds.hf_id, ds.config, [t for t, _ in train], [y for _, y in train],
                      [t for t, _ in test], [y for _, y in test], ds.label_names, dict(ds.split_info))
    removed = {"empty": empty, "conflicting": {"train": tr_conflict, "test": te_conflict},
               "duplicate": {"train": tr_dup, "test": te_dup}, "test_in_train": n_before_leak - len(test)}
    report = {
        "dataset": ds.name,
        "rules": "empty -> conflicting -> duplicate -> test_in_train (see experiments/clean.py)",
        "text_key": "lower-cased, whitespace collapsed; kept texts are the original strings",
        "n_before": {"train": n_train, "test": n_test}, "n_after": {"train": len(train), "test": len(test)},
        "removed": removed,
        "class_counts_before": {"train": _class_counts(ds.train_labels), "test": _class_counts(ds.test_labels)},
        "class_counts_after": {"train": _class_counts(cleaned.train_labels), "test": _class_counts(cleaned.test_labels)},
        "classes_missing_after": {"train": sorted(set(range(len(ds.label_names))) - set(cleaned.train_labels)),
                                  "test": sorted(set(range(len(ds.label_names))) - set(cleaned.test_labels))},
    }
    cleaned.split_info["cleaning"] = {"n_before": report["n_before"], "n_after": report["n_after"], "removed": removed}
    return cleaned, report
