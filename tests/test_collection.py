"""Tests for the dataset collection: cleaning, extra loaders, fair tuning, the collection runner and its report.

The datasets below are tiny test fixtures; no reported number comes from them. Nothing here touches the network.
"""
import json

import numpy as np
import pytest

from experiments import clean, data, data_extra, run_collection, tuning
from experiments.pipeline import load_predictions


def make_ds(train, test, names=("a", "b")):
    return data.Dataset("fx", "none", None, [t for t, _ in train], [y for _, y in train],
                        [t for t, _ in test], [y for _, y in test], list(names), {"train": "x", "test": "y"})


# ---- cleaning ------------------------------------------------------------------------------------------------
def test_clean_each_rule_and_counts():
    train = [("Hello  World", 0), ("hello world", 0),              # duplicate after the key (case + whitespace)
             ("same text", 0), ("SAME TEXT", 1),                   # conflicting labels: both dropped
             ("   ", 1), ("", 0),                                   # empty
             ("only in train", 1), ("leaky one", 0)]
    test = [("leaky ONE", 0), ("fresh test text", 1), ("fresh test text", 1), ("", 1)]
    cleaned, report = clean.clean_dataset(make_ds(train, test))
    assert cleaned.train_texts == ["Hello  World", "only in train", "leaky one"]   # originals kept, order preserved
    assert cleaned.test_texts == ["fresh test text"]
    assert report["removed"] == {"empty": {"train": 2, "test": 1}, "conflicting": {"train": 2, "test": 0},
                                 "duplicate": {"train": 1, "test": 1}, "test_in_train": 1}
    assert report["n_before"] == {"train": 8, "test": 4} and report["n_after"] == {"train": 3, "test": 1}
    assert cleaned.split_info["cleaning"]["n_after"] == {"train": 3, "test": 1}
    assert report["classes_missing_after"] == {"train": [], "test": [0]}


def test_leakage_is_judged_against_dropped_train_texts_too():
    """A test text equal to a train text that was dropped as conflicting must still not stay in test."""
    train = [("clash", 0), ("clash", 1), ("ok", 0)]
    cleaned, report = clean.clean_dataset(make_ds(train, [("clash", 0), ("new", 1)]))
    assert cleaned.test_texts == ["new"] and report["removed"]["test_in_train"] == 1


def test_clean_is_deterministic_and_leaves_clean_data_unchanged():
    train = [("one", 0), ("two", 1), ("three", 0)]
    test = [("four", 1), ("five", 0)]
    a, _ = clean.clean_dataset(make_ds(train, test))
    b, report = clean.clean_dataset(make_ds(train, test))
    assert a.train_texts == b.train_texts == ["one", "two", "three"] and a.test_texts == ["four", "five"]
    assert a.train_labels == [0, 1, 0] and a.test_labels == [1, 0]
    assert all(v == 0 or v == {"train": 0, "test": 0} for v in report["removed"].values())


def test_clean_does_not_edit_the_input_dataset():
    ds = make_ds([("a a", 0), ("a  a", 0)], [("b", 1)])
    clean.clean_dataset(ds)
    assert ds.train_texts == ["a a", "a  a"] and "cleaning" not in ds.split_info


# ---- collection registry -------------------------------------------------------------------------------------
def test_collection_extends_the_original_grid_without_changing_it():
    assert data.DATASET_NAMES == ("tweet_topic", "go_emotions", "ag_news")
    assert data_extra.COLLECTION_NAMES[:3] == data.DATASET_NAMES
    assert len(data_extra.COLLECTION_NAMES) == 8 and len(set(data_extra.COLLECTION_NAMES)) == 8
    assert set(data_extra.EXTRA_LOADERS) == {f"tweet_eval_{t}" for t in data_extra.TWEET_EVAL_TASKS}


def test_load_cleans_then_subsamples(monkeypatch):
    train = [(f"text {i}", i % 2) for i in range(40)] + [("text 0", 0)]
    monkeypatch.setitem(data_extra.EXTRA_LOADERS, "fx", lambda: make_ds(train, [("text 1", 1), ("novel", 0)]))
    ds, report = data_extra.load("fx", True, 10, 1, seed=3)
    assert report["n_after"] == {"train": 40, "test": 1}               # cleaned on the full data
    assert len(ds.train_texts) == 10 and len(ds.test_texts) == 1       # then subsampled
    raw, none_report = data_extra.load("fx", False)
    assert none_report is None and len(raw.train_texts) == 41
    with pytest.raises(KeyError):
        data_extra.load_raw("nope")


# ---- tuning ---------------------------------------------------------------------------------------------------
def toy_text_ds(n=120):
    rng = np.random.default_rng(0)
    good, bad = ["great", "happy", "love", "fun"], ["awful", "sad", "hate", "dull"]
    train, test = [], []
    for i in range(n):
        label = i % 2
        words = rng.choice(good if label == 0 else bad, 3).tolist() + rng.choice(["the", "a", "day"], 2).tolist()
        (train if i < n * 3 // 4 else test).append((f"{' '.join(words)} #{i}", label))
    return make_ds(train, test)


def test_split_indices_are_deterministic_disjoint_and_stratified():
    labels = [0, 1] * 50
    fit, val = tuning.split_indices(labels, seed=0)
    assert (fit, val) == tuning.split_indices(labels, seed=0)
    assert not set(fit) & set(val) and len(fit) + len(val) == 100 and len(val) == 10
    assert sum(labels[i] for i in val) == 5                              # stratified: half of each class
    fit, val = tuning.split_indices([0] * 30 + [1], seed=0)              # a singleton class: falls back, no crash
    assert not set(fit) & set(val)


def test_search_never_sees_the_test_split_and_picks_the_best_validation_score():
    ds = toy_text_ds()
    ds.test_texts = ["POISON"] * len(ds.test_texts)                      # a tokenizer that explodes on test text
    seen = []

    def tokenizer(text):
        seen.append(text)
        return text.split()
    selection = tuning.search(ds, tokenizer, "lr", seed=0)
    assert seen and not any("poison" in t for t in seen)                 # only train texts were ever tokenized
    best = max(s["val_accuracy"] for s in selection["scores"])
    assert selection["selected"]["val_accuracy"] == best
    assert selection["selected"] == next(s for s in selection["scores"] if s["val_accuracy"] == best)   # ties: earliest
    assert len(selection["scores"]) == len(tuning.SEARCH["vectorizer"]) * len(tuning.SEARCH["lr_C"])


def test_cached_tokenizer_gives_identical_tokens_and_calls_once():
    calls = []
    cached = tuning.cached(lambda t: calls.append(t) or t.split())
    assert cached("a b") == ["a", "b"] == cached("a b") and calls == ["a b"]
    cached("a b").append("x")
    assert cached("a b") == ["a", "b"]                                   # callers cannot corrupt the memo


def test_tuned_run_cell_writes_results_and_reuses_a_fresh_cache(tmp_path):
    ds = toy_text_ds()
    first = tuning.run_cell(ds, "whitespace", tmp_path, seed=0, log=lambda *_: None)
    assert [r["classifier"]["name"] for r in first] == ["nb", "lr"]
    for r in first:
        assert r["setting"] == "tuned" and r["tuning"]["selected"] in r["tuning"]["scores"]
        path = tmp_path / f"fx__whitespace__{r['classifier']['name']}.json"
        y_true, y_pred = load_predictions(path)
        assert list(y_true) == ds.test_labels and len(y_pred) == len(ds.test_labels)
        assert abs(float((y_true == y_pred).mean()) - r["metrics"]["accuracy"]) < 1e-12
    stamp = {r["classifier"]["name"]: r["timestamp_utc"] + r["fingerprint"]["config"] for r in first}
    again = tuning.run_cell(ds, "whitespace", tmp_path, seed=0, log=lambda *_: None)
    assert {r["classifier"]["name"]: r["timestamp_utc"] + r["fingerprint"]["config"] for r in again} == stamp
    ds.train_texts[0] = "completely different text"                    # changed data -> new fingerprint -> recomputed
    assert tuning.expected_fingerprint(ds, "whitespace", "lr", 0) != first[1]["fingerprint"]


def test_same_search_for_every_tokenizer():
    """The grid is data, not per-tokenizer code: fingerprints differ only through the tokenizer name."""
    ds = toy_text_ds()
    a, b = (tuning.expected_fingerprint(ds, t, "lr", 0) for t in ("nltk", "lexfront"))
    assert a != b and a["code"] == b["code"]
    assert tuning.hyper_grid("lr") == tuning.SEARCH["lr_C"] and tuning.hyper_grid("nb") == tuning.SEARCH["nb_alpha"]


# ---- runner, statistics and report ------------------------------------------------------------------------------
def test_collection_pipeline_end_to_end_and_report_is_reproducible(tmp_path, monkeypatch):
    base = toy_text_ds(200)
    monkeypatch.setitem(data_extra.EXTRA_LOADERS, "fx", lambda: base)
    monkeypatch.setattr(data_extra, "COLLECTION_NAMES", ("fx",))
    run_collection.run_all(tmp_path, ["fx"], None, None, 0, False, log=lambda *_: None)
    assert (tmp_path / "cleaning" / "fx.json").exists()
    assert len(list((tmp_path / "default").glob("fx__*__*.json"))) == len(run_collection.TOKENIZERS) * 3
    assert len(list((tmp_path / "tuned").glob("fx__*__*.json"))) == len(run_collection.TOKENIZERS) * 2
    written = run_collection.write_stats(tmp_path, n_boot=50)
    assert [p.name for p in written] == ["default.json", "tuned.json"]
    record = json.loads((tmp_path / "stats" / "tuned.json").read_text())
    pair = record["datasets"]["fx"]["pairs"]["tokenizer_only_lr"]
    assert pair["a"] == "nltk__lr" and pair["b"] == "lexfront__lr" and "ci_excludes_zero" in pair["accuracy"]
    report = run_collection.render_report(tmp_path, raw_results=tmp_path / "none")
    assert "## 1. Datasets and cleaning" in report and "| fx |" in report and "## 5." in report
    assert report == run_collection.render_report(tmp_path, raw_results=tmp_path / "none")
    # --stats-only reproduces the stored statistics and report byte for byte, with no data and no fitting
    before = {p: p.read_bytes() for p in (tmp_path / "stats").glob("*.json")}
    run_collection.write_stats(tmp_path, n_boot=50)
    assert before == {p: p.read_bytes() for p in (tmp_path / "stats").glob("*.json")}


def test_stats_only_cli_needs_no_data(tmp_path, monkeypatch):
    base = toy_text_ds(200)
    monkeypatch.setitem(data_extra.EXTRA_LOADERS, "fx", lambda: base)
    monkeypatch.setattr(data_extra, "COLLECTION_NAMES", ("fx",))
    run_collection.run_all(tmp_path, ["fx"], None, None, 0, False, log=lambda *_: None)

    def no_data(*_a, **_k):
        raise AssertionError("--stats-only must not load data")
    monkeypatch.setattr(data_extra, "load", no_data)
    run_collection.main(["--out", str(tmp_path), "--stats-only", "--raw-results", str(tmp_path / "none")])
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "stats" / "default.json").exists()
