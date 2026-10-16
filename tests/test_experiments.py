"""Fast tests for data helpers, tokenizer interface, classifiers, metrics and the eval loop.

The tiny `Dataset` below is a *test fixture* only; no reported number comes from it.
"""
import json

import pytest

from experiments import data
from experiments.classifiers import CLASSIFIER_NAMES, make_classifier
from experiments.metrics import classification_metrics, intrinsic_metrics
from experiments.pipeline import run_cell
from experiments.tokenizers import TOKENIZERS


def toy_dataset():
    pos = ["great day :) @sam", "so happy sooo much", "love it https://x.co", "wonderful awesome day"] * 6
    neg = ["terrible day :(", "so sad and angry", "hate it http://y.co", "awful horrible day"] * 6
    texts = pos + neg
    labels = [0] * len(pos) + [1] * len(neg)
    return data.Dataset("toy", "none", None, texts, labels, texts[:8] + texts[-8:],
                        labels[:8] + labels[-8:], ["pos", "neg"], {"train": "x", "test": "y"})


def test_keep_single_label():
    texts, labels = data.keep_single_label(["a", "b", "c"], [[1], [1, 2], [3]])
    assert (texts, labels) == (["a", "c"], [1, 3])


def test_choose_tweet_topic_splits():
    names = ["train_coling2022", "test_coling2022", "train_2021", "test_2021"]
    assert data.choose_tweet_topic_splits(names, "see COLING2022")["train"] == "train_coling2022"
    fb = data.choose_tweet_topic_splits(names, "no mention here")
    assert fb["train"] == "train_2021" and fb["card_confirms_coling2022"] is False
    with pytest.raises(ValueError):
        data.choose_tweet_topic_splits(["foo"], "")


def test_subsample_is_deterministic_and_bounded():
    ds = toy_dataset()
    a, b = data.subsample(ds, 10, 5, seed=1), data.subsample(ds, 10, 5, seed=1)
    assert a.train_texts == b.train_texts and len(a.train_texts) == 10 and len(a.test_texts) == 5
    assert a.split_info["subsample"]["seed"] == 1
    assert len(data.subsample(ds, None, None).train_texts) == len(ds.train_texts)


@pytest.mark.parametrize("name", list(TOKENIZERS))
def test_tokenizer_interface(name):
    toks = TOKENIZERS[name]("Hello world :) #tag")
    assert isinstance(toks, list) and all(isinstance(t, str) for t in toks) and toks


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == 0.75 and 0 < m["macro_f1"] < 1


def test_intrinsic_metrics_basic():
    texts = ["a b c", "a b d"]
    m = intrinsic_metrics(str.split, texts, texts, throughput_docs=2, repeats=1)
    assert m["oov_rate_tokens"] == 0 and m["fertility"] == 1 and m["vocab_size"] == 4
    m2 = intrinsic_metrics(str.split, ["a b"], ["a z"], throughput_docs=1, repeats=1)
    assert m2["oov_rate_tokens"] == 0.5


def test_run_cell_writes_json_with_full_config(tmp_path):
    ds = toy_dataset()
    records = run_cell(ds, "lexfront", CLASSIFIER_NAMES, tmp_path, log=lambda *_: None)
    assert len(records) == 3
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 3
    rec = json.loads(files[0].read_text())
    for key in ("dataset", "tokenizer", "vectorizer", "classifier", "seed", "metrics", "environment"):
        assert key in rec
    lr = next(r for r in records if r["classifier"]["name"] == "lr")
    assert lr["metrics"]["accuracy"] >= 0.9           # toy problem is trivially separable


def test_run_cell_skips_existing(tmp_path):
    ds = toy_dataset()
    run_cell(ds, "whitespace", ["nb"], tmp_path, log=lambda *_: None)
    path = next(tmp_path.glob("*.json"))
    stamp = path.stat().st_mtime_ns
    run_cell(ds, "whitespace", ["nb"], tmp_path, log=lambda *_: None)
    assert path.stat().st_mtime_ns == stamp


def test_make_classifier_unknown():
    with pytest.raises(ValueError):
        make_classifier("svm")
