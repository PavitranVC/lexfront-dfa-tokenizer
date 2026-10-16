"""BPE and the LexFront + BPE hybrid (P3): training data, pass-through, determinism, interface, fingerprint, grid."""
import json
import os
import subprocess
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from experiments import pipeline, report, run_bpe
from experiments.observations import observation
from experiments.tokenizers import TOKENIZERS, build_tokenizer, subword_record
from lexer.normalize import WORD_LIKE
from lexer.token_spec import CLASS_NAMES
from subword.bpe import STRUCTURAL, BpeConfig, BpeTokenizer, LexFrontBpe, train_bpe
from tests.test_experiments import toy_dataset

QUIET = {"log": lambda *_: None}
TRAIN = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs running fast", "running jumping sitting"] * 5
SMALL = BpeConfig(vocab_size=80)
HYBRID = BpeConfig(vocab_size=80, pre_tokenizer="whitespace_split")


def test_interface_returns_list_of_str():
    for tok in (BpeTokenizer(TRAIN, SMALL), LexFrontBpe(TRAIN, HYBRID)):
        out = tok("The cat is running @bob https://x.co #tag 42 :)")
        assert isinstance(out, list) and out and all(isinstance(t, str) for t in out)


def test_bpe_is_trained_on_train_texts_only(monkeypatch, tmp_path):
    """The factory gets ds.train_texts, and characters that occur only in test are not in the learned vocabulary."""
    ds = toy_dataset()
    ds.test_texts = [t + " qzjk" for t in ds.test_texts]
    seen = []

    def spy(name, train_texts):
        seen.append(list(train_texts))
        return build_tokenizer(name, train_texts)
    monkeypatch.setattr(pipeline, "build_tokenizer", spy)
    pipeline.run_cell(ds, "bpe", ["nb"], tmp_path, **QUIET)
    assert seen == [ds.train_texts]
    vocab = build_tokenizer("bpe", ds.train_texts).model.get_vocab()
    assert not {"q", "z", "j", "k", "##z", "##j", "##k"} & set(vocab)
    assert build_tokenizer("bpe", ds.train_texts)("qzjk") == ["[UNK]"] * 4


def test_structural_tokens_pass_through_the_hybrid_unsplit():
    hybrid = LexFrontBpe(TRAIN, BpeConfig(vocab_size=40, pre_tokenizer="whitespace_split"))   # alphabet only
    out = hybrid("running @someone https://example.com/very/long #superlongtag 1,234 :) \U0001F600 !! §")
    for whole in ("<USER>", "<URL>", "#superlongtag", "<NUM>", ":)", "\U0001F600", "!", "§"):
        assert whole in out
    assert "running" not in out and "r" in out          # word-like tokens are split into pieces
    assert set(STRUCTURAL) | WORD_LIKE == set(CLASS_NAMES) and not set(STRUCTURAL) & WORD_LIKE


def test_training_is_deterministic_and_keeps_the_trainers_vocabulary():
    texts = TRAIN + ["zebra quiz jumps", "quick brown fox"]
    first, second = train_bpe(texts, SMALL), train_bpe(texts, SMALL)
    assert first.to_str() == second.to_str()
    plain = Tokenizer(models.BPE(unk_token="[UNK]", continuing_subword_prefix="##"))
    plain.pre_tokenizer = pre_tokenizers.Whitespace()
    plain.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=80, special_tokens=["[UNK]"], show_progress=False,
                                                         continuing_subword_prefix="##"))
    assert set(plain.get_vocab()) == set(first.get_vocab())       # seeding the alphabet only fixes the ids
    assert json.loads(first.to_str())["added_tokens"] == []


def test_training_does_not_depend_on_the_thread_count():
    code = ("import sys; sys.path.insert(0, '.'); from subword.bpe import BpeConfig, train_bpe; "
            "from tests.test_bpe import TRAIN; print(train_bpe(TRAIN, BpeConfig(vocab_size=80)).to_str())")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "RAYON_NUM_THREADS": n}).stdout for n in ("1", "4")}
    assert len(outs) == 1


def test_fingerprint_changes_with_vocab_size_and_training_text():
    ds = toy_dataset()
    other = toy_dataset()
    other.train_texts = other.train_texts[:-1] + ["a different train text"]
    fp = pipeline.config_fingerprint
    assert fp(ds, "bpe", "lr", 0) != fp(ds, "bpe:v32000", "lr", 0)
    assert fp(ds, "bpe", "lr", 0) != fp(other, "bpe", "lr", 0)
    assert fp(ds, "bpe", "lr", 0) != fp(ds, "lexfront_bpe", "lr", 0)
    rec = subword_record("bpe", ds.train_texts)["subword"]
    assert rec["vocab_size"] == 8000 and rec["train_text_hash"] != subword_record("bpe", other.train_texts)[
        "subword"]["train_text_hash"]
    assert subword_record("nltk", ds.train_texts) == {} and not {"bpe", "lexfront_bpe"} & set(TOKENIZERS)


def test_bpe_grid_stats_table_and_observation(tmp_path, monkeypatch):
    def toy_load(name, *_args, **_kw):
        ds = toy_dataset()
        ds.name = name
        return ds
    monkeypatch.setattr(run_bpe.data, "load", toy_load)
    monkeypatch.setitem(run_bpe.SUBWORD_TOKENIZERS, "bpe", (False, 60))           # tiny vocabularies for speed
    monkeypatch.setitem(run_bpe.SUBWORD_TOKENIZERS, "lexfront_bpe", (True, 60))
    records = run_bpe.run_bpe(tmp_path / "bpe", dataset_names=["ag_news"], **QUIET)
    assert len(records) == 4 and {r["subword"]["hybrid"] for r in records} == {True, False}
    ds = toy_load("ag_news")
    pipeline.run_cell(ds, "nltk", ["lr"], tmp_path / "module1", **QUIET)
    with pytest.raises(FileNotFoundError):                                        # LexFront+LR still missing
        run_bpe.compute(tmp_path / "bpe")
    pipeline.run_cell(ds, "lexfront", ["lr"], tmp_path / "module2", **QUIET)
    run_bpe.write_stats(tmp_path / "bpe", n_boot=50, seed=1)
    boot = report.load_bpe(tmp_path)
    pairs = boot["datasets"]["ag_news"]["pairs"]
    assert list(pairs) == ["bpe_vs_nltk_lr", "hybrid_vs_lexfront_lr", "hybrid_vs_bpe_lr", "hybrid_vs_nltk_lr"]
    assert "mcnemar_accuracy" in pairs["bpe_vs_nltk_lr"] and "mcnemar_accuracy" not in pairs["hybrid_vs_bpe_lr"]
    assert run_bpe.compute(tmp_path / "bpe", n_boot=50, seed=1) == boot
    tables = report.render_all_tables(tmp_path)
    assert tables["bpe_table"].count("\n") == 1 + 4 and tables["bpe_runs"].count("\n") == 2
    text = observation("bpe", tmp_path)
    assert "**ag_news**" in text and "of 8 difference CIs exclude 0" in text and "hypothesis" in text


def test_vocab_size_names():
    assert run_bpe.tokenizer_names() == ("bpe", "lexfront_bpe")
    assert run_bpe.tokenizer_names(32000) == ("bpe:v32000", "lexfront_bpe:v32000")
    with pytest.raises(ValueError):
        run_bpe.tokenizer_names(123)
