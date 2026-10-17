"""WordPiece and Unigram (P4): training data, interface, determinism, fingerprint, grid, stats, table, observation."""
import json
import os
import subprocess
import sys

import pytest

from experiments import pipeline, report, run_subword2
from experiments.observations import observation
from experiments.tokenizers import TOKENIZERS, LIBRARY_SUBWORD, build_tokenizer, subword_record
from subword.unigram import UnigramConfig, UnigramTokenizer, train_unigram
from subword.wordpiece import WordPieceConfig, WordPieceTokenizer, train_wordpiece
from tests.test_bpe import QUIET, TRAIN
from tests.test_experiments import toy_dataset

WP = WordPieceConfig(vocab_size=80)
UNI = UnigramConfig(vocab_size=60)
TEXTS = TRAIN + ["zebra quiz jumps", "quick brown fox"]


@pytest.mark.parametrize("tok", [WordPieceTokenizer(TRAIN, WP), UnigramTokenizer(TRAIN, UNI)])
def test_interface_returns_list_of_str(tok):
    out = tok("The cat is running @bob https://x.co #tag 42 :)")
    assert isinstance(out, list) and out and all(isinstance(t, str) for t in out)


def test_wordpiece_marks_continuations_and_unigram_does_not():
    wp = WordPieceTokenizer(TRAIN, WordPieceConfig(vocab_size=40))("running")
    assert wp[0] == "r" and all(p.startswith("##") for p in wp[1:])
    assert not any(p.startswith("##") for p in UnigramTokenizer(TRAIN, UNI)("running cats"))


def test_unseen_characters_are_unk_and_literal_unk_is_not_special():
    for tok in (WordPieceTokenizer(TRAIN, WP), UnigramTokenizer(TRAIN, UNI)):
        assert "[UNK]" in tok("qzx")
        assert json.loads(tok.model.to_str())["added_tokens"] == []
        assert tok("[UNK]") != ["[UNK]"]


@pytest.mark.parametrize("name", ["wordpiece", "unigram"])
def test_trained_on_train_texts_only(name, tmp_path, monkeypatch):
    """The factory gets ds.train_texts, and characters that occur only in test are not in the learned vocabulary."""
    ds = toy_dataset()
    ds.test_texts = [t + " qzjk" for t in ds.test_texts]
    seen = []

    def spy(tname, train_texts):
        seen.append(list(train_texts))
        return build_tokenizer(tname, train_texts)
    monkeypatch.setattr(pipeline, "build_tokenizer", spy)
    pipeline.run_cell(ds, name, ["nb"], tmp_path, **QUIET)
    assert seen == [ds.train_texts]
    vocab = set(build_tokenizer(name, ds.train_texts).model.get_vocab())
    assert not {"q", "z", "j", "k", "##z", "##j", "##k"} & vocab
    assert "[UNK]" in build_tokenizer(name, ds.train_texts)("qzjk")


def test_factory_accepts_an_iterator():
    for name in LIBRARY_SUBWORD:
        assert build_tokenizer(name, iter(TEXTS)).model.get_vocab_size() > 1


def test_training_is_deterministic():
    assert train_wordpiece(TEXTS, WP).to_str() == train_wordpiece(TEXTS, WP).to_str()
    assert train_unigram(TEXTS, UNI).to_str() == train_unigram(TEXTS, UNI).to_str()
    assert json.loads(train_wordpiece(TEXTS, WP).to_str())["added_tokens"] == []


def test_unigram_canonical_form_rounds_scores_and_fixes_the_order():
    vocab = json.loads(train_unigram(TEXTS, UNI).to_str())["model"]["vocab"]
    assert vocab[0][0] == "[UNK]"
    keys = [(-round(score, UNI.score_decimals), piece) for piece, score in vocab[1:]]
    assert keys == sorted(keys) and all(score == round(score, UNI.score_decimals) for _, score in vocab)


def test_training_does_not_depend_on_the_thread_count():
    code = ("import sys; sys.path.insert(0, '.'); from subword.wordpiece import WordPieceConfig, train_wordpiece; "
            "from subword.unigram import UnigramConfig, train_unigram; from tests.test_subword2 import TEXTS; "
            "print(train_wordpiece(TEXTS, WordPieceConfig(vocab_size=80)).to_str()); "
            "print(train_unigram(TEXTS, UnigramConfig(vocab_size=60)).to_str())")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "RAYON_NUM_THREADS": n}).stdout for n in ("1", "4")}
    assert len(outs) == 1


def test_fingerprint_changes_with_vocab_size_and_training_text(monkeypatch):
    ds, other = toy_dataset(), toy_dataset()
    other.train_texts = other.train_texts[:-1] + ["a different train text"]
    fp = pipeline.config_fingerprint
    before = {name: fp(ds, name, "lr", 0) for name in LIBRARY_SUBWORD}
    for name, (cls, config) in list(LIBRARY_SUBWORD.items()):
        assert before[name] != fp(other, name, "lr", 0)
        rec = subword_record(name, ds.train_texts)["subword"]
        assert rec["model"] == name and rec["vocab_size"] == 8000 and rec["tokenizers"]
        monkeypatch.setitem(LIBRARY_SUBWORD, name, (cls, type(config)(vocab_size=123)))
        assert fp(ds, name, "lr", 0) != before[name]
        assert subword_record(name, ds.train_texts)["subword"]["vocab_size"] == 123
    assert fp(ds, "wordpiece", "lr", 0) != fp(ds, "unigram", "lr", 0)
    assert not set(LIBRARY_SUBWORD) & set(TOKENIZERS)


def test_subword2_grid_stats_table_and_observation(tmp_path, monkeypatch):
    def toy_load(name, *_args, **_kw):
        ds = toy_dataset()
        ds.name = name
        return ds
    monkeypatch.setattr(run_subword2.data, "load", toy_load)
    monkeypatch.setitem(LIBRARY_SUBWORD, "wordpiece", (WordPieceTokenizer, WP))           # tiny vocabularies for speed
    monkeypatch.setitem(LIBRARY_SUBWORD, "unigram", (UnigramTokenizer, UNI))
    records = run_subword2.run_subword2(tmp_path / "subword2", dataset_names=["ag_news"], **QUIET)
    assert len(records) == 4 and {r["subword"]["model"] for r in records} == {"wordpiece", "unigram"}
    ds = toy_load("ag_news")
    pipeline.run_cell(ds, "nltk", ["lr"], tmp_path / "module1", **QUIET)
    with pytest.raises(FileNotFoundError):                                        # BPE+LR still missing
        run_subword2.compute(tmp_path / "subword2")
    pipeline.run_cell(ds, "bpe", ["lr"], tmp_path / "bpe", **QUIET)
    run_subword2.write_stats(tmp_path / "subword2", n_boot=50, seed=1)
    boot = report.load_bpe(tmp_path, "subword2")
    pairs = boot["datasets"]["ag_news"]["pairs"]
    assert list(pairs) == [n for n, *_ in run_subword2.PAIR_SPEC]
    assert {n for n, p in pairs.items() if "mcnemar_accuracy" in p} == {"wordpiece_vs_nltk_lr", "unigram_vs_nltk_lr"}
    assert run_subword2.compute(tmp_path / "subword2", n_boot=50, seed=1) == boot
    tables = report.render_all_tables(tmp_path)
    assert tables["subword_table"].count("\n") == 1 + 5 and tables["subword_runs"].count("\n") == 2
    text = observation("subword", tmp_path)
    assert "**ag_news**" in text and "of 10 difference CIs exclude 0" in text and "hypothesis" in text
    assert "WordPiece+LR vs NLTK+LR" in text and "Unigram+LR vs NLTK+LR" in text
