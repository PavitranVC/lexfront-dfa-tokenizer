"""Tests for the phase A engine (wide search, character baseline, units, statistics). Toy fixtures only; no reported number comes from here."""
import json
import shutil

import numpy as np

from experiments import boost, large, run_boost
from tests.test_large import toy_raw


def toy_dataset():
    return large.build_dataset(toy_raw(n_users=120, per_user=10))[0]


def test_split_is_disjoint_stratified_and_deterministic():
    labels = [i % 2 for i in range(1000)]
    fit, val = boost.split_indices(labels)
    assert not set(fit) & set(val) and (fit, val) == boost.split_indices(labels)
    assert len(val) == 100 and abs(np.mean([labels[i] for i in val]) - 0.5) < 0.02


def test_pick_best_ties_go_to_the_earlier_candidate():
    scores = [{"vectorizer": "a", "value": 1, "val_accuracy": 0.8}, {"vectorizer": "b", "value": 1, "val_accuracy": 0.8},
              {"vectorizer": "c", "value": 1, "val_accuracy": 0.7}]
    assert boost.pick_best(scores)["vectorizer"] == "a"
    scores[2]["val_accuracy"] = 0.9
    assert boost.pick_best(scores)["vectorizer"] == "c"


def test_search_sees_only_train_and_picks_the_argmax():
    ds = toy_dataset()
    sel = boost.search(ds.train_texts, ds.train_labels, {k: boost.WORD_GRID[k] for k in ("w1_raw_df2", "w2_sub_df2")}, "nb",
                       lambda t: t.lower().split())
    assert sel["n_fit"] + sel["n_val"] <= len(ds.train_texts) and len(sel["scores"]) == 2 * len(boost.NB_ALPHA)
    assert sel["selected"]["val_accuracy"] == max(s["val_accuracy"] for s in sel["scores"])


def test_units_resume_and_anchor_equals_the_default_fit(tmp_path):
    ds = toy_dataset()
    assert boost.run_units(ds, ["whitespace", boost.CHAR], tmp_path, log=lambda *_: None) == 5
    assert boost.run_units(ds, ["whitespace", boost.CHAR], tmp_path, log=lambda *_: None) == 0       # all cached
    out = tmp_path / "out"
    large.consolidate(tmp_path, out)
    runs = json.loads((out / "runs.json").read_text())["runs"]
    assert {r["run_id"] for r in runs} == {"whitespace__anchor", "whitespace__wide__lr", "whitespace__wide__nb", "char__wide__lr", "char__wide__nb"}
    records, preds = large.fit_and_evaluate(ds.train_texts, ds.train_labels, {"clean": ds.test_texts}, np.asarray(ds.test_labels),
                                            "whitespace", "default", 0, boost.tuning.cached(boost.build_tokenizer("whitespace", [])), True)
    _, stored = large.load_predictions(out / "predictions" / "whitespace__anchor.json.gz")
    assert np.array_equal(stored["lr"]["clean"], preds["lr"]["clean"])
    (out / "split.json").write_text(json.dumps({"split": {"hf_id": "toy", "repo_sha": "x", "users_in_both_splits": 0},
                                                "cleaning": {"n_after": {"train": len(ds.train_texts), "test": len(ds.test_texts)}}}))
    run_boost.write_outputs(out, None, "toy", n_boot=50)
    report = (out / "REPORT.md").read_text()
    assert "majority-class accuracy" in report and "wide-tuned" in report and "HA" not in report.split("## 6.")[0]
    s = json.loads((out / "stats.json").read_text())
    assert s["accuracy"]["whitespace|wide|lr"]["value"] > s["majority_accuracy"]
    shutil.rmtree(out)


def test_verdict_uses_the_margin_not_zero():
    assert run_boost.verdict([-0.001, 0.002]) == "within"
    assert run_boost.verdict([0.001, 0.004]) == "within"          # excludes 0 but sits inside the margin
    assert run_boost.verdict([0.006, 0.009]) == "beyond (up)"
    assert run_boost.verdict([-0.009, -0.006]) == "beyond (down)"
    assert run_boost.verdict([0.002, 0.008]) == "unclear"


def test_verify_passes_on_a_clean_run_and_fails_when_a_prediction_changes(tmp_path):
    from experiments import verify_boost
    ds, report = large.build_dataset(toy_raw(n_users=120, per_user=10))
    boost.run_units(ds, ["whitespace"], tmp_path / "w", log=lambda *_: None)
    out = tmp_path / "out"
    large.consolidate(tmp_path / "w", out)
    (out / "split.json").write_text(json.dumps(report))
    run_boost.write_outputs(out, None, "toy", n_boot=50)
    assert verify_boost.main(["--out", str(out)]) == 0
    path = out / "predictions" / "whitespace__wide__lr.json.gz"
    y, preds = large.load_predictions(path)
    preds["lr"]["clean"][:] = 1 - preds["lr"]["clean"]                      # flip every stored prediction
    large.save_predictions(path, y, preds)
    assert verify_boost.main(["--out", str(out)]) == 1
