"""Bootstrap, McNemar, prediction sidecars and the case-sensitive grid (tiny synthetic inputs only)."""
import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from experiments import pipeline, report, run_stats, stats
from experiments.classifiers import CASE_SENSITIVE_OVERRIDES, VECTORIZER_PARAMS, make_vectorizer, vectorizer_params
from experiments.observations import observation
from tests.test_report import make_results
from tests.test_experiments import toy_dataset

QUIET = {"log": lambda *_: None}


def test_macro_f1_from_confusion_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true, y_pred = rng.integers(0, 5, 60), rng.integers(0, 6, 60)      # class 5 only predicted, some may be absent
        cm = stats.confusion_matrices(y_true, y_pred, np.arange(60)[None, :], 6)[0]
        expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
        assert stats.macro_f1_from_confusion(cm) == pytest.approx(expected, abs=1e-12)
        assert stats.accuracy_from_confusion(cm) == pytest.approx(np.mean(y_true == y_pred))


def test_batched_confusion_equals_one_by_one():
    rng = np.random.default_rng(1)
    y_true, y_pred = rng.integers(0, 3, 40), rng.integers(0, 3, 40)
    idx = stats.bootstrap_indices(40, stats.CHUNK + 7, rng)                # crosses a chunk boundary
    batched = stats.confusion_matrices(y_true, y_pred, idx, 3)
    for b in (0, stats.CHUNK - 1, stats.CHUNK, len(idx) - 1):
        one = np.zeros((3, 3), dtype=int)
        np.add.at(one, (y_true[idx[b]], y_pred[idx[b]]), 1)
        assert (batched[b] == one).all()


def test_bootstrap_accuracy_ci_known_answer():
    """80% correct out of n = 10,000: the CI is close to 0.8 +- 1.96 * sqrt(0.8 * 0.2 / n)."""
    n = 10_000
    y_true = np.zeros(n, dtype=np.int64)
    y_pred = np.where(np.arange(n) < 8_000, 0, 1)
    idx = stats.bootstrap_indices(n, 2000, np.random.default_rng(0))
    ci = stats.run_summary(y_true, y_pred, idx, 2)["accuracy"]["ci"]
    half = 1.96 * np.sqrt(0.8 * 0.2 / n)
    assert ci[0] == pytest.approx(0.8 - half, abs=0.0015) and ci[1] == pytest.approx(0.8 + half, abs=0.0015)


def test_perfect_predictions_give_degenerate_ci():
    y = np.array([0, 1, 2, 1, 0, 2])
    idx = stats.bootstrap_indices(6, 200, np.random.default_rng(0))
    summary = stats.run_summary(y, y, idx, 3)
    assert summary["accuracy"] == {"value": 1.0, "ci": [1.0, 1.0]}


def test_identical_predictions_difference_ci_contains_zero():
    rng = np.random.default_rng(2)
    y_true, y_pred = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    idx = stats.bootstrap_indices(300, 500, rng)
    diff = stats.paired_difference(y_true, y_pred, y_pred.copy(), idx, 4)
    for metric in ("accuracy", "macro_f1"):
        assert diff[metric]["diff"] == 0 and diff[metric]["ci"] == [0.0, 0.0]
        assert not diff[metric]["ci_excludes_zero"]


def test_clearly_better_system_difference_ci_excludes_zero():
    y_true = np.tile([0, 1], 200)
    good, bad = y_true.copy(), np.zeros_like(y_true)
    idx = stats.bootstrap_indices(len(y_true), 500, np.random.default_rng(3))
    assert stats.paired_difference(y_true, bad, good, idx, 2)["accuracy"]["ci_excludes_zero"]


def test_mcnemar_known_tables():
    """b = 1, c = 6: p = 2 * (C(7,0) + C(7,1)) / 2^7 = 0.125; no discordant pairs: p = 1."""
    y = np.zeros(10, dtype=int)
    a = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0])     # right on 0, 7, 8, 9
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])     # right on 1..9
    result = stats.mcnemar_exact(y, a, b)
    assert (result["a_right_b_wrong"], result["a_wrong_b_right"]) == (1, 6)
    assert result["p_value"] == pytest.approx(0.125)
    assert stats.mcnemar_exact(y, a, a)["p_value"] == 1.0


def test_run_cell_stores_predictions_sidecar(tmp_path):
    ds = toy_dataset()
    record = pipeline.run_cell(ds, "whitespace", ["lr"], tmp_path, **QUIET)[0]
    path = pipeline.result_path(tmp_path, ds.name, "whitespace", "lr")
    assert record["predictions_file"] == "predictions/toy__whitespace__lr.json.gz"
    y_true, y_pred = pipeline.load_predictions(path)
    assert y_true.tolist() == ds.test_labels and len(y_pred) == len(y_true)
    assert record["metrics"]["accuracy"] == pytest.approx(np.mean(y_true == y_pred))
    sidecar = tmp_path / record["predictions_file"]
    first = sidecar.read_bytes()
    pipeline.run_cell(ds, "whitespace", ["lr"], tmp_path, force=True, **QUIET)
    assert sidecar.read_bytes() == first                       # deterministic bytes (no gzip timestamp)
    sidecar.unlink()
    assert not pipeline.is_fresh(path, pipeline.fingerprint(ds, "whitespace", "lr", 0))


def test_run_stats_end_to_end(tmp_path):
    res = make_results(tmp_path)
    boot = run_stats.compute(res, n_boot=50, seed=1)
    d = boot["datasets"]["ag_news"]
    assert set(d["pairs"]) == {name for name, *_ in run_stats.PAIRS}
    assert "mcnemar_accuracy" in d["pairs"]["tokenizer_only_lr"]
    assert boot["config"]["n_boot"] == 50 and boot["config"]["seed"] == 1
    assert run_stats.compute(res, n_boot=50, seed=1) == boot   # fixed seed: same result
    run_stats.write(res, n_boot=50, seed=1)
    tables = report.render_all_tables(res)
    assert tables["ci_table"].count("\n") == 4 and "McNemar" in tables["ci_table"]
    assert "lexfront" in tables["ci_runs"]
    assert "exclude 0" in observation("ci", res)


def test_case_sensitive_grid_differs_only_in_lowercase(tmp_path):
    params = vectorizer_params(CASE_SENSITIVE_OVERRIDES)
    assert params == {**VECTORIZER_PARAMS, "lowercase": False}
    assert make_vectorizer(str.split, params).get_params()["lowercase"] is False
    with pytest.raises(ValueError):
        vectorizer_params({"lowercas": False})
    ds = toy_dataset()
    assert pipeline.config_fingerprint(ds, "nltk", "lr", 0, params) != pipeline.config_fingerprint(ds, "nltk", "lr", 0)


def test_sensitivity_table_and_observation(tmp_path):
    res = make_results(tmp_path)
    ds = toy_dataset()
    ds.name = "ag_news"
    params = vectorizer_params(CASE_SENSITIVE_OVERRIDES)
    for tok, sub in [("whitespace", "module1"), ("nltk", "module1"), ("tweet", "module1"), ("lexfront", "module2")]:
        pipeline.run_cell(ds, tok, ["nb", "lr"], res / "sensitivity_case" / sub, vec_params=params, **QUIET)
    stored = json.loads(pipeline.result_path(res / "sensitivity_case" / "module2", "ag_news", "lexfront", "lr")
                        .read_text())
    assert stored["vectorizer"]["lowercase"] is False
    table = report.render_all_tables(res)["sensitivity_case"]
    assert table.count("\n") == 5 and "LR acc change vs main" in table
    assert "with case kept" in observation("sensitivity_case", res)


def test_sensitivity_ci_table_and_observation(tmp_path):
    res = make_results(tmp_path)
    ds = toy_dataset()
    ds.name = "ag_news"
    params = vectorizer_params(CASE_SENSITIVE_OVERRIDES)
    for tok, sub in [("nltk", "module1"), ("lexfront", "module2")]:
        pipeline.run_cell(ds, tok, ["lr"], res / "sensitivity_case" / sub, vec_params=params, **QUIET)
    assert "No CIs for this grid" in observation("sensitivity_case", res)
    run_stats.write(res, n_boot=50, seed=1)
    run_stats.write(res / "sensitivity_case", n_boot=50, seed=1)
    table = report.render_all_tables(res)["sensitivity_ci"]
    assert table.count("\n") == 2 and "case kept" in table
    text = observation("sensitivity_case", res)
    assert "main grid:" in text and "Whether the tokenizer-only accuracy CI excludes 0" in text
