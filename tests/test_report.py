import json

"""Tables and figures are built from JSON only (fixture JSON written by the real pipeline)."""
from experiments import report
from experiments.figures import build_all_figures
from experiments.pipeline import run_cell
from tests.test_experiments import toy_dataset


def make_results(tmp_path):
    ds = toy_dataset()
    ds.name = "ag_news"
    for tok in ("whitespace", "nltk", "tweet"):
        run_cell(ds, tok, ["majority", "nb", "lr"], tmp_path / "module1", log=lambda *_: None)
    run_cell(ds, "lexfront", ["nb", "lr"], tmp_path / "module2", log=lambda *_: None)
    summary = dict(ds.summary(), train_class_counts={"0": 12, "1": 12}, test_class_counts={"0": 8, "1": 8})
    (tmp_path / "datasets.json").write_text(json.dumps({"datasets": [summary]}))
    return tmp_path


def test_tables_from_json(tmp_path):
    res = make_results(tmp_path)
    tables = report.build_all_tables(res)
    assert "ag_news" in tables["datasets"] and "none" in tables["datasets"]
    assert tables["module1_results"].count("\n") >= 4          # header + sep + 3 tokenizers
    row = tables["existing_vs_proposed"].splitlines()[2]
    assert row.startswith("| ag_news | ") and row.count("/") == 4    # existing, proposed and the two mixed cells
    assert (res / "tables" / "module1_results.md").exists()
    assert "not available" in tables["intrinsic_metrics"]       # no intrinsic JSON in this fixture


def test_figures_from_json(tmp_path):
    res = make_results(tmp_path)
    written = build_all_figures(res)
    assert "module1_accuracy.png" in written and "existing_vs_proposed.png" in written
