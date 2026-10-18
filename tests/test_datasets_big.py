"""Tests for the phase D loaders' pure parts (extraction, threshold, cap, cleaning) and the generated notebook. No network."""
import sys
from pathlib import Path

import pytest

from experiments import datasets_big, run_boost

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
import make_boost_notebook  # noqa: E402


class Table(dict):
    """Minimal stand-in for a `datasets` table: column access, len, select."""
    def __len__(self):
        return len(next(iter(self.values())))

    def select(self, idx):
        return Table({k: [v[i] for i in idx] for k, v in self.items()})


def civil_tables(n=40):
    mk = lambda off: Table({"text": [f"comment number {i + off} about topic {i % 7}" for i in range(n)],        # noqa: E731
                            "toxicity": [0.0, 0.49, 0.5, 0.9][0:4] * (n // 4)})
    return mk(0), mk(1000)


def test_civil_comments_binarizes_at_the_threshold_inclusive():
    texts, labels = datasets_big.extract("civil_comments", Table({"text": ["a", "b", "c", "d"], "toxicity": [0.0, 0.49, 0.5, 0.9]}))
    assert labels == [0, 0, 1, 1] and texts == ["a", "b", "c", "d"]
    with pytest.raises(ValueError):
        datasets_big.extract("nope", Table({}))


def test_cap_is_seeded_sorted_and_a_noop_when_small():
    a = datasets_big.cap_indices(1000, 100, seed=0)
    assert a == datasets_big.cap_indices(1000, 100, seed=0) and a != datasets_big.cap_indices(1000, 100, seed=1)
    assert a == sorted(set(a)) and len(a) == 100
    assert datasets_big.cap_indices(50, 100, 0) == list(range(50)) == datasets_big.cap_indices(50, None, 0)


def test_build_from_tables_caps_cleans_and_reports():
    spec = datasets_big.Spec(**{**datasets_big.SPECS["civil_comments"].__dict__, "max_train": 30, "max_test": 20})
    tr, te = civil_tables()
    te["text"][0] = tr["text"][3]                      # a test text that also occurs in train (leakage rule)
    ds, report = datasets_big.build_from_tables(spec, tr, te, "sha123")
    assert report["split"]["repo_sha"] == "sha123" and report["split"]["author_grouped"] is False
    assert report["split"]["n_rows_before_cap"] == {"train": 40, "test": 40}
    assert len(ds.train_texts) <= 30 and len(ds.test_texts) <= 20 and set(ds.train_labels) <= {0, 1}
    assert not {t.lower() for t in ds.test_texts} & {t.lower() for t in ds.train_texts}
    assert "test_in_train" in report["cleaning"]["removed"]


def test_primary_metric_is_macro_f1_only_for_civil_comments():
    assert run_boost.PRIMARY_METRIC == {"civil_comments": "macro_f1"}


def test_notebook_generator_covers_every_dataset_and_uses_the_dataset_id():
    for name in datasets_big.SPECS:
        cs = make_boost_notebook.cells(name)
        assert any(f'DATASET = "{name}"' in c.source for c in cs) and any("run_boost.main" in c.source for c in cs)


def test_amazon_polarity_joins_title_and_content_and_keeps_labels():
    texts, labels = datasets_big.extract("amazon_polarity", Table({"title": ["Great", ""], "content": ["works well", "broke"], "label": [1, 0]}))
    assert texts == ["Great works well", "broke"] and labels == [1, 0]
    spec = datasets_big.SPECS["amazon_polarity"]
    assert spec.label_names == ("negative", "positive") and spec.max_train == 500_000 and spec.max_test == 100_000
