"""WordPiece and Unigram (P4): {wordpiece, unigram} x {NB, LR} on the three datasets, plus paired stats.

    python -m experiments.run_subword2 [--out results/subword2] [--max-train N] [--max-test N] [--force]
    python -m experiments.run_subword2 --stats-only [--out results/subword2]

NLP concept: like BPE, each tokenizer learns a fixed vocabulary (8000 pieces) from the train split of its dataset
(lowercased as the vectorizer does) and never sees the test split. They differ in how the vocabulary is learned:
WordPiece merges the pair that most raises the likelihood of the text, Unigram prunes a large candidate set by EM
(protocol: docs/versions/v6-wordpiece-unigram.md).

Runs go to <out>/<dataset>__<tokenizer>__<clf>.json with predictions/ sidecars, main-grid TF-IDF settings. The stats
(same bootstrap code, B and seed as `run_stats`) go to <out>/stats/bootstrap.json; the NLTK+LR and BPE+LR runs they
compare against are read from the grids next to <out> (<grid>/module1, <grid>/bpe).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import data, stats
from .pipeline import result_path, run_grid
from .report import DATASET_ORDER
from .run_bpe import grid_dir
from .run_stats import N_BOOT, SEED, dataset_stats
from .tokenizers import SUBWORD2

SUBWORD2_DIR = "results/subword2"
CLASSIFIERS = ("nb", "lr")
LABELS = {"wordpiece_vs_nltk_lr": "WordPiece+LR minus NLTK+LR (tokenizer only)",
          "unigram_vs_nltk_lr": "Unigram+LR minus NLTK+LR (tokenizer only)",
          "wordpiece_vs_bpe_lr": "WordPiece+LR minus BPE+LR",
          "unigram_vs_bpe_lr": "Unigram+LR minus BPE+LR",
          "unigram_vs_wordpiece_lr": "Unigram+LR minus WordPiece+LR"}
# (name, a, b, McNemar?): every difference is b minus a; McNemar on the two comparisons with the main grid's NLTK+LR.
PAIR_SPEC = [("wordpiece_vs_nltk_lr", "nltk__lr", "wordpiece__lr", True),
             ("unigram_vs_nltk_lr", "nltk__lr", "unigram__lr", True),
             ("wordpiece_vs_bpe_lr", "bpe__lr", "wordpiece__lr", False),
             ("unigram_vs_bpe_lr", "bpe__lr", "unigram__lr", False),
             ("unigram_vs_wordpiece_lr", "wordpiece__lr", "unigram__lr", False)]


def run_subword2(out_dir=SUBWORD2_DIR, dataset_names=data.DATASET_NAMES, max_train=None, max_test=None, seed: int = 0,
                 force: bool = False, log=print) -> list[dict]:
    """The grid (main-grid TF-IDF settings); each tokenizer is trained on the train split of its dataset."""
    return run_grid(dataset_names, SUBWORD2, CLASSIFIERS, out_dir, max_train, max_test, seed, force, log)


def dataset_paths(out_dir, dataset: str) -> dict[str, Path]:
    """{"<tokenizer>__<clf>": run JSON} for the four new runs and the NLTK+LR and BPE+LR references; {} if incomplete."""
    paths = {f"{t}__{c}": result_path(out_dir, dataset, t, c) for t in SUBWORD2 for c in CLASSIFIERS}
    grid = grid_dir(out_dir)
    paths["nltk__lr"] = result_path(grid / "module1", dataset, "nltk", "lr")
    paths["bpe__lr"] = result_path(grid / "bpe", dataset, "bpe", "lr")
    return paths if all(p.exists() for p in paths.values()) else {}


def _subword_summary(paths: dict[str, Path]) -> dict:
    """{tokenizer: subword record} of the runs that carry one, as stored in their run JSONs."""
    records = (json.loads(path.read_text()) for path in paths.values())
    return {r["tokenizer"]: r["subword"] for r in records if "subword" in r}


def compute(out_dir=SUBWORD2_DIR, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """The whole stats record; datasets without every run are left out."""
    config = {"results_dir": str(out_dir), "grid_dir": grid_dir(out_dir).as_posix(),
              "method": "percentile bootstrap over test examples", "ci_level": stats.CI_LEVEL, "n_boot": n_boot,
              "seed": seed, "rng": "numpy.random.default_rng(seed), one fresh generator per dataset; all runs of a "
                                   "dataset share its resampled indices (paired)",
              "difference": "b minus a, in fractions (x100 for points)",
              "mcnemar": "exact two-sided binomial test on discordant pairs, accuracy, pairs vs nltk_lr only",
              "multiplicity": "none: every interval is a separate 95% interval",
              "not_covered": "training randomness: one training run per cell (seed 0); training is deterministic",
              "numpy": np.__version__, "pairs": [{"name": n, "a": a, "b": b} for n, a, b, _ in PAIR_SPEC]}
    datasets = {}
    for ds in DATASET_ORDER:
        paths = dataset_paths(out_dir, ds)
        if paths:
            datasets[ds] = {**dataset_stats(paths, n_boot, seed, PAIR_SPEC, LABELS), "subword": _subword_summary(paths)}
    if not datasets:
        raise FileNotFoundError(f"no complete WordPiece/Unigram grid under {out_dir} (with the main and BPE grids in "
                                f"{grid_dir(out_dir)})")
    return {"config": config, "datasets": datasets}


def write_stats(out_dir=SUBWORD2_DIR, n_boot: int = N_BOOT, seed: int = SEED) -> Path:
    out = Path(out_dir) / "stats" / "bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compute(out_dir, n_boot, seed), indent=2) + "\n")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=SUBWORD2_DIR)
    p.add_argument("--datasets", nargs="+", default=list(data.DATASET_NAMES))
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stats-only", action="store_true", help="only recompute <out>/stats from the stored predictions")
    a = p.parse_args(argv)
    if not a.stats_only:
        run_subword2(a.out, a.datasets, a.max_train, a.max_test, a.seed, a.force)
    print("wrote", write_stats(a.out))


if __name__ == "__main__":
    main()
