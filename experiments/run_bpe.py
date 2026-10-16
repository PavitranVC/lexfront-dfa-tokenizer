"""BPE and the LexFront + BPE hybrid (P3): {bpe, lexfront_bpe} x {NB, LR} on the three datasets, plus paired stats.

    python -m experiments.run_bpe [--out results/bpe] [--vocab-size 8000] [--max-train N] [--max-test N] [--force]
    python -m experiments.run_bpe --stats-only [--out results/bpe] [--vocab-size 8000]

NLP concept: a *subword* tokenizer learns its vocabulary from data, so it is trained on each dataset's train split
(lowercased as the vectorizer does) and never sees the test split. The hybrid keeps LexFront's scanner and
normalizer and splits only word-like tokens into BPE pieces (protocol: docs/versions/v5-bpe-hybrid.md).

Runs go to <out>/<dataset>__<tokenizer>__<clf>.json with predictions/ sidecars, main-grid TF-IDF settings. The stats
(same bootstrap code, B and seed as `run_stats`) go to <out>/stats/bootstrap.json; the NLTK+LR and LexFront+LR runs
they compare against are read from the main grid next to <out> (<grid>/module1, <grid>/module2).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import data, stats
from .pipeline import result_path, run_grid
from .report import DATASET_ORDER
from .run_stats import N_BOOT, SEED, dataset_stats
from .tokenizers import BPE_VOCAB_SIZE, SUBWORD_TOKENIZERS

BPE_DIR = "results/bpe"
CLASSIFIERS = ("nb", "lr")
LABELS = {"bpe_vs_nltk_lr": "BPE+LR minus NLTK+LR (tokenizer only)",
          "hybrid_vs_lexfront_lr": "LexFront+BPE+LR minus LexFront+LR",
          "hybrid_vs_bpe_lr": "LexFront+BPE+LR minus BPE+LR",
          "hybrid_vs_nltk_lr": "LexFront+BPE+LR minus NLTK+LR"}


def tokenizer_names(vocab_size: int = BPE_VOCAB_SIZE) -> tuple[str, str]:
    """('bpe', 'lexfront_bpe') at the primary size, ('bpe:v<N>', 'lexfront_bpe:v<N>') otherwise."""
    suffix = "" if vocab_size == BPE_VOCAB_SIZE else f":v{vocab_size}"
    names = (f"bpe{suffix}", f"lexfront_bpe{suffix}")
    if not all(n in SUBWORD_TOKENIZERS for n in names):
        raise ValueError(f"no subword tokenizers registered for vocabulary size {vocab_size}")
    return names


def pair_spec(vocab_size: int = BPE_VOCAB_SIZE) -> list[tuple[str, str, str, bool]]:
    """(name, a, b, McNemar?) with every difference b minus a; McNemar on the first (tokenizer-only) pair."""
    bpe, hybrid = (f"{n}__lr" for n in tokenizer_names(vocab_size))
    return [("bpe_vs_nltk_lr", "nltk__lr", bpe, True), ("hybrid_vs_lexfront_lr", "lexfront__lr", hybrid, False),
            ("hybrid_vs_bpe_lr", bpe, hybrid, False), ("hybrid_vs_nltk_lr", "nltk__lr", hybrid, False)]


def run_bpe(out_dir=BPE_DIR, vocab_size: int = BPE_VOCAB_SIZE, dataset_names=data.DATASET_NAMES, max_train=None,
            max_test=None, seed: int = 0, force: bool = False, log=print) -> list[dict]:
    """The subword grid (main-grid TF-IDF settings); each BPE is trained on the train split of its dataset."""
    return run_grid(dataset_names, tokenizer_names(vocab_size), CLASSIFIERS, out_dir, max_train, max_test, seed,
                    force, log)


def grid_dir(out_dir) -> Path:
    """The main grid the subword runs are compared with: the parent of <out> (results/bpe -> results)."""
    return Path(out_dir).parent


def dataset_paths(out_dir, dataset: str, vocab_size: int) -> dict[str, Path]:
    """{"<tokenizer>__<clf>": run JSON} for the subword runs and the two main-grid LR references; {} if incomplete."""
    paths = {f"{t}__{c}": result_path(out_dir, dataset, t, c) for t in tokenizer_names(vocab_size) for c in CLASSIFIERS}
    grid = grid_dir(out_dir)
    paths["nltk__lr"] = result_path(grid / "module1", dataset, "nltk", "lr")
    paths["lexfront__lr"] = result_path(grid / "module2", dataset, "lexfront", "lr")
    return paths if all(p.exists() for p in paths.values()) else {}


def _subword_summary(paths: dict[str, Path]) -> dict:
    """{tokenizer: subword record} of the subword runs, as stored in their run JSONs."""
    out = {}
    for run, path in paths.items():
        record = json.loads(path.read_text())
        if "subword" in record:
            out[record["tokenizer"]] = record["subword"]
    return out


def compute(out_dir=BPE_DIR, vocab_size: int = BPE_VOCAB_SIZE, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """The whole stats record; datasets without every run are left out."""
    spec = pair_spec(vocab_size)
    config = {"results_dir": str(out_dir), "grid_dir": grid_dir(out_dir).as_posix(), "vocab_size": vocab_size,
              "method": "percentile bootstrap over test examples", "ci_level": stats.CI_LEVEL, "n_boot": n_boot,
              "seed": seed, "rng": "numpy.random.default_rng(seed), one fresh generator per dataset; all runs of a "
                                   "dataset share its resampled indices (paired)",
              "difference": "b minus a, in fractions (x100 for points)",
              "mcnemar": "exact two-sided binomial test on discordant pairs, accuracy, pair bpe_vs_nltk_lr only",
              "multiplicity": "none: every interval is a separate 95% interval",
              "not_covered": "training randomness: one training run per cell (seed 0); BPE training is deterministic",
              "numpy": np.__version__, "pairs": [{"name": n, "a": a, "b": b} for n, a, b, _ in spec]}
    datasets = {}
    for ds in DATASET_ORDER:
        paths = dataset_paths(out_dir, ds, vocab_size)
        if paths:
            datasets[ds] = {**dataset_stats(paths, n_boot, seed, spec, LABELS), "subword": _subword_summary(paths)}
    if not datasets:
        raise FileNotFoundError(f"no complete BPE grid under {out_dir} (with the main grid in {grid_dir(out_dir)})")
    return {"config": config, "datasets": datasets}


def write_stats(out_dir=BPE_DIR, vocab_size: int = BPE_VOCAB_SIZE, n_boot: int = N_BOOT, seed: int = SEED) -> Path:
    out = Path(out_dir) / "stats" / "bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compute(out_dir, vocab_size, n_boot, seed), indent=2) + "\n")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=BPE_DIR)
    p.add_argument("--vocab-size", type=int, default=BPE_VOCAB_SIZE)
    p.add_argument("--datasets", nargs="+", default=list(data.DATASET_NAMES))
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stats-only", action="store_true", help="only recompute <out>/stats from the stored predictions")
    a = p.parse_args(argv)
    if not a.stats_only:
        run_bpe(a.out, a.vocab_size, a.datasets, a.max_train, a.max_test, a.seed, a.force)
    print("wrote", write_stats(a.out, a.vocab_size))


if __name__ == "__main__":
    main()
