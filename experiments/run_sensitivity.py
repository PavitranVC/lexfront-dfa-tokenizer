"""Sensitivity grid (P0): TF-IDF with `lowercase=False` for all four tokenizers x {NB, LR} x the three datasets.

    python -m experiments.run_sensitivity [--out results/sensitivity_case] [--max-train N] [--max-test N]

The main grid lowercases inside TF-IDF for every tokenizer, so LexFront's own lowercase rule never sees an
uppercase letter there. Here case folding is off: LexFront lowercases its WORD-like tokens itself, the baselines stay
case-sensitive. That makes it a robustness check of the main result, not a like-for-like comparison.
Runs are written in the main layout (`module1/` for the baselines, `module2/` for LexFront), so the same loaders work.
"""
import argparse

from . import data
from .classifiers import CASE_SENSITIVE_OVERRIDES, vectorizer_params
from .pipeline import run_grid
from .tokenizers import MODULE1_TOKENIZERS

SENSITIVITY_DIR = "results/sensitivity_case"
CLASSIFIERS = ("nb", "lr")


def run_case_sensitivity(out_dir=SENSITIVITY_DIR, dataset_names=data.DATASET_NAMES, max_train=None, max_test=None,
                         seed: int = 0, force: bool = False, log=print) -> list[dict]:
    """Every tokenizer x {NB, LR} with TF-IDF case folding off (VECTORIZER_PARAMS + CASE_SENSITIVE_OVERRIDES)."""
    params = vectorizer_params(CASE_SENSITIVE_OVERRIDES)
    records = run_grid(dataset_names, MODULE1_TOKENIZERS, CLASSIFIERS, f"{out_dir}/module1", max_train, max_test,
                       seed, force, log, params)
    return records + run_grid(dataset_names, ["lexfront"], CLASSIFIERS, f"{out_dir}/module2", max_train, max_test,
                              seed, force, log, params)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=SENSITIVITY_DIR)
    p.add_argument("--datasets", nargs="+", default=list(data.DATASET_NAMES))
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    run_case_sensitivity(a.out, a.datasets, a.max_train, a.max_test, a.seed, a.force)


if __name__ == "__main__":
    main()
