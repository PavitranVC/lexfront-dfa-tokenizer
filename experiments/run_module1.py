"""Module 1: existing tokenizers x {majority, NB, LR} on the three datasets.

    python -m experiments.run_module1 [--out results/module1] [--max-train N] [--max-test N]
"""
import argparse

from . import data
from .classifiers import CLASSIFIER_NAMES
from .pipeline import run_grid
from .tokenizers import MODULE1_TOKENIZERS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/module1")
    p.add_argument("--datasets", nargs="+", default=list(data.DATASET_NAMES))
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    run_grid(a.datasets, MODULE1_TOKENIZERS, CLASSIFIER_NAMES, a.out, a.max_train, a.max_test, a.seed, a.force)


if __name__ == "__main__":
    main()
