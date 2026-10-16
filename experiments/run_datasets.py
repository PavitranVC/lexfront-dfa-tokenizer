"""Save dataset summaries: python -m experiments.run_datasets [--out results] [--max-train N] [--max-test N]"""
import argparse

from .data import write_summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    a = p.parse_args(argv)
    print(write_summary(a.out, a.max_train, a.max_test))


if __name__ == "__main__":
    main()
