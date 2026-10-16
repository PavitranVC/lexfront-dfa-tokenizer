"""Module 2: LexFront.

    python -m experiments.run_module2 [--out results/module2] [--only dfa intrinsic classify]

* dfa        : state counts + .dot/.png of the minimized DFA  (dfa_states.json)
* intrinsic  : OOV, fertility, vocab size, throughput for 4 tokenizers (intrinsic__<dataset>.json)
* classify   : LexFront + {NB, LR} with exactly the Module 1 settings
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import data
from .dfa_report import build_report
from .metrics import intrinsic_metrics
from .pipeline import environment, run_grid
from .tokenizers import TOKENIZERS


def run_intrinsic(dataset_names, out_dir, max_train=None, max_test=None, seed=0, log=print):
    """Intrinsic metrics for every tokenizer on every dataset (texts lowercased like the vectorizer)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for dname in dataset_names:
        ds = data.load(dname, max_train, max_test, seed)
        train = [t.lower() for t in ds.train_texts]
        test = [t.lower() for t in ds.test_texts]
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "dataset": ds.summary(), "preprocessing": "lowercase", "environment": environment(),
                  "tokenizers": {}}
        for tname, tok in TOKENIZERS.items():
            record["tokenizers"][tname] = intrinsic_metrics(tok, train, test)
            m = record["tokenizers"][tname]
            log(f"{dname:12s} {tname:10s} oov={m['oov_rate_tokens']:.4f} fert={m['fertility']:.3f} "
                f"vocab={m['vocab_size']} tok/s={m['throughput_tokens_per_s']:.0f}")
        (out / f"intrinsic__{dname}.json").write_text(json.dumps(record, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/module2")
    p.add_argument("--datasets", nargs="+", default=list(data.DATASET_NAMES))
    p.add_argument("--only", nargs="+", choices=["dfa", "intrinsic", "classify"],
                   default=["dfa", "intrinsic", "classify"])
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--intrinsic-max-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.intrinsic_max_train is not None and a.max_test is not None:
        # data.subsample draws train then test from one RNG, so a different train size changes the test subset.
        p.error("--intrinsic-max-train cannot be combined with --max-test (the test subset would differ "
                "from the classification runs)")
    if "dfa" in a.only:
        build_report(a.out)
    if "intrinsic" in a.only:
        run_intrinsic(a.datasets, a.out, a.intrinsic_max_train or a.max_train, a.max_test, a.seed)
    if "classify" in a.only:
        run_grid(a.datasets, ["lexfront"], ["nb", "lr"], a.out, a.max_train, a.max_test, a.seed, a.force)


if __name__ == "__main__":
    main()
