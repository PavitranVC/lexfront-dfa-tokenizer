"""Bootstrap CIs and paired tests for the main grid, from the stored per-example predictions.

    python -m experiments.run_stats [--results results] [--boot 2000] [--seed 20261016]

Reads every run JSON under <results>/module1 and <results>/module2 (plus its predictions sidecar) and writes
<results>/stats/bootstrap.json. Per dataset all runs share one matrix of resampled test indices, drawn from
`numpy.random.default_rng(seed)`, so per-run CIs and paired differences use the same resamples. No refitting:
only the test set is resampled, so training randomness is not covered.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import stats
from .pipeline import load_predictions
from .report import DATASET_ORDER

N_BOOT = 2000
SEED = 20261016
# (name, baseline A, LexFront system B, McNemar?) -- every difference is B minus A.
PAIRS = [("tokenizer_only_lr", "nltk__lr", "lexfront__lr", True),
         ("tweet_vs_lexfront_lr", "tweet__lr", "lexfront__lr", False),
         ("existing_vs_proposed", "nltk__nb", "lexfront__lr", False)]
PAIR_LABELS = {"tokenizer_only_lr": "LexFront+LR minus NLTK+LR (tokenizer only)",
               "tweet_vs_lexfront_lr": "LexFront+LR minus TweetTokenizer+LR (tokenizer only)",
               "existing_vs_proposed": "LexFront+LR minus NLTK+NB (proposed minus existing)"}


def find_runs(results_dir) -> dict[str, dict[str, Path]]:
    """{dataset: {"<tokenizer>__<classifier>": run JSON path}} for the main grid."""
    runs: dict[str, dict[str, Path]] = {}
    for sub in ("module1", "module2"):
        for path in sorted((Path(results_dir) / sub).glob("*__*__*.json")):
            dataset, tokenizer, classifier = path.stem.split("__")
            runs.setdefault(dataset, {})[f"{tokenizer}__{classifier}"] = path
    if not runs:
        raise FileNotFoundError(f"no run JSONs under {results_dir}/module1 or module2")
    return runs


def load_dataset_predictions(paths: dict[str, Path]) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """(y_true, {run: y_pred}, number of labels); fails if the runs disagree on the gold labels."""
    y_true, preds, n_labels = None, {}, None
    for run, path in paths.items():
        gold, preds[run] = load_predictions(path)
        if y_true is not None and not np.array_equal(gold, y_true):
            raise ValueError(f"{path}: gold labels differ from the other runs of this dataset")
        y_true = gold
        n_labels = json.loads(path.read_text())["dataset"]["n_labels"]
    return y_true, preds, n_labels


def check_point_estimates(summary: dict, path: Path) -> None:
    """The bootstrap's full-sample metrics must equal the ones the pipeline stored (same predictions)."""
    stored = json.loads(path.read_text())["metrics"]
    for name in stats.METRICS:
        if abs(summary[name]["value"] - stored[name]) > 1e-9:
            raise ValueError(f"{path}: recomputed {name} {summary[name]['value']} != stored {stored[name]}")


def dataset_stats(paths: dict[str, Path], n_boot: int, seed: int) -> dict:
    """Per-run CIs, paired differences and McNemar for one dataset."""
    y_true, preds, k = load_dataset_predictions(paths)
    idx = stats.bootstrap_indices(len(y_true), n_boot, np.random.default_rng(seed))
    runs = {}
    for run, path in paths.items():
        runs[run] = {"run_json": path.as_posix(), **stats.run_summary(y_true, preds[run], idx, k)}
        check_point_estimates(runs[run], path)
    pairs = {}
    for name, a, b, with_mcnemar in PAIRS:
        if a in preds and b in preds:
            pairs[name] = {"a": a, "b": b, "label": PAIR_LABELS[name],
                           **stats.paired_difference(y_true, preds[a], preds[b], idx, k)}
            if with_mcnemar:
                pairs[name]["mcnemar_accuracy"] = stats.mcnemar_exact(y_true, preds[a], preds[b])
    return {"n_test": int(len(y_true)), "n_labels": k, "runs": runs, "pairs": pairs}


def compute(results_dir, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """The whole bootstrap.json record for one results directory."""
    found = find_runs(results_dir)
    config = {"results_dir": str(results_dir), "method": "percentile bootstrap over test examples",
              "ci_level": stats.CI_LEVEL, "n_boot": n_boot, "seed": seed,
              "rng": "numpy.random.default_rng(seed), one fresh generator per dataset; all runs of a dataset "
                     "share its resampled indices (paired)",
              "macro_f1": "mean over classes present in gold or predictions of a resample (= sklearn zero_division=0)",
              "difference": "b minus a, in fractions (x100 for points)",
              "mcnemar": "exact two-sided binomial test on discordant pairs, accuracy, pair tokenizer_only_lr only",
              "not_covered": "training randomness: one training run per cell (seed 0)",
              "numpy": np.__version__, "pairs": [{"name": n, "a": a, "b": b} for n, a, b, _ in PAIRS]}
    order = [d for d in DATASET_ORDER if d in found] + sorted(set(found) - set(DATASET_ORDER))
    return {"config": config, "datasets": {d: dataset_stats(found[d], n_boot, seed) for d in order}}


def write(results_dir, n_boot: int = N_BOOT, seed: int = SEED) -> Path:
    out = Path(results_dir) / "stats" / "bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compute(results_dir, n_boot, seed), indent=2) + "\n")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    print("wrote", write(a.results, a.boot, a.seed))


if __name__ == "__main__":
    main()
