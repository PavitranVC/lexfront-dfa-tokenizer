"""Test-set uncertainty of the stored predictions: percentile bootstrap CIs and an exact McNemar test.

Concepts
* **Percentile bootstrap.** Resample the n test examples with replacement B times, recompute the metric on each
  resample, and take the 2.5th and 97.5th percentiles as a 95% interval. It measures how much a score would move
  if a different test set of the same size were drawn; it says nothing about training randomness.
* **Paired bootstrap.** Two systems scored on the *same* resampled indices, so the CI of their difference keeps the
  correlation between them (both see the same easy and hard examples).
* **Exact McNemar test.** Only the discordant examples matter: b = A right / B wrong, c = A wrong / B right. Under
  "equal accuracy" each discordant example is a fair coin, so p = two-sided binomial tail of min(b, c) in b + c.

Macro-F1 is computed from a confusion matrix built with `np.bincount` for all resamples at once. Classes absent from
both y_true and y_pred of a resample are left out of the mean, which is what scikit-learn's `f1_score(average="macro",
zero_division=0)` does, so the point estimate equals the run JSON's `macro_f1`.
"""
import numpy as np
from scipy.stats import binom

CI_LEVEL = 0.95
CHUNK = 250   # resamples per bincount call (bounds memory: CHUNK x n index matrix)


def bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) matrix of test-example indices drawn with replacement."""
    return rng.integers(0, n, size=(n_boot, n))


def confusion_matrices(y_true, y_pred, idx: np.ndarray, n_classes: int) -> np.ndarray:
    """(len(idx), K, K) confusion matrices (rows = gold, columns = predicted), one per resample row of `idx`."""
    out = np.empty((len(idx), n_classes, n_classes), dtype=np.int64)
    cell = n_classes * n_classes
    for start in range(0, len(idx), CHUNK):
        rows = idx[start:start + CHUNK]
        codes = y_true[rows] * n_classes + y_pred[rows] + cell * np.arange(len(rows))[:, None]
        out[start:start + len(rows)] = np.bincount(codes.ravel(), minlength=cell * len(rows)).reshape(-1, n_classes,
                                                                                                         n_classes)
    return out


def accuracy_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Accuracy = trace / total, for one (K, K) or a stack of (B, K, K) matrices."""
    return np.trace(cm, axis1=-2, axis2=-1) / cm.sum(axis=(-2, -1))


def macro_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 = mean over present classes of 2TP / (2TP + FP + FN); a class is present if it is gold or predicted."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    denom = cm.sum(axis=-1) + cm.sum(axis=-2)          # (TP + FN) + (TP + FP) = 2TP + FP + FN
    present = denom > 0
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=present)
    return f1.sum(axis=-1) / present.sum(axis=-1)


METRICS = {"accuracy": accuracy_from_confusion, "macro_f1": macro_f1_from_confusion}


def n_classes_of(*label_arrays) -> int:
    return int(max(int(np.max(a)) for a in label_arrays)) + 1


def metric_samples(y_true, y_pred, idx: np.ndarray, n_classes: int) -> dict[str, np.ndarray]:
    """{metric: (B,) bootstrap values} for one system on the resamples `idx`."""
    cms = confusion_matrices(y_true, y_pred, idx, n_classes)
    return {name: fn(cms) for name, fn in METRICS.items()}


def point_estimates(y_true, y_pred, n_classes: int) -> dict[str, float]:
    """{metric: value} on the full test set (no resampling)."""
    idx = np.arange(len(y_true))[None, :]
    return {name: float(v[0]) for name, v in metric_samples(y_true, y_pred, idx, n_classes).items()}


def percentile_ci(samples: np.ndarray, level: float = CI_LEVEL) -> list[float]:
    """[lower, upper] percentile interval of the bootstrap samples."""
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(samples, [tail, 100 - tail])
    return [float(lo), float(hi)]


def excludes_zero(ci) -> bool:
    return ci[0] > 0 or ci[1] < 0


def run_summary(y_true, y_pred, idx: np.ndarray, n_classes: int) -> dict:
    """Point estimate and 95% bootstrap CI of accuracy and macro-F1 for one run."""
    points = point_estimates(y_true, y_pred, n_classes)
    samples = metric_samples(y_true, y_pred, idx, n_classes)
    return {name: {"value": points[name], "ci": percentile_ci(samples[name])} for name in METRICS}


def paired_difference(y_true, pred_a, pred_b, idx: np.ndarray, n_classes: int) -> dict:
    """B minus A on the same resamples: point difference, 95% CI, and whether the CI excludes 0, per metric."""
    pa, pb = point_estimates(y_true, pred_a, n_classes), point_estimates(y_true, pred_b, n_classes)
    sa, sb = metric_samples(y_true, pred_a, idx, n_classes), metric_samples(y_true, pred_b, idx, n_classes)
    out = {}
    for name in METRICS:
        ci = percentile_ci(sb[name] - sa[name])
        out[name] = {"diff": pb[name] - pa[name], "ci": ci, "ci_excludes_zero": excludes_zero(ci)}
    return out


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact two-sided McNemar test on accuracy (binomial test on the discordant pairs)."""
    right_a, right_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(right_a & ~right_b))
    c = int(np.sum(~right_a & right_b))
    p = 1.0 if b + c == 0 else min(1.0, 2 * float(binom.cdf(min(b, c), b + c, 0.5)))
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}
