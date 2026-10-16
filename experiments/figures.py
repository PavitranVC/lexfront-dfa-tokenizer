"""Matplotlib figures built from the same JSON files as the tables.

Colors: first four slots of a validated categorical palette, assigned to tokenizers in a
fixed order (whitespace, nltk, tweet, lexfront). Every bar carries its value, so identity
and magnitude never depend on color alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report import (DATASET_ORDER, TOKENIZER_ORDER, load_dfa, load_intrinsic, load_runs)

COLORS = {"whitespace": "#2a78d6", "nltk": "#eb6834", "tweet": "#1baf7a", "lexfront": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


def _grouped_bars(ax, groups, series, value, fmt="{:.1f}"):
    """groups: x categories; series: names; value(group, series) -> float or None."""
    width = 0.8 / max(len(series), 1)
    for j, s in enumerate(series):
        xs, ys = [], []
        for i, g in enumerate(groups):
            v = value(g, s)
            if v is not None:
                xs.append(i - 0.4 + width * (j + 0.5))
                ys.append(v)
        bars = ax.bar(xs, ys, width * 0.9, color=COLORS.get(s, "#888"), label=s)
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)), groups)


def _present(order, values):
    return [x for x in order if x in set(values)]


def figure_module1(results_dir, out_path) -> bool:
    """Accuracy of each existing tokenizer, per dataset, with NB and LR side by side."""
    runs = load_runs(results_dir)
    runs = runs[runs.module == "module1"]
    if runs.empty:
        return False
    datasets, toks = _present(DATASET_ORDER, runs.dataset), _present(TOKENIZER_ORDER, runs.tokenizer)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (clf, label) in zip(axes, [("nb", "Naive Bayes"), ("lr", "Logistic Regression")]):
        def val(d, t, clf=clf):
            hit = runs[(runs.dataset == d) & (runs.tokenizer == t) & (runs.classifier == clf)]
            return 100 * hit.accuracy.iloc[0] if len(hit) else None
        _grouped_bars(ax, datasets, toks, val)
        _style(ax, f"Module 1: accuracy with {label}", "accuracy (%)" if clf == "nb" else "")
    axes[0].legend(frameon=False, fontsize=8, title="tokenizer", title_fontsize=8)
    return _save(fig, out_path)


def figure_existing_vs_proposed(results_dir, out_path) -> bool:
    """Accuracy and macro-F1: existing (NLTK + NB) against proposed (LexFront + LR)."""
    runs = load_runs(results_dir)
    pairs = {"existing: NLTK + NB": ("nltk", "nb"), "proposed: LexFront + LR": ("lexfront", "lr")}
    datasets = [d for d in DATASET_ORDER if d in set(runs.dataset)]
    if not datasets or runs[runs.tokenizer == "lexfront"].empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {"existing: NLTK + NB": COLORS["nltk"], "proposed: LexFront + LR": COLORS["lexfront"]}
    for ax, (metric, label) in zip(axes, [("accuracy", "accuracy (%)"), ("macro_f1", "macro-F1 (%)")]):
        def val(d, s, metric=metric):
            tok, clf = pairs[s]
            hit = runs[(runs.dataset == d) & (runs.tokenizer == tok) & (runs.classifier == clf)]
            return 100 * hit[metric].iloc[0] if len(hit) else None
        width = 0.35
        for j, s in enumerate(pairs):
            present = [(i, val(d, s)) for i, d in enumerate(datasets) if val(d, s) is not None]   # skip missing cells
            xs = [i - width / 2 + j * width for i, _ in present]
            ys = [v for _, v in present]
            bars = ax.bar(xs, ys, width * 0.92, color=colors[s], label=s)
            for b, v in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(datasets)), datasets)
        _style(ax, f"Existing vs proposed: {label.split(' ')[0]}", label)
    axes[0].legend(frameon=False, fontsize=8)
    return _save(fig, out_path)


def figure_intrinsic(results_dir, out_path) -> bool:
    """OOV rate, fertility and vocabulary size for the four tokenizers."""
    df = load_intrinsic(results_dir)
    if df.empty:
        return False
    datasets, toks = _present(DATASET_ORDER, df.dataset), _present(TOKENIZER_ORDER, df.tokenizer)
    panels = [("oov_rate_tokens", "OOV rate (% of test tokens)", 100, "{:.1f}"),
              ("fertility", "fertility (tokens per whitespace word)", 1, "{:.2f}"),
              ("vocab_size", "vocabulary size (train, thousands)", 1e-3, "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title, scale, fmt) in zip(axes, panels):
        def val(d, t, col=col, scale=scale):
            hit = df[(df.dataset == d) & (df.tokenizer == t)]
            return float(hit[col].iloc[0]) * scale if len(hit) else None
        _grouped_bars(ax, datasets, toks, val, fmt)
        _style(ax, title, "")
    axes[0].legend(frameon=False, fontsize=8, title="tokenizer", title_fontsize=8)
    return _save(fig, out_path)


def figure_dfa_states(results_dir, out_path) -> bool:
    """NFA, DFA and minimized-DFA state counts per class and combined."""
    d = load_dfa(results_dir)
    names = list(d["per_class"]) + ["combined"]
    stats = [d["per_class"][n] for n in d["per_class"]] + [d["combined"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.27
    for j, (key, label, color) in enumerate([("nfa_states", "NFA", "#2a78d6"), ("dfa_states", "DFA", "#eb6834"),
                                            ("min_dfa_states", "minimized DFA", "#1baf7a")]):
        xs = [i - width + j * width for i in range(len(names))]
        ys = [s[key] for s in stats]
        bars = ax.bar(xs, ys, width * 0.92, color=color, label=label)
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)), names)
    _style(ax, "State counts through the pipeline", "states")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_path)


def _save(fig, out_path) -> bool:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    return True


FIGURES = {"module1_accuracy.png": figure_module1, "existing_vs_proposed.png": figure_existing_vs_proposed,
           "intrinsic_metrics.png": figure_intrinsic, "dfa_states.png": figure_dfa_states}


def build_all_figures(results_dir, out_dir=None) -> list[str]:
    """Draw every figure whose input JSON exists; return the file names written."""
    out_dir = Path(out_dir or Path(results_dir) / "figures")
    written = []
    for name, fn in FIGURES.items():
        try:
            if fn(results_dir, out_dir / name):
                written.append(name)
        except FileNotFoundError:
            pass
    return written
