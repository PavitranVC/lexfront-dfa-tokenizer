"""Redraw the repo's result figures from the committed JSON at 600 dpi (plus SVG and PDF vector copies) with larger text, for the reports.

Writes to report/figures/hq_*.png only; results/figures is not touched. Run from the repository root:
    python report/make_hq_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt

from experiments import figures
from report.panels import split_panels

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 11.5,
                     "ytick.labelsize": 11.5, "legend.fontsize": 11})


_subplots = plt.subplots


def smaller_subplots(*args, **kwargs):
    """Draw on a smaller canvas so the text is relatively larger once the figure is scaled to the page width."""
    if "figsize" in kwargs:
        kwargs["figsize"] = tuple(0.85 * v for v in kwargs["figsize"])
    return _subplots(*args, **kwargs)


plt.subplots = smaller_subplots


def save_hq(fig, out_path) -> bool:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for text in ax.texts:                      # value labels on the bars
            text.set_fontsize(max(text.get_fontsize(), 9))
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, facecolor="white")                         # lossless PNG for the Word reports
    vec = Path(out_path).parent / "vector"
    vec.mkdir(exist_ok=True)
    for ext in ("svg", "pdf"):                                                # vector copies
        fig.savefig(vec / (Path(out_path).stem + "." + ext), facecolor="white")
    if Path(out_path).stem == "hq_module1_accuracy":            # copy without the module numbering, for the manuscript
        for ax in fig.axes:
            ax.set_title(ax.get_title(loc="left").replace("Module 1: accuracy", "Accuracy"), loc="left")
        fig.savefig(Path(out_path).with_name("tokenizers_accuracy.png"), dpi=600, facecolor="white")
        for ext in ("svg", "pdf"):
            fig.savefig(vec / ("tokenizers_accuracy." + ext), facecolor="white")
    stem = "tokenizers_accuracy" if Path(out_path).stem == "hq_module1_accuracy" else Path(out_path).stem
    if len(fig.axes) > 1:
        split_panels(fig, stem, Path(out_path).parent, 0.72 if stem == "hq_intrinsic_metrics" else 0.75)
        if stem == "hq_existing_vs_proposed":        # flatter copy for the Word manuscript, where the chart sits between two tables
            for _ax in fig.axes: _ax.set_title("", loc="left"); _ax.set_title("")                      # the caption names the panels
            h, l = fig.axes[0].get_legend_handles_labels()
            fig.axes[0].legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, fontsize=9.5, columnspacing=1.0, handlelength=1.2)
            split_panels(fig, stem + "_compact", Path(out_path).parent, 0.66)
    plt.close(fig)
    return True


# Existing tokenizers in grey and orange shades, the proposed LexFront in a clearly different deep red.
figures.COLORS.update({"whitespace": "#8A8F98", "nltk": "#F28E2B", "tweet": "#F4C95D", "lexfront": "#C62828"})
figures._save = save_hq
out = Path(__file__).resolve().parent / "figures"
for name, fn in figures.FIGURES.items():
    fn("results", out / f"hq_{name}")
    print("wrote", out / f"hq_{name}")
