"""Redraw the repo's result figures from the committed JSON at 300 dpi with larger text, for the reports.

Writes to report/figures/hq_*.png only; results/figures is not touched. Run from the repository root:
    python report/make_hq_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt

from experiments import figures

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
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    return True


# Existing tokenizers in grey and orange shades, the proposed LexFront in a clearly different deep red.
figures.COLORS.update({"whitespace": "#8A8F98", "nltk": "#F28E2B", "tweet": "#F4C95D", "lexfront": "#C62828"})
figures._save = save_hq
out = Path(__file__).resolve().parent / "figures"
for name, fn in figures.FIGURES.items():
    fn("results", out / f"hq_{name}")
    print("wrote", out / f"hq_{name}")
