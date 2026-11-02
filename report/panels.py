"""Save every panel of a multi-panel matplotlib figure as its own vector PDF and 600 dpi PNG, for single-column layouts."""
from pathlib import Path


def split_panels(fig, stem, outdir, height_factor=0.62):
    outdir = Path(outdir)
    vec = outdir / "vector"
    vec.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h * height_factor)
    for ax in fig.axes:
        ax.tick_params(labelleft=True)                       # panels share a y axis; each stacked panel needs its own labels
        if not ax.get_ylabel() and fig.axes[0].get_ylabel(): ax.set_ylabel(fig.axes[0].get_ylabel())
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for k, ax in enumerate(fig.axes, 1):
        bb = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.04)
        fig.savefig(outdir / f"{stem}_p{k}.png", dpi=600, bbox_inches=bb, facecolor="white")
        fig.savefig(vec / f"{stem}_p{k}.pdf", bbox_inches=bb, facecolor="white")
    return len(fig.axes)
