"""Draws the architecture figure (Figure 1) in a layered-box style: numbered module boxes with example token chips.

Run from anywhere:  python report/make_architecture.py   ->  report/figures/architecture.png (300 dpi)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

W, H = 160, 75
fig, ax = plt.subplots(figsize=(10.5, 4.9))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
SERIF = "DejaVu Serif"
INK = "#14213d"


def Y(y):                      # image-style y (down) -> axes y (up)
    return H - y


def box(x, y, w, h, fc, ec, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, Y(y + h)), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec=ec, lw=lw, zorder=2))


def chip(x, y, w, h, text, fc, ec, fs=6.8):
    ax.add_patch(FancyBboxPatch((x, Y(y + h)), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(x + w / 2, Y(y + h / 2), text, ha="center", va="center", fontsize=fs, color=INK, zorder=4, family=SERIF)


def badge(x, y, n, color):
    ax.add_patch(Circle((x, Y(y)), 2.5, fc=color, ec="white", lw=1.5, zorder=5))
    ax.text(x, Y(y) - 0.15, str(n), ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=6, family=SERIF)


def title(x, y, text, sub=None, fs=10.6):
    ax.text(x, Y(y), text, ha="left", va="center", fontsize=fs, fontweight="bold", color="#1b2a6b", family=SERIF, zorder=4)
    if sub:
        ax.text(x, Y(y + 5.2 + 1.6 * (sub.count("\n"))), sub, ha="left", va="center", fontsize=8.0, color=INK, family=SERIF, zorder=4,
                linespacing=1.15)


def arrow(points, color="#2b4c8c", lw=1.8, both=False):
    xs, ys = zip(*points)
    if len(points) > 2:
        ax.plot(xs[:-1], [Y(v) for v in ys[:-1]], color=color, lw=lw, zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((points[-2][0], Y(points[-2][1])), (points[-1][0], Y(points[-1][1])), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=color, zorder=1, shrinkA=0, shrinkB=0))
    if both:
        ax.add_patch(FancyArrowPatch((points[1][0], Y(points[1][1])), (points[0][0], Y(points[0][1])), arrowstyle="-|>",
                                     mutation_scale=14, lw=lw, color=color, zorder=1, shrinkA=0, shrinkB=0))


def chain(x, y, label, label_w, names, fc, ec, widths):
    chip(x, y, label_w, 5.2, label, "#f3f6fb", "#8da2c8")
    cx = x + label_w + 4.2
    ax.add_patch(FancyArrowPatch((x + label_w + 0.3, Y(y + 2.6)), (cx - 0.4, Y(y + 2.6)), arrowstyle="-|>", mutation_scale=9, lw=1.2, color=INK, zorder=4))
    for n, w in zip(names, widths):
        chip(cx, y, w, 5.2, n, fc, ec)
        cx += w + 1.4


# ---- Input data
box(1.5, 15.5, 24, 37.5, "#eef2f8", "#6c7a96")
ax.text(13.5, Y(20.2), "Input Data", ha="center", va="center", fontsize=11, fontweight="bold", color=INK, family=SERIF, zorder=4)
for i, label in enumerate(["Raw text", "Tweets", "Comments", "News articles"]):
    yy = 27 + i * 7.2
    if i == 0 or i == 3:       # document / newspaper icon
        ax.add_patch(Rectangle((4.2, Y(yy + 3)), 3.4, 5, fc="white", ec="#3b5ea8", lw=1.2, zorder=4))
        for k in range(3):
            ax.plot([4.9, 6.9], [Y(yy - 1.6 + k * 1.5), Y(yy - 1.6 + k * 1.5)], color="#3b5ea8", lw=0.8, zorder=5)
    else:                      # speech bubble icon
        ax.add_patch(FancyBboxPatch((3.8, Y(yy + 2.6)), 4.6, 4.2, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#3b5ea8", lw=1.2, zorder=4))
        ax.add_patch(Polygon([[4.8, Y(yy + 2.6)], [4.4, Y(yy + 4.4)], [6.0, Y(yy + 2.6)]], fc="white", ec="#3b5ea8", lw=1.0, zorder=4))
        for k in range(3):
            ax.add_patch(Circle((5.0 + k * 1.3, Y(yy + 0.4)), 0.3, fc="#3b5ea8", zorder=5))
    ax.text(9.8, Y(yy + 0.4), label, ha="left", va="center", fontsize=7.8, color=INK, family=SERIF, zorder=4)
ax.text(10.4, Y(yy + 7.2), "....", ha="left", va="center", fontsize=11, color=INK, zorder=4)

# ---- Module boxes
box(34.5, 2.5, 60, 17.5, "#e9f1fd", "#3b6fb6")
badge(36, 4.5, 1, "#2f5fb3")
title(41, 6.4, "Module 1: Existing Tokenizers", "Whitespace | NLTK | TweetTokenizer")
chain(37, 13.8, "Raw text", 11.5, ["Token 1", "Token 2", "Token 3", "..."], "#dbe9fb", "#7fa6dc", [9.2, 9.2, 9.2, 4.4])

box(34.5, 24, 60, 20.5, "#fff0dc", "#e8892b")
badge(36, 26, 2, "#e07b12")
title(41, 28, "Module 2: LexFront Lexer", "11 token classes, longest match\nown regex-NFA-DFA engine")
chain(37, 38.3, "Text Stream", 14.5, ["TOKEN", "NUMBER", "URL", "..."], "#ffe3bf", "#eba95a", [9, 10.5, 7, 4.4])

box(34.5, 47, 66.5, 21, "#e6f5e6", "#3e9b4f")
badge(36, 49, 3, "#2f8a3f")
title(41, 51, "Module 3: Normalizer + Symbol Table", "Six switchable rules\n(placeholders, collapse, lowercase, slang)", fs=9.6)
chain(37, 61.6, "Tokens", 9, ["clean_token1", "clean_token2", "clean_token3", "..."], "#d4efd4", "#7cc088", [13, 13, 13, 4.4])

box(103.5, 46.5, 55, 21.5, "#efe8fa", "#7a55b5")
badge(105, 48.5, 4, "#6a3fb0")
title(110, 50.6, "Module 4: Subword Stage", "BPE | LexFront-BPE hybrid\nWordPiece | Unigram")
chain(106, 61.6, "Normalized Tokens", 19, ["sub_1", "sub_2", "sub_3", "..."], "#e1d4f5", "#a98cd6", [6.3, 6.3, 6.3, 4.4])

box(106.5, 2.5, 52, 15, "#e1f6f2", "#1c9c8a")
badge(108, 4.5, 5, "#12877a")
title(113, 6.4, "Module 5: Evaluation", "Tuned grids, size ladder, noise test,\naccuracy, macro-F1, bootstrap CIs, OOV")

box(102.5, 25.5, 24, 13, "#e9f1fd", "#2f5fb3")
ax.text(114.5, Y(30.6), "TF-IDF", ha="center", va="center", fontsize=11, fontweight="bold", color="#1b2a6b", family=SERIF, zorder=4)
ax.text(114.5, Y(35.6), "(min_df = 2)", ha="center", va="center", fontsize=8.2, color=INK, family=SERIF, zorder=4)

box(132.5, 25, 26, 14, "#fde9ee", "#d6455d")
ax.text(145.5, Y(30.2), "Classifier", ha="center", va="center", fontsize=11, fontweight="bold", color="#8a1c33", family=SERIF, zorder=4)
ax.text(145.5, Y(35.2), "Naive Bayes /\nLogistic Regression", ha="center", va="center", fontsize=7.6, color=INK, family=SERIF, zorder=4, linespacing=1.1)

# ---- Arrows
arrow([(25.5, 33.5), (34.3, 33.5)])
arrow([(25.5, 27), (30.0, 27), (30.0, 11.2), (34.3, 11.2)])
arrow([(65, 20.2), (65, 23.8)])
arrow([(65, 44.7), (65, 46.8)])
arrow([(94.7, 11.2), (101.0, 11.2), (101.0, 25.2)])
arrow([(94.7, 33.5), (102.3, 33.5)])
arrow([(118.5, 38.7), (118.5, 46.3)], both=True)
arrow([(101.2, 57), (103.3, 57)])
arrow([(126.7, 32), (132.3, 32)], color="#d6455d")
arrow([(145.5, 24.8), (145.5, 17.8)])

out = Path(__file__).resolve().parent / "figures" / "architecture.png"
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print("wrote", out)
