import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------- architecture ----------
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(-2, 110); ax.set_ylim(0, 62); ax.axis("off")
def box(x, y, w, h, text, fc="#ffffff", ec="#333333", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#333333"))

box(0, 24, 17, 14, "Raw text\ntweets, comments,\nnews articles", fc="#f2f2f2", fs=8.5)
# module 1 band
box(22, 46, 40, 12, "Module 1: existing tokenizers\nwhitespace | NLTK | TweetTokenizer", fc="#dce9f7")
# module 2
box(22, 26, 40, 14, "Module 2: LexFront lexer\n11 token classes, longest match\nown regex-NFA-DFA engine", fc="#fde9c9")
box(22, 6, 40, 14, "Module 3: normalizer + symbol table\nsix switchable rules\n(placeholders, collapse, lowercase, slang)", fc="#e3f1d8")
box(68, 6, 28, 14, "Module 4: subword stage\nBPE | LexFront+BPE hybrid\nWordPiece | Unigram", fc="#efdcf0")
box(68, 28, 18, 10, "TF-IDF\n(min_df=2)", fc="#f2f2f2")
box(90, 28, 19, 10, "Classifier\nNaive Bayes /\nLogistic Regression", fc="#f2f2f2")
box(68, 46, 41, 12, "Module 5 + evaluation: tuned grids, size ladder,\nnoise test; accuracy, macro-F1, bootstrap CIs, OOV", fc="#f2f2f2", fs=8.5)

arrow(17, 33, 22, 52)      # raw -> M1
arrow(17, 31, 22, 33)      # raw -> M2
arrow(42, 26, 42, 20)      # M2 -> M3
arrow(62, 52, 77, 38)      # M1 -> tfidf
arrow(62, 12, 68, 12)      # M3 -> M4
arrow(62, 30, 68, 32)      # M3/M2 -> tfidf (word-level)
arrow(82, 20, 77, 28)      # M4 -> tfidf
arrow(86, 33, 90, 33)      # tfidf -> clf
arrow(99, 38, 99, 46)      # clf -> eval
ax.text(55, 60.5, "Tokenizer is the only part that changes; TF-IDF, classifiers and metrics stay fixed", ha="center", fontsize=9, style="italic")
plt.savefig("figures/architecture.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------- ablation chart (LR acc change vs full, case kept = main grid values) ----------
import numpy as np
rules = ["no_url", "no_mention", "no_number", "no_elongated", "no_slang", "all_off"]
main = {  # main setting (TF-IDF lowercases): acc change pts, ci lo, hi
 "tweet_topic": [(0.00,0.00,0.00),(0.41,-0.03,0.82),(0.50,-0.03,1.00),(0.09,-0.09,0.26),(-0.06,-0.35,0.24),(1.09,0.50,1.71)],
 "go_emotions": [(0.00,0.00,0.00),(-0.02,-0.13,0.07),(-0.11,-0.26,0.04),(0.00,-0.11,0.13),(-0.04,-0.24,0.15),(-0.17,-0.41,0.07)],
 "ag_news":     [(0.00,0.00,0.00),(0.00,0.00,0.00),(0.07,-0.07,0.20),(0.04,-0.04,0.12),(0.03,-0.03,0.09),(0.11,-0.04,0.25)],
}
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for ax, (ds, vals) in zip(axs, main.items()):
    y = np.arange(len(rules))
    d = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
    ax.errorbar(d, y, xerr=[d-lo, hi-d], fmt="o", color="#1f5fa8", ecolor="#7aa6d6", capsize=3)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(rules if ds == "tweet_topic" else [""]*len(rules)); ax.invert_yaxis()
    ax.set_title(ds); ax.set_xlabel("accuracy change vs full LexFront (points)")
    ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig("figures/ablation.png", dpi=200); plt.close()

# ---------- subword accuracy chart ----------
labels = ["NLTK+LR", "LexFront+LR", "BPE+LR", "LexFront+BPE+LR", "WordPiece+LR", "Unigram+LR"]
data = {
 "tweet_topic": [72.32, 70.73, 71.90, 69.90, 71.61, 73.05],
 "go_emotions": [55.27, 55.36, 55.23, 54.64, 55.12, 55.71],
 "ag_news":     [91.88, 91.88, 91.13, 91.03, 91.47, 90.97],
}
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
cols = ["#e8833a", "#f2b233", "#4a86c5", "#7b5ea7", "#4c9f70", "#b0b0b0"]
for ax, (ds, v) in zip(axs, data.items()):
    b = ax.bar(range(len(v)), v, color=cols)
    lo = min(v) - 2.0; hi = max(v) + 1.2
    ax.set_ylim(lo, hi); ax.set_xticks(range(len(v))); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    for r, val in zip(b, v): ax.text(r.get_x()+r.get_width()/2, val+0.1, f"{val:.2f}", ha="center", fontsize=7)
    ax.set_title(ds); ax.set_ylabel("accuracy (%)")
plt.tight_layout(); plt.savefig("figures/subword.png", dpi=200); plt.close()
print("ok")

# ---------- forest plot: LexFront minus NLTK (LR) across datasets ----------
rows = [  # label, diff, lo, hi
 ("tweet_topic (tuned)", -0.32, -1.15, 0.47), ("go_emotions (tuned)", 0.04, -0.44, 0.55), ("ag_news (tuned)", 0.07, -0.21, 0.34),
 ("tweet_eval sentiment (tuned)", 0.36, -0.28, 0.98), ("tweet_eval emotion (tuned)", 0.00, -1.97, 1.83), ("tweet_eval irony (tuned)", 3.06, 0.00, 6.25),
 ("tweet_eval hate (tuned)", -0.66, -1.68, 0.29), ("tweet_eval offensive (tuned)", -0.24, -1.54, 1.07),
 ("Sentiment140 1.6M (wide)", 0.11, -0.04, 0.25), ("Civil Comments (wide)", -0.00, -0.08, 0.07), ("Amazon polarity (wide)", 0.46, 0.34, 0.57),
]
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(rows))
d = np.array([r[1] for r in rows]); lo = np.array([r[2] for r in rows]); hi = np.array([r[3] for r in rows])
ax.axvspan(-0.5, 0.5, color="#dfeedd", alpha=.7, label="practical margin (+-0.5 points)")
ax.errorbar(d, y, xerr=[d-lo, hi-d], fmt="o", color="#1f5fa8", ecolor="#7aa6d6", capsize=3)
ax.axvline(0, color="#888888", lw=1)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("LexFront+LR minus NLTK+LR, accuracy points [95% CI]"); ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig("figures/forest.png", dpi=200); plt.close()
