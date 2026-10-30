import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10.5})
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# architecture figure: see make_architecture.py

# ---------- ablation chart (LR acc change vs full, case kept = main grid values) ----------
import numpy as np
rules = ["no_url", "no_mention", "no_number", "no_elongated", "no_slang", "all_off"]
main = {  # main setting (TF-IDF lowercases): acc change pts, ci lo, hi
 "tweet_topic": [(0.00,0.00,0.00),(0.41,-0.03,0.82),(0.50,-0.03,1.00),(0.09,-0.09,0.26),(-0.06,-0.35,0.24),(1.09,0.50,1.71)],
 "go_emotions": [(0.00,0.00,0.00),(-0.02,-0.13,0.07),(-0.11,-0.26,0.04),(0.00,-0.11,0.13),(-0.04,-0.24,0.15),(-0.17,-0.41,0.07)],
 "ag_news":     [(0.00,0.00,0.00),(0.00,0.00,0.00),(0.07,-0.07,0.20),(0.04,-0.04,0.12),(0.03,-0.03,0.09),(0.11,-0.04,0.25)],
}
fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.6), sharex=True)
for ax, (ds, vals) in zip(axs, main.items()):
    y = np.arange(len(rules))
    d = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
    ax.errorbar(d, y, xerr=[d-lo, hi-d], fmt="o", color="#C62828", ecolor="#E49A9A", capsize=3)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(rules if ds == "tweet_topic" else [""]*len(rules)); ax.invert_yaxis()
    ax.set_title(ds); ax.set_xlabel("accuracy change (points)")
    ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig("figures/ablation.png", dpi=300); plt.close()

# ---------- subword accuracy chart ----------
labels = ["NLTK+LR", "LexFront+LR", "BPE+LR", "LexFront+BPE+LR", "WordPiece+LR", "Unigram+LR"]
data = {
 "tweet_topic": [72.32, 70.73, 71.90, 69.90, 71.61, 73.05],
 "go_emotions": [55.27, 55.36, 55.23, 54.64, 55.12, 55.71],
 "ag_news":     [91.88, 91.88, 91.13, 91.03, 91.47, 90.97],
}
fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.9))
cols = ["#F28E2B", "#C62828", "#4A86C5", "#7B1E1E", "#4C9F9F", "#8A8F98"]   # existing NLTK orange, proposed LexFront red, hybrid dark red
for ax, (ds, v) in zip(axs, data.items()):
    b = ax.bar(range(len(v)), v, color=cols)
    lo = min(v) - 2.0; hi = max(v) + 1.2
    ax.set_ylim(lo, hi); ax.set_xticks(range(len(v))); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=9)
    for r, val in zip(b, v): ax.text(r.get_x()+r.get_width()/2, val+0.1, f"{val:.2f}", ha="center", fontsize=8)
    ax.set_title(ds); ax.set_ylabel("accuracy (%)")
plt.tight_layout(); plt.savefig("figures/subword.png", dpi=300); plt.close()
print("ok")

# ---------- forest plot: LexFront minus NLTK (LR) across datasets ----------
rows = [  # label, diff, lo, hi
 ("tweet_topic (tuned)", -0.32, -1.15, 0.47), ("go_emotions (tuned)", 0.04, -0.44, 0.55), ("ag_news (tuned)", 0.07, -0.21, 0.34),
 ("tweet_eval sentiment (tuned)", 0.36, -0.28, 0.98), ("tweet_eval emotion (tuned)", 0.00, -1.97, 1.83), ("tweet_eval irony (tuned)", 3.06, 0.00, 6.25),
 ("tweet_eval hate (tuned)", -0.66, -1.68, 0.29), ("tweet_eval offensive (tuned)", -0.24, -1.54, 1.07),
 ("Sentiment140 1.6M (wide)", 0.11, -0.04, 0.25), ("Civil Comments (wide)", -0.00, -0.08, 0.07), ("Amazon polarity (wide)", 0.46, 0.34, 0.57),
 ("Yelp 5 classes (wide)", 0.07, -0.16, 0.31),
]
fig, ax = plt.subplots(figsize=(8, 4.8))
y = np.arange(len(rows))
d = np.array([r[1] for r in rows]); lo = np.array([r[2] for r in rows]); hi = np.array([r[3] for r in rows])
ax.axvspan(-0.5, 0.5, color="#dfeedd", alpha=.7, label="practical margin (+-0.5 points)")
ax.errorbar(d, y, xerr=[d-lo, hi-d], fmt="o", color="#C62828", ecolor="#E49A9A", capsize=3)
ax.axvline(0, color="#888888", lw=1)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("LexFront+LR minus NLTK+LR, accuracy points [95% CI]"); ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig("figures/forest.png", dpi=300); plt.close()
