"""Builds notebooks/LexFront_Review1.ipynb (run: python notebooks/make_notebook.py).

The notebook holds no logic: every code cell calls a function from experiments/ or lexer/.
"""
import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).with_name("LexFront_Review1.ipynb")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = '''# 1. Get the code, install dependencies, download NLTK data.
import os, subprocess, sys
REPO_URL = "https://github.com/PavitranVC/lexfront-dfa-tokenizer.git"
DEV_BRANCH = "claude/lexfront-lexical-analyzer-tl3l0y"   # used only if main does not have the code yet

def find_repo_root():
    here = os.getcwd()
    while here != os.path.dirname(here):
        if os.path.exists(os.path.join(here, "requirements.txt")) and os.path.isdir(os.path.join(here, "lexer")):
            return here
        here = os.path.dirname(here)

root = find_repo_root()
if root is None:                                   # fresh Colab runtime: clone the repo
    subprocess.run(["git", "clone", "-q", REPO_URL], check=True)
    os.chdir("lexfront-dfa-tokenizer")
    if not os.path.isdir("lexer"):
        subprocess.run(["git", "checkout", "-q", DEV_BRANCH], check=True)
else:
    os.chdir(root)
print("working directory:", os.getcwd())

%pip install -q -r requirements.txt
import nltk
for package in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(package, quiet=True)
sys.path.insert(0, os.getcwd())'''

CELLS = [
    md("""# LexFront — Review 1 (Modules 1 and 2)

**Question.** Can a compiler-style lexical analyzer (token classes as regexes → Thompson NFA → DFA → minimized DFA,
with a symbol table and error recovery) work as a *pre-tokenizer* for noisy English text classification,
compared with existing tokenizers when everything downstream is held fixed?

**This notebook.** Module 1 = data + existing tokenizers + classifiers. Module 2 = LexFront itself.
It contains no logic: all code lives in `experiments/` and `lexer/`; the notebook calls it and shows results.
Every number shown is read from a JSON file written by a script that ran.

**`QUICK_MODE`** (next cell) keeps the whole notebook under about 10 minutes on a CPU runtime by using a fixed-seed
subsample of each dataset. Set it to `False` to rerun on the full data (much slower).
The committed full-data results are shown in the appendix at the end."""),
    code(SETUP),
    code('''QUICK_MODE = True      # True: fixed-seed subsample (fast).  False: full datasets (slow).

import pandas as pd
from IPython.display import Image, Markdown, display
from experiments import review1, report
from experiments.observations import observation
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.width", 220)
RES = review1.results_dir(QUICK_MODE)
print("QUICK_MODE =", QUICK_MODE, "| results written to", RES)'''),

    md("""## 1. Datasets

Three Hugging Face datasets, each reduced to *train texts, test texts, labels, label names*:

* `cardiffnlp/tweet_topic_single` — tweets, 6 topics (COLING 2022 split pair when the dataset card confirms it).
* `google-research-datasets/go_emotions` (`simplified`) — Reddit comments; only single-label examples are kept.
* `fancyzhx/ag_news` — news snippets, 4 topics.

The cell loads all three, saves `datasets.json`, and shows the table built from it."""),
    code('''review1.run_datasets_section(QUICK_MODE)
display(Markdown(report.table_datasets(RES)))'''),
    md("### Observations — datasets"),
    code('''display(Markdown(observation("datasets", RES)))'''),

    md("""## 2. Module 1 — existing tokenizers and classifiers

Three tokenizers behind one interface `tokenize(text) -> list[str]`: whitespace split, NLTK `word_tokenize`,
NLTK `TweetTokenizer`. Their tokens feed the *same* TF-IDF vectorizer (`tokenizer=fn`, `token_pattern=None`), then a
majority-class baseline, Multinomial Naive Bayes (the method of the base paper) and Logistic Regression.
Metrics: accuracy and macro-F1. Each run is saved to `results/.../module1/*.json` with its full configuration."""),
    code('''review1.run_module1_section(QUICK_MODE)
out = review1.build_outputs(QUICK_MODE)
display(Markdown(out["tables"]["module1_results"]))
display(Image(f"{RES}/figures/module1_accuracy.png"))'''),
    md("### Observations — Module 1"),
    code('''display(Markdown(observation("module1", RES)))'''),

    md("""## 3. Module 2 — LexFront

LexFront is a lexer written the way a compiler front end is: **token classes in priority order**
(URL, MENTION, HASHTAG, EMOTICON, EMOJI, NUMBER, CONTRACTION, ELONGATED, WORD, PUNCT, and UNKNOWN as the error-recovery
fallback), **longest match with ties to the earlier rule** (PLY), then a **normalizer** (URL→`<URL>`, MENTION→`<USER>`,
NUMBER→`<NUM>`, elongated letters collapsed, WORD lowercased, slang lookup) and a **symbol table**.

### 3a. Side-by-side on eight example tweets
First the plain token lists of all four tokenizers (input lowercased, as the vectorizer does), then LexFront's
class / lexeme / normalized form for each tweet."""),
    code('''display(review1.compare_tokenizers())'''),
    code('''for i, tweet in enumerate(review1.EXAMPLE_TWEETS, 1):
    print(f"--- tweet {i}: {tweet}")
    display(review1.lexfront_table(tweet))'''),
    md("""### 3b. Symbol table
Every normalized token gets an id the first time it is seen; later sightings only increase its frequency."""),
    code('''display(review1.symbol_table_demo())'''),
    md("""### 3c. Our own DFA engine
`lexer/dfa/` is a small lexer generator: a regex parser, Thompson NFA, subset-construction DFA, Moore minimization
and a table-driven scanner. It is applied to **EMOTICON, HASHTAG, MENTION and WORD** only. **ELONGATED is left out:**
its usual regex `(.)\\1{2,}` needs a back-reference, which is not part of regular expressions."""),
    code('''review1.run_module2_experiments(QUICK_MODE)
out = review1.build_outputs(QUICK_MODE)
display(Markdown(out["tables"]["dfa_states"]))
display(Image(f"{RES}/module2/dfa_min.png"))'''),
    md("### Observations — DFA"),
    code('''display(Markdown(observation("dfa", RES)))'''),
    md("""### 3d. Scanner demo with error recovery
The table-driven scanner follows the DFA as far as it can, remembers the last accepting position and backs up to it.
A character that no class accepts (here `§`) becomes an **UNKNOWN** token and scanning continues. The last line checks
that the PLY lexer restricted to the same four classes produces exactly the same tokens."""),
    code('''tokens, agree = review1.scanner_demo()
display(tokens)
print("DFA scanner and PLY lexer agree on this input:", agree)'''),
    md("""### 3e. Intrinsic metrics
Measured on lowercased text for all four tokenizers: **OOV rate** (share of test tokens never seen in train),
**fertility** (tokens per whitespace-separated word), **vocabulary size** (distinct train tokens) and **throughput**
(tokens per second on this machine; it varies between machines)."""),
    code('''display(Markdown(out["tables"]["intrinsic_metrics"]))
display(Image(f"{RES}/figures/intrinsic_metrics.png"))'''),
    md("### Observations — intrinsic metrics"),
    code('''display(Markdown(observation("intrinsic", RES)))'''),

    md("""## 4. Existing vs proposed

**Existing** = NLTK `word_tokenize` + Naive Bayes (the method we start from). **Proposed** = LexFront + Logistic
Regression. The table also shows the two mixed cells (LexFront + NB, NLTK + LR) so a change can be attributed to the
tokenizer or to the classifier. Same TF-IDF settings and seed everywhere."""),
    code('''display(Markdown(out["tables"]["existing_vs_proposed"]))
display(Image(f"{RES}/figures/existing_vs_proposed.png"))'''),
    md("### Observations — existing vs proposed"),
    code('''display(Markdown(observation("existing_vs_proposed", RES)))'''),

    md("""## Appendix — committed full-data results
The tables below are built from the JSON files in `results/` (full datasets, produced by
`python -m experiments.run_module1` and `python -m experiments.run_module2`)."""),
    code('''import os
if os.path.exists("results/datasets.json") and os.path.isdir("results/module1"):
    full = report.build_all_tables("results")
    for name in ["module1_results", "existing_vs_proposed", "intrinsic_metrics"]:
        print(name); display(Markdown(full[name]))
else:
    print("full-data results are not in this checkout")'''),
]


def build() -> Path:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"name": "LexFront_Review1.ipynb"}
    nbf.write(nb, NB_PATH)
    return NB_PATH


if __name__ == "__main__":
    print(build())
