"""Builds notebooks/LexFront_Review1.ipynb (run: python notebooks/make_notebook.py).

`python notebooks/make_notebook.py --check` only compares the cell sources with the committed notebook
(exit 1 if they differ), so CI can tell when the generator and the notebook drift apart.

The notebook holds no logic: every code cell calls a function from experiments/ or lexer/.
"""
import sys
from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).with_name("LexFront_Review1.ipynb")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = '''# 1. Get the code, install dependencies (requirements.txt: version ranges), download NLTK data.
import os, subprocess, sys
REPO_URL = os.environ.get("LEXFRONT_REPO_URL", "https://github.com/PavitranVC/lexfront-dfa-tokenizer.git")   # env: CI/tests
REPO_DIR = "lexfront-dfa-tokenizer"

def find_repo_root():
    """The repo root at or above the working directory, or None."""
    here = os.getcwd()
    while here != os.path.dirname(here):
        if os.path.exists(os.path.join(here, "requirements.txt")) and os.path.exists(os.path.join(here, "lexer", "token_spec.py")):
            return here
        here = os.path.dirname(here)

def update_checkout():
    """Fast-forward the clone to origin. A finished run rewrites tracked files (results/quick/...) and that blocks a plain
    pull, so the local changes are set aside with `git stash` (kept: `git stash list`) and the pull is repeated."""
    def git(*args):
        return subprocess.run(["git", "-c", "user.name=colab", "-c", "user.email=colab@localhost", *args],
                              capture_output=True, text=True)
    pull = git("pull", "-q", "--ff-only")
    if pull.returncode != 0 and git("stash", "push", "--include-untracked", "-q").returncode == 0:
        print("local changes from an earlier run were set aside (git stash list) so the code could be updated")
        pull = git("pull", "-q", "--ff-only")
    if pull.returncode != 0:
        raise RuntimeError("could not update the existing clone, so it would run old code under this newer notebook.\\n"
                           f"Delete the folder (!rm -rf {os.getcwd()}) and run this cell again.\\n" + pull.stderr.strip())

root = find_repo_root()
if root is not None:                               # running inside a checkout (local, CI)
    os.chdir(root)
elif os.path.isdir(REPO_DIR):                      # Colab after a restart or a new session: already cloned
    os.chdir(REPO_DIR)
    update_checkout()                              # old code under this newer notebook would fail further down
else:                                              # fresh Colab runtime: clone the repo
    subprocess.run(["git", "clone", "-q", REPO_URL], check=True)
    os.chdir(REPO_DIR)
if not os.path.exists(os.path.join("lexer", "token_spec.py")):
    raise RuntimeError(f"{os.getcwd()} is not a complete checkout: delete it and run this cell again")
print("working directory:", os.getcwd())
sys.path.insert(0, os.getcwd())
for name in [m for m in sys.modules if m.split(".")[0] in ("experiments", "lexer", "subword")]:
    del sys.modules[name]                          # re-run in a live kernel: do not keep the old code imported

from experiments import setup_check               # standard library only
before_install = setup_check.snapshot()
%pip install -q -r requirements.txt
setup_check.report_after_install(before_install)  # prints versions in use; warns if a restart is needed
import nltk
for package in ["punkt_tab"]:                      # what word_tokenize needs; a failed download must not pass silently
    try:
        nltk.data.find(f"tokenizers/{package}")
    except LookupError:
        nltk.download(package, quiet=True)
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            # Still missing. With a proxy configured NLTK refuses to download (its SSRF guard, CWE-918: the proxy, not
            # NLTK, picks the destination); it detects proxies through urllib (any *_proxy variable, the system settings),
            # so instead of guessing which one is set, retry once with the opt-in, for this one download only.
            import nltk.pathsec as pathsec
            before = os.environ.get("NLTK_ALLOW_PROXIED_URLOPEN"), getattr(pathsec, "ALLOW_PROXIED_FETCH", None)
            print("download of", package, "failed or was refused; retrying once with NLTK's proxy opt-in")
            os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
            pathsec.ALLOW_PROXIED_FETCH = True
            try:
                nltk.download(package, quiet=True)
            finally:
                pathsec.ALLOW_PROXIED_FETCH = before[1]
                if before[0] is None:
                    os.environ.pop("NLTK_ALLOW_PROXIED_URLOPEN", None)
                else:
                    os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = before[0]
        nltk.data.find(f"tokenizers/{package}")    # raises LookupError with the download hint if it is still missing'''

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
FORCE = QUICK_MODE     # recompute every run even if a cached JSON matches (cheap in QUICK_MODE)
# Files each mode rewrites:
#   True : everything under results/quick/ (the committed results/ is untouched).
#   False: in the committed results/: datasets.json, module2/dfa_states.json + dfa_min.dot/.png, module2/intrinsic__*.json
#          (always, new timestamps and throughput), stats/bootstrap.json, ablation/stats/bootstrap.json,
#          bpe/stats/bootstrap.json, subword2/stats/bootstrap.json, tables/ and figures/ (rebuilt from the JSON); module1/, module2/, ablation/ and
#          bpe/ and subword2/ run JSONs and their predictions/
#          sidecars only where the fingerprint no longer matches (FORCE is False). `git checkout results/` undoes it.

import pandas as pd
from IPython.display import Image, Markdown, display
from experiments import review1, report
from experiments.observations import observation
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.width", 220)
RES = review1.results_dir(QUICK_MODE)
print("QUICK_MODE =", QUICK_MODE, "| FORCE =", FORCE, "| results written to", RES)'''),

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
Metrics: accuracy and macro-F1. Each run is saved to `results/.../module1/*.json` with its full configuration and a
fingerprint of that configuration and of the code; a saved run is reused only while the fingerprint still matches
(`FORCE = True` recomputes everything)."""),
    code('''review1.run_module1_section(QUICK_MODE, force=FORCE)
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
    code('''review1.run_module2_experiments(QUICK_MODE, force=FORCE)
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

    md("""## 5. Is the difference larger than test-set noise?

Every run above also saved its test predictions (`predictions/<run>.json.gz` next to its JSON). The cell below
resamples the test examples with replacement (percentile bootstrap, fixed seed; B and the seed are stored in
`stats/bootstrap.json`) and recomputes accuracy and macro-F1 from a confusion matrix for each resample. For a
difference, both systems are scored on the *same* resamples (paired); NLTK+LR vs LexFront+LR also gets an exact
McNemar test. Same code as the full grid (`python -m experiments.run_stats`), here on the QUICK grid.
QUICK uses 500 test examples per dataset (the full test sets have 3399 to 7600), so these intervals are wider (for
accuracy differences about 2 to 6 times the full-grid width in the committed runs) and a difference can even change
sign; the headline CIs are the full-grid ones in
`results/stats/bootstrap.json` (table `ci_table` in the appendix). These intervals cover the draw of the test set only, not training randomness (one training run per cell)."""),
    code('''review1.run_stats_section(QUICK_MODE)
out = review1.build_outputs(QUICK_MODE)
display(Markdown(out["tables"]["ci_table"]))
display(Markdown(out["tables"]["ci_runs"]))'''),
    md("### Observations — confidence intervals"),
    code('''display(Markdown(observation("ci", RES)))'''),

    md("""## 6. Which normalization rule matters? (ablation)

LexFront's normalizer has six switchable rules (URL, MENTION, NUMBER, ELONGATED, lowercase, slang). Each variant
below is LexFront + Logistic Regression with one rule switched off, everything else fixed, compared with full
LexFront on the same resampled test examples (paired bootstrap, as in section 5; McNemar on accuracy).
*Main setting* (TF-IDF lowercases, as the main grid): no_url, no_mention, no_number, no_elongated, no_slang and
all_off (lexer segmentation only; TF-IDF still lowercases). The lowercase rule cannot show an effect there, because TF-IDF lowercases the text
before LexFront sees it, so it is tested in the *case-kept* setting (TF-IDF `lowercase=False`): no_lowercase and
all_off_cs against full_cs. On the QUICK subsample (500 test examples) these intervals are wide; the full-data
ablation is in `results/ablation/` (appendix). With up to 8 comparisons per dataset and no multiplicity correction,
an interval can exclude 0 by chance. The column "rule fires" counts the train / test documents in which the removed
rule changes a token; a rule that never fires on the subsample (0 / 0) gives the full system, not a test of the rule."""),
    code('''review1.run_ablation_section(QUICK_MODE, force=FORCE)
out = review1.build_outputs(QUICK_MODE)
display(Markdown(out["tables"]["ablation_table"]))'''),
    md("### Observations — ablation"),
    code('''display(Markdown(observation("ablation", RES)))'''),

    md("""## 7. Subword tokenizers: BPE and LexFront + BPE

**BPE** (byte-pair encoding, Hugging Face `tokenizers`) learns a vocabulary of 8000 subword pieces from each dataset's
*train* split (lowercased, never the test split) and splits words into vocabulary pieces; a piece that continues a
word carries `##`. **LexFront + BPE** keeps LexFront's scanner and normalizer: structural tokens (`<URL>`, `<USER>`,
`<NUM>`, hashtags, emoticons, emoji, punctuation, UNKNOWN) pass through whole, and only word-like tokens are split by
a BPE trained on the word-like tokens of the train split. Both feed the same TF-IDF + NB / LR as before. The first
cell shows the pieces on three example tweets with a toy BPE trained on the eight example tweets (illustration only).
The second runs the grid on the QUICK subsample and the paired bootstrap against NLTK+LR and LexFront+LR (same code
as section 5); the full-data results are in `results/bpe/` (appendix). BPE's OOV rate in the intrinsic table is near
zero (only characters never seen in train and rare pieces can be OOV), so it is not comparable with the word-level tokenizers."""),
    code('''display(review1.bpe_demo())'''),
    code('''review1.run_bpe_section(QUICK_MODE, force=FORCE)
out = review1.build_outputs(QUICK_MODE)
display(Markdown(out["tables"]["bpe_runs"]))
display(Markdown(out["tables"]["bpe_table"]))'''),
    md("### Observations — BPE and the hybrid"),
    code('''display(Markdown(observation("bpe", RES)))'''),

    md("""## 8. WordPiece and Unigram

Two more learned vocabularies, from the same library and the same protocol as BPE: 8000 pieces per dataset, trained on
the *train* split only (lowercased), `Whitespace` pre-tokenizer, `[UNK]`. **WordPiece** grows its vocabulary by merging
the pair that most raises the likelihood of the text and encodes by greedy longest match (continuing pieces carry
`##`). **Unigram** starts from many candidate pieces, prunes them by EM and encodes by the best-scoring split (Viterbi);
it has no `##`. No hybrid with LexFront. The first cell shows the three on three example tweets (toy vocabularies
trained on the eight example tweets, illustration only). The second runs the grid on the QUICK subsample and the paired
bootstrap against NLTK+LR and BPE+LR; the full-data results are in `results/subword2/` (appendix). OOV in the intrinsic
table is near zero for these too, so it is not comparable with the word-level tokenizers."""),
    code('''display(review1.subword2_demo())'''),
    code('''review1.run_subword2_section(QUICK_MODE, force=FORCE)
out = review1.build_outputs(QUICK_MODE)
display(Markdown(out["tables"]["subword_runs"]))
display(Markdown(out["tables"]["subword_table"]))'''),
    md("### Observations — WordPiece and Unigram"),
    code('''display(Markdown(observation("subword", RES)))'''),

    md("""## Appendix — committed full-data results
The tables below are built from the JSON files in `results/` (full datasets, produced by
`python -m experiments.run_module1`, `python -m experiments.run_module2`, `python -m experiments.run_stats` and
`python -m experiments.run_sensitivity`, `python -m experiments.run_ablation`, `python -m experiments.run_bpe`, `python -m experiments.run_subword2`). `sensitivity_case` is the grid with TF-IDF `lowercase=False`: LexFront's
lowercase rule is active there while the baselines keep case, so it is a robustness check, not a like-for-like."""),
    code('''import os
if os.path.exists("results/datasets.json") and os.path.isdir("results/module1"):
    full = report.render_all_tables("results")      # read only: does not rewrite results/tables
    for name in ["module1_results", "existing_vs_proposed", "intrinsic_metrics", "ci_table", "sensitivity_case",
                 "sensitivity_ci", "ablation_table", "bpe_runs", "bpe_table", "subword_runs", "subword_table"]:
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


def matches_committed() -> bool:
    """True if the committed notebook has exactly the cells (type and source) that CELLS defines."""
    committed = nbf.read(NB_PATH, as_version=4).cells
    return [(c.cell_type, c.source) for c in committed] == [(c.cell_type, c.source) for c in CELLS]


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = matches_committed()
        print("notebook matches make_notebook.py" if ok else "notebook differs: run python notebooks/make_notebook.py")
        sys.exit(0 if ok else 1)
    print(build())
