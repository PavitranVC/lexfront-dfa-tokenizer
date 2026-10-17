"""Builds notebooks/LexFront_Large.ipynb (run: python notebooks/make_large_notebook.py).

`--check` only compares the cell sources with the committed notebook (exit 1 if they differ), like make_notebook.py.
The committed notebook has no outputs: a real run takes hours on Colab, and CI executes the same cells as a dry run by
setting LEXFRONT_LARGE_QUICK=1. The setup cell is the one of LexFront_Review1 (clone or update the repo, install, NLTK data).
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
from make_notebook import SETUP  # noqa: E402

NB_PATH = Path(__file__).with_name("LexFront_Large.ipynb")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("""# LexFront — large-scale test on Sentiment140 (1.6 M tweets)

**What this does.** Runs the tokenizer comparison on a very large noisy dataset: whitespace, NLTK, TweetTokenizer and LexFront
with TF-IDF + Naive Bayes / Logistic Regression, on a user-grouped split (no author in both train and test), at training sizes
10 000, 100 000, 1 000 000 and the full set, plus a test with artificially noisy test text. The protocol and the hypotheses were
written down before the run: `docs/LARGE_PROTOCOL.md`. Every number in the report is read from a JSON file written by this run.

**How to run it.** CPU runtime is enough. Run all cells, once. The real run takes **hours**. Every finished fit is saved, so if
Colab disconnects, reopen this notebook and run all cells again: finished fits are skipped. With `USE_DRIVE = True` they are saved on
your Google Drive (you will be asked to allow access once), otherwise only in the runtime (lost if the runtime is deleted).

**First time?** Set `QUICK_MODE = True` in the next cell for a dry run of a few minutes on a small subsample.
At the end the cells show the report, make a zip you can download, and can optionally push the results to a new branch of the repo."""),
    code(SETUP),
    code('''import os
QUICK_MODE = os.environ.get("LEXFRONT_LARGE_QUICK", "0") == "1"   # True: dry run on a small subsample (minutes). False: the real run (hours).
USE_DRIVE = True        # keep the per-run files on Google Drive so a disconnected session resumes (ignored when not on Colab)
PUSH_RESULTS = False    # True: at the end push results to a NEW branch of the repo (needs a Colab secret GITHUB_TOKEN, see the last cell)

WORK = "../lexfront_large_work_quick" if QUICK_MODE else "../lexfront_large_work"
OUT = "results/quick/large" if QUICK_MODE else "results/large"
if USE_DRIVE and not QUICK_MODE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = "/content/drive/MyDrive/lexfront_large_work"
    except Exception as error:      # not on Colab (ImportError) or the Drive permission was declined
        print("Google Drive not used (" + type(error).__name__ + "): the per-run files stay in the runtime, next to the repository")
print("QUICK_MODE =", QUICK_MODE, "| per-run files:", WORK, "| results:", OUT)'''),
    md("""## Run

Loads the dataset, splits by user, cleans (`experiments/clean.py`), audits it and runs every fit. Progress lines say `done` for a fit
that was just computed and `cached` for one that was found from an earlier session."""),
    code('''from experiments import run_large
run_large.main(["--work", WORK] + (["--quick"] if QUICK_MODE else []))'''),
    md("## Report\n\nGenerated from the JSON files; includes the pre-registered hypotheses with what was observed."),
    code('''from IPython.display import Markdown, display
display(Markdown(open(f"{OUT}/REPORT.md").read()))'''),
    md("## Save the results\n\nA zip of the results folder. On Colab it is downloaded to your computer; copy the files into the repository's `results/large/`."),
    code('''zip_path = run_large.make_zip(OUT, "lexfront_large_results.zip")
print(zip_path, round(os.path.getsize(zip_path) / 1e6, 1), "MB")
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    pass'''),
    md("""## Optional: push the results to a new branch

Only if `PUSH_RESULTS = True`. It commits the results folder to a **new branch** (never `main`) and pushes it, so the results can be reviewed
and merged. It needs a GitHub token stored as a Colab secret named `GITHUB_TOKEN` (key icon in the left bar, switch on "Notebook access").
Use a fine-grained token limited to this one repository with *Contents: read and write*. The token is handed to git through an
askpass helper and is never printed."""),
    code('''if PUSH_RESULTS:
    from google.colab import userdata
    from datetime import datetime, timezone
    branch = "results/large-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    print(run_large.push_results(".", OUT, branch, userdata.get("GITHUB_TOKEN")))
else:
    print("PUSH_RESULTS is False: nothing pushed")'''),
]


def build() -> Path:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"name": "LexFront_Large.ipynb"}
    nbf.write(nb, NB_PATH)
    return NB_PATH


def matches_committed() -> bool:
    committed = nbf.read(NB_PATH, as_version=4).cells
    return [(c.cell_type, c.source) for c in committed] == [(c.cell_type, c.source) for c in CELLS]


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = matches_committed()
        print("large notebook matches make_large_notebook.py" if ok else "large notebook differs: run python notebooks/make_large_notebook.py")
        sys.exit(0 if ok else 1)
    print(build())
