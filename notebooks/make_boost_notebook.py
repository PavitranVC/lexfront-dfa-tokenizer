"""Builds notebooks/LexFront_Boost_<dataset>.ipynb for the phase D datasets (run: python notebooks/make_boost_notebook.py).

`--check` compares the cell sources of every notebook with the committed files (exit 1 if one differs). Committed notebooks have no
outputs; CI executes the same cells as a dry run with LEXFRONT_BOOST_QUICK=1. The setup cell is the one of LexFront_Review1.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from make_notebook import SETUP  # noqa: E402

from experiments import datasets_big  # noqa: E402

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def cells(name: str) -> list:
    spec = datasets_big.SPECS[name]
    cost = ("**Slow dataset:** five-class Logistic Regression over the protocol grid took about 30 minutes per fit set even on a 27 000-example dry run; "
            "expect a full run to take a day or more. Consider asking for a reduced grid (a protocol amendment) before starting.\n") if name == "yelp_full" else ""
    return [
        md(f"""# LexFront — phase D: `{spec.hf_id}`

**What this does.** Runs the tokenizer comparison of `docs/BOOST_PROTOCOL.md` on one more large dataset, with the rules of `docs/D_PROTOCOL.md` (written before any run):
whitespace, NLTK, TweetTokenizer and LexFront, then a tokenizer-free character n-gram baseline, each with TF-IDF + Naive Bayes / Logistic Regression, a default setting
and a wider validation-selected setting (selection uses a slice of the training data only). Labels: {spec.note}. Every number in the report is read from a JSON file written by this run.

**How to run it.** CPU runtime. Run all cells, once. The real run takes **hours** (up to {spec.max_train or 'all'} training examples per fit). Every finished fit is saved; if Colab disconnects,
reopen the notebook and run all cells again: finished fits are skipped. With `USE_DRIVE = True` they are saved on your Google Drive (access is asked once).
{cost}The character baseline needs much memory; if the runtime runs out of memory on it, the report simply has no character row (the protocol says to report it as not run).

**First time?** Set `QUICK_MODE = True` in the next cell for a dry run of a few minutes on a small streamed sample."""),
        code(SETUP),
        code(f'''import os, subprocess, sys
QUICK_MODE = os.environ.get("LEXFRONT_BOOST_QUICK", "0") == "1"   # True: dry run on a small sample (minutes). False: the real run (hours).
USE_DRIVE = True        # keep the per-run files on Google Drive so a disconnected session resumes (ignored when not on Colab)
DATASET = "{name}"

WORK = f"../lexfront_boost_work_{{DATASET}}" + ("_quick" if QUICK_MODE else "")
OUT = ("results/quick/boost_" if QUICK_MODE else "results/boost_") + DATASET
if USE_DRIVE and not QUICK_MODE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = "/content/drive/MyDrive/lexfront_boost_work_" + DATASET
    except Exception as error:      # not on Colab (ImportError) or the Drive permission was declined
        print("Google Drive not used (" + type(error).__name__ + "): the per-run files stay in the runtime, next to the repository")
print("QUICK_MODE =", QUICK_MODE, "| per-run files:", WORK, "| results:", OUT)'''),
        md("## Run: the four word tokenizers\n\nProgress lines say `done` for a fit just computed and `cached` for one found from an earlier session."),
        code('''from experiments import run_boost
base = ["--dataset", DATASET, "--work", WORK, "--out", OUT] + (["--quick"] if QUICK_MODE else [])
run_boost.main(base + ["--tokenizers", *run_boost.boost.WORD_TOKENIZERS])'''),
        md("## Run: the character baseline\n\nIn its own process, so that running out of memory cannot lose the results above. A failure here is allowed and is not an error."),
        code('''done = subprocess.run([sys.executable, "-m", "experiments.run_boost"] + base + ["--tokenizers", run_boost.boost.CHAR])
print("character baseline finished" if done.returncode == 0 else f"character baseline did not finish (exit code {done.returncode}); report it as not run")'''),
        md("## Report\n\nGenerated from the JSON files, including the pre-registered hypotheses."),
        code('''from IPython.display import Markdown, display
display(Markdown(open(f"{OUT}/REPORT.md").read()))'''),
        md("## Save the results\n\nA zip of the results folder. On Colab it is downloaded; copy the files into the repository's `results/` folder (and send them back for review)."),
        code('''from experiments import run_large
zip_path = run_large.make_zip(OUT, f"lexfront_boost_{DATASET}_results.zip")
print(zip_path, round(os.path.getsize(zip_path) / 1e6, 1), "MB")
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    pass'''),
    ]


def path(name: str) -> Path:
    return Path(__file__).with_name(f"LexFront_Boost_{name}.ipynb")


def build() -> list[Path]:
    paths = []
    for name in datasets_big.SPECS:
        nb = nbf.v4.new_notebook(cells=cells(name))
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["colab"] = {"name": path(name).name}
        nbf.write(nb, path(name))
        paths.append(path(name))
    return paths


def matches_committed() -> bool:
    for name in datasets_big.SPECS:
        if not path(name).exists():
            return False
        committed = nbf.read(path(name), as_version=4).cells
        if [(c.cell_type, c.source) for c in committed] != [(c.cell_type, c.source) for c in cells(name)]:
            return False
    return True


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok = matches_committed()
        print("boost notebooks match make_boost_notebook.py" if ok else "boost notebooks differ: run python notebooks/make_boost_notebook.py")
        sys.exit(0 if ok else 1)
    print(*build(), sep="\n")
