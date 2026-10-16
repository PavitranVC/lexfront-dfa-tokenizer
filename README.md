# LexFront

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Review1.ipynb)

A compiler-style lexer (token classes as regexes → Thompson NFA → subset-construction DFA → minimized DFA, plus a
symbol table and error recovery) used as a **pre-tokenizer for noisy English text classification**, compared with
existing tokenizers with the classifier held fixed. Compiler Design course project, Review 1 (Modules 1 and 2).

* Notebook (Colab, CPU): `notebooks/LexFront_Review1.ipynb`
* Study guide for the viva: `docs/LEARN.md`
* Module write-ups: `docs/modules/module1.md`, `docs/modules/module2.md`
* Design versions: `docs/versions/`  · Changelog: `CHANGELOG.md`
* Results (JSON), tables, figures and notes: `results/` (`results/REPORT_NOTES.md`)

## Run locally (Python 3.11 or 3.12, CPU only, no Docker)

Two dependency files:

* `requirements.txt`: compatible version ranges. Install this for Colab or a new environment; a fresh Colab runtime
  already satisfies the numpy/pandas/scipy/scikit-learn/matplotlib ranges, so nothing it loaded is reinstalled.
* `requirements-lock.txt`: the exact versions the committed results were produced with (Python 3.11). CI installs it;
  use it to reproduce `results/` exactly.

Tested: both files install on Python 3.11 and 3.12, and `pytest -q` passes with the lock file, with the lowest
versions `requirements.txt` allows, and with the newest it resolved to (2026-10-16). Python 3.10 is not supported.
Library versions are part of each run's fingerprint, so a run with other versions recomputes instead of reusing a
committed JSON.

```
pip install -r requirements-lock.txt      # or requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords')]"
pytest -q
python -m experiments.run_datasets
python -m experiments.run_module1
python -m experiments.run_module2
python -m experiments.render_docs
```

The notebook runs top to bottom on a fresh Colab runtime; `QUICK_MODE = True` (default) uses a fixed-seed subsample.

## Layout

```
lexer/        token spec, PLY lexer, normalizer, symbol table, LexFront, dfa/ (own regex->DFA engine)
experiments/  data, tokenizers, classifiers, metrics, evaluation loop, reports, figures
tests/        fast pytest suite
results/      JSON results, tables/, figures/, module2/dfa_min.{dot,png}, REPORT_NOTES.md
notebooks/    LexFront_Review1.ipynb (+ make_notebook.py)
docs/         versions/, modules/, LEARN.md
```

Out of scope for review 1: BPE and the LexFront + BPE hybrid, WordPiece, Unigram, ablation, transformers.
