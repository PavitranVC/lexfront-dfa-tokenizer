# LexFront

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Review1.ipynb)

A compiler-style lexer (token classes as regexes → Thompson NFA → subset-construction DFA → minimized DFA, plus a
symbol table and error recovery) used as a **pre-tokenizer for noisy English text classification**, compared with
existing tokenizers with the classifier held fixed. Compiler Design course project, Review 1 (Modules 1 and 2).

## Status at a glance

| Part | State | Where |
|---|---|---|
| Lexer, DFA engine, comparison on small datasets | done | `results/`, `docs/FINAL_SUMMARY.md` |
| Eight-dataset collection with fair tuning | done | `results/collection/REPORT.md` |
| Sentiment140, 1.6 M tweets | done | `results/large/REPORT.md` |
| Phase A: wider settings and character n-grams (Sentiment140) | done | `results/boost_sentiment140/REPORT.md` |
| Phase D: Civil Comments, Amazon polarity, Yelp | done (run on a local machine, see branch `feature/local-run` and `docs/LOCAL_RUN.md` there; Colab was too small for Yelp) | `results/boost_civil_comments/REPORT.md`, `results/boost_amazon_polarity/REPORT.md`, `results/boost_yelp_full/REPORT.md`, `docs/D_PROTOCOL.md` |

Headline numbers (Sentiment140, Logistic Regression, test accuracy %; from the two reports above): at default settings whitespace 79.60, NLTK 79.88,
TweetTokenizer 79.96, LexFront 79.85; with the wider validation-selected settings whitespace 81.68, NLTK 82.73, TweetTokenizer 82.50, LexFront 82.84; a
character n-gram baseline 80.11. LexFront minus NLTK is +0.11 [-0.04, +0.25] points, inside the pre-set practical margin of +-0.5 points. These numbers come from one dataset
and one training run per cell; they do not say why any difference occurs. The reports and protocols state the limits.

## Where to start
* Run it: `notebooks/README.md` (all notebooks, run times, Colab badges) and `docs/RUNNING_ON_COLAB.md` (instructions for someone taking over a run).
* Read it: `docs/README.md` (index of every document).
* Taking over the project: `docs/HANDOFF.md`, then `docs/NEXT_PLAN.md`.

## Main links

* Notebook (Colab, CPU): `notebooks/LexFront_Review1.ipynb`
* Large-scale test on Sentiment140 (1.6 M tweets, Colab, CPU, resumable): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Large.ipynb)  protocol: `docs/LARGE_PROTOCOL.md`, results: `results/large/REPORT.md`
* How it was built (AI-assisted workflow and safeguards): `docs/AI_ASSISTED_DEVELOPMENT.md`
* Study guide for the viva: `docs/LEARN.md`
* One-page honest summary of the results: `docs/FINAL_SUMMARY.md`
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
versions `requirements.txt` allows (both on 3.11 and 3.12), and, on 3.12 only, with the newest versions it resolved to (2026-10-16). Python 3.10 is not supported.
Library versions are part of each run's fingerprint, so a run with other versions recomputes instead of reusing a
committed JSON.

```
pip install -r requirements-lock.txt      # or requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords')]"
pytest -q
python -m experiments.run_datasets
python -m experiments.run_module1
python -m experiments.run_module2
python -m experiments.run_bpe              # BPE + hybrid grid and its stats (P3)
python -m experiments.run_subword2         # WordPiece + Unigram grid and its stats (P4)
python -m experiments.render_docs
python -m experiments.run_boost --stats-only --out results/boost_sentiment140   # phase A report from the committed files
python -m experiments.verify_boost --out results/boost_sentiment140 --ref results/large --protocol docs/BOOST_PROTOCOL.md --base origin/main   # self-check
python -m experiments.run_large           # Sentiment140 large-scale test (--quick for a dry run; see the Colab notebook)
python -m experiments.run_collection      # 8 cleaned datasets, default vs tuned classifiers (about 1.5 h; --stats-only rebuilds the report)
```

The notebook runs top to bottom on a fresh Colab runtime; `QUICK_MODE = True` (default) uses a fixed-seed subsample.
If the Colab runtime kept a clone from an earlier session, the first cell sets that run's local changes aside
(`git stash`) and pulls the current code; if the clone cannot be updated it stops and says how to start clean.

## Layout

```
lexer/        token spec, PLY lexer, normalizer, symbol table, LexFront, dfa/ (own regex->DFA engine)
subword/      BPE, the LexFront + BPE hybrid, WordPiece and Unigram (Hugging Face `tokenizers`, trained on the train split only)
experiments/  data, tokenizers, classifiers, metrics, evaluation loop, reports, figures; clean / data_extra / tuning / run_collection (dataset collection); large / run_large (Sentiment140); boost / run_boost / verify_boost / datasets_big (phase A and D)
tests/        fast pytest suite
results/      JSON results, tables/, figures/, module2/dfa_min.{dot,png}, REPORT_NOTES.md, collection/REPORT.md
notebooks/    five generated notebooks and their generators (see notebooks/README.md)
docs/         index in docs/README.md: protocols, decisions, handoff, audits, versions/, modules/, LEARN.md
```

Out of scope: transformers (they need torch), Hinglish, more seeds. Added since review 1: the per-rule normalization ablation
(`docs/versions/v4-ablation.md`) BPE + the LexFront + BPE hybrid (`docs/versions/v5-bpe-hybrid.md`), and WordPiece and Unigram
(`docs/versions/v6-wordpiece-unigram.md`), and a cleaned eight-dataset collection with fair classifier tuning
(`results/collection/REPORT.md`, `docs/DECISIONS.md`).
