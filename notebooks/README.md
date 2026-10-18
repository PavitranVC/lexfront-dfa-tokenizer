# Notebooks

All notebooks run on a CPU Colab runtime (or locally with Jupyter). They are **generated** by the scripts in this folder, so edit the generator, never the `.ipynb`;
CI checks that each notebook matches its generator (`python notebooks/<generator>.py --check`) and runs a small dry run of each in quick mode.

| Notebook | Open | Generator | What it does | Typical run time |
|---|---|---|---|---|
| `LexFront_Review1.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Review1.ipynb) | `make_notebook.py` | Review-1 grid: whitespace / NLTK / TweetTokenizer / LexFront (+ BPE etc.) on the small datasets | minutes in quick mode, about an hour in full mode |
| `LexFront_Large.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Large.ipynb) | `make_large_notebook.py` | Sentiment140 (1.6 M tweets): size ladder, noise test, tuned vs default | about 1.5 hours of fitting (done; results in `results/large/`) |
| `LexFront_Boost_civil_comments.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Boost_civil_comments.ipynb) | `make_boost_notebook.py` | Phase D: wider settings + character n-gram baseline on Civil Comments | estimate 4 to 7 hours |
| `LexFront_Boost_amazon_polarity.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Boost_amazon_polarity.ipynb) | `make_boost_notebook.py` | Same, Amazon polarity | estimate 4 to 7 hours |
| `LexFront_Boost_yelp_full.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PavitranVC/lexfront-dfa-tokenizer/blob/main/notebooks/LexFront_Boost_yelp_full.ipynb) | `make_boost_notebook.py` | Same, Yelp (5 classes). Slow: read the cost note in `docs/D_PROTOCOL.md` first | estimate 12 to 24 hours or more |

Run-time figures for the phase D rows are estimates, not measurements. Step-by-step instructions for whoever runs them: [`docs/RUNNING_ON_COLAB.md`](../docs/RUNNING_ON_COLAB.md).

## Conventions
* Committed notebooks have no outputs. Results are produced by the run, written to JSON, and the report is generated from the JSON.
* `QUICK_MODE = True` (or `LEXFRONT_*_QUICK=1`) is a dry run on a small sample; its output is not a result and is never committed.
* Every finished fit is saved on its own, so a disconnected run resumes when all cells are run again (keep `USE_DRIVE = True` on Colab).
