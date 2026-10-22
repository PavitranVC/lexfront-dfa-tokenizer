# Running the long experiments on a local machine

This branch (`feature/local-run`) adds what is needed to run the phase D experiments (`civil_comments`, `amazon_polarity`, `yelp_full`) on a
laptop or desktop instead of Colab. Nothing under `notebooks/` changes, and no existing result changes.

## Hardware (what the Colab run could not provide)
| | Colab free tier | Local run |
|---|---|---|
| RAM | about 12 GB, session killed when exceeded | 32 GB recommended: each fit set peaks near 10 GB (measured 10.4 GB on Yelp), and two processes run side by side |
| Session length | about 12 hours, drops when idle | none; finished fits are saved to disk, so a restart resumes |
| CPU | shared, slower cores | fits are single-threaded (`blas_threads` pinned to 1 for reproducibility), so one fast core per process matters more than the core count |
| GPU | not used | not used (scikit-learn, CPU only) |

## Setup
Python 3.11 or 3.12 (3.10 and 3.13 are not supported), then:
```
git clone -b feature/local-run https://github.com/PavitranVC/lexfront-dfa-tokenizer
cd lexfront-dfa-tokenizer
python -m venv .venv            # activate it
pip install -r requirements-lock.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords')]"
```

## Commands (one process per command; at most 2 at once on 32 GB)
```
python -m experiments.run_boost --dataset civil_comments  --work ../civil_work  --out results/boost_civil_comments  --tokenizers whitespace nltk tweet lexfront
python -m experiments.run_boost --dataset amazon_polarity --work ../amazon_work --out results/boost_amazon_polarity --tokenizers whitespace nltk tweet lexfront
python -m experiments.run_boost --dataset yelp_full       --work ../yelp_work   --out results/boost_yelp_full       --tokenizers whitespace nltk
python -m experiments.run_boost --dataset yelp_full       --work ../yelp_work   --out results/boost_yelp_full       --tokenizers tweet lexfront
```
Add `--quick` for a dry run on a small sample (not a result). If a run stops, repeat the same command: finished fits print `cached` and are skipped.
A dataset is complete at 12 files in `<work>/runs/` (3 per word tokenizer); a final run of its command writes `results/boost_<dataset>/REPORT.md`.

## Changes in this branch
* `experiments/peak_memory.py`: `peak_rss_gb()` replaces the direct use of the Unix-only `resource` module in `boost.py` and `large.py`
  (Windows has no `resource`; it now reads the Windows process counters). The recorded field `peak_rss_gb` is unchanged for Linux.
  The Windows branch has not been run on Windows by the author of this change; see the changelog.

## Caveats
* Fits saved by Colab do not carry over: the run fingerprint includes library versions, so a local run recomputes everything.
* Results from a different environment are valid but record their own library versions in each run's `environment` field.
* Yelp's wide Logistic Regression fit took 3.8 to 7.2 hours on Colab; a full Yelp run on any machine is a many-hour job.
