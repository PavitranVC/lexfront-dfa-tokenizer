# Version 5 — BPE and the LexFront + BPE hybrid (P3)

*Status: current. A design addition; v1-v4 stay as they are. Decisions: `docs/DECISIONS.md`, P3 section.*

## Question
Does a learned subword vocabulary (BPE) help the same TF-IDF classifiers, and does putting LexFront's scanner and
normalizer in front of BPE (structural tokens whole, only word-like tokens split) do better than BPE alone or than
LexFront alone?

## What changes from v4
| Item | v4 | v5 (this file) |
|---|---|---|
| Tokenizers | whitespace, NLTK, TweetTokenizer, LexFront (+ ablation variants) | + `bpe`, `lexfront_bpe` (trained per dataset) |
| Tokenizer construction | a fixed function per name | trained ones via the factory `build_tokenizer(name, train_texts)`; interface still `tokenize(text) -> list[str]` |
| Run fingerprint | config + code | + a `subword` record for trained tokenizers (BPE config, vocab size, `tokenizers` version, hash of the training text); `subword/*.py` in the code hash |
| Main, sensitivity, ablation, QUICK grids | as in v4 | unchanged: every metric and prediction identical (checked by script against `main`) |

## Protocol
* **Library.** Hugging Face `tokenizers` (Rust core, CPU, no torch), `models.BPE` + `trainers.BpeTrainer`.
* **Training data.** Every BPE is trained on the **train split only** of its dataset, lowercased as the vectorizer
  does. The test split is never seen by a trainer (a test checks that the factory receives `ds.train_texts` and that
  characters that occur only in test are not in the vocabulary).
* **Vocabulary size.** 8000 for every dataset in the primary grid, chosen once, not tuned. Sensitivity: 32000 with
  the same code (`results/bpe_v32000/`, names `bpe:v32000`, `lexfront_bpe:v32000`).
* **`bpe`.** `Whitespace` pre-tokenizer (`\w+|[^\w\s]+`: words and runs of punctuation), continuing-subword marker
  `##`, UNK token `[UNK]` (a character never seen in train), applied to lowercased text.
* **`lexfront_bpe`.** LexFront (every normalization rule on) scans and normalizes first. Structural tokens pass
  through whole and are never split: URL / MENTION / NUMBER as `<URL>` / `<USER>` / `<NUM>`, HASHTAG, EMOTICON, EMOJI,
  PUNCT and UNKNOWN. Word-like tokens (WORD, CONTRACTION, ELONGATED, after normalization including slang expansion)
  are split by a BPE trained on the word-like normalized forms of the train split (`WhitespaceSplit` pre-tokenizer,
  so a CONTRACTION such as `don't` stays one pre-token). Hashtags pass through whole (one topic feature, as in
  LexFront); an UNKNOWN character passes through as itself (BPE would turn an unseen one into `[UNK]`).
* **Determinism.** With the `##` marker the trainer assigns ids to the marked alphabet in hash-map order and breaks
  ties between equally frequent pairs by id, so two trainings could differ in merge order (observed before the fix,
  at 1 and 4 threads alike). `subword/bpe.py` passes the alphabet the trainer would build anyway, sorted, as its
  special tokens, then removes them from the added-token list. Tests check: two trainings give the same serialized
  tokenizer; `RAYON_NUM_THREADS=1` and `4` give the same result; the vocabulary set equals the unseeded trainer's.
  The thread count is therefore not pinned.
* **Grid.** 3 datasets x {bpe, lexfront_bpe} x {NB, LR}, main setting (TF-IDF `lowercase=True`, `min_df=2`,
  unigrams), seed 0, single-threaded BLAS, full data. Runs: `results/bpe/<dataset>__<tokenizer>__<clf>.json` with
  `predictions/` sidecars. QUICK (notebook, 2000 train / 500 test per dataset): `results/quick/bpe/`.
* **Statistics.** `python -m experiments.run_bpe` reuses `experiments/stats.py` and `run_stats.dataset_stats`: paired
  percentile bootstrap (B = 2000, seed 20261016), accuracy and macro-F1, for bpe+LR minus nltk+LR (with an exact
  McNemar test on accuracy), lexfront_bpe+LR minus lexfront+LR, lexfront_bpe+LR minus bpe+LR and lexfront_bpe+LR
  minus nltk+LR; NLTK+LR and LexFront+LR are the main-grid runs. Output `results/bpe/stats/bootstrap.json`.
  No multiplicity correction (4 pairs x 2 metrics per dataset).
* **Intrinsic metrics.** OOV rate, fertility, vocabulary size and throughput of both new tokenizers in
  `results/module2/intrinsic__<dataset>.json` (also `trained_vocab_size`). BPE's OOV rate is near zero by
  construction: any word splits into pieces seen in train, so OOV is **not comparable** with the word-level
  tokenizers. Throughput depends on the machine.
* **Outputs.** Generated tables `bpe_runs`, `bpe_table` and observation `bpe` (REPORT_NOTES, `docs/modules/module2.md`,
  LEARN viva question, notebook section 7).

## Commands
    python -m experiments.run_bpe                                          # runs (cached by fingerprint) + stats
    python -m experiments.run_bpe --stats-only                             # stats only, from stored predictions
    python -m experiments.run_bpe --out results/bpe_v32000 --vocab-size 32000

## What it cannot show
Training randomness (one run per cell; BPE training itself is deterministic); other vocabulary sizes than 8000 and
32000 or a tuned size; other subword models (WordPiece and Unigram are P4); why a difference appears (any reason is a
hypothesis); how the tokenizers would do with a model that uses token order (TF-IDF is a bag of words).
