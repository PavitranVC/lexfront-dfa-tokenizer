# Version 6 — WordPiece and Unigram (P4)

*Status: current. A design addition; v1-v5 stay as they are. Decisions: `docs/DECISIONS.md`, P4 section.*

## Question
Do two other learned subword vocabularies, WordPiece and Unigram, feed the same TF-IDF classifiers better or worse
than NLTK's word tokenizer and than BPE (v5)? No LexFront hybrid is built for them (the hybrid with BPE was worse in P3).

## What changes from v5
| Item | v5 | v6 (this file) |
|---|---|---|
| Tokenizers | + `bpe`, `lexfront_bpe` | + `wordpiece`, `unigram` (trained per dataset) |
| Factory | `build_tokenizer(name, train_texts)` | same; accepts any iterable of texts (AUDIT_5 A5-1) |
| Fingerprint | `subword` record for BPE | + a `subword` record for the two new ones (model, vocabulary size, pre-tokenizer, library version, training-text hash; Unigram also its trainer settings and `score_decimals`) |
| Intrinsic metrics | 4 + 2 tokenizers | 4 + 2 + 2; the hybrid's piece cache is cleared before each timed repetition (A5-4) |
| Main, sensitivity, ablation, BPE, QUICK grids | as in v5 | unchanged: every metric and prediction identical (checked by script against `main`) |

## Protocol
* **Library.** Hugging Face `tokenizers` (Rust core, CPU, no torch): `models.WordPiece` + `trainers.WordPieceTrainer`,
  `models.Unigram` + `trainers.UnigramTrainer`.
* **Training data.** Each tokenizer is trained on the **train split only** of its dataset, lowercased as the vectorizer
  does; the test split is never seen by a trainer (tests check the factory input and that test-only characters are not
  in the vocabulary).
* **Vocabulary size.** 8000 for every dataset, chosen once, not tuned; no second size.
* **Pre-tokenizer and tokens.** `Whitespace` (`\w+|[^\w\s]+`) for both. `[UNK]` marks a character never seen in train
  (WordPiece: the whole word containing it, as WordPiece does; Unigram: the run of unseen characters; the library
  would return the surface string for Unigram, the wrapper names it `[UNK]`). WordPiece marks continuing pieces `##`;
  Unigram has no marker, so a piece is the same feature inside and at the start of a word.
* **WordPiece** picks merges by pair frequency divided by the frequencies of the two parts and encodes by greedy
  longest match; **Unigram** prunes a large candidate set by EM and encodes by the best-scoring split (Viterbi).
  Trainer settings other than the vocabulary size are the library defaults (Unigram's are written into the config:
  `shrinking_factor` 0.75, `max_piece_length` 16, `n_sub_iterations` 2).
* **Determinism.** Neither trainer is deterministic as shipped (4 trainings, 4 different serialized models each,
  at 1 and at 4 threads). WordPiece: same cause and fix as BPE (sorted alphabet passed first as special tokens, then
  removed from the added tokens). Unigram: EM scores differ in the last digits, and the characters the trainer puts
  back after pruning get scores `lowest + k * 0.0001` in hash-set order. `canonical_unigram` gives those characters the
  block's scores in code-point order, rounds scores to 3 decimals and orders the pieces by (score, piece). The check on
  the full train splits (5 trainings each at `RAYON_NUM_THREADS=1` and 4, three datasets) and its limit are in
  `docs/DECISIONS.md`. The thread count is not pinned.
* **Grid.** 3 datasets x {wordpiece, unigram} x {NB, LR}, main setting (TF-IDF `lowercase=True`, `min_df=2`, unigrams),
  seed 0, single-threaded BLAS, full data: `results/subword2/<dataset>__<tokenizer>__<clf>.json` with `predictions/`
  sidecars. QUICK (notebook, 2000 train / 500 test per dataset): `results/quick/subword2/`.
* **Statistics.** `python -m experiments.run_subword2` reuses `experiments/stats.py` and `run_stats.dataset_stats`:
  paired percentile bootstrap (B = 2000, seed 20261016), accuracy and macro-F1, for wordpiece+LR minus nltk+LR,
  unigram+LR minus nltk+LR, wordpiece+LR minus bpe+LR, unigram+LR minus bpe+LR and unigram+LR minus wordpiece+LR; exact
  McNemar test on accuracy for the first two. NLTK+LR is the main-grid run, BPE+LR the P3 run
  (`results/bpe`). Output `results/subword2/stats/bootstrap.json`. No multiplicity correction (5 pairs x 2 metrics per
  dataset).
* **Intrinsic metrics.** OOV rate, fertility, vocabulary size and throughput of both in
  `results/module2/intrinsic__<dataset>.json` (with `trained_vocab_size`). OOV is near zero for them: words split into
  vocabulary pieces, and only characters never seen in train (`[UNK]`) and rare pieces can be OOV, so it is **not
  comparable** with the word-level tokenizers. Throughput depends on the machine.
* **Outputs.** Generated tables `subword_runs` and `subword_table`, observation `subword` (REPORT_NOTES,
  `docs/modules/module2.md`, LEARN viva question 35, notebook section 8).

## Commands
    python -m experiments.run_subword2                  # runs (cached by fingerprint) + stats
    python -m experiments.run_subword2 --stats-only     # stats only, from stored predictions

## What it cannot show
Training randomness (one run per cell; training itself is deterministic); other vocabulary sizes or tuned
trainer settings; Unigram or WordPiece with a word-boundary symbol or another pre-tokenizer; why a difference appears
(any reason is a hypothesis); how the tokenizers would do with a model that uses token order (TF-IDF is a bag of words).
