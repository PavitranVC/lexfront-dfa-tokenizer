# REPORT NOTES — Review 1 (Modules 1 and 2)

Everything numeric below is generated from the JSON files in `results/` (see `experiments/render_docs.py`).
Blocks between `BEGIN`/`END` markers are regenerated; the text around them is hand-written.

## 1. What was built

* **Module 1** (`experiments/`): loaders for three Hugging Face datasets, three existing tokenizers (whitespace,
  NLTK `word_tokenize`, `TweetTokenizer`) behind one interface, TF-IDF + {majority, Multinomial NB, Logistic
  Regression}, accuracy and macro-F1, one JSON per run with its full configuration.
* **Module 2** (`lexer/`, `experiments/`): LexFront. Token specification with 11 classes, a PLY lexer (longest match,
  ties to the earlier rule, UNKNOWN recovery), a normalizer with six switchable rules and a 30-entry slang table, a
  symbol table, `LexFront.tokenize()` with the baseline interface, **our own DFA engine** (regex parser, Thompson NFA,
  subset construction, Moore minimization, table-driven scanner) applied to EMOTICON, HASHTAG, MENTION and WORD,
  intrinsic metrics (OOV, fertility, vocabulary, throughput), and LexFront + NB / LR on the three datasets.
* **Tests**: fast pytest suite (`pytest -q`), including 31 handwritten LexFront cases and a DFA-vs-PLY agreement test.
* **Notebook**: `notebooks/LexFront_Review1.ipynb` (Colab-ready, `QUICK_MODE` flag).

## 2. Choices made

* **TF-IDF lowercases first** (`lowercase=True`, sklearn default) for *every* tokenizer in the main grid, so LexFront
  sees lowercased text there; its own lowercase rule is inactive in the main grid by design (P0 decision, kept so
  every main number stays comparable). Intrinsic metrics lowercase the text the same way.
* **Case-sensitivity grid** (`results/sensitivity_case/`, `python -m experiments.run_sensitivity`): the same runs with
  TF-IDF `lowercase=False` for all four tokenizers x {NB, LR}. There LexFront's lowercase rule is active, so LexFront's
  WORD tokens are lowercased while the baselines keep case: a robustness check, not a like-for-like comparison.
* **`min_df=2`** in the TF-IDF vectorizer for all runs (keeps AG News features manageable).
* **Naive Bayes and Logistic Regression on TF-IDF**: NB on TF-IDF is a common variant; the base paper used raw
  counts (`CountVectorizer`). We did not run NB on counts.
* **PLY dispatcher.** PLY's master regex is an ordered alternation (first match wins), not longest match. We register one
  catch-all PLY rule whose body tries every class and keeps the longest match (ties to the earlier class). PLY still
  provides the token stream, positions, ignored whitespace and `t_error`.
* **Python `re` greediness.** Per-class matching uses `re.match`; for our patterns the greedy match equals the longest
  match (checked on the four DFA classes by the agreement test; not proven for the other classes).
* **WORD may contain digits** after the first letter (`gr8`, `covid19`); NUMBER starts with a digit, so `5pm` is NUMBER + WORD.
* **Slang table** is 30 entries; multi-word expansions (`idk`) become several tokens.
* **ELONGATED** and WORD-like classes: lowercase and slang apply to WORD, CONTRACTION and ELONGATED.
* **Moore minimization** instead of Hopcroft (simpler; automata are tiny).
* **DFA scope** as specified: EMOTICON, HASHTAG, MENTION, WORD. ELONGATED left out (back-reference).
* **Splits**: see the datasets table; the TweetTopic choice and its reason are stored in `results/datasets.json` (`split_info`).
* **One training run per cell (seed 0); test-set bootstrap CIs.** Every run stores its test predictions
  (`predictions/<run>.json.gz` next to the run JSON); `python -m experiments.run_stats` resamples the test examples to
  get 95% percentile CIs and paired CIs for differences (`results/stats/bootstrap.json`). Training randomness is not
  covered (one run per cell; the NB and LR fits are deterministic and the full grid uses no subsampling).
* **Every fit runs on one BLAS thread** (`threadpoolctl.threadpool_limits(1)`; recorded as `blas_threads = 1` in each
  run JSON and part of its fingerprint). Without it the same seed gave slightly different metrics on a different
  thread count (see "Reproducibility and leakage checks" below). The determinism claim above holds for this pinned
  setting, with the library versions of the fingerprint.

## 3. Fallbacks used

1. **Network block, resolved.** `huggingface.co` was denied by the sandbox network policy for about an hour; the
   dataset-independent code and tests were written meanwhile. After the host was allowed, all experiments ran on the full data.
2. **TweetTopic loading fallback.** `cardiffnlp/tweet_topic_single` ships a loading script, which `datasets` 5.x refuses to run.
   The loader reads the same files from Hugging Face's auto-converted parquet revision (`refs/convert/parquet`). The dataset
   card confirms the COLING 2022 pair (`train_coling2022` 3,598 / `test_coling2022` 3,399, temporal split); the card sentence is
   stored in `results/datasets.json` under `split_info.card_quote`.
3. **NLTK downloader.** The downloader refused the sandbox proxy by default (SSRF guard); the one-off download command was run with
   NLTK's documented opt-in `NLTK_ALLOW_PROXIED_URLOPEN=1`. This is not in the repo or the notebook (Colab has no proxy).
4. **No subsampling** was needed for the committed results: every run finished well under 10 minutes. Only the notebook's
   `QUICK_MODE` subsamples (2,000 train / 500 test per dataset, seed 0) and writes to `results/quick/`.
5. **DFA regex parser fallback (constructor-function ASTs) was not needed**: the parser works and covers all four classes.
6. **Not run / out of scope**: transformers (need torch); Hinglish; multiple seeds. (The per-rule ablation, BPE + the
   LexFront + BPE hybrid and WordPiece + Unigram are now run: sections below.)

## 4. Tables and DRAFT observations

### Datasets
<!-- BEGIN table:datasets -->
| dataset | source | train split | test split | train docs | test docs | labels | largest test class % | subsample |
|---|---|---|---|---|---|---|---|---|
| tweet_topic | cardiffnlp/tweet_topic_single | train_coling2022 | test_coling2022 | 3598 | 3399 | 6 | 40.01 | none (full data) |
| go_emotions | google-research-datasets/go_emotions (simplified) | train | test | 36308 | 4590 | 28 | 34.99 | none (full data) |
| ag_news | fancyzhx/ag_news | train | test | 120000 | 7600 | 4 | 25.00 | none (full data) |
<!-- END table:datasets -->

<!-- BEGIN obs:datasets -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: 3598 train / 3399 test documents, 6 labels; the largest test class holds 40.01% of the test set.
- **go_emotions**: 36308 train / 4590 test documents, 28 labels; the largest test class holds 34.99% of the test set.
- **ag_news**: 120000 train / 7600 test documents, 4 labels; the largest test class holds 25.00% of the test set.
<!-- END obs:datasets -->

### Module 1 results (accuracy / macro-F1, %)
<!-- BEGIN table:module1_results -->
| dataset | tokenizer | Majority acc / F1 | Naive Bayes acc / F1 | Logistic Reg. acc / F1 |
|---|---|---|---|---|
| tweet_topic | whitespace | 40.01 / 9.53 | 70.26 / 28.83 | 70.49 / 35.76 |
| tweet_topic | nltk | 40.01 / 9.53 | 70.99 / 28.56 | 72.32 / 36.95 |
| tweet_topic | tweet | 40.01 / 9.53 | 70.79 / 28.31 | 72.02 / 36.70 |
| go_emotions | whitespace | 34.99 / 1.85 | 37.73 / 4.93 | 53.51 / 31.40 |
| go_emotions | nltk | 34.99 / 1.85 | 37.78 / 4.88 | 55.27 / 33.86 |
| go_emotions | tweet | 34.99 / 1.85 | 37.80 / 4.86 | 55.47 / 34.02 |
| ag_news | whitespace | 25.00 / 10.00 | 90.14 / 90.12 | 91.43 / 91.42 |
| ag_news | nltk | 25.00 / 10.00 | 90.38 / 90.35 | 91.88 / 91.87 |
| ag_news | tweet | 25.00 / 10.00 | 90.22 / 90.20 | 91.84 / 91.83 |
<!-- END table:module1_results -->

<!-- BEGIN obs:module1 -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: best accuracy 72.32% (nltk + LR). Across the three tokenizers LR is +0.93 points vs NB on average; the tokenizer spread is 1.82 points under LR and 0.74 points under NB.
- **go_emotions**: best accuracy 55.47% (tweet + LR). Across the three tokenizers LR is +16.98 points vs NB on average; the tokenizer spread is 1.96 points under LR and 0.07 points under NB.
- **ag_news**: best accuracy 91.88% (nltk + LR). Across the three tokenizers LR is +1.47 points vs NB on average; the tokenizer spread is 0.45 points under LR and 0.24 points under NB.

_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file `stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score moves with the draw of the test set, not with training randomness, which one run per cell cannot show (the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed subsample). A gap whose CI includes 0 should not be read as a real difference. Gaps not in `ci_table` (e.g. spreads between baselines, LR vs NB) have no paired CI and are untested; overlapping per-run CIs (`ci_runs`) are not a test of a difference._
<!-- END obs:module1 -->

### Existing (NLTK + NB) vs proposed (LexFront + LR)
<!-- BEGIN table:existing_vs_proposed -->
| dataset | existing NLTK+NB acc / F1 | proposed LexFront+LR acc / F1 | delta acc (pts) | delta F1 (pts) | LexFront+NB acc / F1 | NLTK+LR acc / F1 |
|---|---|---|---|---|---|---|
| tweet_topic | 70.99 / 28.56 | 70.73 / 36.18 | -0.26 | +7.62 | 70.43 / 27.84 | 72.32 / 36.95 |
| go_emotions | 37.78 / 4.88 | 55.36 / 33.97 | +17.58 | +29.09 | 37.78 / 4.81 | 55.27 / 33.86 |
| ag_news | 90.38 / 90.35 | 91.88 / 91.86 | +1.50 | +1.51 | 90.08 / 90.05 | 91.88 / 91.87 |
<!-- END table:existing_vs_proposed -->

<!-- BEGIN obs:existing_vs_proposed -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: proposed (LexFront+LR) vs existing (NLTK+NB): accuracy -0.26 points, macro-F1 +7.62 points. Split of the accuracy change: switching classifier with NLTK held fixed (NB to LR) +1.32; switching tokenizer with LR held fixed (NLTK to LexFront) -1.59; switching tokenizer with NB held fixed -0.56.
- **go_emotions**: proposed (LexFront+LR) vs existing (NLTK+NB): accuracy +17.58 points, macro-F1 +29.09 points. Split of the accuracy change: switching classifier with NLTK held fixed (NB to LR) +17.49; switching tokenizer with LR held fixed (NLTK to LexFront) +0.09; switching tokenizer with NB held fixed +0.00.
- **ag_news**: proposed (LexFront+LR) vs existing (NLTK+NB): accuracy +1.50 points, macro-F1 +1.51 points. Split of the accuracy change: switching classifier with NLTK held fixed (NB to LR) +1.50; switching tokenizer with LR held fixed (NLTK to LexFront) +0.00; switching tokenizer with NB held fixed -0.30.

_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file `stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score moves with the draw of the test set, not with training randomness, which one run per cell cannot show (the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed subsample). A gap whose CI includes 0 should not be read as a real difference. Gaps not in `ci_table` (e.g. spreads between baselines, LR vs NB) have no paired CI and are untested; overlapping per-run CIs (`ci_runs`) are not a test of a difference._
<!-- END obs:existing_vs_proposed -->

### Intrinsic metrics
<!-- BEGIN table:intrinsic_metrics -->
| dataset | tokenizer | OOV rate % (tokens) | OOV rate % (types) | fertility | vocab size | throughput (tokens/s) |
|---|---|---|---|---|---|---|
| tweet_topic | whitespace | 18.45 | 65.83 | 1.000 | 21653 | 26,769,229 |
| tweet_topic | nltk | 8.74 | 58.23 | 1.431 | 15712 | 243,754 |
| tweet_topic | tweet | 9.86 | 59.72 | 1.376 | 16524 | 295,726 |
| tweet_topic | lexfront | 8.25 | 57.13 | 1.391 | 14483 | 290,545 |
| tweet_topic | bpe | 0.44 | 2.05 | 1.633 | 7413 | 1,039,367 |
| tweet_topic | lexfront_bpe | 2.20 | 26.89 | 1.578 | 10101 | 251,860 |
| tweet_topic | wordpiece | 0.31 | 2.39 | 1.618 | 7267 | 1,266,539 |
| tweet_topic | unigram | 0.26 | 2.12 | 1.834 | 7020 | 927,536 |
| go_emotions | whitespace | 6.04 | 30.11 | 1.000 | 44476 | 25,609,705 |
| go_emotions | nltk | 2.57 | 21.55 | 1.220 | 26410 | 190,698 |
| go_emotions | tweet | 2.56 | 20.84 | 1.195 | 25907 | 273,224 |
| go_emotions | lexfront | 2.45 | 19.89 | 1.195 | 24459 | 281,843 |
| go_emotions | bpe | 0.18 | 0.23 | 1.377 | 7704 | 942,109 |
| go_emotions | lexfront_bpe | 0.23 | 1.97 | 1.311 | 8236 | 237,066 |
| go_emotions | wordpiece | 0.03 | 0.30 | 1.373 | 7585 | 1,155,009 |
| go_emotions | unigram | 0.04 | 0.07 | 1.516 | 7898 | 786,663 |
| ag_news | whitespace | 1.90 | 14.97 | 1.000 | 158733 | 24,396,944 |
| ag_news | nltk | 0.96 | 11.52 | 1.158 | 98994 | 340,852 |
| ag_news | tweet | 0.75 | 9.13 | 1.154 | 86096 | 268,415 |
| ag_news | lexfront | 0.48 | 6.78 | 1.207 | 65337 | 286,773 |
| ag_news | bpe | 0.00 | 0.01 | 1.435 | 7916 | 1,065,421 |
| ag_news | lexfront_bpe | 0.00 | 0.01 | 1.408 | 7989 | 283,587 |
| ag_news | wordpiece | 0.00 | 0.07 | 1.433 | 7806 | 1,096,802 |
| ag_news | unigram | 0.00 | 0.00 | 1.609 | 7998 | 949,478 |
<!-- END table:intrinsic_metrics -->

<!-- BEGIN obs:intrinsic -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: lowest OOV rate among the word-level tokenizers 8.25% (lexfront); smallest vocabulary 14483 (lexfront); LexFront: OOV 8.25%, fertility 1.391, vocabulary 14483; LexFront throughput is 1.19x NLTK's (tokens/s, this machine). Subword tokenizers (trained on the train split; they split words into vocabulary pieces, so only characters never seen in train (`[UNK]`) and rare pieces can be OOV: the OOV rate is near zero and not comparable with the word-level ones above; LexFront + BPE splits only word-like tokens, so its OOV tokens are structural tokens passed through whole, `[UNK]`, or word pieces unseen in train): bpe: OOV 0.44%, fertility 1.633, vocabulary 7413; lexfront_bpe: OOV 2.20%, fertility 1.578, vocabulary 10101; wordpiece: OOV 0.31%, fertility 1.618, vocabulary 7267; unigram: OOV 0.26%, fertility 1.834, vocabulary 7020.
- **go_emotions**: lowest OOV rate among the word-level tokenizers 2.45% (lexfront); smallest vocabulary 24459 (lexfront); LexFront: OOV 2.45%, fertility 1.195, vocabulary 24459; LexFront throughput is 1.48x NLTK's (tokens/s, this machine). Subword tokenizers (trained on the train split; they split words into vocabulary pieces, so only characters never seen in train (`[UNK]`) and rare pieces can be OOV: the OOV rate is near zero and not comparable with the word-level ones above; LexFront + BPE splits only word-like tokens, so its OOV tokens are structural tokens passed through whole, `[UNK]`, or word pieces unseen in train): bpe: OOV 0.18%, fertility 1.377, vocabulary 7704; lexfront_bpe: OOV 0.23%, fertility 1.311, vocabulary 8236; wordpiece: OOV 0.03%, fertility 1.373, vocabulary 7585; unigram: OOV 0.04%, fertility 1.516, vocabulary 7898.
- **ag_news**: lowest OOV rate among the word-level tokenizers 0.48% (lexfront); smallest vocabulary 65337 (lexfront); LexFront: OOV 0.48%, fertility 1.207, vocabulary 65337; LexFront throughput is 0.84x NLTK's (tokens/s, this machine). Subword tokenizers (trained on the train split; they split words into vocabulary pieces, so only characters never seen in train (`[UNK]`) and rare pieces can be OOV: the OOV rate is near zero and not comparable with the word-level ones above; LexFront + BPE splits only word-like tokens, so its OOV tokens are structural tokens passed through whole, `[UNK]`, or word pieces unseen in train): bpe: OOV 0.00%, fertility 1.435, vocabulary 7916; lexfront_bpe: OOV 0.00%, fertility 1.408, vocabulary 7989; wordpiece: OOV 0.00%, fertility 1.433, vocabulary 7806; unigram: OOV 0.00%, fertility 1.609, vocabulary 7998.
<!-- END obs:intrinsic -->

### DFA state counts
<!-- BEGIN table:dfa_states -->
| automaton | alphabet atoms | NFA states | DFA states | minimized DFA states |
|---|---|---|---|---|
| MENTION | 5 | 9 | 3 | 3 |
| HASHTAG | 8 | 9 | 3 | 3 |
| EMOTICON | 17 | 19 | 6 | 5 |
| WORD | 6 | 9 | 3 | 2 |
| combined (MENTION+HASHTAG+EMOTICON+WORD) | 31 | 43 | 12 | 10 |
| ELONGATED, 26 spelled-out branches (extra, not in LexFront) | 26 | 221 | 157 | 54 |
<!-- END table:dfa_states -->

<!-- BEGIN obs:dfa -->
**DRAFT observations (generated from the JSON files):**

- Combined automaton for MENTION, HASHTAG, EMOTICON, WORD: 43 NFA states -> 12 DFA states -> 10 after minimization (31 alphabet atoms).
- Per class (NFA / DFA / minimized): MENTION 9/3/3; HASHTAG 9/3/3; EMOTICON 19/6/5; WORD 9/3/2.
- ELONGATED is written with a back-reference (`[A-Za-z]*([A-Za-z])\1{2,}[A-Za-z]*`) and is not accepted by our regex parser, so it is not in the DFA engine. Spelled out as 26 alternatives (lowercase only) it compiles to 221 NFA / 157 DFA / 54 minimized states.
<!-- END obs:dfa -->

Reading the tables together (hand-written, still DRAFT): among the word-level tokenizers (whitespace, NLTK,
TweetTokenizer, LexFront) LexFront has the lowest OOV rate and the smallest vocabulary on all three datasets (the BPE rows
have an even smaller vocabulary and near-zero OOV by construction, so they are not comparable), but part of that is by construction (placeholders such as `<URL>`/`<NUM>`, elongation collapsing and slang
expansion merge strings), so it is not by itself evidence of better classification. Where the proposed-vs-existing
accuracy change comes from differs per dataset (generated below from the JSON; whether each difference is larger than
test-set noise is in the confidence-interval section that follows):

<!-- BEGIN obs:attribution -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: proposed is 0.26 points **below** existing (classifier switch NB to LR +1.32, tokenizer switch NLTK to LexFront under LR -1.59).
- **go_emotions**: proposed is 17.58 points above existing; almost all of it comes from the classifier (classifier switch NB to LR +17.49, tokenizer switch NLTK to LexFront under LR +0.09).
- **ag_news**: proposed is 1.50 points above existing; almost all of it comes from the classifier (classifier switch NB to LR +1.50, tokenizer switch NLTK to LexFront under LR +0.00).

_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file `stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score moves with the draw of the test set, not with training randomness, which one run per cell cannot show (the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed subsample). A gap whose CI includes 0 should not be read as a real difference. Gaps not in `ci_table` (e.g. spreads between baselines, LR vs NB) have no paired CI and are untested; overlapping per-run CIs (`ci_runs`) are not a test of a difference._
<!-- END obs:attribution -->

### Confidence intervals (test-set bootstrap)
Paired differences, LexFront+LR minus each baseline, in points; percentile bootstrap over the test examples with the
settings recorded in `results/stats/bootstrap.json` (`config`). McNemar is computed for the headline tokenizer-only pair only.

<!-- BEGIN table:ci_table -->
| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | LexFront+LR minus NLTK+LR (tokenizer only) | -1.59 [-2.38, -0.82] | -0.77 [-1.81, +0.32] | yes / no | < 0.001 |
| tweet_topic | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | -1.29 [-2.00, -0.59] | -0.52 [-1.58, +0.57] | yes / no | - |
| tweet_topic | LexFront+LR minus NLTK+NB (proposed minus existing) | -0.26 [-1.47, +0.91] | +7.62 [+5.93, +9.31] | no / yes | - |
| go_emotions | LexFront+LR minus NLTK+LR (tokenizer only) | +0.09 [-0.35, +0.54] | +0.11 [-0.51, +0.77] | no / no | 0.773 |
| go_emotions | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | -0.11 [-0.46, +0.22] | -0.05 [-0.55, +0.47] | no / no | - |
| go_emotions | LexFront+LR minus NLTK+NB (proposed minus existing) | +17.58 [+16.21, +18.95] | +29.09 [+27.32, +31.07] | yes / yes | - |
| ag_news | LexFront+LR minus NLTK+LR (tokenizer only) | +0.00 [-0.32, +0.33] | -0.01 [-0.33, +0.33] | no / no | 1.000 |
| ag_news | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | +0.04 [-0.24, +0.32] | +0.04 [-0.24, +0.31] | no / no | - |
| ag_news | LexFront+LR minus NLTK+NB (proposed minus existing) | +1.50 [+1.00, +1.99] | +1.51 [+1.01, +2.01] | yes / yes | - |
<!-- END table:ci_table -->

<!-- BEGIN obs:ci -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**, LexFront+LR minus NLTK+LR (tokenizer only): accuracy -1.59 points (95% CI -2.38 to -0.82, excludes 0); macro-F1 -0.77 points (95% CI -1.81 to +0.32, includes 0). Exact McNemar test on accuracy: p < 0.001 (discordant test examples: right only with nltk+lr 124, right only with lexfront+lr 70).
- **tweet_topic**, LexFront+LR minus TweetTokenizer+LR (tokenizer only): accuracy -1.29 points (95% CI -2.00 to -0.59, excludes 0); macro-F1 -0.52 points (95% CI -1.58 to +0.57, includes 0).
- **tweet_topic**, LexFront+LR minus NLTK+NB (proposed minus existing): accuracy -0.26 points (95% CI -1.47 to +0.91, includes 0); macro-F1 +7.62 points (95% CI +5.93 to +9.31, excludes 0).
- **go_emotions**, LexFront+LR minus NLTK+LR (tokenizer only): accuracy +0.09 points (95% CI -0.35 to +0.54, includes 0); macro-F1 +0.11 points (95% CI -0.51 to +0.77, includes 0). Exact McNemar test on accuracy: p = 0.773 (discordant test examples: right only with nltk+lr 52, right only with lexfront+lr 56).
- **go_emotions**, LexFront+LR minus TweetTokenizer+LR (tokenizer only): accuracy -0.11 points (95% CI -0.46 to +0.22, includes 0); macro-F1 -0.05 points (95% CI -0.55 to +0.47, includes 0).
- **go_emotions**, LexFront+LR minus NLTK+NB (proposed minus existing): accuracy +17.58 points (95% CI +16.21 to +18.95, excludes 0); macro-F1 +29.09 points (95% CI +27.32 to +31.07, excludes 0).
- **ag_news**, LexFront+LR minus NLTK+LR (tokenizer only): accuracy +0.00 points (95% CI -0.32 to +0.33, includes 0); macro-F1 -0.01 points (95% CI -0.33 to +0.33, includes 0). Exact McNemar test on accuracy: p = 1.000 (discordant test examples: right only with nltk+lr 81, right only with lexfront+lr 81).
- **ag_news**, LexFront+LR minus TweetTokenizer+LR (tokenizer only): accuracy +0.04 points (95% CI -0.24 to +0.32, includes 0); macro-F1 +0.04 points (95% CI -0.24 to +0.31, includes 0).
- **ag_news**, LexFront+LR minus NLTK+NB (proposed minus existing): accuracy +1.50 points (95% CI +1.00 to +1.99, excludes 0); macro-F1 +1.51 points (95% CI +1.01 to +2.01, excludes 0).
- Tokenizer-only comparison (LexFront+LR vs NLTK+LR): 1 of 6 difference CIs (accuracy and macro-F1, all datasets) exclude 0.
- Case-sensitivity grid: whether the tokenizer-only accuracy CI excludes 0 differs between the two grids on 2 of 3 datasets (tweet_topic, go_emotions): the tokenizer-only result depends on how case is handled.
- Settings: percentile bootstrap over test examples, B = 2000, seed 20261016, 95% percentile intervals.

_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file `stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score moves with the draw of the test set, not with training randomness, which one run per cell cannot show (the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed subsample). A gap whose CI includes 0 should not be read as a real difference. Gaps not in `ci_table` (e.g. spreads between baselines, LR vs NB) have no paired CI and are untested; overlapping per-run CIs (`ci_runs`) are not a test of a difference._
<!-- END obs:ci -->

Per-run 95% CIs (NB and LR; the majority runs are in the JSON):

<!-- BEGIN table:ci_runs -->
| dataset | tokenizer | classifier | accuracy % [95% CI] | macro-F1 % [95% CI] |
|---|---|---|---|---|
| tweet_topic | whitespace | Naive Bayes | 70.26 [68.76, 71.76] | 28.83 [27.88, 29.74] |
| tweet_topic | whitespace | Logistic Reg. | 70.49 [68.87, 72.08] | 35.76 [34.24, 37.37] |
| tweet_topic | nltk | Naive Bayes | 70.99 [69.52, 72.49] | 28.56 [27.76, 29.44] |
| tweet_topic | nltk | Logistic Reg. | 72.32 [70.90, 73.87] | 36.95 [35.34, 38.58] |
| tweet_topic | tweet | Naive Bayes | 70.79 [69.29, 72.29] | 28.31 [27.53, 29.17] |
| tweet_topic | tweet | Logistic Reg. | 72.02 [70.58, 73.61] | 36.70 [35.13, 38.28] |
| tweet_topic | lexfront | Naive Bayes | 70.43 [68.90, 71.93] | 27.84 [27.12, 28.62] |
| tweet_topic | lexfront | Logistic Reg. | 70.73 [69.23, 72.29] | 36.18 [34.64, 37.84] |
| go_emotions | whitespace | Naive Bayes | 37.73 [36.34, 39.15] | 4.93 [4.47, 5.47] |
| go_emotions | whitespace | Logistic Reg. | 53.51 [52.11, 54.95] | 31.40 [29.73, 33.40] |
| go_emotions | nltk | Naive Bayes | 37.78 [36.38, 39.19] | 4.88 [4.42, 5.39] |
| go_emotions | nltk | Logistic Reg. | 55.27 [53.81, 56.71] | 33.86 [32.07, 35.92] |
| go_emotions | tweet | Naive Bayes | 37.80 [36.43, 39.24] | 4.86 [4.41, 5.37] |
| go_emotions | tweet | Logistic Reg. | 55.47 [54.03, 56.91] | 34.02 [32.24, 36.04] |
| go_emotions | lexfront | Naive Bayes | 37.78 [36.40, 39.19] | 4.81 [4.37, 5.32] |
| go_emotions | lexfront | Logistic Reg. | 55.36 [53.97, 56.80] | 33.97 [32.12, 36.02] |
| ag_news | whitespace | Naive Bayes | 90.14 [89.49, 90.79] | 90.12 [89.47, 90.75] |
| ag_news | whitespace | Logistic Reg. | 91.43 [90.80, 92.04] | 91.42 [90.79, 92.02] |
| ag_news | nltk | Naive Bayes | 90.38 [89.72, 91.01] | 90.35 [89.70, 90.98] |
| ag_news | nltk | Logistic Reg. | 91.88 [91.25, 92.46] | 91.87 [91.23, 92.44] |
| ag_news | tweet | Naive Bayes | 90.22 [89.54, 90.87] | 90.20 [89.53, 90.85] |
| ag_news | tweet | Logistic Reg. | 91.84 [91.22, 92.43] | 91.83 [91.20, 92.41] |
| ag_news | lexfront | Naive Bayes | 90.08 [89.39, 90.74] | 90.05 [89.37, 90.69] |
| ag_news | lexfront | Logistic Reg. | 91.88 [91.26, 92.47] | 91.86 [91.23, 92.44] |
<!-- END table:ci_runs -->

### Case-sensitivity check (TF-IDF `lowercase=False`)
All four tokenizers with case folding off in TF-IDF. LexFront's own lowercase rule is active here, so LexFront's WORD
tokens are lowercased while the baselines keep case: this is a robustness check of the main result, not a like-for-like
comparison. The last two columns compare each LR cell with the same cell of the main grid (`lowercase=True`).

<!-- BEGIN table:sensitivity_case -->
| dataset | tokenizer | Naive Bayes acc / F1 (case kept) | Logistic Reg. acc / F1 (case kept) | LR acc change vs main (pts) | LR F1 change vs main (pts) |
|---|---|---|---|---|---|
| tweet_topic | whitespace | 69.73 / 28.81 | 69.58 / 34.58 | -0.91 | -1.18 |
| tweet_topic | nltk | 71.08 / 28.63 | 71.61 / 36.04 | -0.71 | -0.91 |
| tweet_topic | tweet | 70.55 / 28.56 | 71.43 / 35.86 | -0.59 | -0.84 |
| tweet_topic | lexfront | 70.43 / 27.99 | 70.70 / 36.27 | -0.03 | +0.09 |
| go_emotions | whitespace | 37.78 / 4.95 | 52.64 / 29.80 | -0.87 | -1.61 |
| go_emotions | nltk | 37.89 / 5.06 | 54.18 / 32.31 | -1.09 | -1.55 |
| go_emotions | tweet | 37.93 / 5.03 | 54.20 / 32.33 | -1.26 | -1.69 |
| go_emotions | lexfront | 37.78 / 4.81 | 55.34 / 34.03 | -0.02 | +0.06 |
| ag_news | whitespace | 90.22 / 90.19 | 91.59 / 91.57 | +0.16 | +0.16 |
| ag_news | nltk | 90.47 / 90.44 | 92.11 / 92.09 | +0.22 | +0.22 |
| ag_news | tweet | 90.42 / 90.39 | 92.08 / 92.06 | +0.24 | +0.24 |
| ag_news | lexfront | 90.08 / 90.05 | 91.92 / 91.90 | +0.04 | +0.04 |
<!-- END table:sensitivity_case -->

Paired bootstrap CIs in this grid (`python -m experiments.run_stats --results results/sensitivity_case`, same code, B and seed as the main grid), next to the same pair in the main grid:

<!-- BEGIN table:sensitivity_ci -->
| dataset | difference | acc pts [95% CI], case kept | acc pts [95% CI], main | F1 pts [95% CI], case kept | F1 pts [95% CI], main | McNemar p (acc), case kept |
|---|---|---|---|---|---|---|
| tweet_topic | LexFront+LR minus NLTK+LR (tokenizer only) | -0.91 [-1.88, +0.09] | -1.59 [-2.38, -0.82] | +0.23 [-0.90, +1.40] | -0.77 [-1.81, +0.32] | 0.073 |
| tweet_topic | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | -0.74 [-1.65, +0.18] | -1.29 [-2.00, -0.59] | +0.41 [-0.77, +1.68] | -0.52 [-1.58, +0.57] | - |
| tweet_topic | LexFront+LR minus NLTK+NB (proposed minus existing) | -0.38 [-1.56, +0.82] | -0.26 [-1.47, +0.91] | +7.64 [+5.96, +9.44] | +7.62 [+5.93, +9.31] | - |
| go_emotions | LexFront+LR minus NLTK+LR (tokenizer only) | +1.15 [+0.46, +1.85] | +0.09 [-0.35, +0.54] | +1.72 [+0.56, +2.97] | +0.11 [-0.51, +0.77] | < 0.001 |
| go_emotions | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | +1.13 [+0.48, +1.81] | -0.11 [-0.46, +0.22] | +1.70 [+0.58, +2.91] | -0.05 [-0.55, +0.47] | - |
| go_emotions | LexFront+LR minus NLTK+NB (proposed minus existing) | +17.45 [+16.08, +18.82] | +17.58 [+16.21, +18.95] | +28.97 [+27.22, +30.95] | +29.09 [+27.32, +31.07] | - |
| ag_news | LexFront+LR minus NLTK+LR (tokenizer only) | -0.18 [-0.55, +0.20] | +0.00 [-0.32, +0.33] | -0.19 [-0.56, +0.19] | -0.01 [-0.33, +0.33] | 0.353 |
| ag_news | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | -0.16 [-0.50, +0.18] | +0.04 [-0.24, +0.32] | -0.16 [-0.51, +0.18] | +0.04 [-0.24, +0.31] | - |
| ag_news | LexFront+LR minus NLTK+NB (proposed minus existing) | +1.45 [+0.93, +1.92] | +1.50 [+1.00, +1.99] | +1.46 [+0.95, +1.94] | +1.51 [+1.01, +2.01] | - |
<!-- END table:sensitivity_ci -->

<!-- BEGIN obs:sensitivity_case -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: LexFront+LR minus NLTK+LR accuracy is -0.91 points with case kept vs -1.59 in the main grid; LR accuracy change from switching TF-IDF case folding off: whitespace -0.91, nltk -0.71, tweet -0.59, lexfront -0.03.
- **go_emotions**: LexFront+LR minus NLTK+LR accuracy is +1.15 points with case kept vs +0.09 in the main grid; LR accuracy change from switching TF-IDF case folding off: whitespace -0.87, nltk -1.09, tweet -1.26, lexfront -0.02.
- **ag_news**: LexFront+LR minus NLTK+LR accuracy is -0.18 points with case kept vs +0.00 in the main grid; LR accuracy change from switching TF-IDF case folding off: whitespace +0.16, nltk +0.22, tweet +0.24, lexfront +0.04.
- **tweet_topic**, LexFront+LR minus NLTK+LR accuracy with case kept: -0.91 points (95% CI -1.88 to +0.09, includes 0); main grid: -1.59 points (95% CI -2.38 to -0.82, excludes 0). Exact McNemar p with case kept = 0.073.
- **go_emotions**, LexFront+LR minus NLTK+LR accuracy with case kept: +1.15 points (95% CI +0.46 to +1.85, excludes 0); main grid: +0.09 points (95% CI -0.35 to +0.54, includes 0). Exact McNemar p with case kept < 0.001.
- **ag_news**, LexFront+LR minus NLTK+LR accuracy with case kept: -0.18 points (95% CI -0.55 to +0.20, includes 0); main grid: +0.00 points (95% CI -0.32 to +0.33, includes 0). Exact McNemar p with case kept = 0.353.
- Whether the tokenizer-only accuracy CI excludes 0 differs between the two grids on 2 of 3 datasets (tweet_topic, go_emotions): the tokenizer-only result depends on how case is handled.

_Robustness check, not a like-for-like comparison: with TF-IDF case folding off, LexFront lowercases its own WORD tokens while the baselines keep case. One training run per cell; the CIs (table `sensitivity_ci`, file `sensitivity_case/stats/bootstrap.json`) are test-set bootstraps with the same code, B and seed as the main grid._
<!-- END obs:sensitivity_case -->

Finding: the usual ELONGATED regex `(.)\1{2,}` needs a back-reference, which is outside regular expressions, so it is
not in the DFA engine. For a finite alphabet and run length 3 the language is still regular when spelled out letter by
letter; the last row of the DFA table is that spelled-out version (lowercase only), shown for comparison.

### Which normalization rule matters? (per-rule ablation)
LexFront + LR with one normalization rule switched off at a time, everything else fixed
(`python -m experiments.run_ablation`, `results/ablation/`, protocol in `docs/versions/v4-ablation.md`). Main setting
(TF-IDF lowercases, as the main grid): no_url, no_mention, no_number, no_elongated, no_slang, and all_off (the lexer's
segmentation only; TF-IDF still lowercases). The lowercase rule cannot have an effect there, because TF-IDF lowercases the text before LexFront
sees it; it is tested in the case-kept setting (TF-IDF `lowercase=False`, as the sensitivity grid): no_lowercase and
all_off_cs against full_cs. Differences are variant minus full LexFront of the same setting, paired bootstrap over the
test examples (same code, B and seed as `ci_table`), with an exact McNemar test on accuracy. The table says which
removals change the scores; it does not say why. The column "rule fires" counts the train / test documents in which
the removed rule changes at least one token: a variant whose rule never fires (0 / 0) is the full system itself and is
not a test of the rule, so the observation lists it apart and leaves it out of the counts.

<!-- BEGIN table:ablation_table -->
| dataset | setting | variant | rule fires, docs train / test | TF-IDF features | acc / F1 % | acc minus full, pts [95% CI] | F1 minus full, pts [95% CI] | CI excludes 0 (acc / F1) | predictions changed | McNemar p (acc) |
|---|---|---|---|---|---|---|---|---|---|---|
| tweet_topic | TF-IDF lowercase (main grid) | full (reference) | - | 5350 | 70.73 / 36.18 | - | - | - | - | - |
| tweet_topic | TF-IDF lowercase (main grid) | no_url | 0 / 0 | 5350 | 70.73 / 36.18 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| tweet_topic | TF-IDF lowercase (main grid) | no_mention | 2414 / 2416 | 5776 | 71.14 / 35.94 | +0.41 [-0.03, +0.82] | -0.24 [-1.07, +0.45] | no / no | 66 | 0.070 |
| tweet_topic | TF-IDF lowercase (main grid) | no_number | 1165 / 1172 | 5461 | 71.23 / 36.16 | +0.50 [-0.03, +1.00] | -0.02 [-0.57, +0.43] | no / no | 130 | 0.068 |
| tweet_topic | TF-IDF lowercase (main grid) | no_elongated | 89 / 76 | 5347 | 70.81 / 36.23 | +0.09 [-0.09, +0.26] | +0.05 [-0.09, +0.19] | no / no | 9 | 0.508 |
| tweet_topic | TF-IDF lowercase (main grid) | no_slang | 247 / 221 | 5373 | 70.67 / 36.04 | -0.06 [-0.35, +0.24] | -0.14 [-0.75, +0.45] | no / no | 34 | 0.845 |
| tweet_topic | TF-IDF lowercase (main grid) | all_off (TF-IDF still lowercases) | 2856 / 2815 | 5907 | 71.82 / 36.37 | +1.09 [+0.50, +1.71] | +0.19 [-0.61, +0.94] | yes / no | 172 | < 0.001 |
| tweet_topic | case kept (TF-IDF lowercase=False) | full_cs (reference) | - | 5313 | 70.70 / 36.27 | - | - | - | - | - |
| tweet_topic | case kept (TF-IDF lowercase=False) | no_lowercase | 3584 / 3381 | 6236 | 69.79 / 35.37 | -0.91 [-1.71, -0.09] | -0.90 [-1.93, +0.04] | yes / no | 288 | 0.036 |
| tweet_topic | case kept (TF-IDF lowercase=False) | all_off_cs | 3598 / 3399 | 6791 | 71.11 / 35.65 | +0.41 [-0.44, +1.27] | -0.62 [-1.89, +0.53] | no / no | 340 | 0.387 |
| go_emotions | TF-IDF lowercase (main grid) | full (reference) | - | 12035 | 55.36 / 33.97 | - | - | - | - | - |
| go_emotions | TF-IDF lowercase (main grid) | no_url | 0 / 0 | 12035 | 55.36 / 33.97 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_mention | 6 / 0 | 12035 | 55.34 / 33.98 | -0.02 [-0.13, +0.07] | +0.01 [-0.05, +0.09] | no / no | 7 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_number | 2963 / 389 | 12218 | 55.25 / 33.91 | -0.11 [-0.26, +0.04] | -0.06 [-0.29, +0.18] | no / no | 23 | 0.267 |
| go_emotions | TF-IDF lowercase (main grid) | no_elongated | 524 / 72 | 12039 | 55.36 / 34.03 | +0.00 [-0.11, +0.13] | +0.05 [-0.09, +0.22] | no / no | 12 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_slang | 1646 / 190 | 12061 | 55.32 / 33.96 | -0.04 [-0.24, +0.15] | -0.02 [-0.27, +0.25] | no / no | 33 | 0.832 |
| go_emotions | TF-IDF lowercase (main grid) | all_off (TF-IDF still lowercases) | 4864 / 622 | 12248 | 55.19 / 33.85 | -0.17 [-0.41, +0.07] | -0.13 [-0.44, +0.19] | no / no | 50 | 0.215 |
| go_emotions | case kept (TF-IDF lowercase=False) | full_cs (reference) | - | 12035 | 55.34 / 34.03 | - | - | - | - | - |
| go_emotions | case kept (TF-IDF lowercase=False) | no_lowercase | 34555 / 4389 | 13854 | 54.27 / 32.13 | -1.07 [-1.72, -0.44] | -1.90 [-3.08, -0.79] | yes / yes | 313 | 0.002 |
| go_emotions | case kept (TF-IDF lowercase=False) | all_off_cs | 34839 / 4420 | 14056 | 54.14 / 32.17 | -1.20 [-1.85, -0.54] | -1.86 [-3.07, -0.73] | yes / yes | 322 | < 0.001 |
| ag_news | TF-IDF lowercase (main grid) | full (reference) | - | 41853 | 91.88 / 91.86 | - | - | - | - | - |
| ag_news | TF-IDF lowercase (main grid) | no_url | 1833 / 99 | 42166 | 91.89 / 91.88 | +0.01 [-0.07, +0.09] | +0.01 [-0.07, +0.09] | no / no | 11 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | no_mention | 0 / 0 | 41853 | 91.88 / 91.86 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | no_number | 51541 / 3273 | 43561 | 91.95 / 91.93 | +0.07 [-0.07, +0.20] | +0.07 [-0.06, +0.20] | no / no | 30 | 0.424 |
| ag_news | TF-IDF lowercase (main grid) | no_elongated | 249 / 13 | 41861 | 91.92 / 91.90 | +0.04 [-0.04, +0.12] | +0.04 [-0.04, +0.12] | no / no | 10 | 0.508 |
| ag_news | TF-IDF lowercase (main grid) | no_slang | 7174 / 442 | 41869 | 91.91 / 91.89 | +0.03 [-0.03, +0.09] | +0.03 [-0.03, +0.09] | no / no | 9 | 0.688 |
| ag_news | TF-IDF lowercase (main grid) | all_off (TF-IDF still lowercases) | 56554 / 3576 | 43898 | 91.99 / 91.97 | +0.11 [-0.04, +0.25] | +0.11 [-0.04, +0.25] | no / no | 36 | 0.200 |
| ag_news | case kept (TF-IDF lowercase=False) | full_cs (reference) | - | 41855 | 91.92 / 91.90 | - | - | - | - | - |
| ag_news | case kept (TF-IDF lowercase=False) | no_lowercase | 120000 / 7600 | 55452 | 92.07 / 92.05 | +0.14 [-0.17, +0.46] | +0.14 [-0.17, +0.46] | no / no | 165 | 0.406 |
| ag_news | case kept (TF-IDF lowercase=False) | all_off_cs | 120000 / 7600 | 57491 | 92.09 / 92.08 | +0.17 [-0.16, +0.49] | +0.17 [-0.16, +0.49] | no / no | 171 | 0.326 |
<!-- END table:ablation_table -->

<!-- BEGIN obs:ablation -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic** (7 variants vs full): accuracy CI excludes 0 for all_off (TF-IDF still lowercases) +1.09 (higher than full), no_lowercase (case kept) -0.91 (lower than full); macro-F1 CI excludes 0 for none. Every other compared interval includes 0. Rule never fires on this data (train and test), so not a test of the rule: no_url (identical to full, left out of the counts).
- **go_emotions** (7 variants vs full): accuracy CI excludes 0 for no_lowercase (case kept) -1.07 (lower than full), all_off_cs (case kept) -1.20 (lower than full); macro-F1 CI excludes 0 for no_lowercase (case kept) -1.90 (lower than full), all_off_cs (case kept) -1.86 (lower than full). Every other compared interval includes 0. Rule never fires on this data (train and test), so not a test of the rule: no_url (identical to full, left out of the counts). Fires in train only, never in test: no_mention (6 train documents).
- **ag_news** (7 variants vs full): accuracy CI excludes 0 for none; macro-F1 CI excludes 0 for none. Every other compared interval includes 0. Rule never fires on this data (train and test), so not a test of the rule: no_mention (identical to full, left out of the counts).
- In all, 6 of 42 difference CIs of the compared variants exclude 0 (3 never-firing variants left out). There is no multiplicity correction: with 42 separate 95% intervals, roughly 2.1 would exclude 0 by chance even if no rule had any effect, if the intervals were independent (they are not: all_off contains every single-rule change), so a single interval that excludes 0 is weak evidence on its own.

_Ablation: each variant is LexFront + LR with the named rule (all_off: every rule) switched off, everything else fixed; in the main setting all_off still has TF-IDF's lowercasing (TF-IDF still lowercases). Differences are variant minus full LexFront of the same setting, paired bootstrap over the test examples (same code, B and seed as `ci_table`). One training run per variant, so training randomness is not covered. The lowercase rule is tested only with case kept: in the main setting TF-IDF lowercases the text before LexFront sees it, so switching the rule off cannot change anything there. A variant whose rule never fires on a dataset (train and test) is the full system itself, so it is listed apart and not counted as a comparison; 0 changed predictions alone does not show that (a rule firing only in train still changes the fit). These numbers say which removals change the scores, not why; any reason given elsewhere is a hypothesis._
<!-- END obs:ablation -->

### BPE and the LexFront + BPE hybrid (P3)
BPE (Hugging Face `tokenizers`, vocabulary 8000, `##` continuation marker, `[UNK]`) trained on each dataset's
lowercased train split only; LexFront + BPE keeps LexFront's structural tokens (`<URL>`, `<USER>`, `<NUM>`, hashtags,
emoticons, emoji, punctuation, UNKNOWN) whole and splits only word-like tokens with a BPE trained on the word-like
tokens of the train split (`python -m experiments.run_bpe`, `results/bpe/`, protocol in `docs/versions/v5-bpe-hybrid.md`).
Same TF-IDF (main setting, TF-IDF lowercases), NB and LR as the main grid. Differences are paired bootstraps over the
test examples (same code, B and seed as `ci_table`) against the main grid's NLTK+LR and LexFront+LR, with an exact
McNemar test for BPE+LR vs NLTK+LR. A vocabulary-size sensitivity run at 32000 is in `results/bpe_v32000/`. In the
intrinsic table BPE's OOV rate is near zero (words split into vocabulary pieces; only characters never seen in train and rare pieces can be OOV), so OOV is not
comparable with the word-level tokenizers.

<!-- BEGIN table:bpe_runs -->
| dataset | BPE vocab | BPE+NB acc / F1 | BPE+LR acc / F1 | LexFront+BPE+NB acc / F1 | LexFront+BPE+LR acc / F1 | NLTK+LR acc / F1 | LexFront+LR acc / F1 |
|---|---|---|---|---|---|---|---|
| tweet_topic | 8000 | 70.52 / 27.88 | 71.90 / 35.09 | 70.02 / 27.12 | 69.90 / 33.56 | 72.32 / 36.95 | 70.73 / 36.18 |
| go_emotions | 8000 | 38.74 / 5.67 | 55.23 / 33.29 | 38.63 / 5.63 | 54.64 / 32.59 | 55.27 / 33.86 | 55.36 / 33.97 |
| ag_news | 8000 | 89.21 / 89.16 | 91.13 / 91.12 | 89.37 / 89.32 | 91.03 / 91.01 | 91.88 / 91.87 | 91.88 / 91.86 |
<!-- END table:bpe_runs -->

<!-- BEGIN table:bpe_table -->
| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | BPE+LR minus NLTK+LR (tokenizer only) | -0.41 [-1.27, +0.44] | -1.86 [-3.28, -0.40] | no / yes | 0.397 |
| tweet_topic | LexFront+BPE+LR minus LexFront+LR | -0.82 [-1.56, -0.06] | -2.62 [-3.90, -1.46] | yes / yes | - |
| tweet_topic | LexFront+BPE+LR minus BPE+LR | -2.00 [-2.91, -1.09] | -1.53 [-2.72, -0.32] | yes / yes | - |
| tweet_topic | LexFront+BPE+LR minus NLTK+LR | -2.41 [-3.35, -1.50] | -3.39 [-4.66, -2.16] | yes / yes | - |
| go_emotions | BPE+LR minus NLTK+LR (tokenizer only) | -0.04 [-0.68, +0.59] | -0.57 [-1.37, +0.19] | no / no | 0.946 |
| go_emotions | LexFront+BPE+LR minus LexFront+LR | -0.72 [-1.31, -0.15] | -1.39 [-2.16, -0.68] | yes / yes | - |
| go_emotions | LexFront+BPE+LR minus BPE+LR | -0.59 [-1.09, -0.09] | -0.71 [-1.42, +0.00] | yes / no | - |
| go_emotions | LexFront+BPE+LR minus NLTK+LR | -0.63 [-1.24, +0.00] | -1.27 [-2.11, -0.46] | no / yes | - |
| ag_news | BPE+LR minus NLTK+LR (tokenizer only) | -0.75 [-1.13, -0.37] | -0.75 [-1.14, -0.37] | yes / yes | < 0.001 |
| ag_news | LexFront+BPE+LR minus LexFront+LR | -0.86 [-1.25, -0.49] | -0.86 [-1.25, -0.48] | yes / yes | - |
| ag_news | LexFront+BPE+LR minus BPE+LR | -0.11 [-0.38, +0.18] | -0.11 [-0.39, +0.17] | no / no | - |
| ag_news | LexFront+BPE+LR minus NLTK+LR | -0.86 [-1.28, -0.43] | -0.86 [-1.29, -0.44] | yes / yes | - |
<!-- END table:bpe_table -->

<!-- BEGIN obs:bpe -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: BPE+LR minus NLTK+LR (tokenizer only): accuracy -0.41 (CI includes 0), macro-F1 -1.86 (CI excludes 0, lower); LexFront+BPE+LR minus LexFront+LR: accuracy -0.82 (CI excludes 0, lower), macro-F1 -2.62 (CI excludes 0, lower); LexFront+BPE+LR minus BPE+LR: accuracy -2.00 (CI excludes 0, lower), macro-F1 -1.53 (CI excludes 0, lower); LexFront+BPE+LR minus NLTK+LR: accuracy -2.41 (CI excludes 0, lower), macro-F1 -3.39 (CI excludes 0, lower). Exact McNemar p (BPE+LR vs NLTK+LR, accuracy) = 0.397.
- **go_emotions**: BPE+LR minus NLTK+LR (tokenizer only): accuracy -0.04 (CI includes 0), macro-F1 -0.57 (CI includes 0); LexFront+BPE+LR minus LexFront+LR: accuracy -0.72 (CI excludes 0, lower), macro-F1 -1.39 (CI excludes 0, lower); LexFront+BPE+LR minus BPE+LR: accuracy -0.59 (CI excludes 0, lower), macro-F1 -0.71 (CI includes 0); LexFront+BPE+LR minus NLTK+LR: accuracy -0.63 (CI includes 0), macro-F1 -1.27 (CI excludes 0, lower). Exact McNemar p (BPE+LR vs NLTK+LR, accuracy) = 0.946.
- **ag_news**: BPE+LR minus NLTK+LR (tokenizer only): accuracy -0.75 (CI excludes 0, lower), macro-F1 -0.75 (CI excludes 0, lower); LexFront+BPE+LR minus LexFront+LR: accuracy -0.86 (CI excludes 0, lower), macro-F1 -0.86 (CI excludes 0, lower); LexFront+BPE+LR minus BPE+LR: accuracy -0.11 (CI includes 0), macro-F1 -0.11 (CI includes 0); LexFront+BPE+LR minus NLTK+LR: accuracy -0.86 (CI excludes 0, lower), macro-F1 -0.86 (CI excludes 0, lower). Exact McNemar p (BPE+LR vs NLTK+LR, accuracy) < 0.001.
- In all, 17 of 24 difference CIs exclude 0 (vocabulary 8000, B = 2000, seed 20261016).

_Subword grid: each BPE (vocabulary size in the last line) is trained on the lowercased train split of its dataset only; LexFront + BPE keeps LexFront's structural tokens whole and splits only word-like tokens. Differences are paired bootstraps over the test examples (same code, B and seed as `ci_table`), against the NLTK+LR and LexFront+LR runs of the main grid. One training run per cell, so training randomness is not covered; no multiplicity correction. These lines say which differences exclude 0 and in which direction, not why; any reason given elsewhere is a hypothesis._
<!-- END obs:bpe -->

Vocabulary-size sensitivity (same code and protocol, vocabulary 32000, `results/bpe_v32000/`):

<!-- BEGIN table:bpe_v32000_runs -->
| dataset | BPE vocab | BPE+NB acc / F1 | BPE+LR acc / F1 | LexFront+BPE+NB acc / F1 | LexFront+BPE+LR acc / F1 | NLTK+LR acc / F1 | LexFront+LR acc / F1 |
|---|---|---|---|---|---|---|---|
| tweet_topic | 32000 | 71.02 / 28.76 | 72.43 / 36.02 | 70.02 / 27.46 | 70.93 / 35.57 | 72.32 / 36.95 | 70.73 / 36.18 |
| go_emotions | 32000 | 37.56 / 4.56 | 54.71 / 32.71 | 37.65 / 4.62 | 54.86 / 33.15 | 55.27 / 33.86 | 55.36 / 33.97 |
| ag_news | 32000 | 90.16 / 90.13 | 92.11 / 92.09 | 90.16 / 90.13 | 91.99 / 91.97 | 91.88 / 91.87 | 91.88 / 91.86 |
<!-- END table:bpe_v32000_runs -->

<!-- BEGIN table:bpe_v32000_table -->
| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | BPE+LR minus NLTK+LR (tokenizer only) | +0.12 [-0.53, +0.74] | -0.94 [-1.95, -0.00] | no / yes | 0.788 |
| tweet_topic | LexFront+BPE+LR minus LexFront+LR | +0.21 [-0.26, +0.68] | -0.61 [-1.39, +0.02] | no / no | - |
| tweet_topic | LexFront+BPE+LR minus BPE+LR | -1.50 [-2.27, -0.74] | -0.44 [-1.18, +0.38] | yes / no | - |
| tweet_topic | LexFront+BPE+LR minus NLTK+LR | -1.38 [-2.21, -0.59] | -1.38 [-2.43, -0.40] | yes / yes | - |
| go_emotions | BPE+LR minus NLTK+LR (tokenizer only) | -0.57 [-1.11, -0.02] | -1.15 [-2.09, -0.31] | yes / yes | 0.045 |
| go_emotions | LexFront+BPE+LR minus LexFront+LR | -0.50 [-0.89, -0.15] | -0.83 [-1.29, -0.42] | yes / yes | - |
| go_emotions | LexFront+BPE+LR minus BPE+LR | +0.15 [-0.35, +0.65] | +0.44 [-0.44, +1.41] | no / no | - |
| go_emotions | LexFront+BPE+LR minus NLTK+LR | -0.41 [-0.92, +0.15] | -0.71 [-1.45, +0.04] | no / no | - |
| ag_news | BPE+LR minus NLTK+LR (tokenizer only) | +0.22 [-0.08, +0.53] | +0.22 [-0.08, +0.52] | no / no | 0.168 |
| ag_news | LexFront+BPE+LR minus LexFront+LR | +0.11 [-0.08, +0.30] | +0.11 [-0.08, +0.31] | no / no | - |
| ag_news | LexFront+BPE+LR minus BPE+LR | -0.12 [-0.37, +0.14] | -0.12 [-0.38, +0.14] | no / no | - |
| ag_news | LexFront+BPE+LR minus NLTK+LR | +0.11 [-0.22, +0.43] | +0.10 [-0.22, +0.43] | no / no | - |
<!-- END table:bpe_v32000_table -->

<!-- BEGIN obs:bpe_v32000 -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: BPE+LR minus NLTK+LR (tokenizer only): accuracy +0.12 (CI includes 0), macro-F1 -0.94 (CI excludes 0, lower); LexFront+BPE+LR minus LexFront+LR: accuracy +0.21 (CI includes 0), macro-F1 -0.61 (CI includes 0); LexFront+BPE+LR minus BPE+LR: accuracy -1.50 (CI excludes 0, lower), macro-F1 -0.44 (CI includes 0); LexFront+BPE+LR minus NLTK+LR: accuracy -1.38 (CI excludes 0, lower), macro-F1 -1.38 (CI excludes 0, lower). Exact McNemar p (BPE+LR vs NLTK+LR, accuracy) = 0.788.
- **go_emotions**: BPE+LR minus NLTK+LR (tokenizer only): accuracy -0.57 (CI excludes 0, lower), macro-F1 -1.15 (CI excludes 0, lower); LexFront+BPE+LR minus LexFront+LR: accuracy -0.50 (CI excludes 0, lower), macro-F1 -0.83 (CI excludes 0, lower); LexFront+BPE+LR minus BPE+LR: accuracy +0.15 (CI includes 0), macro-F1 +0.44 (CI includes 0); LexFront+BPE+LR minus NLTK+LR: accuracy -0.41 (CI includes 0), macro-F1 -0.71 (CI includes 0). Exact McNemar p (BPE+LR vs NLTK+LR, accuracy) = 0.045.
- **ag_news**: BPE+LR minus NLTK+LR (tokenizer only): accuracy +0.22 (CI includes 0), macro-F1 +0.22 (CI includes 0); LexFront+BPE+LR minus LexFront+LR: accuracy +0.11 (CI includes 0), macro-F1 +0.11 (CI includes 0); LexFront+BPE+LR minus BPE+LR: accuracy -0.12 (CI includes 0), macro-F1 -0.12 (CI includes 0); LexFront+BPE+LR minus NLTK+LR: accuracy +0.11 (CI includes 0), macro-F1 +0.10 (CI includes 0). Exact McNemar p (BPE+LR vs NLTK+LR, accuracy) = 0.168.
- In all, 8 of 24 difference CIs exclude 0 (vocabulary 32000, B = 2000, seed 20261016).

_Subword grid: each BPE (vocabulary size in the last line) is trained on the lowercased train split of its dataset only; LexFront + BPE keeps LexFront's structural tokens whole and splits only word-like tokens. Differences are paired bootstraps over the test examples (same code, B and seed as `ci_table`), against the NLTK+LR and LexFront+LR runs of the main grid. One training run per cell, so training randomness is not covered; no multiplicity correction. These lines say which differences exclude 0 and in which direction, not why; any reason given elsewhere is a hypothesis._
<!-- END obs:bpe_v32000 -->

### WordPiece and Unigram (P4)
Two more learned vocabularies, same library, protocol and statistics as BPE (`docs/versions/v6-wordpiece-unigram.md`,
`python -m experiments.run_subword2`, `results/subword2/`): 8000 pieces per dataset, trained on the lowercased train
split only, `Whitespace` pre-tokenizer, same TF-IDF (main setting) with NB and LR, no hybrid with LexFront. WordPiece merges
the pair that most raises the likelihood of the text and encodes by greedy longest match; Unigram prunes a large
candidate set by EM and encodes by the best-scoring split. Differences are paired bootstraps against the main grid's NLTK+LR
and the P3 BPE+LR; exact McNemar on accuracy against NLTK+LR.

<!-- BEGIN table:subword_runs -->
| dataset | vocab | WordPiece+NB acc / F1 | WordPiece+LR acc / F1 | Unigram+NB acc / F1 | Unigram+LR acc / F1 | NLTK+LR acc / F1 | BPE+LR acc / F1 |
|---|---|---|---|---|---|---|---|
| tweet_topic | 8000 | 70.84 / 28.07 | 71.61 / 34.88 | 71.52 / 28.46 | 73.05 / 36.49 | 72.32 / 36.95 | 71.90 / 35.09 |
| go_emotions | 8000 | 38.69 / 5.67 | 55.12 / 33.12 | 38.13 / 5.18 | 55.71 / 35.12 | 55.27 / 33.86 | 55.23 / 33.29 |
| ag_news | 8000 | 88.95 / 88.89 | 91.47 / 91.46 | 89.16 / 89.12 | 90.97 / 90.96 | 91.88 / 91.87 | 91.13 / 91.12 |
<!-- END table:subword_runs -->

<!-- BEGIN table:subword_table -->
| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | WordPiece+LR minus NLTK+LR (tokenizer only) | -0.71 [-1.62, +0.12] | -2.08 [-3.33, -0.87] | no / yes | 0.122 |
| tweet_topic | Unigram+LR minus NLTK+LR (tokenizer only) | +0.74 [-0.24, +1.68] | -0.46 [-1.93, +0.98] | no / no | 0.143 |
| tweet_topic | WordPiece+LR minus BPE+LR | -0.29 [-0.91, +0.29] | -0.21 [-1.08, +0.70] | no / no | - |
| tweet_topic | Unigram+LR minus BPE+LR | +1.15 [+0.24, +2.09] | +1.40 [+0.09, +2.81] | yes / yes | - |
| tweet_topic | Unigram+LR minus WordPiece+LR | +1.44 [+0.56, +2.41] | +1.62 [+0.62, +2.80] | yes / yes | - |
| go_emotions | WordPiece+LR minus NLTK+LR (tokenizer only) | -0.15 [-0.78, +0.48] | -0.74 [-1.57, +0.08] | no / no | 0.687 |
| go_emotions | Unigram+LR minus NLTK+LR (tokenizer only) | +0.44 [-0.28, +1.15] | +1.26 [+0.10, +2.39] | no / yes | 0.266 |
| go_emotions | WordPiece+LR minus BPE+LR | -0.11 [-0.46, +0.26] | -0.17 [-0.62, +0.30] | no / no | - |
| go_emotions | Unigram+LR minus BPE+LR | +0.48 [-0.15, +1.11] | +1.83 [+0.78, +2.96] | no / yes | - |
| go_emotions | Unigram+LR minus WordPiece+LR | +0.59 [-0.04, +1.24] | +2.00 [+0.89, +3.14] | no / yes | - |
| ag_news | WordPiece+LR minus NLTK+LR (tokenizer only) | -0.41 [-0.82, -0.01] | -0.41 [-0.82, -0.02] | yes / yes | 0.049 |
| ag_news | Unigram+LR minus NLTK+LR (tokenizer only) | -0.91 [-1.34, -0.50] | -0.91 [-1.34, -0.50] | yes / yes | < 0.001 |
| ag_news | WordPiece+LR minus BPE+LR | +0.34 [+0.05, +0.66] | +0.34 [+0.05, +0.65] | yes / yes | - |
| ag_news | Unigram+LR minus BPE+LR | -0.16 [-0.57, +0.22] | -0.16 [-0.56, +0.22] | no / no | - |
| ag_news | Unigram+LR minus WordPiece+LR | -0.50 [-0.89, -0.13] | -0.50 [-0.89, -0.13] | yes / yes | - |
<!-- END table:subword_table -->

<!-- BEGIN obs:subword -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: WordPiece+LR minus NLTK+LR (tokenizer only): accuracy -0.71 (CI includes 0), macro-F1 -2.08 (CI excludes 0, lower); Unigram+LR minus NLTK+LR (tokenizer only): accuracy +0.74 (CI includes 0), macro-F1 -0.46 (CI includes 0); WordPiece+LR minus BPE+LR: accuracy -0.29 (CI includes 0), macro-F1 -0.21 (CI includes 0); Unigram+LR minus BPE+LR: accuracy +1.15 (CI excludes 0, higher), macro-F1 +1.40 (CI excludes 0, higher); Unigram+LR minus WordPiece+LR: accuracy +1.44 (CI excludes 0, higher), macro-F1 +1.62 (CI excludes 0, higher). Exact McNemar p (accuracy): WordPiece+LR vs NLTK+LR = 0.122; Unigram+LR vs NLTK+LR = 0.143.
- **go_emotions**: WordPiece+LR minus NLTK+LR (tokenizer only): accuracy -0.15 (CI includes 0), macro-F1 -0.74 (CI includes 0); Unigram+LR minus NLTK+LR (tokenizer only): accuracy +0.44 (CI includes 0), macro-F1 +1.26 (CI excludes 0, higher); WordPiece+LR minus BPE+LR: accuracy -0.11 (CI includes 0), macro-F1 -0.17 (CI includes 0); Unigram+LR minus BPE+LR: accuracy +0.48 (CI includes 0), macro-F1 +1.83 (CI excludes 0, higher); Unigram+LR minus WordPiece+LR: accuracy +0.59 (CI includes 0), macro-F1 +2.00 (CI excludes 0, higher). Exact McNemar p (accuracy): WordPiece+LR vs NLTK+LR = 0.687; Unigram+LR vs NLTK+LR = 0.266.
- **ag_news**: WordPiece+LR minus NLTK+LR (tokenizer only): accuracy -0.41 (CI excludes 0, lower), macro-F1 -0.41 (CI excludes 0, lower); Unigram+LR minus NLTK+LR (tokenizer only): accuracy -0.91 (CI excludes 0, lower), macro-F1 -0.91 (CI excludes 0, lower); WordPiece+LR minus BPE+LR: accuracy +0.34 (CI excludes 0, higher), macro-F1 +0.34 (CI excludes 0, higher); Unigram+LR minus BPE+LR: accuracy -0.16 (CI includes 0), macro-F1 -0.16 (CI includes 0); Unigram+LR minus WordPiece+LR: accuracy -0.50 (CI excludes 0, lower), macro-F1 -0.50 (CI excludes 0, lower). Exact McNemar p (accuracy): WordPiece+LR vs NLTK+LR = 0.049; Unigram+LR vs NLTK+LR < 0.001.
- In all, 16 of 30 difference CIs exclude 0 (B = 2000, seed 20261016).

_WordPiece and Unigram (vocabulary 8000, `Whitespace` pre-tokenizer) are each trained on the lowercased train split of their dataset only, and feed the same TF-IDF + LR / NB as the other tokenizers. Differences are paired bootstraps over the test examples (same code, B and seed as `ci_table`) against the NLTK+LR run of the main grid and the BPE+LR run of the P3 grid. One training run per cell, so training randomness is not covered; no multiplicity correction. These lines say which differences exclude 0 and in which direction, not why; any reason given elsewhere is a hypothesis._
<!-- END obs:subword -->

Reading it (hand-written; the numbers restate the table, the reasons are hypotheses): no WordPiece or Unigram run is above
NLTK+LR in accuracy with a CI that excludes 0, and on ag_news both are below it. Unigram is above WordPiece on tweet_topic
(accuracy +1.44 [+0.56, +2.41]) and below it on ag_news. Unigram's macro-F1 gain over NLTK+LR on go_emotions
(+1.26 [+0.10, +2.39]) is the only interval in the subword experiments that puts a subword tokenizer above NLTK+LR; its lower end
is close to 0 and it is one of 30 intervals without multiplicity correction (about 1.5 would exclude 0 by chance if they
were independent, which they are not). A possible reason (untested): macro-F1 weighs the many small go_emotions
classes, where the way Unigram splits rare words could matter.

### Reproducibility and leakage checks
Hand-written; every number below is in a file named here or in `docs/DECISIONS.md`.

* **Thread nondeterminism and its fix.** Rerunning go_emotions NLTK + LR with 1 / 2 / 4 BLAS threads gave accuracy
  0.552723 / 0.552941 / 0.553159 and macro-F1 0.338600 / 0.338894 / 0.338969: the same seed, a different float summation
  order inside lbfgs. The first committed numbers were the 2-thread ones. Since the fix every vectorize / fit / predict
  call is wrapped in `threadpool_limits(1)`; all run JSONs were regenerated, 12 metric values moved (at most 0.07 points in accuracy, 0.18 in macro-F1;
  old and new values: `docs/DECISIONS.md`, "BLAS threads"), and a rerun with 4 threads reproduced the committed
  go_emotions and ag_news LR cells. These shifts are much smaller than the test-set CIs, but they are the size of some
  of the differences discussed above, which is why the pin matters.
* **The subword vocabularies are trained on the train split only.** `build_tokenizer(name, train_texts)` is the only
  constructor, the fingerprint carries a hash of the training texts, and `tests/test_bpe.py::test_bpe_is_trained_on_train_texts_only`
  checks which texts reach the trainer. End-to-end check (independent audit, `docs/AUDIT_6.md`, tweet_topic): all 3399
  test texts replaced by garbage (characters absent from train, 200k-character documents) and the real grid code path
  run: the serialized WordPiece and Unigram models, their training-text hashes and the TF-IDF vocabularies were
  bit-identical to the run on the real test texts, while 1666 / 1765 of the 3399 test predictions changed (so the test is
  not vacuous); 185 characters that occur only in the real test split are in no vocabulary. The same audit covered the
  other training paths (vectorizer, classifier, intrinsic metrics) by reading the code.
* **Subword training is deterministic here.** As shipped, the BPE and WordPiece trainers gave different merge orders on
  the same text (hash-map order of the `##` alphabet; fixed by passing a sorted alphabet, DECISIONS "P3"/"P4") and the
  Unigram trainer different scores (floating-point order and a hash-set order in the characters it puts back; fixed by
  `canonical_unigram`, scores rounded to 3 decimals). Checked: 6 trainings each in separate processes (1 and 4 Rayon
  threads, different `PYTHONHASHSEED`) gave one byte-identical model per tokenizer, and re-running the tweet_topic
  WordPiece+LR and Unigram+LR cells reproduced the committed metrics, fingerprints and prediction files. Limit: Unigram's
  rounding boundary is 2e-10 from the nearest score against a measured run-to-run noise of at most 4e-11 (about 5x), so
  a different CPU or library version could in principle flip one rounded score; none flipped in the runs above.
* **The confidence intervals were recomputed independently.** A second implementation written from the pseudocode in
  section 5 (same seed, B and resampling scheme) reproduced every stored interval end of `results/subword2/stats/bootstrap.json`
  to 5e-13, the McNemar counts and p-values exactly, and the point metrics exactly (`docs/AUDIT_6.md`).
* **What none of this covers:** training randomness across seeds (one run per cell), other datasets, other vocabulary
  sizes than 8000 and 32000.

## 5. Pseudocode

```text
(1) regex -> NFA -> DFA -> minimized DFA

    input : rules = [(name_0, regex_0), ..., (name_k, regex_k)]   # list order = priority
    output: scanner tables

    asts   <- [parse(regex_i) for each rule]                       # recursive descent
    atoms  <- split the code-point line at every range boundary used in asts
    NFA    <- new start state
    for i, ast in asts:
        (s, t) <- thompson(ast)          # sym: s -a-> t | cat: chain with eps
                                         # alt: s -eps-> each branch -eps-> t
                                         # star/plus/opt: eps loop / skip edges
        add eps edge  NFA.start -> s
        accepting[t]  <- i
    start_set <- eps_closure({NFA.start})
    DFA.states <- {start_set};  worklist <- [start_set]
    while worklist not empty:
        S <- pop(worklist)
        for each atom a:
            T <- eps_closure(move(S, a))
            if T is empty: continue
            if T is new: add to DFA.states and worklist
            DFA.trans[S][a] <- T
    for each DFA state S:
        S.accepts <- min{accepting[q] : q in S and q accepting}   # priority; None if none
    # Moore minimization
    block[S] <- id of (S.accepts)             # initial partition: by accepted rule
    repeat
        signature[S] <- (block[S], sorted (a, block[DFA.trans[S][a]]) pairs)
        block'[S]    <- id of signature[S]
    until number of blocks stops growing
    merge states with equal block; renumber breadth-first from the start state

(2) Longest-match scanner with error recovery

    scan(text):
        pos <- 0
        while pos < len(text):
            if text[pos] is whitespace: pos++ ; continue
            state <- start;  last <- none;  i <- pos
            while i < len(text):
                a <- atom_of(text[i])
                state <- table[state][a]            # DEAD if no edge
                if state == DEAD: break
                i++
                if accepts[state] is not none: last <- (i, accepts[state])
            if last is none:                        # error recovery (panic mode)
                emit (UNKNOWN, text[pos]);  pos++
            else:
                (end, rule) <- last
                emit (class[rule], text[pos:end]);  pos <- end
    # PLY version: same loop, but "table walk" is replaced by trying every class regex at pos,
    # keeping the strictly longest match so that ties stay with the earlier class.

(3) Evaluation loop

    for dataset in [tweet_topic, go_emotions, ag_news]:
        ds <- load(dataset)                         # train/test texts, labels, label names
        (optionally subsample train/test with a fixed seed)
        for tokenizer in [whitespace, nltk, tweet] (module 1) or [lexfront] (module 2):
            X_train <- TfidfVectorizer(tokenizer, token_pattern=None, lowercase=True, min_df=2).fit_transform(train)
            X_test  <- vectorizer.transform(test)
            for clf in [majority, naive_bayes, logistic_regression] (module 1) or [naive_bayes, logistic_regression] (module 2):
                clf.fit(X_train, y_train)
                record <- {config, accuracy(y_test, clf.predict(X_test)), macro_f1(...), timings, versions}
                write record (with a config + code fingerprint) to results/<module>/<dataset>__<tokenizer>__<clf>.json
                write (y_test, predictions) to results/<module>/predictions/<dataset>__<tokenizer>__<clf>.json.gz
                # a cached record is reused only while its fingerprint still matches
    tables, figures, observations are built only from those JSON files

(4) Test-set bootstrap (experiments/stats.py)

    for dataset:
        y <- gold test labels;  pred[run] <- stored predictions of every run
        idx <- B x n matrix of indices drawn with replacement      # default_rng(seed); shared by all runs: paired
        for b in 1..B:
            cm[run][b] <- confusion matrix of (y[idx[b]], pred[run][idx[b]])   # one bincount for all b
            acc = trace(cm) / n;  macro-F1 = mean over present classes of 2TP / (2TP + FP + FN)
        CI(run)        <- 2.5th and 97.5th percentile of metric(cm[run][.])
        CI(B minus A)  <- percentiles of metric(cm[B][.]) - metric(cm[A][.])         # same resamples
        McNemar(A, B)  <- exact binomial test on (#A right & B wrong, #A wrong & B right)
```
