# LexFront: a DFA-based pre-tokenizer for noisy English text classification

Compiler Design course project, Review 2 report.

**Scope of this report.** It covers the core study: the lexer and DFA engine (Modules 1 and 2), the comparison on
three datasets, the case and rule ablations, and the subword extensions (BPE, WordPiece, Unigram). Later extension work
(larger datasets, wider classifier tuning) is not part of this report. All numbers come from JSON files under
`results/`, written by the scripts named in brackets. The full repository history is public, so the extension work is
visible there.

---

## 1. Problem and motivation

Text from social media is noisy: hashtags, @-mentions, URLs, emoticons, elongated words ("soooo"), slang. A generic
tokenizer splits these in ways that can hurt a classifier. A compiler's lexical analyzer solves a similar problem. It
turns characters into tokens using regular expressions, longest-match rules, a symbol table and error recovery.

**Idea.** Build a lexer in the compiler-design style and use it as a pre-tokenizer for text classification. Then
compare it with existing tokenizers while holding the classifier fixed.

**Question.** Does a compiler-style lexer classify noisy text better than existing tokenizers?

## 2. Design (compiler-design content)

### 2.1 Token classes
Eleven classes in priority order (`lexer/token_spec.py`): URL, MENTION, HASHTAG, EMOTICON, EMOJI, NUMBER,
CONTRACTION, ELONGATED, WORD, PUNCT, and UNKNOWN as the error-recovery fallback.

### 2.2 Longest-match lexer with error recovery
The PLY-based lexer (`lexer/ply_lexer.py`) takes the longest match at each position, and ties go to the earlier
class. A character that matches no class becomes an UNKNOWN token and scanning continues (panic-mode recovery).

### 2.3 Our own regex to DFA engine (`lexer/dfa/`)
The pipeline is the textbook one:

1. **Regex parser**: recursive descent to an AST.
2. **Thompson construction**: AST to NFA with epsilon edges.
3. **Subset construction**: NFA to DFA, with epsilon-closure and an alphabet partitioned into atoms.
4. **Moore minimization**: partition refinement, starting from the partition by accepted rule.
5. **Table-driven longest-match scanner** with UNKNOWN recovery.

The engine is used for EMOTICON, HASHTAG, MENTION and WORD. A test checks that it and the PLY lexer give identical
tokens on those four classes. ELONGATED is not in the DFA engine. Its usual regex `(.)\1{2,}` needs a back-reference,
which is outside regular expressions. This is a limit of regular languages, not a bug.

**Automaton sizes** (`results/module2/dfa_states.json`):

| automaton | alphabet atoms | NFA states | DFA states | minimized DFA states |
|---|---|---|---|---|
| MENTION | 5 | 9 | 3 | 3 |
| HASHTAG | 8 | 9 | 3 | 3 |
| EMOTICON | 17 | 19 | 6 | 5 |
| WORD | 6 | 9 | 3 | 2 |
| combined (4 classes) | 31 | 43 | 12 | 10 |
| ELONGATED spelled out as 26 branches (extra, not in LexFront) | 26 | 221 | 157 | 54 |

The combined automaton goes from 43 NFA states to 12 DFA states to 10 after minimization. The spelled-out ELONGATED
automaton shows the blow-up that the construction can cause: 221 to 157 to 54 states.

Figure: `results/module2/dfa_min.png` (minimized DFA), `results/figures/dfa_states.png`.

### 2.4 Normalizer and symbol table
Six switchable rules (`lexer/normalize.py`): URL to `<URL>`, MENTION to `<USER>`, NUMBER to `<NUM>`, elongated runs
collapsed to two letters, lowercasing of word-like tokens, and a slang lookup (30 entries). The symbol table
(`lexer/symbol_table.py`) stores token id, frequency and class. `LexFront.tokenize()` has the same interface as the
baseline tokenizers, so the classifier code is identical for all of them.

## 3. Experimental setup

* **Datasets**: tweet_topic (6 classes, 3598 train / 3399 test), go_emotions (single-label, 28 classes, 36308 / 4590),
  ag_news (4 classes, 120000 / 7600). Full data, no subsampling [`results/datasets.json`].
* **Tokenizers**: whitespace, NLTK `word_tokenize`, NLTK `TweetTokenizer`, LexFront. Extensions: BPE, LexFront + BPE,
  WordPiece, Unigram.
* **Classifiers**: TF-IDF (`min_df=2`) into a majority baseline, Multinomial Naive Bayes (the existing method), and
  Logistic Regression (`max_iter=1000`, seed 0, single-threaded).
* **Metrics**: accuracy and macro-F1. One training run per cell. 95% confidence intervals are paired percentile
  bootstraps over the test examples, with McNemar's test for accuracy [`results/stats/bootstrap.json`].
* **Reproducibility**: every run stores its config and code fingerprint, and the test suite checks the engine.

## 4. Results

### 4.1 Existing tokenizers (Module 1)
Accuracy / macro-F1 in % [`results/module1/`]:

| dataset | tokenizer | Majority | Naive Bayes | Logistic Regression |
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

Figure: `results/figures/module1_accuracy.png`.

### 4.2 Proposed (LexFront + LR) versus existing (NLTK + NB)
[`results/tables/existing_vs_proposed.md`]

| dataset | existing NLTK+NB acc / F1 | proposed LexFront+LR acc / F1 | delta acc (pts) | LexFront+NB | NLTK+LR |
|---|---|---|---|---|---|
| tweet_topic | 70.99 / 28.56 | 70.73 / 36.18 | -0.26 | 70.43 / 27.84 | 72.32 / 36.95 |
| go_emotions | 37.78 / 4.88 | 55.36 / 33.97 | +17.58 | 37.78 / 4.81 | 55.27 / 33.86 |
| ag_news | 90.38 / 90.35 | 91.88 / 91.86 | +1.50 | 90.08 / 90.05 | 91.88 / 91.87 |

Figure: `results/figures/existing_vs_proposed.png`.

### 4.3 Where the gain comes from
Almost all of the "proposed over existing" gain is the classifier change from Naive Bayes to Logistic Regression, not
the tokenizer:

| dataset | NB to LR (NLTK fixed) | NLTK to LexFront (LR fixed) |
|---|---|---|
| tweet_topic | +1.32 | -1.59 [-2.38, -0.82], McNemar p < 0.001 |
| go_emotions | +17.49 | +0.09 [-0.35, +0.54] |
| ag_news | +1.50 | +0.00 [-0.32, +0.33] |

With the tokenizer as the only change, LexFront matches NLTK on go_emotions and ag_news (confidence intervals include
0). On tweet_topic it is 1.59 points lower. We do not claim LexFront is better.

### 4.4 Intrinsic tokenizer metrics
[`results/module2/intrinsic__*.json`, figure `results/figures/intrinsic_metrics.png`]

Among the word-level tokenizers, LexFront has the lowest out-of-vocabulary rate and the smallest vocabulary on all
three datasets (for example ag_news: OOV 0.48% against 0.96% for NLTK, vocabulary 65,337 against 98,994). Part of this
is by construction, since placeholders and collapsing shrink the vocabulary. It is therefore not evidence of better
classification, and the accuracy results above are the check on that.

### 4.5 Case sensitivity and per-rule ablation
`results/sensitivity_case`, `results/ablation`.

* With TF-IDF's own case folding switched off, go_emotions flips to +1.15 points [+0.46, +1.85] for LexFront over NLTK.
  The ablation attributes this to the lowercase rule alone. Removing that rule gives -1.07 [-1.72, -0.44].
* In the main setting, no single-rule removal has an accuracy CI that excludes 0. 6 of 42 compared intervals exclude 0
  in all, and about 2 would be expected by chance.
* The URL rule never fires on tweet_topic and go_emotions, and the mention rule never fires on ag_news, because those
  datasets already mask URLs and handles. Those variants are therefore not tests of the rule.

### 4.6 Subword extensions
`results/bpe`, `results/bpe_v32000`, `results/subword2`.

* **BPE (vocabulary 8000)**: no accuracy gain over NLTK+LR. The intervals include 0 on tweet_topic and go_emotions, and
  ag_news is 0.75 points lower.
* **LexFront + BPE hybrid**: below LexFront alone on all three datasets (17 of 24 intervals exclude 0 at vocabulary
  8000, all pointing down).
* **WordPiece and Unigram (8000)**: none is above NLTK+LR in accuracy with an interval that excludes 0. On ag_news
  both are below.

## 5. Discussion and limitations

* The tokenizer matters little for these datasets and this classifier family. The classifier choice (NB to LR) matters
  far more.
* Reasons for any difference are hypotheses and were not tested.
* Three datasets, one classifier family (TF-IDF + NB/LR), one training run per cell. No claims about transformers,
  Hinglish text, or training-seed variance.
* Fits are single-threaded, because the BLAS thread count changed LR metrics in the last digits (at most 0.07 points).
* Test-set leakage into the subword vocabularies was checked end to end.

## 6. Work remaining before the final review

* Larger-scale evaluation and the effect of wider classifier settings (extension work, not covered in this report).
* Report polish: figures, formatting and the demo script.

## 7. Demo plan

1. Run `notebooks/LexFront_Review1.ipynb` on Colab (CPU, quick mode, runs top to bottom).
2. Show a noisy tweet going through LexFront. Print the token classes, the normalized tokens and the symbol table.
3. Show the minimized DFA (`results/module2/dfa_min.png`) and walk one hashtag through the table-driven scanner.
4. Show the results tables from section 4.

## 8. Repository map

`lexer/` (lexer, DFA engine), `experiments/` (runners), `results/` (JSON, tables, figures), `docs/modules/` (module
write-ups), `docs/LEARN.md` (viva study guide), `tests/`.
