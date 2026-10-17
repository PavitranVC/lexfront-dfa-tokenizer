# LexFront — final summary (one honest page)

Every number below is read from a JSON file written by a script that ran (named in brackets). 95% CIs are percentile
bootstraps over the test examples only; one training run per cell, no multiplicity correction. Reasons are hypotheses.

## What was built
A compiler-style lexer (11 token classes as regexes, PLY longest-match dispatcher, UNKNOWN error recovery, six switchable
normalization rules, symbol table) plus our own regex -> Thompson NFA -> subset-construction DFA -> Moore-minimized DFA
engine for EMOTICON, HASHTAG, MENTION and WORD (tested against PLY). It is used as a pre-tokenizer for TF-IDF + Naive
Bayes / Logistic Regression on three datasets (tweet_topic, go_emotions, ag_news) and compared with whitespace, NLTK,
TweetTokenizer, BPE, LexFront + BPE, WordPiece and Unigram.

## What the data show
* **Tokenizer only (NLTK+LR -> LexFront+LR)**: -1.59 accuracy points on tweet_topic [-2.38, -0.82] (McNemar p < 0.001);
  +0.09 [-0.35, +0.54] on go_emotions; +0.00 [-0.32, +0.33] on ag_news [results/stats/bootstrap.json].
* **"Proposed over existing" (NLTK+NB -> LexFront+LR)** is +17.58 on go_emotions and +1.50 on ag_news, but almost all of it is
  Naive Bayes -> Logistic Regression (+17.49 and +1.50 with NLTK held fixed). On tweet_topic it is -0.26 [-1.47, +0.91].
* **Case**: with TF-IDF case folding off, go_emotions flips to +1.15 [+0.46, +1.85]; the ablation shows this is the
  lowercase rule alone (removing it: -1.07 [-1.72, -0.44]) [results/sensitivity_case, results/ablation].
* **Rules**: in the main setting no single-rule removal has an accuracy CI that excludes 0; 6 of 42 compared CIs exclude 0 in
  all (about 2 expected by chance). The URL rule never fires on tweet_topic and go_emotions, the mention rule never on ag_news
  (those datasets mask URLs and handles), so those variants are not tests of the rule [results/ablation/stats].
* **Subword**: BPE (vocabulary 8000) shows no accuracy gain over NLTK+LR (CIs include 0 on tweet_topic and go_emotions, -0.75 on ag_news; macro-F1 -1.86 on tweet_topic);
  LexFront + BPE is below LexFront alone on all three datasets (17 of 24 CIs exclude 0 at 8000, all pointing down; 8 of 24 at 32000)
  [results/bpe, results/bpe_v32000]. WordPiece and Unigram (8000): none is above NLTK+LR in accuracy with a CI that excludes 0;
  on ag_news both are below; the only interval above NLTK+LR is Unigram's macro-F1 on go_emotions (+1.26 [+0.10, +2.39], 1 of 30)
  [results/subword2/stats].
* **Intrinsic**: among the word-level tokenizers LexFront has the lowest OOV rate and the smallest vocabulary on all three
  datasets; part of that is by construction (placeholders, collapsing), so it is not evidence of better classification.

## Dataset collection, cleaning and tuning (added afterwards, `results/collection/REPORT.md`)
Five tweet_eval tasks were added (eight datasets), cleaned (empty, conflicting, duplicate and train-duplicated test texts removed; the
largest cut is tweet_eval_hate, test 2970 -> 2731) and rerun with the default settings and with validation-selected TF-IDF / `C` / `alpha`,
the same search for every tokenizer. Tuning raised accuracy in 44 of 64 cells and lowered it in 16 (mean +1.63 points). It qualifies two
statements above: with tuned Naive Bayes the NLTK+NB to NLTK+LR gap on go_emotions shrinks (17.45 -> 10.49 points), and LexFront+LR minus
NLTK+LR has no CI excluding 0 on any of the eight datasets (default settings: 4 of 8, two up and two down; tweet_topic -1.59 -> -0.32
[-1.15, +0.47]). One run per cell, one validation split, an NB `alpha` grid whose edge values were often chosen (a wider grid was not run); reasons are hypotheses.

## What we do not claim
Better, first or novel. That LexFront is worse in general (three datasets, one classifier family, one run per cell). Why any
difference appears (all reasons are untested hypotheses). Anything about transformers, Hinglish or multiple seeds.

## Reproducibility
Fits are single-threaded (the BLAS thread count changed LR metrics in the last digits: 12 values moved by at most 0.07
accuracy points; DECISIONS "BLAS threads"). Runs carry config + code fingerprints. Subword trainers are not deterministic as shipped;
fixed and checked (docs/AUDIT_6.md). Test-set leakage into the subword vocabularies was checked end to end (garbage test
texts leave the trained models bit-identical).
