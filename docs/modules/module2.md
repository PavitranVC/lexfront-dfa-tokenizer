# Module 2 — LexFront (the new contribution)

## What it does
A compiler-style lexical analyzer used as a pre-tokenizer.

* **Token classes** in priority order (`lexer/token_spec.py`): URL, MENTION, HASHTAG, EMOTICON, EMOJI, NUMBER,
  CONTRACTION, ELONGATED, WORD, PUNCT, and UNKNOWN as the error-recovery fallback.
* **PLY lexer** (`lexer/ply_lexer.py`): longest match, ties to the earlier class; an unmatched character becomes
  UNKNOWN and scanning continues.
* **Normalizer** (`lexer/normalize.py`): URL→`<URL>`, MENTION→`<USER>`, NUMBER→`<NUM>`, elongated runs collapsed to
  2 letters, WORD-like tokens lowercased, slang lookup from `lexer/slang_table.csv` (30 entries). Each rule can be
  switched off (`NormalizeConfig`).
* **Symbol table** (`lexer/symbol_table.py`): token → id, frequency, class.
* **`LexFront.tokenize()`** (`lexer/lexfront.py`): same interface as the baselines.
* **Own DFA engine** (`lexer/dfa/`): regex parser → Thompson NFA → subset construction → Moore minimization →
  table-driven longest-match scanner with UNKNOWN recovery. Applied to EMOTICON, HASHTAG, MENTION and WORD; a test
  checks that it and the PLY lexer give identical tokens on those four classes.
* **ELONGATED is not in the DFA engine**: its usual regex `(.)\1{2,}` needs a back-reference, which is outside
  regular expressions.

Run: `python -m experiments.run_module2` (`--only dfa intrinsic classify` to pick parts).

## Pseudocode
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

```

## Results
Generated from `results/module2/*.json`. Minimized DFA: `results/module2/dfa_min.dot` and `dfa_min.png`.

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

<!-- BEGIN table:intrinsic_metrics -->
| dataset | tokenizer | OOV rate % (tokens) | OOV rate % (types) | fertility | vocab size | throughput (tokens/s) |
|---|---|---|---|---|---|---|
| tweet_topic | whitespace | 18.45 | 65.83 | 1.000 | 21653 | 11,774,749 |
| tweet_topic | nltk | 8.74 | 58.23 | 1.431 | 15712 | 106,738 |
| tweet_topic | tweet | 9.86 | 59.72 | 1.376 | 16524 | 217,245 |
| tweet_topic | lexfront | 8.25 | 57.13 | 1.391 | 14483 | 162,224 |
| go_emotions | whitespace | 6.04 | 30.11 | 1.000 | 44476 | 19,737,435 |
| go_emotions | nltk | 2.57 | 21.55 | 1.220 | 26410 | 137,377 |
| go_emotions | tweet | 2.56 | 20.84 | 1.195 | 25907 | 122,889 |
| go_emotions | lexfront | 2.45 | 19.89 | 1.195 | 24459 | 163,800 |
| ag_news | whitespace | 1.90 | 14.97 | 1.000 | 158733 | 12,746,416 |
| ag_news | nltk | 0.96 | 11.52 | 1.158 | 98994 | 222,434 |
| ag_news | tweet | 0.75 | 9.13 | 1.154 | 86096 | 170,507 |
| ag_news | lexfront | 0.48 | 6.78 | 1.207 | 65337 | 175,741 |
<!-- END table:intrinsic_metrics -->

<!-- BEGIN obs:intrinsic -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: lowest OOV rate 8.25% (lexfront); smallest vocabulary 14483 (lexfront); LexFront: OOV 8.25%, fertility 1.391, vocabulary 14483; LexFront throughput is 1.52x NLTK's (tokens/s, this machine).
- **go_emotions**: lowest OOV rate 2.45% (lexfront); smallest vocabulary 24459 (lexfront); LexFront: OOV 2.45%, fertility 1.195, vocabulary 24459; LexFront throughput is 1.19x NLTK's (tokens/s, this machine).
- **ag_news**: lowest OOV rate 0.48% (lexfront); smallest vocabulary 65337 (lexfront); LexFront: OOV 0.48%, fertility 1.207, vocabulary 65337; LexFront throughput is 0.79x NLTK's (tokens/s, this machine).
<!-- END obs:intrinsic -->

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

_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file `stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score moves with the draw of the test set, not with training randomness, which one run per cell cannot show (the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed subsample). A gap whose CI includes 0 should not be read as a real difference._
<!-- END obs:existing_vs_proposed -->

### Is the difference larger than test-set noise?
Paired 95% bootstrap CIs over the test examples (`python -m experiments.run_stats`, `results/stats/bootstrap.json`).
They cover the draw of the test set, not training randomness (one training run per cell).

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
- Settings: percentile bootstrap over test examples, B = 2000, seed 20261016, 95% percentile intervals.

_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file `stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score moves with the draw of the test set, not with training randomness, which one run per cell cannot show (the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed subsample). A gap whose CI includes 0 should not be read as a real difference._
<!-- END obs:ci -->

### Case-sensitivity check (TF-IDF `lowercase=False`)
In the main grid TF-IDF lowercases before any tokenizer runs, so LexFront's lowercase rule is inactive there by design.
This grid switches TF-IDF case folding off for all four tokenizers (`python -m experiments.run_sensitivity`,
`results/sensitivity_case/`). LexFront's rule is then active: its WORD tokens are lowercased while the baselines keep
case, so this is a robustness check, not a like-for-like comparison.

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
-
<!-- END table:sensitivity_ci -->

<!-- BEGIN obs:sensitivity_case -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: LexFront+LR minus NLTK+LR accuracy is -0.91 points with case kept vs -1.59 in the main grid; LR accuracy change from switching TF-IDF case folding off: whitespace -0.91, nltk -0.71, tweet -0.59, lexfront -0.03.
- **go_emotions**: LexFront+LR minus NLTK+LR accuracy is +1.15 points with case kept vs +0.09 in the main grid; LR accuracy change from switching TF-IDF case folding off: whitespace -0.87, nltk -1.09, tweet -1.26, lexfront -0.02.
- **ag_news**: LexFront+LR minus NLTK+LR accuracy is -0.18 points with case kept vs +0.00 in the main grid; LR accuracy change from switching TF-IDF case folding off: whitespace +0.16, nltk +0.22, tweet +0.24, lexfront +0.04.

_Robustness check, not a like-for-like comparison: with TF-IDF case folding off, LexFront lowercases its own WORD tokens while the baselines keep case. One training run per cell, no confidence intervals for this grid._
<!-- END obs:sensitivity_case -->

### Which normalization rule matters? (per-rule ablation)
LexFront + LR with one normalization rule switched off at a time, everything else fixed
(`python -m experiments.run_ablation`, `results/ablation/`, protocol in `docs/versions/v4-ablation.md`). Main setting
(TF-IDF lowercases, as the main grid): no_url, no_mention, no_number, no_elongated, no_slang, and all_off (the lexer's
segmentation only). The lowercase rule cannot have an effect there, because TF-IDF lowercases the text before LexFront
sees it; it is tested in the case-kept setting (TF-IDF `lowercase=False`, as the sensitivity grid): no_lowercase and
all_off_cs against full_cs. Differences are variant minus full LexFront of the same setting, paired bootstrap over the
test examples (same code, B and seed as `ci_table`), with an exact McNemar test on accuracy. The table says which
removals change the scores; it does not say why.

<!-- BEGIN table:ablation_table -->
-
<!-- END table:ablation_table -->

<!-- BEGIN obs:ablation -->
-
<!-- END obs:ablation -->
