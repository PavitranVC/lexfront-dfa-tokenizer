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

_Single run per cell (seed 0), no confidence intervals: gaps of about one point or less should not be read as real differences._
<!-- END obs:existing_vs_proposed -->
