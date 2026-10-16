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
_pending_
<!-- END table:dfa_states -->

<!-- BEGIN obs:dfa -->
_pending_
<!-- END obs:dfa -->

<!-- BEGIN table:intrinsic_metrics -->
_pending_
<!-- END table:intrinsic_metrics -->

<!-- BEGIN obs:intrinsic -->
_pending_
<!-- END obs:intrinsic -->

<!-- BEGIN table:existing_vs_proposed -->
_pending_
<!-- END table:existing_vs_proposed -->

<!-- BEGIN obs:existing_vs_proposed -->
_pending_
<!-- END obs:existing_vs_proposed -->
