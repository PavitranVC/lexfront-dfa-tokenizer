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

* **TF-IDF lowercases first** (`lowercase=True`, sklearn default) for *every* tokenizer, so LexFront sees lowercased
  text; its own lowercase rule is therefore inactive in the experiments and covered only by unit tests.
  Intrinsic metrics lowercase the text the same way.
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
* **Single seed (0), no confidence intervals.**

## 3. Fallbacks used

FALLBACKS_PLACEHOLDER

## 4. Tables and DRAFT observations

### Datasets
<!-- BEGIN table:datasets -->
_not available: FileNotFoundError(2, 'No such file or directory')_
<!-- END table:datasets -->

<!-- BEGIN obs:datasets -->
**DRAFT observations (generated from the JSON files):**

_not available: FileNotFoundError(2, 'No such file or directory')_
<!-- END obs:datasets -->

### Module 1 results (accuracy / macro-F1, %)
<!-- BEGIN table:module1_results -->
|  |
||
<!-- END table:module1_results -->

<!-- BEGIN obs:module1 -->
**DRAFT observations (generated from the JSON files):**


_Single run per cell (seed 0), no confidence intervals: gaps of about one point or less should not be read as real differences._
<!-- END obs:module1 -->

### Existing (NLTK + NB) vs proposed (LexFront + LR)
<!-- BEGIN table:existing_vs_proposed -->
|  |
||
<!-- END table:existing_vs_proposed -->

<!-- BEGIN obs:existing_vs_proposed -->
**DRAFT observations (generated from the JSON files):**


_Single run per cell (seed 0), no confidence intervals: gaps of about one point or less should not be read as real differences._
<!-- END obs:existing_vs_proposed -->

### Intrinsic metrics
<!-- BEGIN table:intrinsic_metrics -->
_not available: AttributeError("'DataFrame' object has no attribute 'oov_rate_tokens'")_
<!-- END table:intrinsic_metrics -->

<!-- BEGIN obs:intrinsic -->
**DRAFT observations (generated from the JSON files):**


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

Finding: the usual ELONGATED regex `(.)\1{2,}` needs a back-reference, which is outside regular expressions, so it is
not in the DFA engine. For a finite alphabet and run length 3 the language is still regular when spelled out letter by
letter; the last row of the DFA table is that spelled-out version (lowercase only), shown for comparison.

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
            for clf in [majority, naive_bayes, logistic_regression]:
                clf.fit(X_train, y_train)
                record <- {config, accuracy(y_test, clf.predict(X_test)), macro_f1(...), timings, versions}
                write record to results/<module>/<dataset>__<tokenizer>__<clf>.json
    tables, figures, observations are built only from those JSON files
```
