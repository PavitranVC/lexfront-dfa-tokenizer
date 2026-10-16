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
6. **Not run / out of scope**: BPE, LexFront + BPE hybrid, WordPiece, Unigram, ablation, transformers; Hinglish; multiple seeds.

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
| go_emotions | whitespace | 34.99 / 1.85 | 37.73 / 4.93 | 53.44 / 31.23 |
| go_emotions | nltk | 34.99 / 1.85 | 37.78 / 4.88 | 55.29 / 33.89 |
| go_emotions | tweet | 34.99 / 1.85 | 37.80 / 4.86 | 55.40 / 33.99 |
| ag_news | whitespace | 25.00 / 10.00 | 90.14 / 90.12 | 91.43 / 91.42 |
| ag_news | nltk | 25.00 / 10.00 | 90.38 / 90.35 | 91.88 / 91.87 |
| ag_news | tweet | 25.00 / 10.00 | 90.22 / 90.20 | 91.83 / 91.81 |
<!-- END table:module1_results -->

<!-- BEGIN obs:module1 -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: best accuracy 72.32% (nltk + LR). Across the three tokenizers LR is +0.93 points vs NB on average; the tokenizer spread is 1.82 points under LR and 0.74 points under NB.
- **go_emotions**: best accuracy 55.40% (tweet + LR). Across the three tokenizers LR is +16.94 points vs NB on average; the tokenizer spread is 1.96 points under LR and 0.07 points under NB.
- **ag_news**: best accuracy 91.88% (nltk + LR). Across the three tokenizers LR is +1.46 points vs NB on average; the tokenizer spread is 0.45 points under LR and 0.24 points under NB.

_Single run per cell (seed 0), no confidence intervals: gaps of about one point or less should not be read as real differences._
<!-- END obs:module1 -->

### Existing (NLTK + NB) vs proposed (LexFront + LR)
<!-- BEGIN table:existing_vs_proposed -->
| dataset | existing NLTK+NB acc / F1 | proposed LexFront+LR acc / F1 | delta acc (pts) | delta F1 (pts) | LexFront+NB acc / F1 | NLTK+LR acc / F1 |
|---|---|---|---|---|---|---|
| tweet_topic | 70.99 / 28.56 | 70.73 / 36.18 | -0.26 | +7.62 | 70.43 / 27.84 | 72.32 / 36.95 |
| go_emotions | 37.78 / 4.88 | 55.32 / 33.95 | +17.54 | +29.07 | 37.78 / 4.81 | 55.29 / 33.89 |
| ag_news | 90.38 / 90.35 | 91.87 / 91.85 | +1.49 | +1.50 | 90.08 / 90.05 | 91.88 / 91.87 |
<!-- END table:existing_vs_proposed -->

<!-- BEGIN obs:existing_vs_proposed -->
**DRAFT observations (generated from the JSON files):**

- **tweet_topic**: proposed (LexFront+LR) vs existing (NLTK+NB): accuracy -0.26 points, macro-F1 +7.62 points. Split of the accuracy change: switching classifier with NLTK held fixed (NB to LR) +1.32; switching tokenizer with LR held fixed (NLTK to LexFront) -1.59; switching tokenizer with NB held fixed -0.56.
- **go_emotions**: proposed (LexFront+LR) vs existing (NLTK+NB): accuracy +17.54 points, macro-F1 +29.07 points. Split of the accuracy change: switching classifier with NLTK held fixed (NB to LR) +17.52; switching tokenizer with LR held fixed (NLTK to LexFront) +0.02; switching tokenizer with NB held fixed +0.00.
- **ag_news**: proposed (LexFront+LR) vs existing (NLTK+NB): accuracy +1.49 points, macro-F1 +1.50 points. Split of the accuracy change: switching classifier with NLTK held fixed (NB to LR) +1.50; switching tokenizer with LR held fixed (NLTK to LexFront) -0.01; switching tokenizer with NB held fixed -0.30.

_Single run per cell (seed 0), no confidence intervals: gaps of about one point or less should not be read as real differences._
<!-- END obs:existing_vs_proposed -->

### Intrinsic metrics
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

Reading the tables together (hand-written, still DRAFT): LexFront has the lowest OOV rate and the smallest vocabulary on
all three datasets, but part of that is by construction (placeholders such as `<URL>`/`<NUM>`, elongation collapsing and slang
expansion merge strings), so it is not by itself evidence of better classification. Where the proposed-vs-existing
accuracy change comes from differs per dataset (generated below from the JSON; no confidence intervals were computed):

<!-- BEGIN obs:attribution -->
<!-- END obs:attribution -->

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
            for clf in [majority, naive_bayes, logistic_regression] (module 1) or [naive_bayes, logistic_regression] (module 2):
                clf.fit(X_train, y_train)
                record <- {config, accuracy(y_test, clf.predict(X_test)), macro_f1(...), timings, versions}
                write record (with a config + code fingerprint) to results/<module>/<dataset>__<tokenizer>__<clf>.json
                # a cached record is reused only while its fingerprint still matches
    tables, figures, observations are built only from those JSON files
```
