# LexFront study guide (for the viva)

Read top to bottom once, then use the viva questions in section 6 as a self-test. Every concept comes with
a pointer into our own code. Results are *not* quoted here: they live in `results/` and the generated tables;
anything that depends on a number says "see the table".

## 1. The idea in one paragraph

A **lexer** (scanner) is the first phase of a compiler: it turns a character stream into tokens. Compilers
build lexers from regular expressions: regex → NFA → DFA → minimized DFA → table-driven scanner. Noisy text
(tweets, comments) has the same problem a programming language has — URLs, @mentions, #hashtags, emoticons,
numbers, "sooooo" — so we ask: *if we tokenize noisy text with a real compiler-style lexer plus a
normalizer, does a downstream text classifier do better than with the usual tokenizers?* We keep everything
else fixed (TF-IDF, classifier, seed, splits) so that only the tokenizer changes.

## 2. Concepts, each with an example from our code

**Token, lexeme, token class.** A *lexeme* is the matched text (`sooooo`), a *token class* is its category
(`ELONGATED`), a *token* is the pair (plus position). `lexer/dfa/scanner.py::Token(cls, lexeme, pos)`.

**Regular expression.** A pattern built from literals, concatenation, `|`, `*`, `+`, `?`, character classes.
Example: HASHTAG is `#[letters digits _]+` in `lexer/token_spec.py`.

**Token specification and priority.** One regex per class, listed in priority order (`SPECS` in
`lexer/token_spec.py`). When two classes match the same length the earlier one wins.

**Maximal munch / longest match.** At each position take the longest lexeme any class matches. `don't`
is one CONTRACTION (5 characters) rather than WORD `don` (3). Implemented in `lexer/ply_lexer.py::t_ANY`
(and in the DFA scanner by remembering the last accepting state).

**PLY.** A Python Lex-Yacc library. PLY's own master regex is an *ordered alternation* (first matching rule
wins), which is not longest match; so we use PLY for the token stream, positions, ignored whitespace and the
`t_error` hook, and one dispatcher rule that tries every class and keeps the longest match.

**Thompson NFA.** Each regex operator becomes a small automaton with one start and one end state, glued with
ε-edges. `lexer/dfa/nfa.py::_fragment`. Our variant creates two states per AST node, which is why the NFA for
MENTION has 9 states (see the DFA state-count table) even though a hand-drawn one has fewer.

**ε-closure and subset construction.** An NFA can be in many states at once; a DFA state *is a set of NFA
states*. `lexer/dfa/dfa.py::epsilon_closure`, `subset_construction`. The DFA is deterministic: one move per symbol.

**Rule priority inside the DFA.** A DFA state accepts the *lowest-numbered* rule among the accepting NFA
states in its set (`_accepting_rule`). That is how "earlier rule wins ties" survives determinization.

**DFA minimization.** Merge states that behave identically. We use Moore's partition refinement
(`dfa.py::_refine`): start by splitting states by which rule they accept, then split any block whose members
go to different blocks on some symbol, until nothing changes. Hopcroft's algorithm is faster; Moore is simpler
and our automata are tiny.

**Alphabet partition (symbol equivalence classes).** Unicode has over a million code points, so we cut the
code-point line at every range boundary used by any rule; each piece ("atom") gets one DFA column.
`lexer/dfa/alphabet.py`. Our four-class automaton has 31 atoms.

**Table-driven scanner.** The DFA as a 2-D array `table[state][atom]`; the scanner loop just indexes it.
`lexer/dfa/scanner.py::TableScanner`.

**Error recovery (panic mode).** If no class accepts the character at the current position we emit an
`UNKNOWN` token for that one character and continue after it (`scan` in both scanners). A lexer never stops
on bad input.

**Symbol table.** token → id, frequency, class; first sighting assigns the id.
`lexer/symbol_table.py`. In real compilers it also holds types and scopes; ours is the lexical part only.

**Normalization.** A semantic action attached to a token: URL→`<URL>`, MENTION→`<USER>`, NUMBER→`<NUM>`,
elongated letters collapsed to 2, lowercase, slang lookup (`lexer/slang_table.csv`). Each rule has an on/off
switch (`NormalizeConfig`). `lexer/normalize.py`.

**TF-IDF, Naive Bayes, Logistic Regression, macro-F1.** TF-IDF weights a token by how often it appears in a
document and how rare it is across documents. Multinomial NB is the existing method (base paper); LR is a
discriminative linear classifier. Accuracy = share correct; macro-F1 = unweighted mean of per-class F1, so
small classes count as much as big ones. `experiments/classifiers.py`, `metrics.py`.

**Intrinsic metrics.** Measure the tokenizer without a classifier: *OOV rate* (test tokens unseen in train),
*fertility* (tokens per whitespace word), *vocabulary size*, *throughput*. `experiments/metrics.py`.

**Regular vs non-regular.** A back-reference such as `(.)\1{2,}` ("a letter repeated 3+ times") is not part of
the regular-expression formalism, so our regex parser rejects it and the DFA engine does not include ELONGATED.
(Subtle point: for a *finite* alphabet and a *fixed* run length the language is still regular — spell it out
as `aaa|bbb|…|zzz`. We measured that: see `results/module2/dfa_states.json`, key
`elongated_expanded_lowercase`. It needs a much bigger automaton than the other classes.)

## 3. One tweet traced through every stage

Input: `OMG u r sooooo cool!!! @bob https://t.co/x1 :-) §`

1. **Preprocessing** (classifier pipeline only): TF-IDF lowercases the text before calling the tokenizer.
2. **Scanning (PLY, `ply_lexer.py`).** Whitespace is skipped. At each position the dispatcher tries every
   class and keeps the longest match (ties → earlier class). Output (class, lexeme):
   `WORD OMG`, `WORD u`, `WORD r`, `ELONGATED sooooo`, `WORD cool`, `PUNCT !` ×3, `MENTION @bob`,
   `URL https://t.co/x1`, `EMOTICON :-)`, `UNKNOWN §`.
   Note `cool!!!` is split: WORD takes letters/digits only, each `!` is a PUNCT.
3. **Normalization (`normalize.py`).** `OMG`→lowercase `omg`→slang→`oh my god` (three tokens); `u`→`you`;
   `r`→`are`; `sooooo`→`soo`; `@bob`→`<USER>`; URL→`<URL>`; emoticon and `§` unchanged.
4. **Symbol table.** If one is attached, each normalized token is added: first time → new id, frequency 1.
5. **Result of `tokenize()`:** `oh my god you are soo cool ! ! ! <USER> <URL> :-) §`.
   For comparison, on the lowercased text: whitespace gives `cool!!!`, `@bob`, `https://t.co/x1` as single tokens;
   NLTK breaks the URL into `https`, `:`, `//t.co/x1` and the emoticon into `:`, `-`, `)`; TweetTokenizer
   keeps URL, mention and emoticon intact but does not normalize them.
6. **Features and classification.** TF-IDF turns each document's tokens into a sparse vector; the classifier maps it to a label.

**Inside the DFA** (four-class automaton, 10 minimized states): scanning `#hi` goes q0 → q1 (`#`) → q6 (`h`) →
q6 (`i`); q6 accepts HASHTAG, and at end of input the last accepting position is 3, so the lexeme is `#hi`.
Scanning `:-)` goes q0 → q2 → q7 → q8 and q8 accepts EMOTICON. `@bob` ends in q9, accepting MENTION. State
numbers come from the committed `results/module2/dfa_min.png`.

## 4. File-by-file map

| Path | What it is |
|---|---|
| `lexer/token_spec.py` | The lexical specification: classes, regexes, priority, which go into the DFA |
| `lexer/ply_lexer.py` | PLY lexer: longest match, ties to earlier rule, UNKNOWN recovery |
| `lexer/normalize.py`, `slang_table.csv` | Normalization rules (each switchable) and slang lookup |
| `lexer/symbol_table.py` | token → id, frequency, class |
| `lexer/lexfront.py` | `tokenize(text) -> list[str]`, the same interface as the baselines |
| `lexer/dfa/regex_parser.py` | Regex text → AST (plus constructor functions) |
| `lexer/dfa/alphabet.py` | Splits Unicode into atoms |
| `lexer/dfa/nfa.py` | Thompson construction |
| `lexer/dfa/dfa.py` | Subset construction and minimization |
| `lexer/dfa/scanner.py` | Table-driven longest-match scanner with UNKNOWN recovery |
| `lexer/dfa/compiler.py` | Pipeline driver, state counts, Graphviz export |
| `lexer/prototype/mini_lexgen.py` | Earlier one-file prototype (kept for reference) |
| `experiments/data.py` | Loads the three Hugging Face datasets |
| `experiments/tokenizers.py` | Whitespace, NLTK, TweetTokenizer, LexFront behind one interface |
| `experiments/classifiers.py`, `metrics.py` | TF-IDF + majority/NB/LR; accuracy, macro-F1, intrinsic metrics |
| `experiments/pipeline.py` | The evaluation loop; one JSON per run |
| `experiments/run_module1.py`, `run_module2.py`, `dfa_report.py`, `run_datasets.py` | Command-line entry points |
| `experiments/report.py`, `figures.py`, `observations.py` | Tables, figures, observations — all from JSON |
| `experiments/review1.py` | Functions the notebook calls |
| `notebooks/LexFront_Review1.ipynb` (+ `make_notebook.py`) | The Colab notebook and its generator |
| `results/` | JSON results, tables, figures, `REPORT_NOTES.md` |
| `docs/versions/` | Frozen design versions v1–v3 |
| `docs/modules/` | Module 1 and Module 2 write-ups |
| `tests/` | Fast pytest suite |

## 5. How to run

```
pip install -r requirements.txt
python -m experiments.run_datasets
python -m experiments.run_module1
python -m experiments.run_module2
python -m experiments.render_docs     # refresh tables/observations inside the docs
pytest -q
```

## 6. Viva questions with honest answers

**1. What is a lexer and why does a compiler need one?**
It converts characters to tokens so the parser deals with a small alphabet of token classes instead of raw characters. It also strips whitespace/comments and reports bad characters.

**2. Walk me through regex → minimized DFA.**
Parse the regex to an AST; Thompson's construction builds an NFA with ε-edges; subset construction makes each DFA state a set of NFA states; minimization merges states that accept the same rule and agree on every transition. See sections 2–3.

**3. Why longest match, and what if two rules match the same length?**
Otherwise `don't` would split as `don` + `'t`. On equal length the earlier rule wins (priority), implemented by taking the lowest rule index among accepting NFA states.

**4. Why not just use PLY's default rule order?**
PLY's master regex is an ordered alternation: the first rule that matches wins even if a later one would match longer. So we added a dispatcher that picks the longest. That is a deliberate design choice, noted in `results/REPORT_NOTES.md`.

**5. Why write your own DFA engine if PLY exists?**
The course is about how lexer generators work, so we implemented the pipeline ourselves and tested it against PLY on the four classes it covers. PLY is what the actual tokenizer uses. The DFA engine is not the production path: a test checks that both give identical tokens on four classes.

**6. How do you know your DFA is correct?**
Tests: the textbook `(a|b)*abb` minimizes to 4 states; the minimized DFA gives the same tokens as the unminimized one on random strings; and the DFA scanner equals the PLY lexer on handwritten and 1,500 random strings over the four classes.

**7. Which classes are in the DFA engine and why only those?**
EMOTICON, HASHTAG, MENTION, WORD, as the project scope states. The others are left out by scope. URL, NUMBER, EMOJI and PUNCT could probably be written in our regex subset (URL would need negated classes, which our parser rejects); CONTRACTION as written uses look-ahead.

**8. Why is ELONGATED not in the DFA?**
Its usual regex uses a back-reference, which regular expressions do not have. Honest nuance: with a finite alphabet and run length 3 it is regular if you enumerate letters; we measured that and the automaton is much larger (see the dfa_states table).

**9. What is error recovery in your lexer?**
Panic mode: an unmatched character becomes an UNKNOWN token and scanning resumes after it. Example: `§`, or a CJK character.

**10. What does the symbol table do?**
Maps each normalized token to an integer id, a frequency and the class it was first seen as. It is filled during tokenization when attached. The classification pipeline does not use it; TF-IDF builds its own vocabulary.

**11. Isn't this just TweetTokenizer?**
Partly, and we should say so. TweetTokenizer also keeps URLs, mentions, hashtags and emoticons whole. What LexFront adds: (a) a formal, testable token specification with longest-match/priority semantics and an UNKNOWN recovery path; (b) a normalization layer TweetTokenizer lacks (URL/USER/NUM placeholders, slang, switchable rules); (c) a lexer-generator pipeline we built and verified. Whether that improves classification is an empirical question; see the existing-vs-proposed table, and note that part of any gain can come from the classifier, which is why the table has the mixed cells.

**12. What is new here?**
Not a new algorithm: lexers and subset construction are textbook. The contribution is the *application and controlled comparison*: a compiler-style lexer with normalization as a pre-tokenizer for noisy text, evaluated against existing tokenizers with the classifier held fixed on three datasets, plus intrinsic metrics. We do not claim more than the data show.

**13. Why does the existing-vs-proposed table include LexFront+NB and NLTK+LR?**
The headline pair changes two things at once (tokenizer and classifier). The extra cells separate the effects.

**14. Why TF-IDF with `tokenizer=fn, token_pattern=None`?**
So every tokenizer plugs into the identical vectorizer. `token_pattern=None` silences the warning that the default pattern is ignored when a tokenizer is given.

**15. Why lowercase before tokenizing?**
TF-IDF's `lowercase=True` runs first for all tokenizers (the sklearn default, as in the base paper's CountVectorizer). It keeps the comparison fair; it also means LexFront's own lowercase rule is inactive in the experiments (it is covered by unit tests and the notebook demo).

**16. Why accuracy and macro-F1?**
Accuracy is the headline in the base paper; macro-F1 shows whether small classes are handled, which matters on imbalanced data such as GoEmotions.

**17. Why a majority baseline?**
It is the floor: a classifier worse than "always predict the most common class" has learned nothing useful.

**18. Why did you keep only single-label GoEmotions examples?**
The task here is single-label classification; multi-label examples would need a different protocol.

**19. Which TweetTopic split did you use and why?**
Recorded in each result JSON and in `results/datasets.json` (`split_info`): the COLING 2022 pair if the dataset card confirms it, otherwise a recorded fallback.

**20. Why only one seed? Are differences significant?**
Time. Classifiers here are deterministic, and subsampling uses a fixed seed, but we have no confidence intervals; differences of about a point should not be trusted. The observation cells say so.

**21. What is OOV rate and why does a tokenizer affect it?**
Share of test tokens never seen in training. Placeholders (`<URL>`, `<NUM>`) and elongation collapsing merge many rare strings into frequent ones, which tends to lower OOV; see the intrinsic table for the actual values.

**22. What is fertility?**
Tokens produced per whitespace-separated word. Above 1 means the tokenizer splits words (e.g. punctuation); slang expansion (`idk` → 4 tokens) also raises it.

**23. How does throughput compare, and is it fair?**
We time each tokenizer on the same test texts, best of 3 runs, on this machine; it varies between machines. LexFront tries every class at each position in Python, so it is not built for speed.

**24. Why Moore and not Hopcroft?**
Simplicity; automata here have at most a few hundred states. Hopcroft is O(n log n) and would matter for large specifications.

**25. How are alphabets with Unicode handled without a million columns?**
By partitioning the code-point line into atoms at every range boundary; the transition table has one column per atom.

**26. What are the known weaknesses of LexFront?**
Longest-match with fixed regexes: `5pm` becomes NUMBER `5` + WORD `pm`; `:Dog` begins with the emoticon `:D`; slang is a 30-entry table; no handling of Hinglish or code-mixed text (dropped from scope in v3); `y`→`why` is a questionable slang entry.

**27. Why was Hinglish dropped?**
`docs/versions/v3-english-three-datasets.md`: it needs annotated data and its own slang handling, which does not fit the time. The design change is recorded rather than editing v2.

**28. What happens to emoji?**
They have their own class; each emoji (with optional skin-tone/variation selector) is one token.

**29. How do I reproduce a number in the report?**
Every number comes from a JSON file in `results/`, written by a script (`python -m experiments.run_module1` etc.); the tables are generated from those files by `experiments/report.py`.

**30. What is out of scope for this review?**
BPE and the LexFront + BPE hybrid, WordPiece, Unigram, ablations and transformers.
