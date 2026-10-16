# REVIEW 1 PLAN: Modules 1 and 2 by 17:00 today

Read CLAUDE.md first. Today's goal: implement and run **Module 1** and **Module 2** and produce
report-ready tables and figures. Work through the phases in order **without waiting for my
confirmation**. Only stop to ask me if you are truly blocked (missing credentials, a decision
that changes scope). Commit to git at the end of every phase.

Time boxes are hard limits. If a step is stuck for more than 20 minutes, take its listed
fallback and note it in `results/REPORT_NOTES.md`. A working simpler version beats a broken
ambitious one.

## Ground rules for today

- **No Docker today.** Use a local virtual environment on CPU. GPU is not needed for Modules 1-2.
- Never invent or estimate a number. Every number in tables, figures and notes comes from a
  JSON file in `results/` written by a script you ran.
- Keep code small and readable. Docstrings name the compiler concept the code implements.
- Write pytest tests for each module, but keep them fast (under 1 minute total).
- Keep runs fast: if any single experiment takes more than 10 minutes on CPU, subsample the
  training set (fixed seed) and record the subsample size in the config.

## Phase 0 (09:30-10:00): environment

1. Create and activate a venv (Windows Git Bash: `python -m venv .venv && source .venv/Scripts/activate`;
   Linux/Mac: `source .venv/bin/activate`). Add `.venv/` to `.gitignore`.
2. `pip install pandas numpy scikit-learn scipy nltk datasets ply tokenizers matplotlib pytest tqdm emoji graphviz`
3. Download NLTK data: `punkt`, `punkt_tab`, `stopwords`.
4. Check that the Graphviz `dot` command exists. If not, still produce `.dot` files and tell me
   to install Graphviz; do not block on it.
5. Smoke test: import every package, print versions to `results/env.json`.

## Phase 1 (10:00-12:00): MODULE 1, data pipeline and existing methodology

Goal: reproduce the existing approach (the base paper's NLTK tokenization + Naive Bayes) and
strong baselines on our 3 datasets, with honest evaluation.

1. `experiments/data.py`: one loader per dataset, each returning `(train_texts, train_labels,
   test_texts, test_labels, label_names)`:
   - TweetTopic: `cardiffnlp/tweet_topic_single`. Read the dataset card and use the split
     pair intended for the COLING 2022 paper if the card confirms it (e.g. `train_coling2022`
     / `test_coling2022`). Write the chosen split names and why into REPORT_NOTES.md.
   - GoEmotions: `google-research-datasets/go_emotions`, config `simplified`. Keep only
     examples with exactly one label. Use its train and test splits.
   - AG News: `fancyzhx/ag_news`, train and test splits.
   - Cache to `data/`. Record row counts and class counts per split in `results/datasets.json`.
2. `experiments/tokenizers_baseline.py`: wrapper functions with one interface
   `tokenize(text) -> list[str]` for: whitespace split, NLTK `word_tokenize`, NLTK `TweetTokenizer`.
3. `experiments/classify.py`: given a tokenizer function, build TF-IDF features
   (`TfidfVectorizer(tokenizer=..., lowercase=True, token_pattern=None)`), then run:
   - majority-class baseline (`DummyClassifier(strategy="most_frequent")`)
   - Multinomial Naive Bayes (the existing methodology)
   - Logistic Regression (`max_iter=1000`)
   Report accuracy and macro-F1. Use 3 seeds where randomness exists (e.g. training-set
   subsampling or solver shuffling); if a run is fully deterministic, run once and say so.
4. `experiments/run_module1.py`: run every tokenizer x classifier x dataset combination and
   save each result to `results/module1/<dataset>_<tokenizer>_<classifier>.json` with the full
   config (dataset, split, tokenizer, classifier, seeds, subsample size, library versions).
5. Tests: loaders return non-empty data, labels within range, no empty texts; tokenizers return lists.
6. Commit: "Module 1: data pipeline and baselines".

Fallback: if a dataset fails to download, continue with the other two and note it.

## Phase 2 (12:00-15:00): MODULE 2, LexFront lexical analyzer (new contribution)

### 2a. Token specification and PLY lexer (12:00-13:00)

1. `lexer/token_spec.py`: the single source of truth. A list of token classes in priority order,
   each with name, regex, and normalization action:
   URL, MENTION, HASHTAG, EMOTICON, EMOJI, NUMBER, CONTRACTION, ELONGATED, WORD, PUNCT,
   and UNKNOWN as the fallback (error recovery). Start from the draft table in CLAUDE.md.
2. `lexer/ply_lexer.py`: PLY lexer built from the spec. Longest match, ties go to the earlier rule.
   Unmatched characters become UNKNOWN tokens and scanning continues (panic-mode style recovery).
   Output: list of `(token_class, lexeme, normalized_form)`.
3. `lexer/normalize.py`: URL -> `<URL>`, MENTION -> `<USER>`, NUMBER -> `<NUM>`, ELONGATED
   collapses runs of 3+ letters to 2, WORD lowercased; slang table lookup on WORD tokens.
   Create `lexer/slang_table.csv` with about 30 common abbreviations as a starter
   (u->you, ur->your, pls->please, idk->i don't know, etc.); my teammates will extend it.
   Each normalization is a switch (dict of booleans) so the ablation can turn rules off later.
4. `lexer/symbol_table.py`: hash table mapping each normalized token to id, frequency and token
   class. Built from training data; becomes the vocabulary.
5. `lexer/lexfront.py`: `tokenize(text) -> list[str]` returning normalized forms, same interface
   as the baselines.
6. Tests: at least 25 tricky cases (URLs, `@user`, `#tag`, `:-)`, `<3`, emoji, `sooo`, `can't`,
   `2,000`, mixed punctuation, empty string, unknown characters). Expected outputs written by hand.

### 2b. Own DFA engine for 3-4 token classes (13:00-14:30)

Extend `lexer/prototype/mini_lexgen.py` into a package `lexer/dfa/`:

1. `regex_parser.py`: parse a small regex syntax (literals, character classes like `[a-z]`,
   concatenation, `|`, `*`, `+`, `?`, parentheses) into an AST.
   Fallback if this takes too long: build ASTs with constructor functions as the prototype does.
2. `thompson.py`: AST -> NFA (Thompson's construction).
3. `subset.py`: NFA -> DFA (subset construction with epsilon-closure). Accepting states carry
   the highest-priority token class.
4. `minimize.py`: DFA minimization (Hopcroft or partition refinement).
5. `scanner.py`: table-driven longest-match scanner with UNKNOWN error recovery.
6. Apply it to 4 classes: EMOTICON, HASHTAG, MENTION, WORD (add URL only if time allows).
   Do not include ELONGATED; the "letter repeated 3+ times" rule needs a back-reference in Python
   regex and blows up as a DFA. Note this in REPORT_NOTES.md as a finding.
7. Save NFA, DFA and minimized-DFA state and transition counts to `results/module2/dfa_states.json`.
   Export `.dot` and `.png` (if Graphviz exists) of the minimized DFA to `results/figures/`.
8. Equivalence test: on the shared test cases restricted to these 4 classes, the DFA scanner and
   the PLY lexer must produce identical tokens. Report any mismatch rather than hiding it.

### 2c. Module 2 experiments (14:30-15:00)

1. `experiments/intrinsic.py`: for whitespace, NLTK, TweetTokenizer and LexFront, on each
   dataset's test texts: OOV rate (vs training vocabulary), fertility (tokens per whitespace word),
   vocabulary size, throughput (tokens/second, same machine). Save to `results/module2/intrinsic_*.json`.
2. `experiments/run_module2.py`: LexFront with Naive Bayes and Logistic Regression on all 3
   datasets, same splits and settings as Module 1. Save like Module 1.
3. Commit: "Module 2: LexFront lexer, DFA engine, experiments".

## Phase 3 (15:00-16:30): report-ready outputs

1. `experiments/make_tables.py`: read ONLY the JSON files and write to `results/tables/`:
   - `datasets.md`: dataset, text type, train/test sizes, number of classes
   - `module1_results.md`: accuracy and macro-F1 for every tokenizer x classifier x dataset,
     with the majority baseline in each block
   - `existing_vs_proposed.md`: existing method (NLTK + Naive Bayes) vs proposed (LexFront +
     Logistic Regression) vs LexFront + Naive Bayes, per dataset, same splits
   - `intrinsic.md`: OOV, fertility, vocabulary size, throughput per tokenizer and dataset
   - `dfa_states.md`: NFA, DFA and minimized state counts
2. `experiments/make_figures.py` writing PNGs to `results/figures/`:
   - grouped bar chart of macro-F1 per dataset per tokenizer (Logistic Regression)
   - bar chart existing vs proposed per dataset
   - OOV rate per tokenizer per dataset
   Label axes with units; include the majority baseline as a reference line where sensible.
3. `results/REPORT_NOTES.md`: a factual summary for the report writers:
   - what was implemented in Modules 1 and 2 (file names, one line each)
   - split choices, subsample sizes, seeds, anything that failed or used a fallback
   - for each table: 2-3 bullet "observations" stating only what the numbers show
     (e.g. "LexFront + LR macro-F1 is X vs Y for NLTK + NB on TweetTopic"), no speculation.
     Mark each as DRAFT for human review.
   - pseudocode (plain text, compiler-textbook style) for: (1) regex -> NFA -> DFA -> minimized DFA,
     (2) longest-match scanner with error recovery, (3) Module 1 evaluation loop.
4. Commit: "Report-ready tables and figures".

## Phase 4 (16:30-17:00): freeze

1. Run the full test suite and every experiment script once more from a clean state.
2. Update README.md with exact commands to reproduce Modules 1 and 2.
3. Final commit and push. Print a short summary: what is done, what used a fallback, what failed.

## Out of scope today (Modules 3-4, final review)

BPE and LexFront + BPE hybrid, WordPiece/Unigram, ablation runs, Bluesky test, embeddings,
transformer fine-tuning, Docker GPU setup.

## Documentation and GitHub (do this alongside every phase, not at the end)

The repo must tell the full story of the project. GitHub renders Mermaid diagrams inside
Markdown, so draw every architecture diagram as a ```mermaid block (no external tools needed).

1. `docs/versions/` with one file per version, each containing: what changed, why, and its
   own architecture diagram (Mermaid). Create these now from CLAUDE.md and the proposal:
   - `v1-base-paper.md`: blog corpus, NLTK + PorterStemmer + CountVectorizer + Naive Bayes,
     35% (40 labels) -> 54% (3 merged labels). Diagram of that pipeline.
   - `v2-lexfront-proposal.md`: compiler-style lexer as pre-tokenizer, Hinglish + 4 datasets,
     9 tokenizers. Diagram.
   - `v3-english-trimmed.md`: English only, 3 datasets, 5 core tokenizers, LR; why Hinglish
     was cut (scope, unconfirmed dataset). Diagram (compiler phases -> classifiers).
   - When code changes the design today (a fallback, a dropped class), add `v3.1-...md`
     with the updated diagram. Never edit an old version file; add a new one.
2. `CHANGELOG.md`: dated entries (newest first) for every phase and design change, linking
   to the version file. Also the "version history" table the report needs.
3. `docs/LEARN.md`, written for me (a student) to understand and defend the project in a viva:
   - The project in plain words (1 paragraph), and how it maps to compiler phases
   - Each concept explained simply with a tiny worked example from OUR code: regular
     expressions, NFA, Thompson's construction, epsilon-closure, subset construction, DFA
     minimization, longest match + rule priority, symbol table, error recovery, why
     back-references are not regular, TF-IDF, Naive Bayes, Logistic Regression, macro-F1,
     majority baseline, OOV rate, fertility
   - A walkthrough: what happens to one example tweet at each stage, with real output
   - A file-by-file map: what each source file does, in one or two lines
   - "Viva questions": at least 25 likely questions with short answers, including tough ones
     ("isn't this just NLTK's TweetTokenizer?", "why not just use BERT?", "why did accuracy
     not improve?", "what is new here?") answered honestly
4. `docs/modules/module1.md` and `module2.md`: explanation, pseudocode, how to run, results
   (copied from `results/tables`, never typed by hand).
5. GitHub: commit after every phase with clear messages. If a remote named `origin` exists,
   push after every commit; if not, remind me at the end of Phase 0 to create the repo
   `lexfront-dfa-tokenizer` (private) and run
   `git remote add origin <url>` and `git push -u origin main`.
