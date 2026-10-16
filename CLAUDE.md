# LexFront: compiler-style pre-tokenizer for noisy social media text

Compiler Design team project (3 people, 14 days, done alongside other coursework).
Full proposal: https://claude.ai/code/artifact/392bb277-7697-4b90-b7f1-7dd7a7656c00

## What we are building

A lexical analyzer built the way a compiler front end is built (token classes as regular
expressions -> Thompson NFA -> subset-construction DFA -> minimized DFA, plus a symbol table
and error recovery), used as a **pre-tokenizer** for text classification. We compare it with
other tokenizers on noisy English text, holding the classifier fixed.

Research question: does a compiler-style lexer, used as a pre-tokenizer, improve tokenization
quality and downstream classification on noisy English social media text?

## Scope

**Must have**
- PLY lexer with the 11 token classes below (used for all experiments)
- Our own regex -> NFA -> DFA -> minimized-DFA engine for 3-4 token classes only
  (starting point: `lexer/prototype/mini_lexgen.py`; report state counts at each stage)
- 5 tokenizers: NLTK `word_tokenize`, NLTK `TweetTokenizer`, LexFront, BPE, LexFront + BPE
- 3 datasets: TweetTopic, GoEmotions (single-label comments only), AG News (clean control)
- Classifiers: majority baseline + Logistic Regression on TF-IDF
- Ablation: switch off one token class (or slang expansion) at a time

**Only if ahead at the day-7 scope check**
- WordPiece and Unigram tokenizers; Twokenize baseline
- 2024 Bluesky posts for an OOV-rate test (aggregate numbers only, never quote posts)
- Sentence embeddings (all-MiniLM-L6-v2) + Logistic Regression
- One transformer fine-tune (e.g. BERTweet on TweetTopic) on the local GPU

**Cut:** Hinglish, large-model fine-tuning. The Blog Authorship Corpus is optional.

## Token classes (priority order = rule order)

URL, MENTION, HASHTAG, EMOTICON, EMOJI, NUMBER, CONTRACTION, ELONGATED, WORD, PUNCT,
UNKNOWN (fallback / error recovery). Scanner uses longest match, ties go to the earlier rule.
Slang (`u` -> `you`) is NOT its own regex: WORD tokens are looked up in a slang table,
like keyword lookup in a compiler's symbol table.

Note: "letter repeated 3+ times" needs a back-reference in Python regex, which is not regular.
In the DFA engine it must be a union over letters; leave it out of the engine if the state
count explodes, and report that as a finding.

## Repo layout

- `lexer/` PLY lexer, DFA engine, symbol table, slang table; `lexer/prototype/` = DFA prototype
- `subword/` BPE (and optional WordPiece/Unigram) training and the LexFront + BPE hybrid
- `data/` downloaded data and caches (git-ignored)
- `experiments/` one script per experiment
- `results/` JSON output of every run (committed)
- `report/` figures and report sources
- `tests/` pytest tests

## Environment

Everything runs in Docker on a laptop (RTX 5070 Ti 12 GB, 32 GB RAM).
- CPU: `docker compose up --build`
- GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`
- Shell: `docker compose exec lab bash`; tests: `docker compose exec lab pytest`
- GPU check: `python -c "import torch; print(torch.cuda.is_available())"`
- RTX 50-series needs PyTorch >= 2.7 built for CUDA 12.8 (the Dockerfile's default).

Load datasets with Hugging Face `datasets`:
`cardiffnlp/tweet_topic_single`, `google-research-datasets/go_emotions` (config `simplified`),
`fancyzhx/ag_news`. Check each dataset card for its recommended train/test split before
using it; do not invent split names.

## Rules for every run

- Change one thing at a time: the tokenizer varies, classifier and split stay fixed.
- 3 random seeds per run; report mean and spread.
- Main metric: macro-F1. Always report the majority-class baseline next to it.
- Every experiment writes `results/<experiment>/<timestamp>.json` with the full config
  (dataset, split, tokenizer, classifier, seed, library versions) and the metrics.
- Same vocabulary size for all learned tokenizers (start at 16,000).

## Integrity rules (non-negotiable)

- Never type, estimate or invent a result. Numbers in the report come only from `results/`.
- If something fails or a result is disappointing, say so plainly; negative results are valid.
- Do not claim LexFront is "first" or "novel": regex tweet tokenizers and automaton-based
  tokenizers already exist (see the proposal's credibility section).

## How to work with us (we are learning)

- We vibe-code but must be able to explain every module in the viva.
- Before implementing a compiler concept (Thompson, subset construction, minimization,
  longest match, symbol table, error recovery), explain it in 3-5 lines, then implement.
- Prefer small, readable, well-named functions over clever code. Docstrings name the
  compiler concept the code implements.
- Work in small steps; run tests after each step.
- When a decision is ours to make (scope, which dataset split, what to cut), ask.

## Milestones

- After day 4 (35% checkpoint): literature notes, 3 datasets loaded, baseline table
  (majority + LR), PLY lexer passing unit tests.
- After day 7 (scope check): DFA engine on 3-4 classes, BPE and hybrid working.
  Decide whether to add the optional extras.
- Day 13: draft report. Day 14: buffer and viva prep.

## Documentation rules

- Every design change gets a new file in `docs/versions/` with its own Mermaid architecture
  diagram; old version files are never edited. Log it in `CHANGELOG.md`.
- Keep `docs/LEARN.md` current: when you add a concept or file, explain it there in plain
  words with an example from our code, and add viva questions.
