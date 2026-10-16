# Version 4 — tokenizer line-up from 5 to 8

*Status: current (plan). Supersedes the tokenizer row of v3; v1-v3 stay as they are.*

## Changes from v3
| Item | v3 | v4 |
|---|---|---|
| Tokenizers compared | **5**: whitespace, NLTK `word_tokenize`, `TweetTokenizer`, LexFront, LexFront + BPE hybrid (planned) | **8**: the same four existing/LexFront tokenizers, plus **BPE**, **LexFront + BPE hybrid**, **WordPiece**, **Unigram** |
| Datasets, language, packaging | 3 English datasets, Python 3.11 CPU, Colab | unchanged |

`docs/DECISIONS.md` ("Scope cuts", 9 -> 5) records the v3 count; P3 (BPE, hybrid) and P4 (WordPiece, Unigram) in
`docs/PIPELINE.md` bring it to 8.

## Line-up
| # | Tokenizer | Kind | Status at v4 | Pipeline step |
|---|---|---|---|---|
| 1 | whitespace | rule (split on spaces) | done (Module 1) | – |
| 2 | NLTK `word_tokenize` | rule (Treebank) | done (Module 1) | – |
| 3 | NLTK `TweetTokenizer` | rule (regex, tweets) | done (Module 1) | – |
| 4 | LexFront | compiler-style lexer + normalizer | done (Module 2) | – |
| 5 | BPE | learned subword, trained on the train split only | planned | P3 |
| 6 | LexFront + BPE hybrid | LexFront pre-tokenizes, BPE splits words; placeholders/UNKNOWN pass-through to be decided in P3 | planned | P3 |
| 7 | WordPiece | learned subword (library trainer), train split only | planned | P4 |
| 8 | Unigram | learned subword (library trainer), train split only | planned | P4 |

All eight use the same interface `tokenize(text) -> list[str]` and the same evaluation loop (TF-IDF settings,
classifiers, seed). No numbers exist for 5-8 yet.
