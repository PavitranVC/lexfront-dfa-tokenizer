# Version 2 — LexFront proposal

*Status: frozen. Do not edit; add a new version file when the design changes.*

## What it is
The original project proposal. A compiler-style lexical analyzer (token classes
written as regexes, compiled to a Thompson NFA, then a DFA, then a minimized
DFA, with a symbol table and error recovery) used as a pre-tokenizer for noisy
text classification.

| Item | Value |
|---|---|
| Languages | English and **Hinglish** (romanized Hindi mixed with English) |
| Datasets | 4 (one of them Hinglish) |
| Comparison | LexFront against existing tokenizers, classifier held fixed |
| Packaging | Docker (CPU and GPU compose files) |

## Architecture

```mermaid
flowchart LR
    T[Noisy text<br/>English + Hinglish] --> R
    subgraph LEX[LexFront lexer]
        R[Token regexes] --> N[Thompson NFA]
        N --> D[Subset construction DFA]
        D --> M[Minimized DFA]
        M --> S[Longest-match scanner<br/>+ error recovery]
        S --> ST[(Symbol table)]
    end
    ST --> V[TF-IDF features]
    V --> C[Classifier<br/>fixed across tokenizers]
    C --> E[Accuracy / macro-F1]
```

## Why we move on
See `v3-english-three-datasets.md`: Hinglish needs its own annotated data and
slang handling, which does not fit the available time, and the fourth dataset
was dropped with it.
