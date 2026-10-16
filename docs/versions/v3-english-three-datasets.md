# Version 3 — English only, three datasets, five tokenizers

*Status: current. Do not edit once superseded; add a v4 file instead.*

## Changes from v2
| Item | v2 | v3 |
|---|---|---|
| Language | English + Hinglish | **English only** |
| Datasets | 4 | **3** (TweetTopic, GoEmotions single-label, AG News) |
| Tokenizers compared | existing tokenizers + LexFront | **5**: whitespace, NLTK `word_tokenize`, `TweetTokenizer`, LexFront, and LexFront + BPE hybrid (planned) |
| Packaging | Docker | **Docker dropped**; plain Python 3.11, CPU only |
| Presentation | – | **Google Colab notebook** |

Five-tokenizer line-up: the first four exist in the review-1 scope (Modules 1
and 2). BPE and the LexFront + BPE hybrid are planned for later and are *out of
scope for review 1*.

## Architecture

```mermaid
flowchart TB
    subgraph Data
      D1[TweetTopic single]:::d
      D2[GoEmotions simplified<br/>single-label only]:::d
      D3[AG News]:::d
    end
    subgraph Tokenizers["Tokenizers (interface: tokenize(text) -> list[str])"]
      T1[whitespace]
      T2[NLTK word_tokenize]
      T3[TweetTokenizer]
      T4[LexFront]
      T5[LexFront + BPE<br/>later]
    end
    subgraph LexFront
      RX[Token spec<br/>URL MENTION HASHTAG EMOTICON EMOJI<br/>NUMBER CONTRACTION ELONGATED WORD PUNCT UNKNOWN]
      PLY[PLY lexer<br/>longest match, ties to earlier rule]
      DFA[Own DFA engine<br/>regex parser -> NFA -> DFA -> min DFA -> scanner]
      NORM[Normalizer<br/>switchable rules + slang table]
      SYM[(Symbol table)]
      RX --> PLY --> NORM --> SYM
      RX -. 4 regular classes .-> DFA
      DFA -. agreement test .- PLY
    end
    Data --> Tokenizers
    T4 --- LexFront
    Tokenizers --> TF[TF-IDF<br/>tokenizer=fn]
    TF --> C1[Majority baseline]
    TF --> C2[Multinomial NB<br/>existing method]
    TF --> C3[Logistic Regression]
    C1 & C2 & C3 --> M[Accuracy + macro-F1<br/>results/*.json]
    Tokenizers --> I[Intrinsic metrics<br/>OOV, fertility, vocab, throughput]
    M & I --> COLAB[Colab notebook<br/>LexFront_Review1.ipynb]
    classDef d fill:#eef,stroke:#88a;
```

## Scope for review 1 (30–40 %)
Module 1 (data, three baseline tokenizers, three classifiers) and Module 2
(LexFront, own DFA engine, intrinsic metrics, LexFront + NB/LR).
Out of scope today: BPE, LexFront + BPE hybrid, WordPiece, Unigram, ablation,
transformers.
