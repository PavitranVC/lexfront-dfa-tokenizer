# Version 1 — Base paper (reference point)

*Status: frozen. Do not edit; add a new version file when the design changes.*

## What it is
The paper our project starts from. It classifies blog posts with a classic
bag-of-words pipeline.

| Item | Value |
|---|---|
| Corpus | Blog corpus |
| Tokenizer | NLTK |
| Stemmer | `PorterStemmer` |
| Features | scikit-learn `CountVectorizer` |
| Classifier | Multinomial Naive Bayes |
| Result, 40 labels | 35 % accuracy |
| Result, labels merged to 3 | 54 % accuracy |

(The figures are the ones quoted from the paper by the team; we did not re-run
this version.)

## Architecture

```mermaid
flowchart LR
    A[Blog posts] --> B[NLTK tokenizer]
    B --> C[PorterStemmer]
    C --> D[CountVectorizer<br/>bag of words]
    D --> E[Multinomial Naive Bayes]
    E --> F[Label<br/>40 classes, or 3 after merging]
```

## Why we move on
The tokenizer is treated as a fixed, uninteresting preprocessing step. Our
project asks whether a compiler-style lexer designed for noisy text can do
better at that step, with everything else held fixed.
