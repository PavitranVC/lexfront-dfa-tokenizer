| dataset | tokenizer | Naive Bayes acc / F1 (case kept) | Logistic Reg. acc / F1 (case kept) | LR acc change vs main (pts) | LR F1 change vs main (pts) |
|---|---|---|---|---|---|
| tweet_topic | whitespace | 69.73 / 28.81 | 69.58 / 34.58 | -0.91 | -1.18 |
| tweet_topic | nltk | 71.08 / 28.63 | 71.61 / 36.04 | -0.71 | -0.91 |
| tweet_topic | tweet | 70.55 / 28.56 | 71.43 / 35.86 | -0.59 | -0.84 |
| tweet_topic | lexfront | 70.43 / 27.99 | 70.70 / 36.27 | -0.03 | +0.09 |
| go_emotions | whitespace | 37.78 / 4.95 | 52.64 / 29.80 | -0.87 | -1.61 |
| go_emotions | nltk | 37.89 / 5.06 | 54.18 / 32.31 | -1.09 | -1.55 |
| go_emotions | tweet | 37.93 / 5.03 | 54.20 / 32.33 | -1.26 | -1.69 |
| go_emotions | lexfront | 37.78 / 4.81 | 55.34 / 34.03 | -0.02 | +0.06 |
| ag_news | whitespace | 90.22 / 90.19 | 91.59 / 91.57 | +0.16 | +0.16 |
| ag_news | nltk | 90.47 / 90.44 | 92.11 / 92.09 | +0.22 | +0.22 |
| ag_news | tweet | 90.42 / 90.39 | 92.08 / 92.06 | +0.24 | +0.24 |
| ag_news | lexfront | 90.08 / 90.05 | 91.92 / 91.90 | +0.04 | +0.04 |
