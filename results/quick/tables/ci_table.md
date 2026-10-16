| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | LexFront+LR minus NLTK+LR (tokenizer only) | +0.40 [-1.40, +2.20] | -0.40 [-1.75, +0.79] | no / no | 0.839 |
| tweet_topic | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | +1.20 [-0.60, +3.20] | +0.25 [-1.03, +1.60] | no / no | - |
| tweet_topic | LexFront+LR minus NLTK+NB (proposed minus existing) | +0.40 [-2.80, +3.40] | +4.66 [+2.21, +7.17] | no / yes | - |
| go_emotions | LexFront+LR minus NLTK+LR (tokenizer only) | +0.60 [-0.20, +1.60] | +0.16 [-0.35, +0.65] | no / no | 0.375 |
| go_emotions | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | +0.60 [-0.20, +1.60] | +0.11 [-0.38, +0.60] | no / no | - |
| go_emotions | LexFront+LR minus NLTK+NB (proposed minus existing) | +8.00 [+5.40, +10.60] | +11.15 [+7.81, +14.24] | yes / yes | - |
| ag_news | LexFront+LR minus NLTK+LR (tokenizer only) | -0.60 [-2.20, +1.20] | -0.55 [-2.25, +1.17] | no / no | 0.648 |
| ag_news | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | -0.80 [-2.40, +0.80] | -0.71 [-2.42, +0.92] | no / no | - |
| ag_news | LexFront+LR minus NLTK+NB (proposed minus existing) | -2.20 [-4.40, +0.00] | -2.15 [-4.39, +0.00] | no / no | - |
