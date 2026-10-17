| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | WordPiece+LR minus NLTK+LR (tokenizer only) | +0.20 [-2.00, +2.60] | -0.06 [-1.80, +1.74] | no / no | 1.000 |
| tweet_topic | Unigram+LR minus NLTK+LR (tokenizer only) | +2.80 [+0.20, +5.40] | +2.16 [+0.29, +4.17] | yes / yes | 0.054 |
| tweet_topic | WordPiece+LR minus BPE+LR | -0.80 [-2.60, +1.00] | -0.15 [-1.57, +1.34] | no / no | - |
| tweet_topic | Unigram+LR minus BPE+LR | +1.80 [-0.60, +4.40] | +2.07 [+0.19, +4.12] | no / yes | - |
| tweet_topic | Unigram+LR minus WordPiece+LR | +2.60 [+0.20, +5.20] | +2.22 [+0.43, +4.14] | yes / yes | - |
| go_emotions | WordPiece+LR minus NLTK+LR (tokenizer only) | +0.60 [-0.60, +2.00] | +0.51 [-0.11, +1.33] | no / no | 0.581 |
| go_emotions | Unigram+LR minus NLTK+LR (tokenizer only) | +0.40 [-1.40, +2.20] | -1.91 [-4.10, +0.78] | no / no | 0.832 |
| go_emotions | WordPiece+LR minus BPE+LR | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | - |
| go_emotions | Unigram+LR minus BPE+LR | -0.20 [-2.00, +1.60] | -2.42 [-4.66, +0.30] | no / no | - |
| go_emotions | Unigram+LR minus WordPiece+LR | -0.20 [-2.00, +1.60] | -2.42 [-4.66, +0.30] | no / no | - |
| ag_news | WordPiece+LR minus NLTK+LR (tokenizer only) | +0.20 [-1.40, +2.00] | +0.25 [-1.49, +2.03] | no / no | 1.000 |
| ag_news | Unigram+LR minus NLTK+LR (tokenizer only) | -1.20 [-3.20, +0.80] | -1.16 [-3.11, +0.84] | no / no | 0.327 |
| ag_news | WordPiece+LR minus BPE+LR | +1.00 [-0.60, +2.60] | +0.97 [-0.63, +2.63] | no / no | - |
| ag_news | Unigram+LR minus BPE+LR | -0.40 [-2.60, +2.00] | -0.44 [-2.61, +1.86] | no / no | - |
| ag_news | Unigram+LR minus WordPiece+LR | -1.40 [-3.40, +0.80] | -1.41 [-3.56, +0.79] | no / no | - |
