| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | BPE+LR minus NLTK+LR (tokenizer only) | +1.00 [-1.20, +3.20] | +0.09 [-1.61, +1.82] | no / no | 0.487 |
| tweet_topic | LexFront+BPE+LR minus LexFront+LR | -0.60 [-2.20, +1.20] | -0.66 [-2.11, +0.79] | no / no | - |
| tweet_topic | LexFront+BPE+LR minus BPE+LR | -1.20 [-3.20, +0.80] | -1.14 [-2.60, +0.08] | no / no | - |
| tweet_topic | LexFront+BPE+LR minus NLTK+LR | -0.20 [-2.40, +2.00] | -1.06 [-2.93, +0.74] | no / no | - |
| go_emotions | BPE+LR minus NLTK+LR (tokenizer only) | +0.60 [-0.60, +2.00] | +0.51 [-0.11, +1.33] | no / no | 0.581 |
| go_emotions | LexFront+BPE+LR minus LexFront+LR | +0.40 [-0.40, +1.40] | +0.31 [-0.13, +0.91] | no / no | - |
| go_emotions | LexFront+BPE+LR minus BPE+LR | +0.40 [-1.00, +1.80] | -0.04 [-0.79, +0.65] | no / no | - |
| go_emotions | LexFront+BPE+LR minus NLTK+LR | +1.00 [-0.20, +2.40] | +0.47 [-0.20, +1.22] | no / no | - |
| ag_news | BPE+LR minus NLTK+LR (tokenizer only) | -0.80 [-2.80, +1.00] | -0.72 [-2.63, +1.13] | no / no | 0.541 |
| ag_news | LexFront+BPE+LR minus LexFront+LR | -0.60 [-2.00, +0.80] | -0.57 [-1.99, +0.81] | no / no | - |
| ag_news | LexFront+BPE+LR minus BPE+LR | -0.40 [-1.80, +1.00] | -0.40 [-1.82, +1.10] | no / no | - |
| ag_news | LexFront+BPE+LR minus NLTK+LR | -1.20 [-3.20, +0.80] | -1.12 [-3.05, +0.87] | no / no | - |
