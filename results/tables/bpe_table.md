| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | BPE+LR minus NLTK+LR (tokenizer only) | -0.41 [-1.27, +0.44] | -1.86 [-3.28, -0.40] | no / yes | 0.397 |
| tweet_topic | LexFront+BPE+LR minus LexFront+LR | -0.82 [-1.56, -0.06] | -2.62 [-3.90, -1.46] | yes / yes | - |
| tweet_topic | LexFront+BPE+LR minus BPE+LR | -2.00 [-2.91, -1.09] | -1.53 [-2.72, -0.32] | yes / yes | - |
| tweet_topic | LexFront+BPE+LR minus NLTK+LR | -2.41 [-3.35, -1.50] | -3.39 [-4.66, -2.16] | yes / yes | - |
| go_emotions | BPE+LR minus NLTK+LR (tokenizer only) | -0.04 [-0.68, +0.59] | -0.57 [-1.37, +0.19] | no / no | 0.946 |
| go_emotions | LexFront+BPE+LR minus LexFront+LR | -0.72 [-1.31, -0.15] | -1.39 [-2.16, -0.68] | yes / yes | - |
| go_emotions | LexFront+BPE+LR minus BPE+LR | -0.59 [-1.09, -0.09] | -0.71 [-1.42, +0.00] | yes / no | - |
| go_emotions | LexFront+BPE+LR minus NLTK+LR | -0.63 [-1.24, +0.00] | -1.27 [-2.11, -0.46] | no / yes | - |
| ag_news | BPE+LR minus NLTK+LR (tokenizer only) | -0.75 [-1.13, -0.37] | -0.75 [-1.14, -0.37] | yes / yes | < 0.001 |
| ag_news | LexFront+BPE+LR minus LexFront+LR | -0.86 [-1.25, -0.49] | -0.86 [-1.25, -0.48] | yes / yes | - |
| ag_news | LexFront+BPE+LR minus BPE+LR | -0.11 [-0.38, +0.18] | -0.11 [-0.39, +0.17] | no / no | - |
| ag_news | LexFront+BPE+LR minus NLTK+LR | -0.86 [-1.28, -0.43] | -0.86 [-1.29, -0.44] | yes / yes | - |
