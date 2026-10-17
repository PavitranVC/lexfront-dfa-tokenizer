| dataset | difference | accuracy pts [95% CI] | macro-F1 pts [95% CI] | CI excludes 0 (acc / F1) | McNemar p (acc) |
|---|---|---|---|---|---|
| tweet_topic | BPE+LR minus NLTK+LR (tokenizer only) | +0.12 [-0.53, +0.74] | -0.94 [-1.95, -0.00] | no / yes | 0.788 |
| tweet_topic | LexFront+BPE+LR minus LexFront+LR | +0.21 [-0.26, +0.68] | -0.61 [-1.39, +0.02] | no / no | - |
| tweet_topic | LexFront+BPE+LR minus BPE+LR | -1.50 [-2.27, -0.74] | -0.44 [-1.18, +0.38] | yes / no | - |
| tweet_topic | LexFront+BPE+LR minus NLTK+LR | -1.38 [-2.21, -0.59] | -1.38 [-2.43, -0.40] | yes / yes | - |
| go_emotions | BPE+LR minus NLTK+LR (tokenizer only) | -0.57 [-1.11, -0.02] | -1.15 [-2.09, -0.31] | yes / yes | 0.045 |
| go_emotions | LexFront+BPE+LR minus LexFront+LR | -0.50 [-0.89, -0.15] | -0.83 [-1.29, -0.42] | yes / yes | - |
| go_emotions | LexFront+BPE+LR minus BPE+LR | +0.15 [-0.35, +0.65] | +0.44 [-0.44, +1.41] | no / no | - |
| go_emotions | LexFront+BPE+LR minus NLTK+LR | -0.41 [-0.92, +0.15] | -0.71 [-1.45, +0.04] | no / no | - |
| ag_news | BPE+LR minus NLTK+LR (tokenizer only) | +0.22 [-0.08, +0.53] | +0.22 [-0.08, +0.52] | no / no | 0.168 |
| ag_news | LexFront+BPE+LR minus LexFront+LR | +0.11 [-0.08, +0.30] | +0.11 [-0.08, +0.31] | no / no | - |
| ag_news | LexFront+BPE+LR minus BPE+LR | -0.12 [-0.37, +0.14] | -0.12 [-0.38, +0.14] | no / no | - |
| ag_news | LexFront+BPE+LR minus NLTK+LR | +0.11 [-0.22, +0.43] | +0.10 [-0.22, +0.43] | no / no | - |
