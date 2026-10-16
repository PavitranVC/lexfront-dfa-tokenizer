| dataset | difference | acc pts [95% CI], case kept | acc pts [95% CI], main | F1 pts [95% CI], case kept | F1 pts [95% CI], main | McNemar p (acc), case kept |
|---|---|---|---|---|---|---|
| tweet_topic | LexFront+LR minus NLTK+LR (tokenizer only) | -0.91 [-1.88, +0.09] | -1.59 [-2.38, -0.82] | +0.23 [-0.90, +1.40] | -0.77 [-1.81, +0.32] | 0.073 |
| tweet_topic | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | -0.74 [-1.65, +0.18] | -1.29 [-2.00, -0.59] | +0.41 [-0.77, +1.68] | -0.52 [-1.58, +0.57] | - |
| tweet_topic | LexFront+LR minus NLTK+NB (proposed minus existing) | -0.38 [-1.56, +0.82] | -0.26 [-1.47, +0.91] | +7.64 [+5.96, +9.44] | +7.62 [+5.93, +9.31] | - |
| go_emotions | LexFront+LR minus NLTK+LR (tokenizer only) | +1.15 [+0.46, +1.85] | +0.09 [-0.35, +0.54] | +1.72 [+0.56, +2.97] | +0.11 [-0.51, +0.77] | < 0.001 |
| go_emotions | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | +1.13 [+0.48, +1.81] | -0.11 [-0.46, +0.22] | +1.70 [+0.58, +2.91] | -0.05 [-0.55, +0.47] | - |
| go_emotions | LexFront+LR minus NLTK+NB (proposed minus existing) | +17.45 [+16.08, +18.82] | +17.58 [+16.21, +18.95] | +28.97 [+27.22, +30.95] | +29.09 [+27.32, +31.07] | - |
| ag_news | LexFront+LR minus NLTK+LR (tokenizer only) | -0.18 [-0.55, +0.20] | +0.00 [-0.32, +0.33] | -0.19 [-0.56, +0.19] | -0.01 [-0.33, +0.33] | 0.353 |
| ag_news | LexFront+LR minus TweetTokenizer+LR (tokenizer only) | -0.16 [-0.50, +0.18] | +0.04 [-0.24, +0.32] | -0.16 [-0.51, +0.18] | +0.04 [-0.24, +0.31] | - |
| ag_news | LexFront+LR minus NLTK+NB (proposed minus existing) | +1.45 [+0.93, +1.92] | +1.50 [+1.00, +1.99] | +1.46 [+0.95, +1.94] | +1.51 [+1.01, +2.01] | - |
