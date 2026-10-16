| dataset | setting | variant | TF-IDF features | acc / F1 % | acc minus full, pts [95% CI] | F1 minus full, pts [95% CI] | CI excludes 0 (acc / F1) | predictions changed | McNemar p (acc) |
|---|---|---|---|---|---|---|---|---|---|
| tweet_topic | TF-IDF lowercase (main grid) | full (reference) | 5350 | 70.73 / 36.18 | - | - | - | - | - |
| tweet_topic | TF-IDF lowercase (main grid) | no_url | 5350 | 70.73 / 36.18 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| tweet_topic | TF-IDF lowercase (main grid) | no_mention | 5776 | 71.14 / 35.94 | +0.41 [-0.03, +0.82] | -0.24 [-1.07, +0.45] | no / no | 66 | 0.070 |
| tweet_topic | TF-IDF lowercase (main grid) | no_number | 5461 | 71.23 / 36.16 | +0.50 [-0.03, +1.00] | -0.02 [-0.57, +0.43] | no / no | 130 | 0.068 |
| tweet_topic | TF-IDF lowercase (main grid) | no_elongated | 5347 | 70.81 / 36.23 | +0.09 [-0.09, +0.26] | +0.05 [-0.09, +0.19] | no / no | 9 | 0.508 |
| tweet_topic | TF-IDF lowercase (main grid) | no_slang | 5373 | 70.67 / 36.04 | -0.06 [-0.35, +0.24] | -0.14 [-0.75, +0.45] | no / no | 34 | 0.845 |
| tweet_topic | TF-IDF lowercase (main grid) | all_off | 5907 | 71.82 / 36.37 | +1.09 [+0.50, +1.71] | +0.19 [-0.61, +0.94] | yes / no | 172 | < 0.001 |
| tweet_topic | case kept (TF-IDF lowercase=False) | full_cs (reference) | 5313 | 70.70 / 36.27 | - | - | - | - | - |
| tweet_topic | case kept (TF-IDF lowercase=False) | no_lowercase | 6236 | 69.79 / 35.37 | -0.91 [-1.71, -0.09] | -0.90 [-1.93, +0.04] | yes / no | 288 | 0.036 |
| tweet_topic | case kept (TF-IDF lowercase=False) | all_off_cs | 6791 | 71.11 / 35.65 | +0.41 [-0.44, +1.27] | -0.62 [-1.89, +0.53] | no / no | 340 | 0.387 |
| go_emotions | TF-IDF lowercase (main grid) | full (reference) | 12035 | 55.36 / 33.97 | - | - | - | - | - |
| go_emotions | TF-IDF lowercase (main grid) | no_url | 12035 | 55.36 / 33.97 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_mention | 12035 | 55.34 / 33.98 | -0.02 [-0.13, +0.07] | +0.01 [-0.05, +0.09] | no / no | 7 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_number | 12218 | 55.25 / 33.91 | -0.11 [-0.26, +0.04] | -0.06 [-0.29, +0.18] | no / no | 23 | 0.267 |
| go_emotions | TF-IDF lowercase (main grid) | no_elongated | 12039 | 55.36 / 34.03 | +0.00 [-0.11, +0.13] | +0.05 [-0.09, +0.22] | no / no | 12 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_slang | 12061 | 55.32 / 33.96 | -0.04 [-0.24, +0.15] | -0.02 [-0.27, +0.25] | no / no | 33 | 0.832 |
| go_emotions | TF-IDF lowercase (main grid) | all_off | 12248 | 55.19 / 33.85 | -0.17 [-0.41, +0.07] | -0.13 [-0.44, +0.19] | no / no | 50 | 0.215 |
| go_emotions | case kept (TF-IDF lowercase=False) | full_cs (reference) | 12035 | 55.34 / 34.03 | - | - | - | - | - |
| go_emotions | case kept (TF-IDF lowercase=False) | no_lowercase | 13854 | 54.27 / 32.13 | -1.07 [-1.72, -0.44] | -1.90 [-3.08, -0.79] | yes / yes | 313 | 0.002 |
| go_emotions | case kept (TF-IDF lowercase=False) | all_off_cs | 14056 | 54.14 / 32.17 | -1.20 [-1.85, -0.54] | -1.86 [-3.07, -0.73] | yes / yes | 322 | < 0.001 |
| ag_news | TF-IDF lowercase (main grid) | full (reference) | 41853 | 91.88 / 91.86 | - | - | - | - | - |
| ag_news | TF-IDF lowercase (main grid) | no_url | 42166 | 91.89 / 91.88 | +0.01 [-0.07, +0.09] | +0.01 [-0.07, +0.09] | no / no | 11 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | no_mention | 41853 | 91.88 / 91.86 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | no_number | 43561 | 91.95 / 91.93 | +0.07 [-0.07, +0.20] | +0.07 [-0.06, +0.20] | no / no | 30 | 0.424 |
| ag_news | TF-IDF lowercase (main grid) | no_elongated | 41861 | 91.92 / 91.90 | +0.04 [-0.04, +0.12] | +0.04 [-0.04, +0.12] | no / no | 10 | 0.508 |
| ag_news | TF-IDF lowercase (main grid) | no_slang | 41869 | 91.91 / 91.89 | +0.03 [-0.03, +0.09] | +0.03 [-0.03, +0.09] | no / no | 9 | 0.688 |
| ag_news | TF-IDF lowercase (main grid) | all_off | 43898 | 91.99 / 91.97 | +0.11 [-0.04, +0.25] | +0.11 [-0.04, +0.25] | no / no | 36 | 0.200 |
| ag_news | case kept (TF-IDF lowercase=False) | full_cs (reference) | 41855 | 91.92 / 91.90 | - | - | - | - | - |
| ag_news | case kept (TF-IDF lowercase=False) | no_lowercase | 55452 | 92.07 / 92.05 | +0.14 [-0.17, +0.46] | +0.14 [-0.17, +0.46] | no / no | 165 | 0.406 |
| ag_news | case kept (TF-IDF lowercase=False) | all_off_cs | 57491 | 92.09 / 92.08 | +0.17 [-0.16, +0.49] | +0.17 [-0.16, +0.49] | no / no | 171 | 0.326 |
