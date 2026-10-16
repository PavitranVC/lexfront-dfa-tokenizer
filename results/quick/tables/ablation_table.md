| dataset | setting | variant | TF-IDF features | acc / F1 % | acc minus full, pts [95% CI] | F1 minus full, pts [95% CI] | CI excludes 0 (acc / F1) | predictions changed | McNemar p (acc) |
|---|---|---|---|---|---|---|---|---|---|
| tweet_topic | TF-IDF lowercase (main grid) | full (reference) | 3534 | 66.60 / 31.30 | - | - | - | - | - |
| tweet_topic | TF-IDF lowercase (main grid) | no_url | 3534 | 66.60 / 31.30 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| tweet_topic | TF-IDF lowercase (main grid) | no_mention | 3783 | 67.20 / 31.30 | +0.60 [-0.40, +1.60] | +0.00 [-0.84, +0.68] | no / no | 7 | 0.453 |
| tweet_topic | TF-IDF lowercase (main grid) | no_number | 3610 | 65.20 / 30.91 | -1.40 [-3.20, +0.20] | -0.39 [-1.62, +0.89] | no / no | 33 | 0.167 |
| tweet_topic | TF-IDF lowercase (main grid) | no_elongated | 3532 | 67.00 / 31.46 | +0.40 [+0.00, +1.00] | +0.16 [+0.00, +0.41] | no / no | 2 | 0.500 |
| tweet_topic | TF-IDF lowercase (main grid) | no_slang | 3551 | 67.20 / 31.78 | +0.60 [+0.00, +1.40] | +0.48 [+0.00, +1.23] | no / no | 5 | 0.250 |
| tweet_topic | TF-IDF lowercase (main grid) | all_off | 3874 | 65.40 / 30.52 | -1.20 [-3.00, +0.60] | -0.78 [-1.75, +0.08] | no / no | 27 | 0.263 |
| tweet_topic | case kept (TF-IDF lowercase=False) | full_cs (reference) | 3514 | 66.80 / 31.39 | - | - | - | - | - |
| tweet_topic | case kept (TF-IDF lowercase=False) | no_lowercase | 4009 | 65.80 / 31.11 | -1.00 [-3.00, +1.00] | -0.28 [-2.06, +1.53] | no / no | 46 | 0.442 |
| tweet_topic | case kept (TF-IDF lowercase=False) | all_off_cs | 4340 | 66.00 / 31.57 | -0.80 [-3.20, +1.80] | +0.18 [-1.66, +2.03] | no / no | 69 | 0.644 |
| go_emotions | TF-IDF lowercase (main grid) | full (reference) | 1779 | 43.60 / 14.06 | - | - | - | - | - |
| go_emotions | TF-IDF lowercase (main grid) | no_url | 1779 | 43.60 / 14.06 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_mention | 1779 | 43.60 / 14.06 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_number | 1813 | 43.40 / 13.90 | -0.20 [-0.60, +0.00] | -0.16 [-0.54, +0.00] | no / no | 1 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_elongated | 1777 | 43.60 / 14.06 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | no_slang | 1791 | 43.40 / 13.90 | -0.20 [-0.60, +0.00] | -0.16 [-0.56, +0.08] | no / no | 3 | 1.000 |
| go_emotions | TF-IDF lowercase (main grid) | all_off | 1823 | 43.40 / 13.90 | -0.20 [-0.60, +0.00] | -0.16 [-0.56, +0.08] | no / no | 3 | 1.000 |
| go_emotions | case kept (TF-IDF lowercase=False) | full_cs (reference) | 1779 | 43.60 / 14.06 | - | - | - | - | - |
| go_emotions | case kept (TF-IDF lowercase=False) | no_lowercase | 1930 | 41.00 / 11.06 | -2.60 [-4.20, -1.00] | -3.00 [-5.79, -0.62] | yes / yes | 25 | 0.002 |
| go_emotions | case kept (TF-IDF lowercase=False) | all_off_cs | 1971 | 41.00 / 11.06 | -2.60 [-4.20, -1.00] | -3.00 [-5.79, -0.62] | yes / yes | 25 | 0.002 |
| ag_news | TF-IDF lowercase (main grid) | full (reference) | 5450 | 85.20 / 85.14 | - | - | - | - | - |
| ag_news | TF-IDF lowercase (main grid) | no_url | 5452 | 85.20 / 85.14 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | no_mention | 5450 | 85.20 / 85.14 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | no_number | 5597 | 86.20 / 86.17 | +1.00 [-0.20, +2.20] | +1.03 [-0.11, +2.26] | no / no | 11 | 0.180 |
| ag_news | TF-IDF lowercase (main grid) | no_elongated | 5449 | 85.20 / 85.14 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no / no | 0 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | no_slang | 5452 | 85.40 / 85.37 | +0.20 [+0.00, +0.60] | +0.23 [+0.00, +0.71] | no / no | 1 | 1.000 |
| ag_news | TF-IDF lowercase (main grid) | all_off | 5600 | 86.20 / 86.17 | +1.00 [-0.20, +2.20] | +1.03 [-0.11, +2.26] | no / no | 11 | 0.180 |
| ag_news | case kept (TF-IDF lowercase=False) | full_cs (reference) | 5452 | 85.20 / 85.14 | - | - | - | - | - |
| ag_news | case kept (TF-IDF lowercase=False) | no_lowercase | 6202 | 84.80 / 84.76 | -0.40 [-2.00, +1.20] | -0.38 [-2.03, +1.28] | no / no | 22 | 0.804 |
| ag_news | case kept (TF-IDF lowercase=False) | all_off_cs | 6350 | 84.40 / 84.29 | -0.80 [-2.80, +1.20] | -0.85 [-2.91, +1.20] | no / no | 30 | 0.557 |
