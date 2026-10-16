# Version 4 — per-rule normalization ablation (P2)

*Status: current. A design addition next to `v4-tokenizer-lineup.md`; v1-v3 stay as they are.*

## Question
Which of LexFront's six normalization rules changes the downstream classification scores, and in which direction?

## What changes from v3
| Item | v3 | v4 (this file) |
|---|---|---|
| LexFront configurations evaluated | one (every rule on) | the full configuration plus one variant per rule switched off, and every rule off |
| Main grid, sensitivity grid | as in P0/P1 | unchanged: every metric identical (checked by script) |

## Protocol
* **System.** LexFront + TF-IDF + Logistic Regression (single-threaded, seed 0, the main-grid settings). Naive Bayes is
  not ablated. Three datasets, full data (QUICK subsample only in the notebook).
* **Rules** (`lexer/normalize.py`, `NormalizeConfig`): url, mention, number, elongated, lowercase, slang. A variant
  switches exactly one rule off (`lexfront:no_<rule>`); `lexfront:all_off` switches all six off, so its tokens are the
  lexer's lexemes (segmentation only). Variants are registered in `experiments/tokenizers.py` (`LEXFRONT_VARIANTS`)
  and are not part of the main tokenizer list.
* **Main setting** (TF-IDF `lowercase=True`, as the main grid): full, no_url, no_mention, no_number, no_elongated,
  no_slang, all_off (TF-IDF still lowercases). The lowercase rule cannot show an effect here: TF-IDF lowercases the text before LexFront sees it.
* **Case-kept setting** (TF-IDF `lowercase=False`, as the sensitivity grid): full_cs, no_lowercase, all_off_cs.
* **Comparison.** Each variant minus the full variant of its setting, accuracy and macro-F1: paired percentile
  bootstrap over the test examples, the same code, B and seed as P1 (`experiments/stats.py`); exact McNemar test on
  accuracy. No multiplicity correction: 8 comparisons per dataset x 2 metrics, so some intervals can exclude 0 by chance.
* **Cache and reuse.** The run fingerprint includes the `NormalizeConfig` of a LexFront tokenizer, so two variants
  never share a cache entry. The full variants are recomputed inside `results/ablation/` rather than copied; the stats
  file records whether their predictions are identical to the full-LexFront+LR runs of the main and sensitivity grids
  (`reference_vs_grid`).
* **Outputs.** `results/ablation/{main,case}/<dataset>__<tokenizer>__lr.json` (a `:` in a variant name becomes `-` in
  the file name) with `predictions/` sidecars; statistics in `results/ablation/stats/bootstrap.json`; generated table
  `ablation_table` and observation `ablation` (REPORT_NOTES, `docs/modules/module2.md`, LEARN Q33, notebook section 6).

## Commands
    python -m experiments.run_ablation                 # runs (cached by fingerprint) + statistics
    python -m experiments.run_ablation --stats-only    # statistics only, from the stored predictions

## What it cannot show
Training randomness (one run per variant); interactions between rules beyond all-off; why a rule matters (any reason
is a hypothesis). It also cannot test a rule that never fires on a dataset: then the variant is the full system. How
often each rule fires is now counted (AUDIT_4 A4-1): per setting, dataset and split, the documents and lexer tokens
whose normalized form differs from full LexFront (`<out>/<setting>/firing__<dataset>.json`, copied into
`stats/bootstrap.json` as `rule_fires`, table column "rule fires"). On the full data the URL rule never fires in
tweet_topic or go_emotions and the mention rule never fires in ag_news (train and test); the mention rule fires in 6
go_emotions train documents and no test document. Never-firing variants are listed apart in the observation and left
out of the comparison and chance counts. "0 changed predictions" is not a reliable sign of a never-firing rule: a rule
that fires only in train still changes the fit. In the main setting all_off keeps TF-IDF's lowercasing (TF-IDF still
lowercases).
