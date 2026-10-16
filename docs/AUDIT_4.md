# Audit #4 — diff audit of P2 (per-rule ablation) and the AUDIT_3 fixes

*Auditor role, read-only on code. Scope: `git diff 97e026a..182a88f` (PR #13), `results/**` checked by script only.*
*Token estimate: about 75k tokens of the 120k budget. One pass plus one verification pass of the top finding.*

## Verdict: **PASS** (0 blocking, 1 major, 1 minor)

No number is wrong. Every stored CI, McNemar test and pre-existing metric I recomputed matches. The major finding is
about how the results are described: three ablation "comparisons" pit full LexFront against itself, because the
removed rule never fires on that dataset. The generated text calls these "not distinguishable from full", when the
plain fact is that the rule never triggers in this data.

## Scope
Covered: the per-rule ablation (code, runs, stats, generated table and observation, v4-ablation protocol, LEARN Q33,
REPORT_NOTES and module2 blocks, DECISIONS, HANDOFF numbers line), the A3-1..A3-5 fixes (CAVEAT text, QUICK stats text
in the notebook, `sensitivity_ci`, the CI rerun-and-diff step, deletion of `n_classes_of`), and the safety checks.

Not covered: line-by-line review of the generated `.ipynb` JSON (only `make_notebook.py` and running the QUICK ablation
section); a full, non-QUICK rerun of the ablation training (only the QUICK set was force-retrained); McNemar and
bootstrap recomputation for tweet_topic and ag_news ablation contrasts (task asked for go_emotions only); the
`report.py` table renderer beyond checking `render_docs` gives no diff.

## What passed
| # | Check | Command / method | Result |
|---|---|---|---|
| 0 | Test suite | `.venv/bin/python -m pytest -q` | `151 passed in 6.43s` |
| 1a | Independent paired bootstrap, go_emotions (own code: sklearn `accuracy_score`/`f1_score(average="macro")`, `default_rng(12345)`, B=2000; no `experiments.*` import) | scratch `recompute.py` | no_lowercase − full_cs acc mine [-1.699, -0.392] vs stored [-1.721, -0.436]; F1 [-3.031, -0.742] vs [-3.085, -0.788]. all_off_cs − full_cs acc [-1.852, -0.566] vs [-1.852, -0.544]; F1 [-2.988, -0.805] vs [-3.065, -0.729]. all_off − full (main) acc [-0.414, +0.065] vs [-0.414, +0.065]; F1 [-0.434, +0.167] vs [-0.443, +0.193]. Largest endpoint gap 0.077 pts = 0.14 × bootstrap SD (0.564). **Tolerance: 0.25 × bootstrap SD per endpoint** (two independent B=2000 percentile runs). All within it. |
| 1b | McNemar from discordant pairs (`scipy.stats.binomtest`) | same | (140, 91) p=0.001532; (145, 90) p=0.000405; (20, 12) p=0.2153: identical to stored |
| 1c | y_true of every variant equals the full variant's | same, all 30 ablation run sidecars | all equal |
| 2 | Rule-firing counts per dataset, on the text LexFront sees (lowercased in main, raw in case) | scratch `fires.py` (`LexFront(all_off).analyze`) | see finding A4-1 |
| 3a | Each switch changes output; `all_off` = six off | `tests/test_ablation.py` (passes); `NormalizeConfig.all_off() == NormalizeConfig(**{r: False})` asserted | pass |
| 3b | Fingerprints distinct and include NormalizeConfig | 30 ablation run JSONs: all have a `normalize` key, 30 distinct `fingerprint.config`; `config_fingerprint` adds `normalize_record` | pass |
| 3c | Lowercase rule inert in main setting | `full(t.lower()) != no_lowercase(t.lower())` over all train+test texts | 0 / 6997, 0 / 40898, 0 / 127600: inert, as the text says; `lowercase` only ablated in `case` (SETTINGS) |
| 4a | Generated observation vs stats JSON | read against `results/ablation/stats/bootstrap.json` | "6 of 48 CIs exclude 0", the per-dataset lists, directions and HANDOFF line 4 all correct |
| 4b | A3-3 `sensitivity_ci` | own bootstrap (rng 777, B=2000), LexFront+LR − NLTK+LR acc, case kept | tweet_topic -0.91 [-1.82, +0.06] (table [-1.88, +0.09]); go_emotions +1.15 [+0.48, +1.85] ([+0.46, +1.85]); ag_news -0.18 [-0.54, +0.17] ([-0.55, +0.20]); point values exact, endpoints within 0.15 SD |
| 4c | A3-2 notebook text "about 2 to 6 times the full-grid width" | ratio of QUICK/full accuracy-difference CI widths | 1.89 to 5.80: fine |
| 4d | Causal language | grep of v4-ablation, LEARN, CHANGELOG, HANDOFF, observations | the text says which removals change scores and labels reasons as hypotheses; no unsupported causal claim found (but see A4-1) |
| 5a | Pre-existing metrics unchanged | scratch `premetrics.py`: each of the 95 modified results JSONs at `97e026a` vs `182a88f`, recursive diff | changed keys only `fingerprint/code` (90), `fingerprint/config` (18), `normalize` (18, new), `timestamp_utc`, `timing_seconds/*`, intrinsic `throughput_*`; **0 metric, CI or prediction changes**; `results/stats/bootstrap.json` and all pre-existing prediction sidecars unmodified; `quick/tables/intrinsic_metrics.md` differs only in the throughput column |
| 5b | A3-4 CI step from a clean checkout | `git clone` → checkout 182a88f → the five commands of the new CI step → `git diff --exit-code results`; then `render_docs` | diff rc=0; no diff from render_docs either |
| 5c | QUICK ablation in the notebook | `review1.run_ablation_section(True, force=…)` in the clean clone | cached 14 s, forced retrain 28 s; forced rerun's stats identical to the committed `results/quick/ablation/stats/bootstrap.json` |
| 5d | Repo growth | `du` of the changed files; `results/ablation` | 384 KB for `results/ablation`; about 1.4 MB for all changed files (11.3k inserted lines, mostly the notebook and results JSON) |
| 5e | Dead code, silent exceptions, tests | read the diff | `n_classes_of` removed and unused; the only new `except` is `FileNotFoundError` → visible "_No CIs…_" text (tested); `test_ablation` asserts real behaviour (one assert, `compute()==written`, re-reads its own output, which is weak but harmless) |

## Findings
| id | severity | file:line | evidence (command → output) | suggested fix | status |
|---|---|---|---|---|---|
| A4-1 | major | `experiments/observations.py:268` (generated "Every other interval includes 0", in REPORT_NOTES / module2 / LEARN blocks); `docs/LEARN.md:261` ("every other removal is not distinguishable from full LexFront"); `docs/versions/v4-ablation.md:41` ("how often URLs occur in a dataset, not measured here"); `docs/DECISIONS.md:25-26` | `fires.py` (LexFront all_off `analyze` on train+test): **URL tokens: tweet_topic 0 train / 0 test, go_emotions 0 / 0** (ag_news 2237 / 117); **MENTION: ag_news 0 / 0; go_emotions 6 train docs (labels 15:1, 18:1, 27:4) / 0 test**. Verification by another path: `full(t.lower()) != variant(t.lower())` → no_url tweet_topic 0/3598 train, 0/3399 test; no_url go_emotions 0/36308, 0/4590; no_mention ag_news 0/120000, 0/7600; no_mention go_emotions 6/36308, 0/4590. So 3 of the 24 "comparisons" are the identical system (CI [+0.00, +0.00], 0 predictions changed), yet the text counts them as intervals that "include 0" and as removals "not distinguishable" from full. That describes a rule that never triggers as a rule with no measurable effect. DECISIONS' proxy ("a rule that never fires is visible as 0 changed predictions") is not reliable either: go_emotions no_mention changes 7 test predictions although MENTION never occurs in the test set. Those 7 changes come only from the training fit (6 train docs). | Record per dataset and split how often each rule fires (count of tokens or docs where the variant's tokens differ from full) in `bootstrap.json` and as a table column. In the observation, list the variants identical to full separately: "rule never fires on this dataset (train and test): not a test of the rule". Leave them out of the comparison and multiplicity counts. Fix LEARN Q33, v4-ablation "What it cannot show" and DECISIONS:25 to match. | CONFIRMED |
| A4-2 | minor | `experiments/observations.py:270` | Stats JSON: 3 variants have 0 changed predictions, so 6 of the 48 CIs are the degenerate [0, 0] and cannot exclude 0. The comparisons are also not independent (all_off contains every single-rule change, all_off_cs contains no_lowercase). So "about 2.4 would exclude 0 by chance" is overstated and assumes independence without saying so. | Count only non-identical variants (42 CIs → about 2.1) and write "roughly, if the intervals were independent". | CONFIRMED |

## Notes (no finding)
* The `full` variants are re-trained, not copied. `reference_vs_grid.identical_predictions` is `true` for all 6
  (setting, dataset) pairs, so the ablation reference equals the main and sensitivity grid runs.
* In the main setting `all_off` still has lowercasing, done by TF-IDF. The text says "the lexer's segmentation only",
  and the protocol explains that TF-IDF lowercases first, so this is consistent but easy to misread. A reader-facing
  "(TF-IDF still lowercases)" next to `all_off` would help.
