# HANDOFF (written by P2, 2026-10-16, status ready)
DONE: P2 per-rule ablation, LexFront+LR, 3 datasets full data: main setting (TF-IDF lowercase) full/no_url/no_mention/no_number/
no_elongated/no_slang/all_off; case-kept setting full_cs/no_lowercase/all_off_cs; paired bootstrap + McNemar vs full. AUDIT_3 A3-1..A3-5 fixed.
NUMBERS: results/ablation/stats/bootstrap.json (table ablation_table, obs:ablation): 6 of 48 diff CIs exclude 0: no_lowercase (3, lower), all_off_cs
(2, lower, go_emotions), all_off (1, higher, tweet_topic acc); none on ag_news. results/sensitivity_case/stats/bootstrap.json (table sensitivity_ci): tokenizer-only
verdict differs from the main grid on 2 of 3 datasets. QUICK: results/quick/ablation/stats/bootstrap.json.
SAFETY: all 90 main/QUICK/sensitivity runs regenerated (--force); every metric and stored prediction identical to main (script compare).
OPEN: one training run per cell; no multiplicity correction (stated); no reason given for why a rule matters (hypotheses only).
DECISIONS: docs/DECISIONS.md "P2" section; protocol docs/versions/v4-ablation.md.
NEXT: P3 BPE + LexFront+BPE hybrid (propose placeholder/UNKNOWN pass-through first).
FILES TOUCHED: lexer/normalize.py, experiments/{tokenizers,pipeline,report,observations,review1,stats,run_ablation}.py, tests/{test_ablation,test_stats}.py,
.github/workflows/ci.yml, notebooks/*, results/** (ablation/, sensitivity_case/stats/, regenerated runs, tables), results/REPORT_NOTES.md,
docs/{modules/*,LEARN,DECISIONS,HANDOFF,HANDOFF_LOG,versions/v4-ablation}.md, CHANGELOG.md
CHECK: `pytest -q`, `python notebooks/make_notebook.py --check`, `python -m experiments.render_docs && git diff --exit-code`, CI stats step.
