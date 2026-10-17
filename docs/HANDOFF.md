# HANDOFF (written by P3, 2026-10-17, status ready)
DONE: AUDIT_4 A4-1/A4-2 fixed (ablation rule-firing counts `rule_fires`, never-firing variants listed apart). P3: `subword/bpe.py` (BPE, LexFront+BPE
hybrid, deterministic training), factory `build_tokenizer(name, train_texts)`, `python -m experiments.run_bpe` (3 datasets x {bpe, lexfront_bpe} x {NB, LR}).
NUMBERS: results/bpe/stats/bootstrap.json (tables bpe_runs, bpe_table, obs:bpe), vocab 8000: 17 of 24 diff CIs exclude 0, all lower (hybrid < LexFront+LR on
all 3 datasets, acc and F1). results/bpe_v32000/stats/bootstrap.json: 8 of 24 exclude 0, all lower. results/ablation/stats/bootstrap.json: 6 of 42 compared CIs
exclude 0, 3 never-firing variants apart. QUICK: results/quick/bpe/stats/bootstrap.json. SAFETY: 315 pre-existing JSON/prediction files identical to main (script).
OPEN: one training run per cell; no multiplicity correction; two vocab sizes only, not tuned; why splitting words hurts is a hypothesis.
DECISIONS: docs/DECISIONS.md "P3" section (hashtags + UNKNOWN pass through whole; hybrid BPE uses WhitespaceSplit; sorted alphabet seeded for determinism).
NEXT: P4 WordPiece and Unigram, same protocol as docs/versions/v5-bpe-hybrid.md (factory, train split only, fingerprint, determinism check first).
FILES TOUCHED: subword/{__init__,bpe}.py, experiments/{tokenizers,pipeline,run_bpe,run_stats,run_module2,run_ablation,report,observations,review1}.py,
tests/{test_bpe,test_ablation,test_audit_fixes}.py, requirements*.txt, .github/workflows/ci.yml, notebooks/*, results/** (bpe/, bpe_v32000/, quick/bpe/,
ablation firing, regenerated runs), README.md, results/REPORT_NOTES.md, docs/{modules/module2,LEARN,DECISIONS,HANDOFF,HANDOFF_LOG,versions/v4-ablation,versions/v5-bpe-hybrid}.md, CHANGELOG.md
CHECK: `pytest -q` (161 passed), `python notebooks/make_notebook.py --check`, `python -m experiments.render_docs && git diff --exit-code`, CI stats step.
