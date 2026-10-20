# Handoff log (append-only; sessions write one line, never read this file)

Format: `date | id | status | one-line summary | commit`

2026-10-16 | setup | done | DECISIONS.md, PIPELINE.md, HANDOFF.md created | (see git log)
2026-10-16 | FIXAUDIT | blocked | audit fixes done; full-data rerun changed 5 LR metrics, not committed (docs/FIXAUDIT_BLOCKED.md) | feature/fixaudit-1
2026-10-16 | FIXAUDIT | ready | BLAS threads pinned to 1, all results regenerated and reproducible across thread counts; blocker resolved | feature/fixaudit-1
2026-10-16 | FIXAUDIT2 | ready | AUDIT_2 findings fixed; deps split ranges/lock; fingerprint adds data hash + versions; all run JSONs regenerated, metrics identical | feature/fixaudit2-1
2026-10-16 | P0P1 | ready | main grid kept; case-sensitive grid; predictions stored; bootstrap CIs + McNemar; all metrics identical | feature/p0p1-1
2026-10-16 | P2 | ready | per-rule ablation (LR, 3 datasets, CIs+McNemar), sensitivity CIs, AUDIT_3 fixes; all metrics identical | feature/p2-1
2026-10-17 | P3 | ready | AUDIT_4 fixes (rule firing counts); BPE + LexFront+BPE hybrid grid (8000, 32000), stats, docs, notebook; pre-existing metrics identical | 58e125d
2026-10-17 | COLLECTION | ready | cleaned 8-dataset collection (tweet_eval x5 added), fair validation-tuned NB/LR, generated report; pre-existing metrics identical | feature/fix-lexfront-notebook-setup-brk26t
2026-10-17 | LARGE | pipeline ready, awaiting owner's Colab run | Sentiment140 ladder + noise test + Colab notebook + CI dry run; results/large not yet produced | feature/fix-lexfront-notebook-setup-brk26t
2026-10-18 | LARGE | done | Sentiment140 full results merged; checks reproduce; verdicts in results/large/REPORT.md | feature/fix-lexfront-notebook-setup-brk26t
