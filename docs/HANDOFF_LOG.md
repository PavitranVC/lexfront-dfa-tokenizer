# Handoff log (append-only; sessions write one line, never read this file)

Format: `date | id | status | one-line summary | commit`

2026-10-16 | setup | done | DECISIONS.md, PIPELINE.md, HANDOFF.md created | (see git log)
2026-10-16 | FIXAUDIT | blocked | audit fixes done; full-data rerun changed 5 LR metrics, not committed (docs/FIXAUDIT_BLOCKED.md) | claude/fixaudit-1
2026-10-16 | FIXAUDIT | ready | BLAS threads pinned to 1, all results regenerated and reproducible across thread counts; blocker resolved | claude/fixaudit-1
2026-10-16 | FIXAUDIT2 | ready | AUDIT_2 findings fixed; deps split ranges/lock; fingerprint adds data hash + versions; all run JSONs regenerated, metrics identical | claude/fixaudit2-1
2026-10-16 | P0P1 | ready | main grid kept; case-sensitive grid; predictions stored; bootstrap CIs + McNemar; all metrics identical | claude/p0p1-1
