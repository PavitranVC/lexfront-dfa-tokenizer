# Prompt pipeline (one fresh Claude Code session per prompt)

Time figures are estimates, not measurements. State lives in the repo, so each session starts cold.
Skip P5/B5 unless P0-P4 are done by about Oct 28. Run P6 only after P2, P3 and P4 are final.

## Header (paste at the top of every prompt)
> LexFront advisor rules: propose code in 2-3 lines and wait for my OK. Design changes get a `docs/versions/v4-*.md`
> (or the next number) plus a CHANGELOG entry. No invented numbers or references. Read only what you need
> (`docs/HANDOFF.md` first, then `docs/DECISIONS.md` only if needed, `git log --oneline -5`). Do not re-read the
> source tree. Before finishing, overwrite `docs/HANDOFF.md` using the template in `docs/PIPELINE.md`. Commit and push to
> the current branch. Reply using the reply template, under 10 lines.

## Buffer header (paste at the top of every B prompt)
> Buffer pass, no new features. Run `pytest -q`, the touched experiment runner on QUICK_MODE, and the notebook in
> QUICK_MODE. Report failures with the exact error. Fix only what is broken, propose each fix in 2-3 lines first.
> Re-read the diff of the previous prompt adversarially (what would make CI reject it? any number not from JSON?).
> Commit and push. Reply in under 10 lines.

## Handoff file (the only memory between sessions)
`docs/HANDOFF.md` is overwritten (not appended) at the end of every session; max 15 lines; facts only, no prose.
```
# HANDOFF (written by <P/B id>, <date>)
DONE: <1 line, what now exists, with file paths>
NUMBERS: <only figures that later prompts need, each with its JSON path; "none" if none>
OPEN: <failing tests / known bugs, exact error text; "none" if none>
DECISIONS: <new choices made, 1 line each, also added to docs/DECISIONS.md>
NEXT: <next prompt id and the first thing it must do>
FILES TOUCHED: <paths>
CHECK: <the one command that proves it works, e.g. `pytest -q` and expected result>
```

## Reply template (what each session says back to you, under 10 lines)
```
STATUS: done | blocked | needs OK
DID: <1-2 lines>
NUMBERS: <key figures only, from JSON>
RISKS: <what could be wrong, or "none">
NEEDS FROM YOU: <decision or paste, or "nothing">
COMMIT: <short hash>
```
When you paste a reply into a new chat with me (the advisor), paste this block only.

## Sequence

| # | Prompt | Model / effort | Est. time |
|---|---|---|---|
| P0 | Resolve TF-IDF lowercasing vs LexFront's lowercase rule. Options: `lowercase=False` in TF-IDF, or drop the rule from the ablation. Pick one, rerun affected results, log it in `docs/DECISIONS.md`. | Sonnet / low | 0.75 h |
| P1 | Check whether per-example predictions are stored; if not, propose the smallest runner change. Then add bootstrap CIs and a paired test for NLTK+LR vs LexFront+LR on all 3 datasets. Add a CI table to `results/REPORT_NOTES.md`. | Sonnet / medium | 3 h |
| B1 | Buffer: check P0-P1 (stats correctness, seeds, tests, notebook). | Sonnet / medium | 0.75 h |
| P2 | Per-rule normalization ablation: switch each of the 6 rules off one at a time, 3 datasets, LR, with CIs from P1. Write `docs/versions/v4-ablation.md` and a CHANGELOG entry. | Sonnet / medium | 4 h |
| B2 | Buffer: check P2 (every rule really toggles, results from JSON, runtime). | Sonnet / medium | 0.75 h |
| P3 | Add BPE and LexFront + BPE hybrid. First propose how placeholders (`<URL>`, `<NUM>`) and UNKNOWN pass through BPE. Same tokenizer interface and eval loop. Update the version file and CHANGELOG. | Opus / high | 4.5 h |
| B3 | Buffer: check P3 (interface, tests for BPE edge cases, vocab sizes, CI). | Opus / medium | 1 h |
| P4 | Add WordPiece and Unigram via library trainers. Same interface and eval loop; train only on the train split. | Sonnet / medium | 2.25 h |
| B4 | Buffer: check P4 (no test-set leakage into trainers, results, CI). | Sonnet / medium | 0.5 h |
| P5 | (optional) DistilBERT reference on 1-2 datasets, subsampled, Colab GPU. Label it as a reference, not a tokenizer comparison. | Sonnet / medium | 5 h |
| B5 | Buffer: check P5 (subsample seed, memory, numbers only from JSON). | Sonnet / medium | 0.75 h |
| P6 | Regenerate all tables and figures from JSON, run the notebook in QUICK_MODE, confirm CI green. | Haiku / low | 1 h |
| P7 | Consistency pass over `REPORT_NOTES.md`, `LEARN.md`, README, `docs/DECISIONS.md`. Flag every claim the numbers or CIs do not support; never claim "first" or "novel". | Sonnet / medium | 1.5 h |

## Totals (estimates)
* Must-haves (P0, P1, B1, P2, B2, P6, P7): about 11.5 h.
* Plus BPE and WordPiece/Unigram (P3, B3, P4, B4): about 19.75 h.
* Everything including DistilBERT (P5, B5): about 25.5 h.

## Rules
* One phase per session; end each with commit, push and a short report.
* Paste back short summaries, not full JSON, unless something looks wrong.
* Never start a P prompt while the previous B prompt has open failures.
