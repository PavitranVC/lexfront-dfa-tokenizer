# Decisions log

Newest first. One short entry each: decision, reason, source. Gaps filled on 2026-10-16 from the team's notes; items
already recorded elsewhere are only pointed to. Citations below are from abstracts only and were not re-checked here.

## 2026-10-16 (FIXAUDIT, fixes for docs/AUDIT.md)

### M1: result cache keyed by a fingerprint
- Decision: every run JSON stores `fingerprint = {config, code}` (hash of data subset, tokenizer, vectorizer and
  classifier params, seed, sklearn/nltk versions; hash of `lexer/` and the experiment modules that affect a run).
  `run_cell` reuses a JSON only if both match. The notebook sets `FORCE = QUICK_MODE`, so the CI notebook job and the
  committed notebook recompute every QUICK run. The 33 QUICK run JSONs were regenerated (metrics identical); the full-data
  rerun changed 5 LR metrics, so full JSONs are left as on main (BLOCKED, see `docs/FIXAUDIT_BLOCKED.md`).
- Reason: the file name alone hid stale results (AUDIT M1). No test asserts that committed JSONs match the current
  code hash: that would force every later code change to rerun the full grid; a stale JSON is recomputed on next run.

### M2: proposed-vs-existing attribution is generated, per dataset
- Decision: new `obs:attribution` block (`experiments/observations.py`) replaces the hand-written sentence in
  `results/REPORT_NOTES.md` s.4.
- Reason: the old text said the classifier switch explains "almost all of the gain"; on TweetTopic proposed is below existing.

### m1: intrinsic subsample vs classification subsample
- Decision: `run_module2` rejects `--intrinsic-max-train` together with `--max-test`; `data.subsample` is unchanged.
- Reason: separate RNGs would change the committed QUICK subsets and every QUICK number; the bug is latent.

### m2: slang lookup and the lowercase switch
- Decision: the slang lookup is exact (`lexeme in table`); with `lowercase=False`, `OMG` stays `OMG`, `omg` expands.
- Reason: keeps the two rules independent for the P2 ablation. No result changes (TF-IDF lowercases first; P0 decides that).

### m3/m4: emoji sequences, NFD, `www.`, `heyyy'll`
- Decision: documented as limits in `docs/LEARN.md` Q26/Q28 and pinned by tests; no scanner change.
- Reason: changing the spec would change every LexFront number; out of scope for an audit-fix task.

### m6: dependencies
- Decision: `requirements.txt` pins the versions the results were produced with; pip `graphviz` dropped (unused);
  CI installs the apt package `graphviz` so `dfa_min.png` can be rendered by `dot`.

### m8/m9: errors and side effects in reporting
- Decision: tables and observations turn only `FileNotFoundError` into "_not available_"; other errors raise.
  `report.render_all_tables` builds tables without writing; `render_docs` and the notebook appendix use it.

### m10: CI
- Decision: `permissions: contents: read`, `concurrency`, a step that re-renders docs and checks `git diff --exit-code`,
  and a step that rebuilds the notebook from `make_notebook.py` and compares cell sources.

### m11: overnight mode and tokenizer line-up
- Decision: `docs/PIPELINE.md` gains an "Overnight mode" section that overrides its header; tokenizer plan 5 -> 8 in
  `docs/versions/v4-tokenizer-lineup.md`.

### m12/m13
- Decision: a test checks the DFA numbers and state ids hand-typed in `docs/LEARN.md`; the existing-vs-proposed figure
  skips a missing cell instead of drawing a 0 bar.

## 2026-10-16

### Proposal document (external)
- Decision: the full proposal lives outside the repo.
- Source: https://claude.ai/code/artifact/392bb277-7697-4b90-b7f1-7dd7a7656c00

### Next-phase order
- Decision: (1) bootstrap confidence intervals + paired test (NLTK+LR vs LexFront+LR) -> (2) normalization ablation
  -> (3) BPE and LexFront + BPE hybrid -> (4) WordPiece/Unigram -> (5) optional transformer baseline -> (6) a
  `docs/versions/v4-*.md` file for any design change.
- Reason: results are single-seed, so uncertainty comes first; the ablation depends on the item-6 decision below.

### Honest headline
- Decision: report LexFront as giving lower OOV rate and smaller vocabulary on all 3 datasets. Accuracy, per dataset:
  on GoEmotions and AG News the gain of proposed over existing comes almost entirely from NB -> LR; on TweetTopic
  proposed is 0.26 points below existing. Single runs, no CIs. (Corrected by FIXAUDIT, AUDIT M2.)
- Reason: see `results/REPORT_NOTES.md` section 4, block `obs:attribution` (generated from the run JSONs).

### Open: TF-IDF lowercasing vs LexFront's lowercase rule
- Decision: undecided; must be settled before the ablation (options: `lowercase=False` in TF-IDF, or drop the rule
  from the ablation).
- Reason: TF-IDF lowercases first, so the rule is inactive in all experiments. Source: `results/REPORT_NOTES.md` s.2.

### Credibility check (abstracts only)
- Decision: position the work as a combination, never claim "first" or "novel".
- Reason: regex tweet tokenizers exist (NLTK TweetTokenizer, CMU Twokenize, ekphrasis); automaton-based tokenization
  exists (Fast WordPiece, EMNLP 2021); normalization gave small downstream gains (MultiLexNorm, W-NUT 2021);
  pre-tokenization evidence is mixed (COLING 2025 vs Vemula et al. 2025). Our gap: compiler-built DFA pre-tokenizer
  compared on noisy classification, with a per-rule ablation. Self-assessed novelty ~3-4/10; fit for Compiler Design ~9/10.
- Source: team literature check (abstracts only, full papers not read).

### Scope cuts
- Decision: tokenizers 9 -> 5; datasets 4 -> 3; Logistic Regression only, with Naive Bayes as the existing method;
  transformer fine-tuning cut (optional baseline later); Docker dropped for Colab.
- Reason: 14-day side project. Source: team decision; datasets/Docker also in `docs/versions/v3-*.md`.

### Hinglish dropped
- Decision: English only.
- Reason: SentiMix dataset download unconfirmed; closest prior work (HingLM, NeurIPS 2026 workshop) is very recent;
  too much scope. (v3 and `docs/LEARN.md` Q27 give only the annotation/time reason.)
- Source: team decision.

### DFA engine covers 4 classes only
- Already recorded: `results/REPORT_NOTES.md` s.4 (ELONGATED needs a back-reference, not regular). No change.

### Origin: base paper
- Decision: start from a blog classification report (NLTK + PorterStemmer + CountVectorizer + Naive Bayes; 35% on
  40 labels, 54% after manually merging to 3).
- Reason: its weaknesses: no baseline, accuracy only, possible author leakage, misused terms, code that didn't run.
  We therefore add majority baseline, macro-F1, fixed classifier, and runnable code. (`v1-base-paper.md` has only the figures.)
- Source: team reading of the report; figures not re-run.
