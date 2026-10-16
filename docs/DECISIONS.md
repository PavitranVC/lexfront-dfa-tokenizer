# Decisions log

Newest first. One short entry each: decision, reason, source. Gaps filled on 2026-10-16 from the team's notes; items
already recorded elsewhere are only pointed to. Citations below are from abstracts only and were not re-checked here.

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
- Decision: report LexFront as giving lower OOV rate and smaller vocabulary on all 3 datasets; the accuracy gain over
  NLTK+NB comes mainly from NB -> LR, not the tokenizer. Single runs, no CIs.
- Reason: see `results/REPORT_NOTES.md` section 4 (e.g. LexFront vs NLTK under LR: -1.59 / +0.02 / -0.01 pts).

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
