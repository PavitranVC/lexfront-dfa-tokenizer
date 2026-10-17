# Audit #5 — diff review of P3 (`8b4d73e..d826546`, PR #15)

Auditor role, read-only on code; the only file this audit changes is this one. Scripts used are quoted inline; they lived
in a scratch directory outside the repo. Environment: Python 3.11, `requirements-lock.txt` (tokenizers 0.23.3),
`pytest -q` → **161 passed**.

**Token estimate:** about 85k tokens (budget 100k).

## Scope

Covered:
- P1 leakage: factory, runner path (`pipeline.vectorize`), fingerprint/`subword_record`, intrinsic path, hybrid corpus.
- P2 determinism: repeated training, `RAYON_NUM_THREADS=1` and `=4`, full re-run of one dataset cell vs committed JSON.
- P3 independent statistics (own bootstrap/macro-F1/McNemar code) and the 17/24 and 8/24 counts.
- P4 A4 fixes: per-rule firing counts recomputed independently (main setting, 3 datasets × 2 splits × 6 variants);
  generated "never fires" wording.
- Hybrid passthrough, `list[str]` interface, `[UNK]` and `##` collisions, OOV "near zero by construction" claim,
  generated observations and docs wording, pre-existing metrics vs `8b4d73e`, repo size, CI, tests (incl. mutation
  check), `tokenizers` wheel for 3.11/3.12, notebook section 7, dead code / silent exceptions.

Not covered:
- Firing counts of the **case** setting (`results/ablation/case`, `no_lowercase`, `all_off_cs`): not recomputed.
- Full re-run of the ag_news and go_emotions BPE cells and of `results/bpe_v32000` (only tweet_topic re-run end to end;
  the others checked through the stats recomputation and the CI stats-only regeneration).
- Executing the notebook (only `make_notebook.py --check`); line-by-line review of `docs/modules/module2.md`,
  `CHANGELOG.md`, `HANDOFF*.md` beyond the claims grepped below.
- Installing on Python 3.12 (wheel availability checked, not a full 3.12 test run).

## What passed

| check | command / method | result |
|---|---|---|
| Test suite | `.venv/bin/python -m pytest -q` | 161 passed |
| Leakage, model | `leak.py`: tweet_topic, train both tokenizers through `pipeline.vectorize`, replace every test text with mutated text (`"zzqx ÆØÅ " + reversed + " ☃☃ qwxj"`), retrain | `bpe` and `lexfront_bpe` models **bit-identical** (`to_str()` sha `56d10eab…`, `0197b3cb…`) |
| Leakage, corpus | spy on `train_bpe`, compare its input with an independently built corpus (`[t.lower() for t in train]` for BPE; LexFront word-like forms of lowercased train texts for the hybrid) | equal for both; 185 test-only characters, none in either vocabulary (neither as `c` nor `##c`) |
| Leakage, fingerprint | `subword_record` hash with mutated test texts; stored `train_text_hash` in `results/bpe/tweet_topic__bpe__lr.json` | `train_text_hash` unchanged by test edits; equals the train-only hash and differs from the train+test hash. (Run fingerprint changes, as it should, via `data_hash`.) Intrinsic path (`run_module2.py:39-44`) also passes only `train`. |
| Determinism | `det.py` run under `RAYON_NUM_THREADS=1` and `=4`: `run_bpe(..., ("tweet_topic",), force=True)` into a scratch dir | same model hashes in both runs and in the leakage run; all 4 cells (`bpe`/`lexfront_bpe` × NB/LR) `metrics` **and** fingerprint equal to the committed JSON; prediction sidecar byte-identical (`cmp`) |
| The fix is needed | `unseeded.py`: 4 trainings with a plain `BpeTrainer` (no sorted alphabet) on tweet_topic train | 4 distinct models → the hash-map-order bug is real, and the seeding removes it |
| Tests catch a regression | mutation: seed only `[UNK]` instead of `sorted_alphabet(...)`, run `pytest tests/test_bpe.py -k "deterministic or thread"` ×3, then restore | 2 failed each time (code restored, `git status` clean) |
| Independent stats (ag_news) | `stats_mine.py`: own accuracy / macro-F1, B=2000, seed 20261017, percentile CI | BPE−NLTK acc −0.750 [−1.145, −0.368] (stored [−1.132, −0.368]); F1 −0.754 [−1.154, −0.373]. Hybrid−LexFront acc −0.855 [−1.250, −0.461] (stored [−1.250, −0.487]); F1 −0.856. Hybrid−BPE acc −0.105 [−0.368, +0.158] (stored [−0.382, +0.184]); F1 −0.109. Point differences identical, CIs agree within Monte-Carlo noise, same exclusion verdict for all 6. |
| McNemar | own exact binomial | 144 / 87, p = 0.000214 = stored |
| CI counts | own count over `results/*/stats/bootstrap.json`, also checks `ci_excludes_zero` flag consistency | vocab 8000: **17/24**, 0 up / 17 down; vocab 32000: **8/24**, 0 up / 8 down; quick: 0/24. Matches PR. Per-dataset signs also support LEARN Q34 ("hybrid below LexFront on all three at 8000; below BPE or not distinguishable"; "at 32000 most ag_news / tweet_topic gaps include 0": 12 of 16). |
| A4 firing counts | `fires.py`: per lexer token compare `normalize_token(cls, lexeme, variant)` with the full form on lowercased texts; doc = any token differs | **36/36 equal** to stored `rule_fires` (e.g. go_emotions no_mention 6/6 train, 0/0 test; ag_news no_url 1833/2237 train, 99/117 test) |
| A4 wording | `grep -n "never fires" results/REPORT_NOTES.md docs/LEARN.md` | "Rule never fires on this data (train and test), so not a test of the rule: no_url (identical to full, left out of the counts)" for tweet_topic/go_emotions, no_mention for ag_news; go_emotions no_mention listed as "Fires in train only, never in test (6 train documents)". Consistent with the recount. |
| Pre-existing metrics | `unchanged.py` / `unch2.py` over `git diff 8b4d73e d826546 -- results` | 150 pre-existing run JSONs changed only in `fingerprint`, `timestamp_utc`, `timing_seconds`; **0 metric diffs**; 0 prediction sidecars changed; intrinsic JSONs: only throughput changed for the 4 old tokenizers; ablation `bootstrap.json` identical apart from the added `rule_fires`; nothing removed |
| CI regeneration | the three new `run_bpe --stats-only` lines + `render_docs` + `make_notebook.py --check` locally | `git status` clean; "notebook matches make_notebook.py" |
| Interface | `hyb.py` | both return `list` of `str` |
| Hybrid passthrough | `hyb.py` on adversarial text and the tweet_topic test split | classes: `STRUCTURAL ∪ WORD_LIKE` = all 11 lexer classes, disjoint (also tested). Hashtags (`#happy`), emoji, emoticons (`:d`), UNKNOWN (`中`, `文`) pass whole; slang expansions (`idk` → `i do not know`) are split per word; no word form contains a space; every word piece is in the BPE vocab. One pass-through form equals a vocab entry (`’`, PUNCT, 126 test tokens), but BPE never emits `’` at word start (word-like lexemes start with a letter), so no real feature collision. |
| `[UNK]` / `##` | same | literal `[UNK]` lexes to `[`, `un`, `##k`, `]` (lowercased, no collision); literal `##ing` → hybrid `#`, `#ing`; pure BPE `#`, `###`, `in`, `##g` (Whitespace pre-tokenizer separates punctuation runs from letters, so a literal can never produce a `##<letter>` piece). Hybrid `[UNK]` share on tweet_topic test: 8 / 149,688 tokens. |
| `tokenizers` wheel | `pip download tokenizers==0.23.3 --only-binary=:all: --python-version 3.12` | `cp310-abi3` manylinux wheel → installs on 3.11 and 3.12; lock and range (`>=0.22,<0.24`) consistent |
| Repo size | `git ls-tree -r -l` sum of `results/` | 1.74 MB → 2.05 MB (+0.31 MB; 36 new gz sidecars); whole `.git` pack 3.05 MiB. Sensible: sidecars are what the stats CI step regenerates from. |
| CI workflow | diff of `.github/workflows/ci.yml` | only 3 stats-only lines added, covered by `git diff --exit-code results` |
| Dead code / silent exceptions | `git diff 8b4d73e..d826546 -U0 -- '*.py' \| grep except\|pass$\|TODO` | none added |
| Notebook §7 | `make_notebook.py` diff | says train split only, toy demo labelled "illustration only", OOV not comparable; no causal claim |
| Docs causality | grep of added doc lines for because/hurt/helps/cause/better | the only "why" (splitting words hurts) is labelled a hypothesis (LEARN Q34, HANDOFF OPEN, obs:bpe footer) |

## Findings

| id | severity | file:line | evidence | suggested fix | status |
|---|---|---|---|---|---|
| A5-1 | minor | `experiments/tokenizers.py:239-243` | `build_tokenizer` hashes `train_texts` (`text_hash` → `list(texts)`) and then passes the same object to the constructor; an iterator is consumed by the hash, so the tokenizer is trained on nothing. `build_tokenizer('lexfront_bpe', iter(texts))` → vocab size 1, `('hello')` → `['[UNK]']*5`, no error; with a list: vocab 51. No current caller passes an iterator (`Dataset` stores lists, `run_intrinsic` builds a list), so committed results are not affected. | `train_texts = list(train_texts)` as the first line of `build_tokenizer`. | CONFIRMED |
| A5-2 | minor | `experiments/observations.py:128`; `docs/LEARN.md:275`; `run_module2.py:32`; `subword/bpe.py:4-5` | "BPE splits every word into pieces seen in train, so its OOV rate is near zero by construction". Not exact: characters absent from train become `[UNK]`, and some vocabulary pieces are never produced on train. tweet_topic test, pure BPE: 679 OOV tokens of 154,901, of which **427 are `[UNK]`**, the rest pieces such as `bon`, `##asures` unseen on train (`hyb.py`). "Near zero" (0.44 % vs LexFront 8.25 %) is fair; "every word … pieces seen in train" is not. The hybrid sentence ("its OOV tokens are structural tokens … or `[UNK]`") likewise omits unseen word pieces. | Reword to "splits words into vocabulary pieces; only characters never seen in train (`[UNK]`) and rare pieces can be OOV, so its OOV rate is near zero"; same for the hybrid sentence. | CONFIRMED |
| A5-3 | minor | `experiments/observations.py:135-143` (generated line in `results/REPORT_NOTES.md:152-154`, `docs/modules/module2.md:129-131`) | Each line begins "lowest OOV rate 8.25% (lexfront)" and then lists `bpe: OOV 0.44%` in the same bullet. The docstring says "among the four word-level tokenizers", the generated text does not. | "lowest OOV rate among the word-level tokenizers …". | CONFIRMED (text read) |
| A5-4 | minor | `subword/bpe.py:109-113` with `experiments/metrics.py:18-45` (intrinsic table `throughput` column) | The hybrid memoizes word pieces; throughput is timed after the same tokenizer already processed train and test, so the cache is warm. tweet_topic, 1000 test docs: cold 209k tok/s vs warm 266k tok/s (**1.27×**, `thr.py`). LexFront and the baselines have no such cache, so `lexfront_bpe`'s throughput in `results/tables/intrinsic_metrics.md` is not like-for-like. No doc draws a conclusion from it. | Clear `_pieces` before timing, or footnote the column ("hybrid: warm piece cache"). | CONFIRMED |

No blocking or major finding.

## Verdict

**PASS** — 0 blocking, 0 major, 4 minor. Leakage is excluded by experiment (bit-identical models under mutated test
texts, training corpus = train split only, including the hybrid's word-like corpus); training and run metrics are
deterministic across repeats and thread counts and reproduce the committed JSON; headline statistics, McNemar, the
17/24 and 8/24 counts and all 36 main-setting firing counts reproduce with independent code; pre-existing metrics are
unchanged.
