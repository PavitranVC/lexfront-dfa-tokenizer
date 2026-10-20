# AUDIT #1 — full codebase (main @ 6e96521)

Auditor role, overnight pipeline. Adversarial pass, then one verification pass on the top findings. No code was changed;
this file is the only change on branch `audit/round-1`.

**Verdict: PASS** (0 blocking, 3 major, 13 minor). No confirmed blocking finding. The three majors should be fixed
before the next numbers are produced (P0/P1), because two of them affect what CI and the notebook actually verify.

## Scope

Covered (each file read once): `lexer/` (all, except `lexer/prototype/mini_lexgen.py`), `experiments/` (all),
`tests/` (all), `notebooks/make_notebook.py`, `.github/workflows/ci.yml`, `requirements.txt`, `.gitignore`, `README.md`,
`CHANGELOG.md`, `docs/LEARN.md`, `docs/PIPELINE.md`, `docs/DECISIONS.md`, `docs/HANDOFF.md`, `docs/HANDOFF_LOG.md`,
`docs/modules/*`, `docs/versions/*`, `results/REPORT_NOTES.md` (hand-written parts; generated blocks checked by script).
`results/*.json` and the tables were checked only by script, never read by eye.

Not covered: `lexer/prototype/mini_lexgen.py` (marked "kept for reference", not imported anywhere); `FIRST_PROMPT.md`;
figure PNGs (not compared pixel by pixel; only their generator code was read); full-data reruns of GoEmotions and AG News
(only the full TweetTopic grid and the whole QUICK grid were rerun); the committed notebook's outputs beyond counting the
`acc=` log lines and checking for error outputs; the literature citations in `docs/DECISIONS.md` (not verifiable offline).

Token use: about 175k of the 300k budget.

## What was run and passed (no finding)

| Check | Command / method | Result |
|---|---|---|
| Fresh venv + tests | `python3.11 -m venv .venv && pip install -r requirements.txt`; NLTK data; `pytest -q` | `99 passed in 2.90s` |
| Docs regenerate cleanly | `python -m experiments.render_docs && git diff --exit-code` | exit 0, no diff (tables in `results/tables/` also unchanged) |
| Tables rebuild from JSON | `report.build_all_tables(base, tmpdir)` vs committed, for `results` and `results/quick` | 5/5 identical in both |
| QUICK results reproduce | fresh clone, `run_grid(..., 2000, 500, seed 0, force=True)`, Module 1 + LexFront | 33/33 metric dicts identical to `results/quick/` |
| Full results reproduce (sample) | fresh clone, full TweetTopic, 4 tokenizers x 3 classifiers, `force=True` | 11/11 identical to committed (LexFront+majority is not run by design) |
| Settings identical across tokenizers | script over all 33 run JSONs per base | one (n_train, n_test, subsample, seed, vectorizer params, sklearn, python) tuple per dataset |
| Majority baseline sane | expected = test share of the largest *train* class (from `datasets.json`) | matches every majority run |
| Sizes consistent | `datasets.json` vs run JSONs vs intrinsic JSONs | 3598/3399, 36308/4590, 120000/7600; class counts sum to sizes |
| Leakage | read `pipeline.vectorize`, `metrics.intrinsic_metrics`, `data.subsample`, `keep_single_label` | TF-IDF fit on train only; intrinsic vocab from train only; default hyper-parameters, nothing tuned on test |
| TweetTopic split | `datasets.json` split_info | `train_coling2022`/`test_coling2022`, `card_confirms_coling2022: true` |
| DFA state counts / LEARN claims | `compile_rules(dfa_rules())` and per class; DFA traces of `#hi`, `:-)`, `@bob` | 31 atoms, 43/12/10 states, MENTION NFA 9, expanded ELONGATED 221/157/54; traces q0→q1→q6→q6, q0→q2→q7→q8, ends q9: all as LEARN.md says |
| DFA = PLY on unicode whitespace | NBSP, U+2028, U+001C, ZWSP, BOM | identical tokens |
| Hand-typed numbers | script | REPORT_NOTES "-1.59" and "within 0.02" correct (TT -1.59, GE +0.02, AG -0.01); "31 handwritten cases", "30-entry slang", "11 classes", "under 10 minutes" (max run 67 s) all correct |
| LEARN.md 30 viva answers | read against code + probes (`5pm`, `:Dog`, traced tweet, NLTK/Tweet outputs) | correct except Q28 nuance (m3) |
| "first"/"novel" | grep README, docs, REPORT_NOTES | no such claim (only the rule forbidding it) |
| Secrets / unsafe calls | grep for keys/secrets, `eval`, `exec`, `pickle`, `shell=True` | none; the only `subprocess` calls use argument lists |
| Notebook generator | `make_notebook.CELLS` vs committed `.ipynb` | 32/32 cells, sources identical, no error outputs |
| CI on main | GitHub Actions, run 38899887960 | success (both jobs, about 2 min in total) |

For P1: per-example predictions are **not** stored; a run JSON holds only
`timestamp_utc, dataset, tokenizer, vectorizer, classifier, seed, metrics, timing_seconds, environment`.

## Findings

| id | sev | file:line | evidence (command → output) | suggested fix | status |
|---|---|---|---|---|---|
| M1 | major | `experiments/pipeline.py:62-63` (with committed `results/quick/**.json`, 49 files) | Fresh clone, `review1.run_module1_section(True, log=lines.append)` → `acc lines: 0`: none of the 27 QUICK runs is recomputed; the cached JSON is read back. The same holds for the CI `notebook` job and for the CHANGELOG claim "re-ran it from a fresh clone … identical quick-run metrics". So CI never runs the classification path end to end. The cache key is only the file name; changing tokenizer code, vectorizer params or subsample size silently reuses stale results. | Run the notebook (and the CI job) with `force=True`, or write QUICK results to a temp dir in CI. Also store a config/code fingerprint in each JSON and recompute when it differs. | CONFIRMED |
| M2 | major | `results/REPORT_NOTES.md:167`; `docs/DECISIONS.md:19-21` | Script: TweetTopic proposed − existing = **-0.26** acc (clf NB→LR +1.32, tokenizer NLTK→LexFront under LR -1.59). The hand-written text says "the classifier switch accounts for almost all of the gain of proposed over existing", and DECISIONS calls it "the accuracy gain over NLTK+NB". On TweetTopic there is no gain: proposed is below existing. The claim holds only for GoEmotions (+17.54, clf +17.52) and AG News (+1.49, clf +1.50). | State it per dataset: "On GoEmotions and AG News the gain comes almost entirely from NB→LR; on TweetTopic proposed is 0.26 points below existing." Better: generate this sentence in `observations.py`. | CONFIRMED |
| M3 | major | `notebooks/make_notebook.py:25` (cell 1) | Simulated a Colab session restart (repo already cloned in `/content`, cwd reset to `/content`): `find_repo_root()` only walks *up*, returns None, `git clone` → `CalledProcessError … returned non-zero exit status 128`. Restarting the session is common on Colab (for example after `%pip`). | If `lexfront-dfa-tokenizer/` exists, `chdir` into it (and `git pull`) instead of cloning. | CONFIRMED |
| m1 | minor | `experiments/run_module2.py:55`, `experiments/data.py:64` | One RNG serves train then test, so the test subset depends on `max_train`. Toy: `subsample(ds,10,5).test_texts` = `['te17','te27',…]` vs `subsample(ds,20,5)` = `['te18','te39',…]`. `--intrinsic-max-train N` with `--max-test` therefore measures intrinsic metrics on a different test set from the classification runs. Latent: committed results use the full data. | Use separate RNGs (`Random(seed)` for train, `Random(seed+1)` for test), or forbid `--intrinsic-max-train` together with `--max-test`. | CONFIRMED |
| m2 | minor | `lexer/normalize.py:63` | `LexFront(NormalizeConfig(lowercase=False)).tokenize("OMG U")` → `['oh','my','god','you']`. Slang lookup ignores case even with lowercase off, so the two switches interact. This matters for the P2 ablation. | Decide and document; for example, look up `lexeme` as is when `lowercase=False`. Add a test. | CONFIRMED |
| m3 | minor | `lexer/token_spec.py` (EMOJI); `docs/LEARN.md:229` (Q28) | `PlyLexer().scan`: `👨‍👩` → EMOJI, UNKNOWN `‍`, EMOJI; flag `🇮🇳` → two EMOJI tokens; NFD `é` → WORD `e` + UNKNOWN; `ﬁne` → UNKNOWN `ﬁ` + WORD. Q28 says "each emoji … is one token", which does not hold for ZWJ sequences or flags. | Mention the limits in Q28 / weaknesses (Q26); optionally NFC-normalize input and allow ZWJ/regional-indicator sequences. | CONFIRMED |
| m4 | minor | `lexer/token_spec.py:44` | `"www."` → ELONGATED `www` → normalized `ww`; `"heyyy'll"` → CONTRACTION (longer), so no elongation collapse. | Accept, or list as edge cases in Q26; add tests. | CONFIRMED |
| m5 | minor | `results/REPORT_NOTES.md:240`, `docs/modules/module1.md:29` | Pseudocode (3) applies `[majority, naive_bayes, logistic_regression]` to LexFront too; the code runs only `["nb","lr"]` for LexFront (`run_module2.py`), and there is no `*__lexfront__majority.json`. | Say "majority/NB/LR (module 1), NB/LR (module 2)". | CONFIRMED |
| m6 | minor | `requirements.txt:1-23`; notebook cell 1 `%pip install -q -r requirements.txt` | No version pins (installed here: scikit-learn 1.9.1, datasets 5.1.0, nltk 3.10.3, numpy 2.4.6, pandas 3.0.6, ply 3.11). Pip `graphviz` is unused (`grep -rn "import graphviz"` → nothing); the code shells out to the `dot` binary, which is absent here and in CI (`which dot` → empty). So `build_report` writes `"png": null` (fresh `dfa_states.json` differs from the committed one only in `files.png`), and the notebook shows the stale committed PNG. | Pin versions (or add a lock/constraints file recorded in the results); drop pip `graphviz` or `apt-get install graphviz` in CI. | CONFIRMED |
| m7 | minor | `tests/test_report.py:26`; `tests/test_experiments.py:55`; `tests/test_review1.py` | `"LexFront+LR" in t or "proposed" in t` is always true (the column header "proposed LexFront+LR acc / F1" always exists), so it is a tautology. `0 < macro_f1 < 1` instead of the exact 0.7333. Unused `observation_text`. Gaps: no PLY test for unicode whitespace (NBSP/U+3000), none for slang with `lowercase=False`, CONTRACTION vs ELONGATED, subsample independence (m1), or stale-cache detection (M1). | Assert real values (row content, exact F1); add the missing cases. | CONFIRMED |
| m8 | minor | `experiments/report.py:174`, `experiments/observations.py:120` | Broad `except (KeyError, ValueError, IndexError, AttributeError)` turns a schema or code bug into `_not available: …_` text inside the rendered docs. `render_docs` still exits 0, so the provenance check only catches it if someone reads the diff. | Catch only `FileNotFoundError`; let other errors fail. | PLAUSIBLE |
| m9 | minor | `notebooks/make_notebook.py:14,150` | Dead fallback to the old branch `feature/lexfront-lexer`. The appendix calls `report.build_all_tables("results")`, which rewrites the committed `results/tables/*.md` as a side effect of viewing. | Remove the fallback; call the `table_*` functions without writing. | PLAUSIBLE |
| m10 | minor | `.github/workflows/ci.yml:8` | No `permissions:` block (default token scope), no `concurrency`, and no check that `render_docs` leaves `git diff` clean or that `make_notebook.py` output equals the committed notebook. | Add `permissions: contents: read`; add a "render_docs + git diff --exit-code" step and a notebook-generator diff step. | PLAUSIBLE |
| m11 | minor | `docs/PIPELINE.md:4,7,15`, reply template; `docs/versions/v3-*.md`; `docs/DECISIONS.md:37` | Inconsistent with the autonomous overnight run, where "decide, log in docs/DECISIONS.md, proceed" replaces the header: "propose code in 2-3 lines and wait for my OK" (l.7), "propose each fix in 2-3 lines first" (l.15), and reply `STATUS: needs OK` all assume a human. "Commit and push to the current branch" vs the PR-per-role flow. Date gate "by about Oct 28" (l.4) is meaningless in a one-night run. No auditor role or `docs/AUDIT.md` in the sequence. v3 fixes "5 tokenizers" and DECISIONS says "9 -> 5", but P3+P4 add BPE, hybrid, WordPiece and Unigram (8 total), so this needs a v4 file. | Add an "overnight mode" note to PIPELINE that overrides the header, adds the audit step and replaces the date gate; record the tokenizer-count change in a v4 version file. | CONFIRMED |
| m12 | minor | `experiments/render_docs.py:17`; `docs/LEARN.md` | LEARN.md is not in `DEFAULT_FILES`, yet it hand-types numbers (MENTION 9 NFA states, 31 atoms, 10 minimized states, state ids q1/q2/q4/q6/q7/q8/q9). All of them match today (see the table above), but nothing stops them drifting if the spec changes. | Generate these via an `obs:`/table block, or add a test asserting them. | CONFIRMED |
| m13 | minor | `experiments/figures.py:85` | `val(d, s) or 0` draws a missing cell as a 0 % bar labelled "0.0", indistinguishable from a real zero. | Skip missing bars (as `_grouped_bars` does). | PLAUSIBLE |

## Notes

* No confirmed blocking finding. The results are reproducible (QUICK 33/33, full TweetTopic 11/11), there is no
  train/test leakage, and every number in prose that I checked traces to JSON.
* The TF-IDF lowercase interaction is already documented (REPORT_NOTES s.2, LEARN Q15, DECISIONS "Open") and is P0's
  job, so it is not listed again; m2 should be settled at the same time.
