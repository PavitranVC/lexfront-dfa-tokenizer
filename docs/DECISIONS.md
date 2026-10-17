# Decisions log

Newest first. One short entry each: decision, reason, source. Gaps filled on 2026-10-16 from the team's notes; items
already recorded elsewhere are only pointed to. Citations below are from abstracts only and were not re-checked here.

## 2026-10-17 (large-scale Sentiment140 test)

### Scope and protocol (owner's choices; `docs/LARGE_PROTOCOL.md`, written before the full run)
- Decision: one huge noisy dataset, run by the owner on Google Colab only, finished within a day. Sentiment140 (1.6 M tweets) from
  `bdanko/sentiment140`, because it is the largest genuinely noisy tweet set that loads from parquet here (the official
  `stanfordnlp/sentiment140` is script-based and refused by `datasets` >= 4). Provenance (repo commit SHA) is stored in `split.json`.
- Split by user (5% of users held out), not by tweet: author leakage was a flaw of the base report. Cleaning with `clean.py` after the split.
- Size ladder 10k / 100k / 1M / full with random subsets (5, 5, 3, 1) instead of one run, so the effect of data size has a spread;
  `tuned` setting only at 10k, 100k and full (cost). Test text is also scored after seeded synthetic noise (light 0.10, heavy 0.30 of the
  eligible words), because noise robustness is the claim LexFront was built for; training text is never corrupted.
- Practical margin of +-0.5 accuracy points fixed in advance, because with ~80 k test tweets every difference has a narrow CI.
  Hypotheses H1-H3 fixed in advance and reported as observed / not observed with the rule that decides it.
- Colab design: one JSON per fit set (`--work`, on Google Drive in the notebook) so a disconnected session resumes; results consolidated
  into `results/large/` (few files, easy to commit); optional push to a NEW branch with a token from a Colab secret (never `main`),
  token passed through `GIT_ASKPASS`, scrubbed from error text. Memory: training texts are not memoized, only the test texts are.
- The notebook is generated (`notebooks/make_large_notebook.py`, checked in CI) and committed without outputs. CI executes the same
  cells as a dry run (`LEXFRONT_LARGE_QUICK=1`, Python 3.12, `requirements.txt` ranges, empty directory).
- Limits: one source, labels derived from emoticons by the dataset's authors, synthetic noise, subsets of one size overlap, no transformers.

## 2026-10-17 (dataset collection: cleaning and fair tuning)

### Scope (owner's choices)
- Decision: add English noisy-text datasets; keep the work additive (every committed number under `results/` stays identical,
  `compare_results` reports 0 differences); "performance" means classifier accuracy, with the same effort for every tokenizer.
  Speed work and Hinglish are not part of this step.
- Collection: the original three plus five tweet_eval tasks (sentiment, emotion, irony, hate, offensive), official train/test
  splits. Chosen because they are tweets (the noise LexFront targets), load from the Hub here, and have labelled test sets
  (SST-2's test labels are hidden). Caveats kept in the code docstring: `hate` has a train/test distribution shift, `irony` and
  `offensive` have small test sets, and the five tasks share a source, so they are not five independent domains.

### Cleaning (`experiments/clean.py`)
- Rules, in order: drop empty texts; drop texts that are equal after lower-casing and whitespace collapsing but have different labels
  (ambiguous gold label); keep the first of equal texts with one label; drop test texts that also occur in train (judged against
  every non-empty train text, including ones dropped as conflicting). Texts are never edited: emoticons, hashtags, casing and
  misspellings are what the lexer is tested on, so normalizing them here would remove the thing being measured.
- Cleaning runs on the full data and a subsample is drawn afterwards. Each result records the cleaning counts in its dataset record.
- The cleaned test sets differ from the raw ones, so cleaned-vs-raw accuracy is not a like-for-like "gain from cleaning"; the report says so.

### Fair tuning (`experiments/tuning.py`)
- Same grid for every (dataset, tokenizer, classifier): TF-IDF word 1-grams vs 1-2-grams with sublinear tf, and LR `C` in {1, 3, 10} or
  NB `alpha` in {0.1, 0.3, 1}. Candidates are scored on a validation slice carved from train (10%, at most 5000; the search fits on
  at most 20000 examples), the best validation accuracy wins (ties go to the earlier, simpler candidate), the winner is refit on all of
  train and the test split is scored once. The test split is never used to choose anything (a test asserts it).
- The main grid is untouched: new code lives in new files that are not part of the code fingerprint of the existing runs.
- Limits: one seed, one validation split, small grid; tuned and default can differ by less than the noise (the report shows the sign
  counts, not only the mean). The report also counts how often the search chose an edge of the grid: NB `alpha` often did, so a wider
  grid might choose differently; the committed results do not test that (an unsaved spot check on a few datasets changed the
  selection on one of them, so it is not a result). Validation is carved from train, so under a train/test shift (`hate`) a
  validation-selected setting can do worse on test than the default.

## 2026-10-17 (P4: AUDIT_5 fixes; WordPiece and Unigram)

### AUDIT_5 fixes (stage A)
- A5-1: `build_tokenizer` makes `list(train_texts)` first (an iterator was consumed by the training-text hash and the
  tokenizer then trained on nothing); a test passes an iterator.
- A5-2/A5-3: the OOV statement is reworded everywhere it was generated or written (observation `intrinsic`, `run_module2`
  docstring, LEARN, module2 / REPORT_NOTES text, notebook): subword tokenizers split words into vocabulary pieces, only
  characters never seen in train (`[UNK]`) and rare pieces can be OOV, so the OOV rate is near zero (not "every word is
  split into pieces seen in train"); the generated line says "lowest OOV rate among the word-level tokenizers". The
  P3 text in `docs/versions/v5-bpe-hybrid.md` and in the P3 section below is left as written (earlier versions stay
  untouched); this entry and `v6-wordpiece-unigram.md` supersede that wording.
- A5-4: the hybrid's piece cache is cleared before every timed repetition of the intrinsic throughput (`clear_cache`), so
  `lexfront_bpe` is timed cold like the tokenizers without a cache.

### AUDIT_6 (independent check of P4; `docs/AUDIT_6.md`)
- Leakage, determinism and independent CI recomputation passed. One low note, kept as is: Unigram's rounding to 3 decimals has
  a margin of 2e-10 (nearest score) against numerical noise of at most 4e-11, so a different CPU or library version could
  flip one rounded score. Reason for not changing it: any edit under `subword/` changes the code fingerprint of every
  committed run. Mitigation: documented here, in the v6 file and in LEARN Q38; if it ever flips, the symptom is a changed
  serialized Unigram and a fingerprint-stale rerun, not a silent wrong result.

### P4 protocol (conductor's decisions)
- WordPiece and Unigram from the Hugging Face `tokenizers` trainers (`WordPieceTrainer`, `UnigramTrainer`), same protocol
  as BPE: trained only on the train split (lowercased as the vectorizer does), one vocabulary size 8000 for every
  dataset, not tuned, `Whitespace` pre-tokenizer, `[UNK]`; WordPiece continuation marker `##`; Unigram has no marker
  (no word-boundary symbol such as SentencePiece's, because the conductor fixed the pre-tokenizer).
- No hybrids with LexFront for these two (the P3 result already shows the hybrid with BPE is worse).
- Interface unchanged (`tokenize(text) -> list[str]`, factory `build_tokenizer`); the fingerprint carries the tokenizer
  config, vocabulary size, library version and training-text hash. Both names are added to the intrinsic metrics.
- Grid: 3 datasets x {wordpiece, unigram} x {NB, LR}, main setting, full data, `results/subword2/`; QUICK in
  `results/quick/subword2/`. Statistics: paired bootstrap (accuracy, macro-F1; same code, B and seed) for wordpiece+LR and
  unigram+LR minus nltk+LR and minus bpe+LR, and unigram+LR minus wordpiece+LR; exact McNemar on accuracy for the two
  against nltk+LR. Generated table `subword_table` (and `subword_runs`), observation `subword`.
- Determinism was checked before any run (4 trainings each, `RAYON_NUM_THREADS=1` and `4`, tweet_topic 6000 documents):
  plain `WordPieceTrainer` and plain `UnigramTrainer` are both NOT deterministic (4 distinct serialized models each).
  WordPiece has the same hash-map-order cause as BPE and is fixed the same way (sorted alphabet passed first as
  special tokens, then removed from the added tokens): identical at 1 and 4 threads. Unigram has two sources: the EM
  scores differ in the last digits (up to about 4e-11, floating-point summation order), and the characters that pruning
  dropped but the trainer must keep are put back at the bottom with scores `lowest + k * 0.0001`, k following the
  iteration order of a hash set (found by inspecting the vocabularies of two trainings: the set of pieces was identical,
  the 1e-4 steps were assigned to different characters). Pinned by canonicalizing the trained model
  (`canonical_unigram`): the single characters of that bottom block get the block's scores back in code-point order,
  every score is rounded to 3 decimals, pieces are ordered by (-score, piece) with `[UNK]` first. The configuration
  (`score_decimals`) is in the fingerprint. An unseen span is named `[UNK]` (the library would return its surface
  string for Unigram). Limit: a score within about 1e-10 of a rounding boundary could still round differently (a few
  in 10 000 trainings by a rough estimate). Check on the full train splits of the three datasets: 5 trainings each at
  `RAYON_NUM_THREADS=1` and 5 at 4 gave one identical serialized model per dataset (10 of 10 equal); without the
  canonical form the same check gave different models for tweet_topic and go_emotions. WordPiece: 2 trainings
  each at 1 and 4 threads gave one identical model per dataset. The checks are scripted in the scratchpad of the
  session and repeated as tests on tiny corpora (`tests/test_subword2.py`); the full-data runs are not in CI.

## 2026-10-16 (P3: AUDIT_4 fixes; BPE and the LexFront + BPE hybrid)

### AUDIT_4 fixes (stage A)
- A4-1: `run_ablation` counts, per setting, dataset and split, how often each removed rule fires: documents whose token
  list and lexer tokens whose normalized form differ from full LexFront, on the text LexFront sees in that setting
  (lowercased in main, as stored in case). One lexer scan per document, every configuration normalizes the same
  tokens. Stored in `<out>/<setting>/firing__<dataset>.json` (stamped with data hash and code fingerprint, recomputed
  when either changes, so `--stats-only` needs no dataset) and copied into `stats/bootstrap.json` (`rule_fires`);
  table column "rule fires, docs train / test". The observation lists variants with 0 firing tokens in train and test
  apart ("rule never fires on this data: not a test of the rule") and leaves them out of the comparison and chance
  counts. A variant that fires only in train (go_emotions no_mention, 6 train documents) stays a comparison, since
  the fit differs, and is named as train-only. The P2 proxy "0 changed predictions = never fires" is withdrawn: it is
  not reliable (no_mention on go_emotions changes test predictions while the rule never fires in the test set).
  LEARN Q33, `v4-ablation.md` ("What it cannot show") and module2 / REPORT_NOTES prose say the same.
- A4-2: the chance statement counts only the compared (non-identical) variants and says "roughly, if the intervals
  were independent" (they are not: all_off contains every single-rule change). "(TF-IDF still lowercases)" is added
  next to all_off of the main setting in the table, observation, caveat, module2, REPORT_NOTES, notebook and
  `v4-ablation.md`.

### P3 protocol (decisions by the conductor, logged before coding)
- Library: Hugging Face `tokenizers` (Rust, CPU, no torch); `requirements.txt` range `>=0.22,<0.24`, lock `0.23.3`.
  Checked: `tokenizers` installs on Python 3.11 (lock venv) and 3.12 (0.23.3 resolved); `tests/test_bpe.py` also
  passes with the lowest allowed 0.22.0. Note: 0.22.x requires `huggingface-hub<1.0`, which conflicts with
  `datasets` 5.x (`huggingface-hub>=1.31`), so with the lock's `datasets` pip resolves 0.23.x.
- Training data: every BPE is trained only on the train texts of its dataset, lowercased as the vectorizer does;
  never on test. Fixed vocabulary size 8000 for all datasets in the primary grid (no tuning); sensitivity grid at
  32000 with the same code in `results/bpe_v32000/`.
- `bpe`: BPE model with the library's `Whitespace` pre-tokenizer (`\w+|[^\w\s]+`: words and punctuation runs),
  continuing-subword marker `##`, UNK token `[UNK]`, applied to lowercased text.
- `lexfront_bpe` (hybrid): LexFront (every rule on) scans and normalizes first. Structural tokens pass through whole
  and are never split: `<URL>`, `<USER>`, `<NUM>`, HASHTAG, EMOTICON, EMOJI, PUNCT, UNKNOWN. Word-like tokens (WORD,
  CONTRACTION, ELONGATED, after normalization incl. slang expansion) go through a BPE trained on the word-like tokens
  of the train split.
- Grid: 3 datasets x {bpe, lexfront_bpe} x {NB, LR}, main setting (TF-IDF `lowercase=True`), full data, in
  `results/bpe/` (QUICK: `results/quick/bpe/`). Paired bootstrap (B = 2000, seed 20261016, as P1) on accuracy and
  macro-F1 for bpe+LR - nltk+LR (plus exact McNemar on accuracy), lexfront_bpe+LR - lexfront+LR, lexfront_bpe+LR -
  bpe+LR, lexfront_bpe+LR - nltk+LR. Generated table `bpe_table` and observation `bpe` (from the JSON only).
- Intrinsic metrics for both new tokenizers in the usual `module2/intrinsic__<dataset>.json`. BPE's OOV rate is near
  zero by construction (any word splits into known pieces), so OOV is not comparable with the word-level tokenizers.

### P3 implementation choices (this session)
- Hashtags pass through whole (default): a hashtag is a topic marker whose identity matters as one feature, and
  LexFront already treats it as one structural token; splitting it would mix `#monday` with the word `monday`.
- UNKNOWN (a character no token class accepts) passes through as its own lexeme. BPE would map a character it never
  saw to `[UNK]` and lose its identity; TF-IDF's `min_df=2` already drops one-off characters.
- The hybrid's BPE uses the `WhitespaceSplit` pre-tokenizer, not `Whitespace`: its input is already one word-like token
  per piece, and `Whitespace` would split the apostrophe out of a CONTRACTION (`don't` -> `don`, `'`, `t`) that the
  lexer deliberately kept whole. Its training text is, per train document, the word-like normalized forms joined by
  spaces.
- Determinism (checked before coding, scratch script, tokenizers 0.23.3): with the `##` marker, two trainings on the
  same text gave different token ids and merge order, at 1 and at 4 threads alike (in that check the vocabulary and
  merge *sets* matched; a later unit test showed that ties at the vocabulary cutoff can change even the set). Without
  the marker the result was identical. Cause: the marked alphabet (`##a`, ...) gets its ids in the
  order of a Rust hash map, and ties between equally frequent pairs are broken by id. Fix: the alphabet the trainer
  would build anyway (every character, plus `##` + every character seen inside a pre-token) is passed first, sorted,
  as the trainer's special tokens, so every id is fixed; afterwards these entries are dropped from the added-token
  list (they stay ordinary vocabulary entries). With the fix, repeated trainings and RAYON_NUM_THREADS=1 / 2 / 4 gave
  identical vocab, ids and merges, so the thread count is not pinned; tests check that two trainings serialize
  identically, that the seeded alphabet equals the alphabet the unseeded trainer builds itself (no extra symbol), and
  that the result does not depend on RAYON_NUM_THREADS.
- Tokenizers that need training are built by a factory `build_tokenizer(name, train_texts)`
  (`experiments/tokenizers.py`); the interface stays `tokenize(text) -> list[str]`. Names: `bpe`, `lexfront_bpe`
  (8000) and `bpe:v32000`, `lexfront_bpe:v32000`. The run JSON and the config fingerprint carry a `subword` record
  (model, vocab size, pre-tokenizer, marker, UNK, hybrid pass-through classes, `tokenizers` version and the SHA-256 of
  the lowercased train texts); untrained tokenizers get no extra key. `subword/*.py` joins the code fingerprint.
- Stats reuse `experiments/stats.py` and `run_stats.dataset_stats` (now taking the pair list as an argument; the main
  grid's pairs are unchanged); the nltk+LR and lexfront+LR runs come from the main grid (`module1/`, `module2/`).

## 2026-10-16 (P2: per-rule normalization ablation; AUDIT_3 minor fixes)

### P2 protocol (decisions by the conductor)
- Decision: LexFront + LR only, 3 datasets, full data. Main setting (TF-IDF `lowercase=True`): full, no_url,
  no_mention, no_number, no_elongated, no_slang, all_off. Case-kept setting (TF-IDF `lowercase=False`): full_cs,
  no_lowercase, all_off_cs. Each variant minus the full variant of its setting: paired bootstrap (same code, B = 2000,
  seed 20261016 as P1) on accuracy and macro-F1, exact McNemar on accuracy, no multiplicity correction (said in the
  generated observation). Protocol: `docs/versions/v4-ablation.md`.
- Reason: the lowercase rule cannot act in the main setting (TF-IDF lowercases the text first), so it is only tested
  with case kept.

### P2 implementation
- Decision: variants are tokenizer names `lexfront:no_<rule>` and `lexfront:all_off` (`LEXFRONT_VARIANTS` in
  `experiments/tokenizers.py`, not in `TOKENIZERS`, so the main grid and intrinsic metrics do not pick them up). The
  config fingerprint and the run JSON of every LexFront run hold its `NormalizeConfig` (`normalize`); baselines get no
  extra key. In file names `:` becomes `-` (valid on Windows); main-grid file names do not change.
- Decision: the full variants are recomputed inside `results/ablation/` (not copied); `results/ablation/stats/
  bootstrap.json` records per setting and dataset whether their predictions are identical to the full-LexFront+LR run
  of the main or sensitivity grid (`reference_vs_grid`).
- Decision: the ablation table also shows the TF-IDF vocabulary size and how many test predictions changed.
  (Corrected by AUDIT_4 A4-1, see the P3 section: "0 changed predictions" is not a reliable sign that a rule never
  fires; firing is now counted directly.)

### Safety check
- The code fingerprint changed (tokenizers, pipeline, normalize), so all main, QUICK and sensitivity runs were
  regenerated with `--force` and compared by script with `main` (every field except timestamp, timing, fingerprint,
  environment, predictions_file and the new `normalize`; predictions compared too): see the P2 CHANGELOG entry.

### AUDIT_3 minor findings
- A3-1: CAVEAT adds that gaps without a paired CI are untested and that overlapping per-run CIs are not a test.
- A3-2: notebook section 5 says QUICK has 500 test examples, so its CIs are wider and can change sign. The audit
  suggested "2 to 3 times wider"; the committed JSONs give 1.9 to 5.8 times for accuracy differences (ag_news is the
  widest), so the text says "about 2 to 6 times".
- A3-3: `run_stats` now also runs on `results/sensitivity_case` (`sensitivity_case/stats/bootstrap.json`); new
  generated table `sensitivity_ci` (case-kept CI next to the main-grid CI) and lines in `obs:sensitivity_case` and
  `obs:ci` stating on how many datasets the tokenizer-only "excludes 0" verdict differs between the two grids.
  A separate table rather than extra columns: `sensitivity_case` has one row per tokenizer, the CIs are per pair.
- A3-4: CI reruns `run_stats` (main, QUICK, sensitivity) and `run_ablation --stats-only` (full and QUICK) and fails
  on any diff under `results/`.
- A3-5: `stats.n_classes_of` (no caller) deleted.

## 2026-10-16 (P0+P1: case sensitivity, stored predictions, bootstrap CIs)

### P0: TF-IDF lowercasing vs LexFront's lowercase rule (decision by the conductor)
- Decision: keep the main grid exactly as it was (TF-IDF `lowercase=True` for every tokenizer; LexFront's own
  lowercase rule is inactive there by design, so every committed main number stays identical and comparable). Add a
  sensitivity grid with TF-IDF `lowercase=False` for all four tokenizers x {NB, LR} x 3 datasets in
  `results/sensitivity_case/{module1,module2}/` (`python -m experiments.run_sensitivity`).
- Reason: in the sensitivity grid LexFront's rule is active, so LexFront's WORD tokens are lowercased while the
  baselines keep case; it is a robustness check, not a like-for-like comparison (said in the docs next to its table).
- How: `VECTORIZER_PARAMS` stays the single source; the grid passes `vectorizer_params(CASE_SENSITIVE_OVERRIDES)`
  (`{"lowercase": False}`), which goes into the run JSON and the config fingerprint. Results: table and observation
  `sensitivity_case` (generated).

### P1a: per-example predictions stored for every run
- Decision: `predictions/<run>.json.gz` next to each run JSON holds `{y_true, y_pred}`; the run JSON names it in
  `predictions_file` (relative to its own directory). gzip header has mtime 0 and no file name, so identical
  predictions give identical bytes. A cached run without its sidecar is recomputed.
- Reason: test-set statistics without refitting; JSON stays readable; small files (whole full grid about 0.2 MB).
- Safety check: changing `pipeline.py` changed every code fingerprint; all 66 main run JSONs (full + QUICK) were
  regenerated with `--force` and compared by script with `main`, ignoring timestamp, timing, fingerprint and the new
  `predictions_file`: no difference in any metric or other field.

### P1b/P1c: bootstrap CIs and paired tests
- Decision: `experiments/stats.py` + `python -m experiments.run_stats` -> `results/stats/bootstrap.json`. Percentile
  bootstrap over test examples, B = 2000, `numpy.random.default_rng(20261016)` per dataset; all runs of a dataset share
  the resampled indices, so differences are paired. Accuracy and macro-F1 from bincount confusion matrices (macro-F1
  averages classes present in gold or predictions, = sklearn `zero_division=0`; the script checks the point estimate
  equals the stored metric). Pairs (LexFront+LR minus A): A = NLTK+LR (headline, plus exact McNemar), TweetTokenizer+LR,
  NLTK+NB. Tables `ci_table`, `ci_runs`, observation `ci`; the shared CAVEAT now describes these CIs.
- Reason: the CIs cover test-set sampling only; training randomness is not covered (one run per cell; NB/LR fits are
  deterministic; no subsampling in the full grid). McNemar only on the headline pair to limit the number of tests.

## 2026-10-16 (FIXAUDIT2, fixes for docs/AUDIT_2.md)

### A2-M1/A2-M2: two dependency files
- Decision: `requirements.txt` = compatible ranges (numpy>=1.26,<3, pandas>=2.2,<4, scipy>=1.12,<2,
  scikit-learn>=1.5,<2, matplotlib>=3.8,<4, threadpoolctl>=3.1,<4, ipykernel>=6; `nltk==3.10.3` and `ply==3.11` stay
  exact); `requirements-lock.txt` = the exact pins behind the committed results, installed by CI. The notebook installs
  the ranges, prints the numpy/pandas/scikit-learn versions in use and warns when pip changed an already-imported
  package (`experiments/setup_check.py`).
- Reason: exact pins forced reinstalls inside a running Colab kernel. Verified: lowest allowed versions and the lock
  install and pass `pytest -q` on Python 3.11 and 3.12; newest resolved versions on 3.12. Python 3.10 not claimed.

### A2-m1: VECTORIZER_PARAMS is the single source
- Decision: `make_vectorizer` builds `TfidfVectorizer(**vectorizer_kwargs())` from the dict (`ngram_range` as tuple).

### A2-m2: data content in the fingerprint; no HF `revision=` pin
- Decision: config fingerprint includes `Dataset.content_hash()` (SHA-256 of train/test texts and labels), also stored as
  `data_hash` in each run JSON. HF revisions not pinned: the content hash already detects an upstream edit, and a pin
  would need commit SHAs looked up by hand. `Dataset` now stores plain lists (datasets 5 Columns were slow to hash).

### A2-m3: library versions in the fingerprint
- Decision: numpy, scipy and threadpoolctl versions join sklearn and nltk in the config fingerprint (`library_versions()`).

### A2-m4: thread-pin test works on a 1-CPU host
- Decision: the probe runs inside `threadpool_limits(2)` and also probes the tokenizer inside `vectorize`. Mutation check
  under `taskset -c 0`: removing either pin fails the test.

### A2-m5: what full mode rewrites is documented
- Decision: comment next to `QUICK_MODE` lists the files each mode rewrites; no skip logic for `run_intrinsic`.
- Reason: throughput is machine-dependent anyway; a cache for it would add code for little gain.

### A2-m6/A2-m7: CI concurrency and notebook setup
- Decision: `cancel-in-progress` only for pull_request events; the setup cell prints a warning when `git pull` fails and
  stops if `lexer/token_spec.py` is missing after `chdir`.

### Results regenerated
- All run JSONs (full and QUICK) regenerated with `--force` for the new fingerprints; every metric identical to main
  (checked by script). Intrinsic JSONs are not fingerprinted and were not rerun.

## 2026-10-16 (FIXAUDIT, fixes for docs/AUDIT.md)

### BLAS threads: every fit is single-threaded (decision by the conductor)
- Decision: `experiments/pipeline.py` wraps vectorize/fit/predict in `threadpoolctl.threadpool_limits(1)`; each run JSON
  records `environment.blas_threads = 1` and the config fingerprint includes it. All full-data and QUICK run JSONs were
  regenerated with `--force` and accepted as the committed results.
- Reason: the native thread count changes the float summation order in lbfgs. go_emotions nltk LR rerun with
  OMP/OPENBLAS/MKL_NUM_THREADS = 1 / 2 / 4 gave accuracy 0.552723 / 0.552941 / 0.553159 and macro-F1
  0.338600 / 0.338894 / 0.338969; the old committed value (0.552941 / 0.338894) is the 2-thread one. After the fix, all
  8 go_emotions and ag_news LR cells rerun with 4 threads were identical to the committed JSONs.
- Changed metrics, old (multi-threaded) -> new (single-threaded); all other run JSONs have identical metrics:

| run JSON | metric | old | new |
|---|---|---|---|
| results/module1/ag_news__tweet__lr.json | accuracy | 0.918289 | 0.918421 |
| results/module1/ag_news__tweet__lr.json | macro_f1 | 0.918149 | 0.918276 |
| results/module1/go_emotions__nltk__lr.json | accuracy | 0.552941 | 0.552723 |
| results/module1/go_emotions__nltk__lr.json | macro_f1 | 0.338894 | 0.338600 |
| results/module1/go_emotions__tweet__lr.json | accuracy | 0.554031 | 0.554684 |
| results/module1/go_emotions__tweet__lr.json | macro_f1 | 0.339879 | 0.340230 |
| results/module1/go_emotions__whitespace__lr.json | accuracy | 0.534423 | 0.535076 |
| results/module1/go_emotions__whitespace__lr.json | macro_f1 | 0.312263 | 0.314023 |
| results/module2/ag_news__lexfront__lr.json | accuracy | 0.918684 | 0.918816 |
| results/module2/ag_news__lexfront__lr.json | macro_f1 | 0.918504 | 0.918630 |
| results/module2/go_emotions__lexfront__lr.json | accuracy | 0.553159 | 0.553595 |
| results/module2/go_emotions__lexfront__lr.json | macro_f1 | 0.339532 | 0.339734 |

### M1: result cache keyed by a fingerprint
- Decision: every run JSON stores `fingerprint = {config, code}` (hash of data subset, tokenizer, vectorizer and
  classifier params, seed, sklearn/nltk versions; hash of `lexer/` and the experiment modules that affect a run).
  `run_cell` reuses a JSON only if both match. The notebook sets `FORCE = QUICK_MODE`, so the CI notebook job and the
  committed notebook recompute every QUICK run. All run JSONs were regenerated single-threaded (see "BLAS threads" below).
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
- Update (claims pass, 2026-10-17): confidence intervals now exist (`results/stats/bootstrap.json`): NLTK to LexFront
  with LR is -1.59 accuracy points on tweet_topic (CI excludes 0) and shows no detectable difference on go_emotions and
  ag_news. "Lower OOV and smaller vocabulary on all 3 datasets" holds among the word-level tokenizers only (BPE has a
  smaller vocabulary and near-zero OOV by construction), and it is not evidence of better classification.

### Open: TF-IDF lowercasing vs LexFront's lowercase rule
- Decision: undecided; must be settled before the ablation (options: `lowercase=False` in TF-IDF, or drop the rule
  from the ablation).
- Reason: TF-IDF lowercases first, so the rule is inactive in all experiments. Source: `results/REPORT_NOTES.md` s.2.
- Update (resolved in P0/P2): the main grid keeps TF-IDF `lowercase=True` (every earlier number stays comparable); the
  lowercase rule is tested in the case-kept setting (`results/sensitivity_case/`, `results/ablation/` no_lowercase).

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
