# AUDIT 6 — P4 (WordPiece and Unigram): leakage, determinism, independent CIs

Date: 2026-10-17. Auditor: one independent agent working on a copy of branch `feature/finish-3-p4` (HEAD d352e4a), no tracked
file changed. Result: all three checks pass; one low-severity robustness note.

## 1. Leakage (train only): PASS
* Code read: `vectorize()` builds the tokenizer with `build_tokenizer(name, ds.train_texts)` and fits TF-IDF on the train
  texts; test texts only go through `transform`; `clf.fit` gets train features and labels; test labels are used only for
  scoring and the sidecar; the intrinsic-metrics path trains with the train split and tokenizes the test texts; the
  trained-tokenizer memo is keyed by the hash of the training texts. `config_fingerprint` includes `ds.content_hash()`
  (test texts too) but only to invalidate the cache; the `subword` record is train-only. No path from test texts or labels to a
  trainer or a fit.
* Empirical (tweet_topic, real `run_cell`): all 3399 test texts replaced by garbage (CJK, Thai, Devanagari, emoji absent from
  train; 68 documents of about 200k characters). WordPiece and Unigram: `model.to_str()` bit-identical, `subword` record
  (config + train_text_hash) identical, TF-IDF vocabulary identical (6624 and 5391 terms); 1666 and 1765 of 3399 test
  predictions changed (not vacuous). 185 characters occur only in the real test split; none is in either vocabulary.

## 2. Determinism: PASS
* 6 trainings per tokenizer in separate processes (`RAYON_NUM_THREADS` 1,1,1,4,4,4; `PYTHONHASHSEED` 10..45):
  WordPiece one byte-identical model (sha256[:12] f6d0642978b3), Unigram one (fa111461e886).
* Re-running tweet_topic WordPiece+LR and Unigram+LR through `run_cell` into a temp dir reproduced the committed metrics
  (wp 0.716093 / 0.348773, ug 0.730509 / 0.364935), fingerprints, n_features and predictions; the sidecars are byte-identical.
* The raw Unigram trainer is not deterministic (4 trainings with the canonical form bypassed): same piece set, 14-15 pieces
  (the restored-character block) differ by up to 1.3e-3, all others by at most 4.0e-11. After `canonical_unigram` all are identical.
* LOW finding: the rounding margin is tight. Minimum distance of `score * 1000` to a half-integer over the 8000 pieces is
  1.98e-7 (2e-10 in score units; pieces `turkeyhill`, `an`, `partie`), against run-to-run noise of at most 4e-11 (about 5x).
  The closest pieces varied by about 1e-15 in the runs, so nothing flipped, but another CPU or `tokenizers` version could in
  principle flip one rounded score. Not changed (a code change would invalidate every result fingerprint); recorded in
  DECISIONS and LEARN Q38 as a known weak point.

## 3. Independent recomputation of the CIs: PASS
A second bootstrap (written from the pseudocode in `results/REPORT_NOTES.md` section 5, `default_rng(20261016)`, B = 2000, one
index matrix per dataset shared by all runs, accuracy and macro-F1 over present classes, `np.percentile`) on the stored
sidecars reproduced all interval ends and point differences of `results/subword2/stats/bootstrap.json` (5 pairs x 3 datasets x
2 metrics, and the per-run CIs) with a maximum absolute deviation of 4.9e-13; McNemar discordant counts and exact two-sided p
(`scipy.stats.binomtest`) match exactly; accuracy and macro-F1 from sklearn on the sidecars equal the run JSONs; no sign or
"CI excludes 0" flag differs; the sidecars of the compared runs have equal length and identical `y_true`.

## Notes
* INFO: the train-only claim applies to the `subword` record and to the trained models; the overall fingerprint `config` hash also
  covers the test texts (cache invalidation only).
