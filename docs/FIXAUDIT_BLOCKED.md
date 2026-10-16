# FIXAUDIT blocked: full-data regeneration changed 5 metrics

`python -m experiments.run_module1 --force` and `python -m experiments.run_module2 --only classify --force` (fresh venv,
pinned requirements = the versions in the committed JSONs: Python 3.11.17, sklearn 1.9.1; `classifier`, `vectorizer`
and `environment` fields identical to main) changed these Logistic Regression metrics vs `results/` on main:

| run JSON | metric | main | regenerated |
|---|---|---|---|
| results/module1/ag_news__nltk__lr.json | accuracy / macro_f1 | 0.918816 / 0.918705 | 0.920132 / 0.920022 |
| results/module1/ag_news__tweet__lr.json | accuracy / macro_f1 | 0.918289 / 0.918149 | 0.918421 / 0.918276 |
| results/module1/go_emotions__nltk__lr.json | accuracy / macro_f1 | 0.552941 / 0.338894 | 0.553159 / 0.338969 |
| results/module1/go_emotions__whitespace__lr.json | macro_f1 | 0.312263 | 0.312370 |
| results/module2/go_emotions__lexfront__lr.json | accuracy / macro_f1 | 0.553159 / 0.339532 | 0.553813 / 0.340009 |

All other 61 run JSONs (all TweetTopic, all NB/majority, all 33 QUICK runs) are identical. A second rerun of AG News
into a scratch directory gave the regenerated values again (0.9201 / 0.9184), so the new values are reproducible on
this machine; the committed ones were presumably produced on different hardware (LR/lbfgs numerics). The audit
reproduced only full TweetTopic, which is unaffected.

State of the branch: code, docs, tests, CI and the QUICK results + notebook are done; the full-data JSONs are left as on
main (no fingerprint), so `tests/test_audit_fixes.py::test_committed_runs_carry_fingerprint[results]` fails and the
`obs:attribution` block in `results/REPORT_NOTES.md` is not rendered yet. Decision needed: accept the regenerated
full-data numbers (and re-render docs), or keep main's numbers and add fingerprints without rerunning.
