# How this project was built: AI-assisted development

LexFront was built with Claude Code (Anthropic) as the coding assistant. The human author owned the problem, the
scope and every decision about what to run and what to claim. The assistant wrote and edited code and docs inside a
workflow designed to keep its work checkable. This page describes that workflow so a reader can judge it.

## The workflow (context engineering)

| Practice | Where to see it |
|---|---|
| **State lives in the repo, not in the chat.** Every session starts cold, so a short status file and an append-only log carry the context. | [`HANDOFF.md`](HANDOFF.md), [`HANDOFF_LOG.md`](HANDOFF_LOG.md) |
| **One task per session**, with a written prompt pipeline, so each session gets a small, well-defined context. | [`PIPELINE.md`](PIPELINE.md) |
| **Decisions are written down with reasons**, so later sessions follow them instead of re-deciding. | [`DECISIONS.md`](DECISIONS.md) |
| **Protocols and hypotheses are written before any run**, so results cannot quietly rewrite the claim. Amendments are labelled as such. | [`LARGE_PROTOCOL.md`](LARGE_PROTOCOL.md), [`BOOST_PROTOCOL.md`](BOOST_PROTOCOL.md), [`D_PROTOCOL.md`](D_PROTOCOL.md) |
| **Independent adversarial audits** of the code and the claims, each followed by a fix pass. | [`AUDIT.md`](AUDIT.md) to [`AUDIT_6.md`](AUDIT_6.md) |
| **Results are reproducible and generated.** Runs carry config and code fingerprints, reports are generated from JSON, and CI regenerates them and fails on a diff. | `results/`, `.github/workflows/ci.yml` |
| **Honest reporting.** Null and negative results are reported as such, with the limits stated. | [`FINAL_SUMMARY.md`](FINAL_SUMMARY.md) |

## What the human did and what the assistant did

* **Human:** chose the research question and datasets, set the scope, ran the large Colab experiments, approved or
  rejected proposed changes (for example, accepting the logistic regression anchor tolerance in phase A), and decides
  what is claimed.
* **Assistant:** wrote the lexer, the DFA engine, the experiment runners, tests, CI, generated reports and drafts of the
  documentation, under the rules above.

Commits made with the assistant carry a `Co-Authored-By` trailer, so the history shows which changes it helped write.
