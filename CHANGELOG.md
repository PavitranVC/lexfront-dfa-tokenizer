# Changelog

Newest first. Dates are the day the change was made (IST workday).

## 2026-10-16
- Module 2 core: token spec (11 classes), PLY longest-match lexer with UNKNOWN recovery, normalizer (6 switchable
  rules, 30-entry slang table), symbol table, `LexFront.tokenize()`, own DFA engine (regex parser, Thompson NFA,
  subset construction, Moore minimization, table-driven scanner), DFA state counts + `.dot`/`.png`.
- Module 1 / experiment code: dataset loaders, tokenizers, classifiers, metrics, evaluation loop, runners.
- Reports: tables, figures and observations built from JSON only; doc renderer; Colab notebook generator.
- Docs: `docs/LEARN.md`, `docs/modules/`, README with Colab badge, `results/REPORT_NOTES.md` skeleton.
- Blocker: `huggingface.co` denied by the sandbox network policy, so dataset experiments have not run yet.
- Phase 0: removed Docker files (`Dockerfile`, both compose files, `.dockerignore`).
- Phase 0: rewrote `requirements.txt` (CPU only, no torch, no transformers).
- Phase 0: added `docs/versions/` v1 (base paper), v2 (LexFront proposal),
  v3 (English only, 3 datasets, 5 tokenizers, Docker dropped, Colab).
