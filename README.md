# LexFront

A compiler-style lexer (regex -> NFA -> DFA -> minimized DFA) used as a pre-tokenizer for
noisy English text classification. Compiler Design team project.

- Project context and rules: `CLAUDE.md`
- First Claude Code prompt: `FIRST_PROMPT.md`
- DFA prototype: `python lexer/prototype/mini_lexgen.py dfa.dot && dot -Tpng dfa.dot -o dfa.png`

## Quick start

    docker compose up --build                                              # CPU
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA GPU

Open http://localhost:8888, or get a shell with `docker compose exec lab bash`.
Give Docker Desktop at least 8 GB of memory (Settings -> Resources).
