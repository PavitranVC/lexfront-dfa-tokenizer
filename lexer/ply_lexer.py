"""PLY-based lexer with longest-match selection and UNKNOWN error recovery.

Compiler concepts: *lexical analysis with maximal munch* and *panic-mode recovery*.

PLY's own master regex is an ordered alternation, i.e. "first rule that matches wins",
which is not longest match. So PLY is used for what it is good at (the token stream,
positions, ignored whitespace, `t_error` hook) while a single dispatcher rule picks
the winning class: try every class pattern at the current position, keep the longest
match, and let the earlier class win ties. A character that no class accepts becomes
an UNKNOWN token and scanning continues after it.
"""
import re
from ply import lex
from ply.lex import TOKEN

from .dfa.scanner import Token
from .token_spec import UNKNOWN, real_specs

_WHITESPACE_IGNORED_BY_PLY = " \t\r\n"


class _Rules:
    """PLY rule container (PLY discovers `tokens`, `t_*` attributes by introspection)."""

    t_ignore = _WHITESPACE_IGNORED_BY_PLY

    def __init__(self, specs):
        self.compiled = [(s.name, re.compile(s.pattern, s.flags)) for s in specs]
        self.tokens = tuple(s.name for s in specs) + (UNKNOWN, "ANY")

    @TOKEN(r"\S")                       # PLY rule: one catch-all that does the dispatching
    def t_ANY(self, t):
        text, start = t.lexer.lexdata, t.lexpos
        best_name, best_end = UNKNOWN, start + 1          # fallback: one char, UNKNOWN
        longest = 0
        for name, regex in self.compiled:                   # priority order
            m = regex.match(text, start)
            if m and m.end() - start > longest:             # strictly longer: ties keep earlier
                longest = m.end() - start
                best_name, best_end = name, m.end()
        t.type, t.value = best_name, text[start:best_end]
        t.lexer.lexpos = best_end
        return t

    def t_error(self, t):
        # Only reached for non-ASCII whitespace (e.g. NBSP): skip it like PLY's ignore.
        t.lexer.skip(1)


class PlyLexer:
    """Longest-match lexer over the given token specs (default: all real classes)."""

    def __init__(self, specs=None):
        rules = _Rules(specs if specs is not None else real_specs())
        self._lexer = lex.lex(object=rules, reflags=re.UNICODE, errorlog=lex.NullLogger())

    def scan(self, text: str) -> list[Token]:
        """Tokenize `text` into (class, lexeme, position) tuples."""
        self._lexer.input(text)
        out = []
        while (tok := self._lexer.token()) is not None:
            out.append(Token(tok.type, tok.value, tok.lexpos))
        return out
