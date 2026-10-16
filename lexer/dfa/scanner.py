"""Table-driven longest-match scanner with error recovery.

Compiler concepts: *maximal munch* (longest match) and *panic-mode error recovery*.
The scanner walks the DFA transition table as far as it can, remembers the last
accepting position, and on getting stuck backs up to it. If no prefix was accepted
the character becomes an UNKNOWN token and scanning resumes at the next character.
"""
from typing import NamedTuple

from .alphabet import AtomLookup

UNKNOWN = "UNKNOWN"
DEAD = -1


class Token(NamedTuple):
    cls: str
    lexeme: str
    pos: int


class TableScanner:
    """Scanner driven by a dense `table[state][atom] -> state | DEAD`."""

    def __init__(self, dfa, atoms, class_names):
        self.class_names = list(class_names)
        self.lookup = AtomLookup(atoms)
        self.accept = list(dfa.accept)
        self.start = dfa.start
        self.table = [[state.get(a, DEAD) for a in range(len(atoms))] for state in dfa.trans]

    def longest_match(self, text: str, pos: int):
        """(end, rule) of the longest accepted prefix starting at `pos`, or None."""
        state, last = self.start, None
        i = pos
        while i < len(text):
            atom = self.lookup(text[i])
            state = self.table[state][atom] if atom >= 0 else DEAD
            if state == DEAD:
                break
            i += 1
            if self.accept[state] is not None:
                last = (i, self.accept[state])
        return last

    def scan(self, text: str) -> list[Token]:
        """Split `text` into tokens; whitespace is skipped, bad characters -> UNKNOWN."""
        tokens, pos = [], 0
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
                continue
            match = self.longest_match(text, pos)
            if match is None:                      # error recovery: skip one character
                tokens.append(Token(UNKNOWN, text[pos], pos))
                pos += 1
            else:
                end, rule = match
                tokens.append(Token(self.class_names[rule], text[pos:end], pos))
                pos = end
        return tokens
