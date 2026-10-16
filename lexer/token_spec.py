"""Token classes of LexFront, in priority order.

Compiler concept: the *lexical specification* -- one regular expression per token
class. When two classes match the same length, the earlier one wins; otherwise the
longest match wins (maximal munch). UNKNOWN has no pattern: it is the error-recovery
fallback for a character that no class accepts.

Patterns are written in the subset understood by our own regex parser
(`lexer/dfa/regex_parser.py`) *and* by Python's `re`, so one source of truth feeds both
the PLY lexer and the DFA engine. Classes marked `dfa=False` use features the DFA
engine does not support (look-ahead, back-references, shorthand classes).
"""
import re
from dataclasses import dataclass

LATIN = "A-Za-zÀ-ÖØ-öø-ɏ"   # ASCII + Latin-1/Extended letters
EMOJI_BASE = ("\U0001F300-\U0001FAFF☀-➿⭐⭕←-⇿"
              "⌀-⏿\U0001F000-\U0001F2FF‼⁉©®™")
EMOJI_MODIFIERS = "️\U0001F3FB-\U0001F3FF"
PUNCT_CHARS = "!-/:-@\\[-`{-~‘’“”–—…«»¡¿"

UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TokenSpec:
    name: str
    pattern: str | None        # None for UNKNOWN
    flags: int = 0             # Python `re` flags only (DFA classes need none)
    dfa: bool = False          # compiled by our own DFA engine as well?
    note: str = ""


SPECS = (
    TokenSpec("URL", r"(?:https?://|www\.)\S*[A-Za-z0-9/_~#=&%+-]",
              note="http(s):// or www. up to the last URL-ish character"),
    TokenSpec("MENTION", r"@[A-Za-z0-9_]+", dfa=True),
    TokenSpec("HASHTAG", f"#[{LATIN}0-9_]+", dfa=True),
    TokenSpec("EMOTICON", r"([:;=][-']?[)(DdPp/|\]\[]|<3)", dfa=True),
    TokenSpec("EMOJI", f"[{EMOJI_BASE}][{EMOJI_MODIFIERS}]*"),
    TokenSpec("NUMBER", r"[0-9]+(?:[.,][0-9]+)*", note="1 | 3.5 | 1,000"),
    TokenSpec("CONTRACTION", r"[A-Za-z]+['’](?:t|s|re|ve|ll|d|m)(?![A-Za-z0-9])",
              flags=re.IGNORECASE, note="don't, it's, we're; uses look-ahead"),
    TokenSpec("ELONGATED", r"[A-Za-z]*([A-Za-z])\1{2,}[A-Za-z]*", flags=re.IGNORECASE,
              note="a letter repeated 3+ times; needs a back-reference, so not a "
                   "regular expression and left out of the DFA engine"),
    TokenSpec("WORD", f"[{LATIN}][{LATIN}0-9]*", dfa=True),
    TokenSpec("PUNCT", f"[{PUNCT_CHARS}]"),
    TokenSpec(UNKNOWN, None, note="error-recovery fallback"),
)

CLASS_NAMES = tuple(s.name for s in SPECS)
DFA_CLASSES = tuple(s.name for s in SPECS if s.dfa)
_BY_NAME = {s.name: s for s in SPECS}


def get_spec(name: str) -> TokenSpec:
    return _BY_NAME[name]


def real_specs(names=None) -> list[TokenSpec]:
    """Specs that have a pattern (everything except UNKNOWN), in priority order."""
    wanted = set(names) if names is not None else None
    return [s for s in SPECS if s.pattern is not None and (wanted is None or s.name in wanted)]
