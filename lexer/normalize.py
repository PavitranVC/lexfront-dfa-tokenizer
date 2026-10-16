"""Token normalization: the step after scanning that maps a lexeme to its canonical form.

Compiler concept: *semantic action attached to a token* (what a Lex rule's `{ ... }`
block does). Every rule can be switched on or off independently via `NormalizeConfig`.

    URL       -> <URL>        MENTION -> <USER>       NUMBER -> <NUM>
    ELONGATED -> runs of 3+ identical letters shortened to 2 ("sooooo" -> "soo")
    lowercase -> word-like tokens (WORD, CONTRACTION, ELONGATED) are lowercased
    slang     -> lookup in slang_table.csv ("u" -> "you"); may expand to several tokens
"""
import csv
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

SLANG_PATH = Path(__file__).with_name("slang_table.csv")
WORD_LIKE = {"WORD", "CONTRACTION", "ELONGATED"}
_RUN = re.compile(r"([A-Za-z])\1{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizeConfig:
    """Switches for each normalization rule (all on by default)."""
    url: bool = True
    mention: bool = True
    number: bool = True
    elongated: bool = True
    lowercase: bool = True
    slang: bool = True

    def switched(self, **changes) -> "NormalizeConfig":
        return replace(self, **changes)


@lru_cache(maxsize=None)
def load_slang(path: str = str(SLANG_PATH)) -> dict[str, str]:
    """Read the slang CSV into {slang: expansion}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["slang"]: row["expansion"] for row in csv.DictReader(f)}


def collapse_elongation(word: str) -> str:
    """Shorten every run of 3+ identical letters to exactly 2."""
    return _RUN.sub(lambda m: m.group(1) * 2, word)


def normalize_token(cls: str, lexeme: str, config: NormalizeConfig | None = None) -> list[str]:
    """Normalized form of one token, as a list (slang may expand to several words)."""
    config = config or NormalizeConfig()
    if cls == "URL" and config.url:
        return ["<URL>"]
    if cls == "MENTION" and config.mention:
        return ["<USER>"]
    if cls == "NUMBER" and config.number:
        return ["<NUM>"]
    if cls not in WORD_LIKE:
        return [lexeme]
    if cls == "ELONGATED" and config.elongated:
        lexeme = collapse_elongation(lexeme)
    if config.lowercase:
        lexeme = lexeme.lower()
    if config.slang and lexeme.lower() in load_slang():
        return load_slang()[lexeme.lower()].split()
    return [lexeme]
