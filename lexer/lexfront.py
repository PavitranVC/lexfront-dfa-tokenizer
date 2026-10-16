"""LexFront: the PLY longest-match lexer + normalizer + symbol table behind one function.

`LexFront.tokenize(text) -> list[str]` has the same interface as the baseline
tokenizers, so it can be dropped into the TF-IDF pipeline unchanged.
"""
from .normalize import NormalizeConfig, normalize_token
from .ply_lexer import PlyLexer
from .symbol_table import SymbolTable


class LexFront:
    """Compiler-style pre-tokenizer: scan -> normalize -> (optionally) record symbols."""

    def __init__(self, config: NormalizeConfig | None = None, symbols: SymbolTable | None = None):
        self.config = config or NormalizeConfig()
        self.symbols = symbols
        self._lexer = PlyLexer()

    def analyze(self, text: str) -> list[tuple[str, str, list[str]]]:
        """(class, lexeme, normalized tokens) for every token of `text`."""
        return [(t.cls, t.lexeme, normalize_token(t.cls, t.lexeme, self.config))
                for t in self._lexer.scan(text)]

    def tokenize(self, text: str) -> list[str]:
        """Normalized tokens of `text`; fills the symbol table when one is attached."""
        out = []
        for cls, _, forms in self.analyze(text):
            for form in forms:
                if self.symbols is not None:
                    self.symbols.add(form, cls)
                out.append(form)
        return out

    __call__ = tokenize


_DEFAULT = None


def tokenize(text: str) -> list[str]:
    """Module-level convenience: LexFront with the default configuration."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LexFront()
    return _DEFAULT.tokenize(text)
