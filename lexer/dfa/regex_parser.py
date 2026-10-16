"""Regex parser: pattern text -> AST.

Compiler concept: *syntax analysis*. A recursive-descent parser for the tiny regex
language used by the token classes:

    alt   := cat ('|' cat)*
    cat   := rep*
    rep   := atom ('*' | '+' | '?')*
    atom  := '(' alt ')' | '[' class ']' | '\\' char | literal

Supported: literals, ``[a-z]`` classes (ranges and single characters), concatenation,
``|``, ``*``, ``+``, ``?`` and parentheses. Deliberately rejected, so a pattern can
never mean something different here than in Python's ``re``: ``.``, ``[^...]``,
``{m,n}``, back-references and ``\\d``-style escapes.

The AST can also be built directly with the constructor functions at the bottom
(``lit``, ``charclass``, ``cat``, ``alt``, ``star``, ``plus``, ``opt``).
"""
from dataclasses import dataclass

Range = tuple[int, int]  # inclusive code-point interval

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}
_META = set("()[]|*+?\\.")


class RegexSyntaxError(ValueError):
    """Raised for a pattern outside the supported regex subset."""


@dataclass(frozen=True)
class Chars:
    """Leaf: one character drawn from a set of code-point ranges."""
    ranges: tuple[Range, ...]


@dataclass(frozen=True)
class Empty:
    """Leaf: the empty string (epsilon)."""


@dataclass(frozen=True)
class Cat:
    parts: tuple


@dataclass(frozen=True)
class Alt:
    parts: tuple


@dataclass(frozen=True)
class Star:
    child: object


@dataclass(frozen=True)
class Plus:
    child: object


@dataclass(frozen=True)
class Opt:
    child: object


# ---- constructor functions (build ASTs without parsing text) -----------------------
def lit(ch: str) -> Chars:
    """AST for one literal character."""
    return Chars(((ord(ch), ord(ch)),))


def charclass(*ranges: Range) -> Chars:
    """AST for a character class from (lo, hi) code-point pairs."""
    return Chars(tuple(sorted(ranges)))


def cat(*parts):
    return parts[0] if len(parts) == 1 else Cat(tuple(parts))


def alt(*parts):
    return parts[0] if len(parts) == 1 else Alt(tuple(parts))


def star(x):
    return Star(x)


def plus(x):
    return Plus(x)


def opt(x):
    return Opt(x)


# ---- recursive-descent parser ------------------------------------------------------
class _Parser:
    def __init__(self, text: str):
        self.text, self.i = text, 0

    def peek(self):
        return self.text[self.i] if self.i < len(self.text) else None

    def take(self) -> str:
        ch = self.text[self.i]
        self.i += 1
        return ch

    def fail(self, msg: str):
        raise RegexSyntaxError(f"{msg} at position {self.i} in {self.text!r}")

    def parse(self):
        node = self.parse_alt()
        if self.peek() is not None:
            self.fail(f"unexpected {self.peek()!r}")
        return node

    def parse_alt(self):
        parts = [self.parse_cat()]
        while self.peek() == "|":
            self.take()
            parts.append(self.parse_cat())
        return alt(*parts)

    def parse_cat(self):
        parts = []
        while self.peek() not in (None, "|", ")"):
            parts.append(self.parse_rep())
        return cat(*parts) if parts else Empty()

    def parse_rep(self):
        node = self.parse_atom()
        while self.peek() in ("*", "+", "?"):
            node = {"*": Star, "+": Plus, "?": Opt}[self.take()](node)
        return node

    def parse_atom(self):
        ch = self.peek()
        if ch == "(":
            self.take()
            node = self.parse_alt()
            if self.peek() != ")":
                self.fail("missing ')'")
            self.take()
            return node
        if ch == "[":
            return self.parse_class()
        if ch == "\\":
            return lit(self.parse_escape())
        if ch in ("*", "+", "?"):
            self.fail(f"nothing to repeat before {ch!r}")
        if ch == ".":
            self.fail("wildcard '.' is not supported; escape it as '\\.'")
        if ch in ("]", "{", "}"):
            self.fail(f"unsupported character {ch!r}; escape it")
        return lit(self.take())

    def parse_escape(self) -> str:
        self.take()  # the backslash
        if self.peek() is None:
            self.fail("dangling backslash")
        ch = self.take()
        if ch.isalnum() and ch not in _ESCAPES:
            self.fail(f"escape \\{ch} is not supported")
        return _ESCAPES.get(ch, ch)

    def parse_class_char(self) -> str:
        return self.parse_escape() if self.peek() == "\\" else self.take()

    def parse_class(self):
        self.take()  # '['
        if self.peek() == "^":
            self.fail("negated classes are not supported")
        ranges = []
        while self.peek() not in (None, "]"):
            lo = self.parse_class_char()
            hi = lo
            if self.peek() == "-" and self.text[self.i + 1:self.i + 2] not in ("", "]"):
                self.take()
                hi = self.parse_class_char()
            if ord(hi) < ord(lo):
                self.fail(f"bad range {lo}-{hi}")
            ranges.append((ord(lo), ord(hi)))
        if self.peek() is None or not ranges:
            self.fail("empty or unterminated class")
        self.take()  # ']'
        return charclass(*ranges)


def parse(pattern: str):
    """Parse a regex string into an AST."""
    return _Parser(pattern).parse()
