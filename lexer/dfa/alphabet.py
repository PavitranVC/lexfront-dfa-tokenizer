"""Alphabet partition (character classes).

Compiler concept: *symbol equivalence classes*. Instead of one DFA column per
Unicode code point we split the code-point line at every range boundary used by
any rule. Every resulting interval ("atom") behaves identically in every rule, so
the automaton needs one column per atom, not per character.
"""
from bisect import bisect_right

from .regex_parser import Alt, Cat, Chars, Empty, Opt, Plus, Range, Star


def collect_ranges(node) -> list[Range]:
    """All code-point ranges mentioned anywhere in an AST."""
    if isinstance(node, Chars):
        return list(node.ranges)
    if isinstance(node, Empty):
        return []
    if isinstance(node, (Cat, Alt)):
        return [r for part in node.parts for r in collect_ranges(part)]
    if isinstance(node, (Star, Plus, Opt)):
        return collect_ranges(node.child)
    raise TypeError(f"unknown AST node {node!r}")


def partition(asts) -> list[Range]:
    """Disjoint atoms that cover every range of every AST, sorted by code point."""
    ranges = [r for ast in asts for r in collect_ranges(ast)]
    cuts = sorted({lo for lo, _ in ranges} | {hi + 1 for _, hi in ranges})
    atoms = []
    for lo, nxt in zip(cuts, cuts[1:]):
        if any(a <= lo and nxt - 1 <= b for a, b in ranges):
            atoms.append((lo, nxt - 1))
    return atoms


def atoms_of(ranges, atoms: list[Range]) -> list[int]:
    """Indices of the atoms covered by a set of ranges."""
    return [i for i, (lo, hi) in enumerate(atoms)
            if any(a <= lo and hi <= b for a, b in ranges)]


class AtomLookup:
    """Maps a character to its atom index (or -1 if no rule mentions it)."""

    def __init__(self, atoms: list[Range]):
        self.atoms = atoms
        self.starts = [lo for lo, _ in atoms]

    def __call__(self, ch: str) -> int:
        code = ord(ch)
        i = bisect_right(self.starts, code) - 1
        return i if i >= 0 and code <= self.atoms[i][1] else -1
