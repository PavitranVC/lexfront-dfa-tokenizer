"""Pipeline driver: rules -> parse -> NFA -> DFA -> minimized DFA -> scanner.

Compiler concept: the whole *lexer-generator* pipeline (what Lex/Flex do), plus
statistics (state counts) and a Graphviz export of the minimized DFA.
"""
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import alphabet
from .dfa import DFA, minimize, subset_construction
from .nfa import NFA, build_nfa
from .regex_parser import parse
from .scanner import TableScanner


@dataclass
class Compiled:
    """Everything produced by compiling a rule list."""
    names: list
    patterns: list
    atoms: list
    nfa: NFA
    dfa: DFA
    min_dfa: DFA

    def scanner(self) -> TableScanner:
        return TableScanner(self.min_dfa, self.atoms, self.names)

    def stats(self) -> dict:
        return {"rules": self.names, "alphabet_atoms": len(self.atoms),
                "nfa_states": self.nfa.n_states, "dfa_states": self.dfa.n_states,
                "min_dfa_states": self.min_dfa.n_states}


def compile_rules(rules, asts=None) -> Compiled:
    """Compile `[(name, pattern), ...]` (priority = list order).

    `asts` may supply pre-built ASTs (constructor-function fallback); otherwise each
    pattern string is parsed.
    """
    names = [n for n, _ in rules]
    patterns = [p for _, p in rules]
    asts = asts or [parse(p) for p in patterns]
    atoms = alphabet.partition(asts)
    nfa = build_nfa(asts, atoms)
    dfa = subset_construction(nfa, len(atoms))
    return Compiled(names, patterns, atoms, nfa, dfa, minimize(dfa))


# ---- Graphviz export -----------------------------------------------------------------
def _char_label(code: int) -> str:
    ch = chr(code)
    if ch.isprintable() and not ch.isspace() and code < 0x250:
        return ch.replace("\\", "\\\\").replace('"', '\\"')
    return f"U+{code:04X}"


def _interval_label(lo: int, hi: int) -> str:
    return _char_label(lo) if lo == hi else f"{_char_label(lo)}-{_char_label(hi)}"


def _merged_intervals(atoms, ids) -> list:
    """Merge adjacent atom intervals so labels stay short."""
    spans = []
    for lo, hi in (atoms[i] for i in sorted(ids)):
        if spans and spans[-1][1] + 1 == lo:
            spans[-1] = (spans[-1][0], hi)
        else:
            spans.append((lo, hi))
    return spans


def _symbol_groups(c: Compiled) -> dict:
    """Group atoms whose column of the transition table is identical in every state."""
    groups = {}
    for a in range(len(c.atoms)):
        column = tuple(state.get(a) for state in c.min_dfa.trans)
        groups.setdefault(column, []).append(a)
    return groups


def to_dot(c: Compiled, max_label: int = 34) -> str:
    """Graphviz DOT text for the minimized DFA (accepting states show their rule)."""
    d = c.min_dfa
    lines = ["digraph DFA {", 'rankdir=LR; bgcolor="white"; dpi=160;',
             'node [fontname="Helvetica", fontsize=13, shape=circle];',
             'edge [fontname="Helvetica", fontsize=11];', "start [shape=point, width=0.08];"]
    for s in range(d.n_states):
        if d.accept[s] is None:
            lines.append(f"q{s};")
        else:
            lines.append(f'q{s} [shape=doublecircle, xlabel="{c.names[d.accept[s]]}"];')
    lines.append(f"start -> q{d.start};")
    edges = {}
    for column, ids in _symbol_groups(c).items():
        text = " ".join(_interval_label(lo, hi) for lo, hi in _merged_intervals(c.atoms, ids))
        text = text if len(text) <= max_label else text[:max_label] + "..."
        for s, t in enumerate(column):
            if t is not None:
                edges.setdefault((s, t), []).append(text)
    for (s, t), labels in sorted(edges.items()):
        lines.append(f'q{s} -> q{t} [label="{" | ".join(labels)}"];')
    lines.append("}")
    return "\n".join(lines)


def write_dot_and_png(c: Compiled, dot_path, png_path) -> bool:
    """Write the .dot file and render the .png with Graphviz; False if `dot` is missing."""
    Path(dot_path).write_text(to_dot(c))
    if shutil.which("dot") is None:
        return False
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    return True
