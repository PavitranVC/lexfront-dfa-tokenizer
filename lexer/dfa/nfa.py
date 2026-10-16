"""Thompson's construction: AST -> NFA.

Compiler concept: *regular expression -> nondeterministic finite automaton*. Each AST
node becomes a fragment with one start and one end state, glued together with
epsilon edges. Several rules are joined by a fresh start state with an epsilon edge
to each rule's fragment; each rule's end state remembers its rule index so the
priority (earlier rule wins ties) can be resolved later.
"""
from dataclasses import dataclass, field

from .alphabet import atoms_of
from .regex_parser import Alt, Cat, Chars, Empty, Opt, Plus, Star


@dataclass
class NFA:
    eps: list = field(default_factory=list)      # state -> set of epsilon targets
    edges: list = field(default_factory=list)    # state -> {atom: set of targets}
    start: int = 0
    accepts: dict = field(default_factory=dict)  # state -> rule index

    def new_state(self) -> int:
        self.eps.append(set())
        self.edges.append({})
        return len(self.eps) - 1

    @property
    def n_states(self) -> int:
        return len(self.eps)


def _fragment(nfa: NFA, node, atoms) -> tuple[int, int]:
    """Build the NFA fragment for `node`; return (start, end)."""
    s, t = nfa.new_state(), nfa.new_state()
    if isinstance(node, Chars):
        for a in atoms_of(node.ranges, atoms):
            nfa.edges[s].setdefault(a, set()).add(t)
    elif isinstance(node, Empty):
        nfa.eps[s].add(t)
    elif isinstance(node, Cat):
        cur = s
        for part in node.parts:
            ps, pt = _fragment(nfa, part, atoms)
            nfa.eps[cur].add(ps)
            cur = pt
        nfa.eps[cur].add(t)
    elif isinstance(node, Alt):
        for part in node.parts:
            ps, pt = _fragment(nfa, part, atoms)
            nfa.eps[s].add(ps)
            nfa.eps[pt].add(t)
    elif isinstance(node, (Star, Plus, Opt)):
        cs, ct = _fragment(nfa, node.child, atoms)
        nfa.eps[s].add(cs)
        nfa.eps[ct].add(t)
        if isinstance(node, (Star, Plus)):
            nfa.eps[ct].add(cs)          # loop back
        if isinstance(node, (Star, Opt)):
            nfa.eps[s].add(t)            # skip entirely
    else:
        raise TypeError(f"unknown AST node {node!r}")
    return s, t


def build_nfa(asts, atoms) -> NFA:
    """Combined NFA for rules `asts` (rule index = priority, lower wins)."""
    nfa = NFA()
    nfa.start = nfa.new_state()
    for rule, ast in enumerate(asts):
        s, t = _fragment(nfa, ast, atoms)
        nfa.eps[nfa.start].add(s)
        nfa.accepts[t] = rule
    return nfa
