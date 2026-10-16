"""Subset construction and DFA minimization.

Compiler concepts: *NFA -> DFA by subset construction* and *DFA minimization* (Moore's
partition refinement). A DFA state is a set of NFA states; it accepts rule r where r
is the lowest rule index among the accepting NFA states it contains (rule priority).
Minimization starts from the partition "by accepted rule" -- so two rules are never
merged -- and splits blocks until all members agree on where every symbol leads.
"""
from dataclasses import dataclass, field

from .nfa import NFA


@dataclass
class DFA:
    trans: list = field(default_factory=list)   # state -> {atom: state}
    accept: list = field(default_factory=list)  # state -> rule index or None
    start: int = 0

    @property
    def n_states(self) -> int:
        return len(self.trans)


def epsilon_closure(nfa: NFA, states) -> frozenset:
    """All NFA states reachable from `states` using only epsilon edges."""
    stack, seen = list(states), set(states)
    while stack:
        for nxt in nfa.eps[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return frozenset(seen)


def _move(nfa: NFA, states, atom: int) -> set:
    return {t for s in states for t in nfa.edges[s].get(atom, ())}


def _accepting_rule(nfa: NFA, subset):
    rules = [nfa.accepts[s] for s in subset if s in nfa.accepts]
    return min(rules) if rules else None


def subset_construction(nfa: NFA, n_atoms: int) -> DFA:
    """Determinize `nfa`; only reachable subsets become DFA states."""
    start = epsilon_closure(nfa, {nfa.start})
    index = {start: 0}
    dfa = DFA(trans=[{}], accept=[_accepting_rule(nfa, start)])
    work = [start]
    while work:
        subset = work.pop()
        for atom in range(n_atoms):
            target = epsilon_closure(nfa, _move(nfa, subset, atom))
            if not target:
                continue
            if target not in index:
                index[target] = len(dfa.trans)
                dfa.trans.append({})
                dfa.accept.append(_accepting_rule(nfa, target))
                work.append(target)
            dfa.trans[index[subset]][atom] = index[target]
    return dfa


def _refine(dfa: DFA) -> list[int]:
    """Moore partition refinement; returns state -> block id."""
    block = {}
    for s, rule in enumerate(dfa.accept):
        block[s] = block.setdefault(("rule", rule), len(block))
    block = {s: block[s] for s in range(dfa.n_states)}
    n_blocks = len(set(block.values()))
    while True:
        signature = {}
        new = {}
        for s in range(dfa.n_states):
            sig = (block[s], tuple(sorted((a, block[t]) for a, t in dfa.trans[s].items())))
            new[s] = signature.setdefault(sig, len(signature))
        block = new
        if len(signature) == n_blocks:
            return [block[s] for s in range(dfa.n_states)]
        n_blocks = len(signature)


def minimize(dfa: DFA) -> DFA:
    """Merge equivalent states; states are renumbered breadth-first from the start."""
    block = _refine(dfa)
    rep = {}
    for s, b in enumerate(block):
        rep.setdefault(b, s)
    order, queue, seen = [], [block[dfa.start]], {block[dfa.start]}
    while queue:
        b = queue.pop(0)
        order.append(b)
        for a in sorted(dfa.trans[rep[b]]):
            nb = block[dfa.trans[rep[b]][a]]
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    new_id = {b: i for i, b in enumerate(order)}
    out = DFA(start=0)
    for b in order:
        s = rep[b]
        out.trans.append({a: new_id[block[t]] for a, t in dfa.trans[s].items()})
        out.accept.append(dfa.accept[s])
    return out
