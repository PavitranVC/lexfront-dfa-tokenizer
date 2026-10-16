"""Tiny lexer generator: regex -> Thompson NFA -> subset-construction DFA -> Hopcroft-style minimization.
Symbols are character CLASSES (disjoint), so edges stay readable in diagrams.
Rules (priority = order): EMOTICON, HASHTAG, WORD.
"""
import itertools, json, sys

# ---- alphabet of disjoint classes -------------------------------------------
CLASSES = {
    "eyes": set(":;="),     # [:;=]
    "nose": set("-"),       # -
    "mouth": set(")(DP"),   # [)(DP]
    "hash": set("#"),       # #
    "letter": set("abcdefghijklmnopqrstuvwxyz"),
}

# ---- regex AST (written as constructors, mirrors the regex text) -----------
def sym(c): return ("sym", c)
def cat(*xs): return ("cat", list(xs))
def alt(*xs): return ("alt", list(xs))
def star(x): return ("star", x)
def plus(x): return cat(x, star(x))
def opt(x): return alt(x, ("eps",))

RULES = [
    ("EMOTICON", "[:;=]-?[)(DP]", cat(sym("eyes"), opt(sym("nose")), sym("mouth"))),
    ("HASHTAG",  "#[a-z]+",       cat(sym("hash"), plus(sym("letter")))),
    ("WORD",     "[a-z]+",        plus(sym("letter"))),
]

# ---- Thompson's construction --------------------------------------------------
class NFA:
    def __init__(self): self.n = 0; self.edges = []  # (src, label|None, dst)
    def new(self): self.n += 1; return self.n - 1

def thompson(nfa, node):
    k = node[0]
    if k == "sym":
        s, t = nfa.new(), nfa.new(); nfa.edges.append((s, node[1], t)); return s, t
    if k == "eps":
        s, t = nfa.new(), nfa.new(); nfa.edges.append((s, None, t)); return s, t
    if k == "cat":
        s, t = thompson(nfa, node[1][0])
        for sub in node[1][1:]:
            s2, t2 = thompson(nfa, sub); nfa.edges.append((t, None, s2)); t = t2
        return s, t
    if k == "alt":
        s, t = nfa.new(), nfa.new()
        for sub in node[1]:
            s2, t2 = thompson(nfa, sub); nfa.edges += [(s, None, s2), (t2, None, t)]
        return s, t
    if k == "star":
        s, t = nfa.new(), nfa.new(); s2, t2 = thompson(nfa, node[1])
        nfa.edges += [(s, None, s2), (t2, None, t), (s, None, t), (t2, None, s2)]
        return s, t
    raise ValueError(k)

nfa = NFA(); start = nfa.new(); accept = {}  # nfa accept state -> rule index
for i, (name, _, ast) in enumerate(RULES):
    s, t = thompson(nfa, ast); nfa.edges.append((start, None, s)); accept[t] = i

def eclose(states):
    stack, seen = list(states), set(states)
    while stack:
        u = stack.pop()
        for a, l, b in nfa.edges:
            if a == u and l is None and b not in seen: seen.add(b); stack.append(b)
    return frozenset(seen)

def move(states, label): return {b for a, l, b in nfa.edges if a in states and l == label}

# ---- subset construction --------------------------------------------------------
d0 = eclose({start}); dstates = [d0]; dtrans = {}; work = [d0]
while work:
    S = work.pop()
    for c in CLASSES:
        T = eclose(move(S, c))
        if not T: continue
        if T not in dstates: dstates.append(T); work.append(T)
        dtrans[(dstates.index(S), c)] = dstates.index(T)

def tag(S):  # highest-priority rule accepted in this DFA state (rule order = priority)
    hits = [accept[q] for q in S if q in accept]
    return min(hits) if hits else None
dtags = [tag(S) for S in dstates]

# ---- minimization (partition refinement, Moore/Hopcroft style) -------------------
part = {}
for i, t in enumerate(dtags): part.setdefault(("acc", t), []).append(i)
blocks = list(part.values())
changed = True
while changed:
    changed = False; new = []
    idx = {s: bi for bi, B in enumerate(blocks) for s in B}
    for B in blocks:
        groups = {}
        for s in B:
            sig = tuple(idx.get(dtrans.get((s, c))) if (s, c) in dtrans else None for c in CLASSES)
            groups.setdefault(sig, []).append(s)
        if len(groups) > 1: changed = True
        new += list(groups.values())
    blocks = new
idx = {s: bi for bi, B in enumerate(blocks) for s in B}
mstart = idx[0]
mtrans = {(idx[s], c): idx[t] for (s, c), t in dtrans.items()}
mtags = {idx[s]: dtags[s] for s in range(len(dstates))}

# rename minimized states in BFS order from the start: q0, q1, ...
order, q = [mstart], [mstart]
while q:
    u = q.pop(0)
    for c in CLASSES:
        v = mtrans.get((u, c))
        if v is not None and v not in order: order.append(v); q.append(v)
name = {s: f"q{i}" for i, s in enumerate(order)}

summary = {
    "rules": [(n, r) for n, r, _ in RULES],
    "nfa_states": nfa.n,
    "dfa_states": len(dstates),
    "min_dfa_states": len(blocks),
    "transitions": sorted([(name[s], c, name[t]) for (s, c), t in mtrans.items()]),
    "accepting": {name[s]: RULES[t][0] for s, t in mtags.items() if t is not None},
}
print(json.dumps(summary, indent=1))

# ---- scanner check: longest match + priority -------------------------------------
def cls(ch):
    for c, chars in CLASSES.items():
        if ch in chars: return c
def scan(text):
    out, i = [], 0
    while i < len(text):
        if text[i] == " ": i += 1; continue
        s, j, last = mstart, i, None
        while j < len(text) and (c := cls(text[j])) and (s, c) in mtrans:
            s = mtrans[(s, c)]; j += 1
            if mtags[s] is not None: last = (j, RULES[mtags[s]][0])
        if last: out.append((last[1], text[i:last[0]])); i = last[0]
        else: out.append(("UNKNOWN", text[i])); i += 1   # panic-mode style recovery
    return out
print(scan("#mood :-) so good ;P !"))

# ---- DOT output -------------------------------------------------------------------
LBL = {"eyes": "[:;=]", "nose": "-", "mouth": "[)(DP]", "hash": "#", "letter": "[a-z]"}
lines = ['digraph DFA {', 'rankdir=LR; bgcolor="white"; dpi=200;',
         'node [fontname="Helvetica", fontsize=13, shape=circle, width=0.55, fixedsize=true];',
         'edge [fontname="Helvetica", fontsize=12];', 'start [shape=point, width=0.08];']
for s in order:
    t = mtags[s]
    if t is not None:
        lines.append(f'{name[s]} [shape=doublecircle, width=0.62, xlabel=<<font point-size="11">{RULES[t][0]}</font>>];')
    else:
        lines.append(f'{name[s]};')
lines.append(f'start -> {name[mstart]};')
pairs = {}
for (s, c), t in mtrans.items(): pairs.setdefault((name[s], name[t]), []).append(LBL[c])
for (a, b), ls in sorted(pairs.items()):
    lines.append(f'{a} -> {b} [label="{", ".join(ls)}"];')
lines.append('}')
open(sys.argv[1], "w").write("\n".join(lines))
