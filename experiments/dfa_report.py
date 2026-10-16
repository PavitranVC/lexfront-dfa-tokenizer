"""Compile the four regular token classes with our own DFA engine and save the evidence.

Writes `dfa_states.json`, `dfa_min.dot` and `dfa_min.png` into the output directory.
Also compiles an *expanded* ELONGATED (26 alternatives, one per lowercase letter) to put a
real number on the finding below.

Finding: ELONGATED is usually written `(.)\\1{2,}`. The back-reference `\\1` is not part of
regular expressions, so that pattern cannot go into a DFA as written. For a *finite*
alphabet and a fixed run length the language is still regular: it can be spelled out as
`aaa|bbb|...|zzz`. That costs a larger automaton instead of a back-reference.
"""
import json
import string
from pathlib import Path

from lexer.dfa.compiler import compile_rules, write_dot_and_png
from lexer.token_spec import DFA_CLASSES, get_spec, real_specs

EXPANDED_ELONGATED = ("[a-z]*(" + "|".join(c * 3 for c in string.ascii_lowercase) + ")[a-z]*")


def dfa_rules(names=DFA_CLASSES) -> list[tuple[str, str]]:
    """[(class name, pattern)] in priority order for the given classes."""
    return [(s.name, s.pattern) for s in real_specs(names)]


def build_report(out_dir) -> dict:
    """Compile, count states, and write the JSON/.dot/.png files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined = compile_rules(dfa_rules())
    per_class = {n: compile_rules(dfa_rules([n])).stats() for n in DFA_CLASSES}
    expanded = compile_rules([("ELONGATED_expanded_lowercase", EXPANDED_ELONGATED)]).stats()
    png_ok = write_dot_and_png(combined, out / "dfa_min.dot", out / "dfa_min.png")
    report = {
        "classes": DFA_CLASSES,
        "patterns": {n: get_spec(n).pattern for n in DFA_CLASSES},
        "combined": combined.stats(),
        "per_class": per_class,
        "elongated_backreference_pattern": get_spec("ELONGATED").pattern,
        "elongated_expanded_lowercase": {"pattern_alternatives": 26, **expanded},
        "files": {"dot": "dfa_min.dot", "png": "dfa_min.png" if png_ok else None},
        "minimizer": "Moore partition refinement",
    }
    (out / "dfa_states.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(build_report("results/module2"), indent=2))
