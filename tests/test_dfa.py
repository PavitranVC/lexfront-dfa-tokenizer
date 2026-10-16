"""Tests for our own DFA engine and its agreement with the PLY lexer."""
import random

import pytest

from lexer.dfa import regex_parser as rp
from lexer.dfa.compiler import compile_rules, to_dot
from lexer.dfa.regex_parser import RegexSyntaxError, parse
from lexer.dfa.scanner import TableScanner
from lexer.ply_lexer import PlyLexer
from lexer.token_spec import DFA_CLASSES, get_spec, real_specs


def scan(rules, text):
    return [(t.cls, t.lexeme) for t in compile_rules(rules).scanner().scan(text)]


# ---- parser ------------------------------------------------------------------------
def test_parse_structure():
    assert parse("a|b") == rp.Alt((rp.lit("a"), rp.lit("b")))
    assert parse("ab*") == rp.Cat((rp.lit("a"), rp.Star(rp.lit("b"))))
    assert parse("[a-c]+") == rp.Plus(rp.charclass((97, 99)))
    assert parse("(ab)?") == rp.Opt(rp.Cat((rp.lit("a"), rp.lit("b"))))
    assert parse("") == rp.Empty()


@pytest.mark.parametrize("bad", ["(ab", "a)", "*a", ".", "[^a]", "a{2}", "[]", "[b-a]", "\\d", "a\\"])
def test_parse_rejects_unsupported(bad):
    with pytest.raises(RegexSyntaxError):
        parse(bad)


def test_constructor_functions_equal_parser():
    built = rp.cat(rp.charclass((ord("a"), ord("z"))), rp.star(rp.lit("x")))
    assert built == parse("[a-z]x*")


# ---- construction pipeline ----------------------------------------------------------
def test_single_rule_scanning():
    rules = [("AB", "(a|b)+c?")]
    assert scan(rules, "abac bb") == [("AB", "abac"), ("AB", "bb")]


def test_longest_match_and_priority():
    rules = [("KW", "if"), ("ID", "[a-z]+")]
    assert scan(rules, "if iffy") == [("KW", "if"), ("ID", "iffy")]   # tie -> KW, longer -> ID


def test_unknown_recovery():
    rules = [("ID", "[a-z]+")]
    assert scan(rules, "ab?cd") == [("ID", "ab"), ("UNKNOWN", "?"), ("ID", "cd")]


def test_backs_up_to_last_accept():
    # "ab" is accepted; "abc" is a dead end after "abx" -> scanner must fall back to "ab"
    rules = [("T", "ab|abcd")]
    assert scan(rules, "abcx") == [("T", "ab"), ("UNKNOWN", "c"), ("UNKNOWN", "x")]


def test_minimization_reduces_states():
    c = compile_rules([("X", "(a|b)*abb")])
    assert c.min_dfa.n_states <= c.dfa.n_states
    assert c.min_dfa.n_states == 4          # the textbook DFA for (a|b)*abb


def test_minimized_dfa_accepts_same_language():
    c = compile_rules([("X", "(a|b)*abb"), ("Y", "b+")])
    full = TableScanner(c.dfa, c.atoms, c.names)
    rng = random.Random(3)
    for _ in range(300):
        text = "".join(rng.choice("ab ") for _ in range(rng.randint(0, 12)))
        assert full.scan(text) == c.scanner().scan(text), repr(text)


def test_dot_export_mentions_every_rule():
    c = compile_rules([(s.name, s.pattern) for s in real_specs(DFA_CLASSES)])
    dot = to_dot(c)
    assert dot.startswith("digraph DFA") and all(n in dot for n in DFA_CLASSES)


# ---- agreement: DFA scanner vs PLY lexer on the 4 regular classes ------------------------
RULES = [(s.name, s.pattern) for s in real_specs(DFA_CLASSES)]
SAMPLES = [
    "#mood :-) so good ;P !", "@bob_1 #Tag word", "café =D <3 :/", "a1b2 :Dog", "#", "@", "::))",
    "hello world § 中", ":-", ";'(", "x@y#z", "Word123 456",
]


def ply_four_classes():
    return PlyLexer([get_spec(n) for n in DFA_CLASSES])


@pytest.mark.parametrize("text", SAMPLES)
def test_agreement_on_samples(text):
    dfa_tokens = compile_rules(RULES).scanner().scan(text)
    assert dfa_tokens == ply_four_classes().scan(text)


def test_agreement_on_random_strings():
    rng = random.Random(7)
    alphabet = list("abzAZ019_#@:;=-')(DdPp/|][< 3é§\n")
    scanner, ply = compile_rules(RULES).scanner(), ply_four_classes()
    for _ in range(1500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
        assert scanner.scan(text) == ply.scan(text), repr(text)


def test_elongated_is_not_in_the_dfa_engine():
    assert "ELONGATED" not in DFA_CLASSES
    assert "\\1" in get_spec("ELONGATED").pattern       # needs a back-reference
    with pytest.raises(RegexSyntaxError):
        parse(get_spec("ELONGATED").pattern)
