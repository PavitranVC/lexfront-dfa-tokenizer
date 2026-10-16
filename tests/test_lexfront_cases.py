"""Hand-written LexFront test cases: (input, expected normalized tokens)."""
import pytest

from lexer.lexfront import LexFront
from lexer.normalize import NormalizeConfig, collapse_elongation, load_slang, normalize_token
from lexer.ply_lexer import PlyLexer
from lexer.symbol_table import SymbolTable

CASES = [
    # --- plain words and punctuation
    ("hello world", ["hello", "world"]),
    ("Hello, World!", ["hello", ",", "world", "!"]),
    ("", []),
    ("   \t\n ", []),
    # --- URL / MENTION / HASHTAG
    ("see https://example.com/a?b=1 now", ["see", "<URL>", "now"]),
    ("go to www.test.org.", ["go", "to", "<URL>", "."]),
    ("thanks @Alice_99!", ["thanks", "<USER>", "!"]),
    ("#BlackFriday deals", ["#BlackFriday", "deals"]),
    ("@a @b", ["<USER>", "<USER>"]),
    # --- emoticons and emoji
    ("nice :)", ["nice", ":)"]),
    ("sad :-( today", ["sad", ":-(", "today"]),
    ("lol ;P", ["lol", ";P"]),
    ("I <3 it", ["i", "<3", "it"]),
    ("fun \U0001F602\U0001F602", ["fun", "\U0001F602", "\U0001F602"]),
    ("ok \U0001F44D\U0001F3FD", ["ok", "\U0001F44D\U0001F3FD"]),
    # --- numbers
    ("I have 3 cats", ["i", "have", "<NUM>", "cats"]),
    ("it costs 1,000.50 dollars", ["it", "costs", "<NUM>", "dollars"]),
    # --- contractions
    ("I can't go", ["i", "can't", "go"]),
    ("She's here, we're late", ["she's", "here", ",", "we're", "late"]),
    ("it’s fine", ["it’s", "fine"]),
    # --- elongation
    ("sooooo good", ["soo", "good"]),
    ("Yesss!!!", ["yess", "!", "!", "!"]),
    ("good food", ["good", "food"]),            # double letters are NOT elongated
    # --- slang
    ("u r great", ["you", "are", "great"]),
    ("idk tbh", ["i", "do", "not", "know", "to", "be", "honest"]),
    ("b4 gr8", ["before", "great"]),
    # --- alphanumerics and accents
    ("covid19 news", ["covid19", "news"]),
    ("café crème", ["café", "crème"]),
    # --- error recovery: unknown characters become UNKNOWN tokens and scanning continues
    ("a § b", ["a", "§", "b"]),
    ("p中q", ["p", "中", "q"]),
    # --- mixed tweet
    ("OMG u wont believe this!!! https://t.co/xyz @bob :)",
     ["oh", "my", "god", "you", "wont", "believe", "this", "!", "!", "!", "<URL>", "<USER>", ":)"]),
]


@pytest.fixture(scope="module")
def lexfront():
    return LexFront()


@pytest.mark.parametrize("text,expected", CASES)
def test_tokenize(lexfront, text, expected):
    assert lexfront.tokenize(text) == expected


def test_at_least_25_cases():
    assert len(CASES) >= 25


def test_classes_of_a_tweet(lexfront):
    classes = [c for c, _, _ in lexfront.analyze("@bob #fun sooo :) 5 don't http://x.io hi . §")]
    assert classes == ["MENTION", "HASHTAG", "ELONGATED", "EMOTICON", "NUMBER",
                       "CONTRACTION", "URL", "WORD", "PUNCT", "UNKNOWN"]


def test_longest_match_beats_earlier_rule():
    # WORD "don" is shorter than CONTRACTION "don't": longest match wins over rule order
    assert [t.cls for t in PlyLexer().scan("don't")] == ["CONTRACTION"]


def test_tie_goes_to_earlier_rule():
    # ELONGATED and WORD both match "sooo" with length 4: ELONGATED is listed first
    assert [t.cls for t in PlyLexer().scan("sooo")] == ["ELONGATED"]


def test_positions_are_correct():
    toks = PlyLexer().scan("ab  cd")
    assert [(t.lexeme, t.pos) for t in toks] == [("ab", 0), ("cd", 4)]


# ---- normalization switches ---------------------------------------------------------
@pytest.mark.parametrize("switch,text,expected", [
    ("url", "http://a.com", ["http://a.com"]),
    ("mention", "@bob", ["@bob"]),
    ("number", "42", ["42"]),
    ("elongated", "sooo", ["sooo"]),
    ("lowercase", "Hello", ["Hello"]),
    ("slang", "u", ["u"]),
])
def test_each_rule_can_be_switched_off(switch, text, expected):
    lf = LexFront(NormalizeConfig().switched(**{switch: False}))
    assert lf.tokenize(text) == expected


def test_collapse_elongation():
    assert collapse_elongation("coooool") == "cool"
    assert collapse_elongation("aaabbb") == "aabb"
    assert collapse_elongation("book") == "book"


def test_slang_table_size():
    assert 25 <= len(load_slang()) <= 40


def test_normalize_token_unknown_class_passthrough():
    assert normalize_token("PUNCT", "!") == ["!"]


# ---- symbol table ------------------------------------------------------------------
def test_symbol_table_counts_and_ids():
    table = SymbolTable()
    LexFront(symbols=table).tokenize("go go :) @a @b")
    assert table["go"].frequency == 2 and table["go"].id == 0
    assert table["<USER>"].frequency == 2 and table["<USER>"].cls == "MENTION"
    assert table[":)"].cls == "EMOTICON"
    assert len(table) == 3
    assert table.rows(1)[0]["frequency"] == 2
