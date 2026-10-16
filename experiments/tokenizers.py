"""All tokenizers behind one interface: `tokenize(text) -> list[str]`.

Compiler concept: the *scanner* stage. Three existing methods and our LexFront lexer
are interchangeable, which is what lets the downstream classifier stay fixed.
"""
from collections.abc import Callable

from nltk.tokenize import TweetTokenizer, word_tokenize

from lexer.lexfront import LexFront
from lexer.normalize import RULES, NormalizeConfig

Tokenizer = Callable[[str], list[str]]
_TWEET = TweetTokenizer()


def whitespace_tokenize(text: str) -> list[str]:
    """Split on runs of whitespace (the weakest baseline)."""
    return text.split()


def nltk_tokenize(text: str) -> list[str]:
    """NLTK `word_tokenize` (Punkt sentence split + Treebank-style word split)."""
    return word_tokenize(text)


def tweet_tokenize(text: str) -> list[str]:
    """NLTK `TweetTokenizer` with its default settings."""
    return _TWEET.tokenize(text)


_LEXFRONT: LexFront | None = None


def lexfront_tokenize(text: str) -> list[str]:
    """LexFront with every normalization rule on."""
    global _LEXFRONT
    if _LEXFRONT is None:
        _LEXFRONT = LexFront()
    return _LEXFRONT.tokenize(text)


TOKENIZERS: dict[str, Tokenizer] = {
    "whitespace": whitespace_tokenize,
    "nltk": nltk_tokenize,
    "tweet": tweet_tokenize,
    "lexfront": lexfront_tokenize,
}
MODULE1_TOKENIZERS = ("whitespace", "nltk", "tweet")

# Ablation variants (P2): the same lexer with one normalization rule off, or all of them off.
# They are not in TOKENIZERS, so the main grid and the intrinsic metrics never pick them up.
LEXFRONT_VARIANTS: dict[str, NormalizeConfig] = {
    **{f"lexfront:no_{rule}": NormalizeConfig().switched(**{rule: False}) for rule in RULES},
    "lexfront:all_off": NormalizeConfig.all_off(),
}
_VARIANT_LEXERS: dict[str, LexFront] = {}


def normalize_config(name: str) -> NormalizeConfig | None:
    """The NormalizeConfig behind a LexFront tokenizer name; None for the baselines."""
    if name == "lexfront":
        return NormalizeConfig()
    return LEXFRONT_VARIANTS.get(name)


def get_tokenizer(name: str) -> Tokenizer:
    """A tokenizer by name: one of TOKENIZERS or a LexFront ablation variant ('lexfront:no_url', ...)."""
    if name in TOKENIZERS:
        return TOKENIZERS[name]
    if name not in _VARIANT_LEXERS:
        _VARIANT_LEXERS[name] = LexFront(LEXFRONT_VARIANTS[name])
    return _VARIANT_LEXERS[name].tokenize
