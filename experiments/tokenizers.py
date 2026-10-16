"""All tokenizers behind one interface: `tokenize(text) -> list[str]`.

Compiler concept: the *scanner* stage. Three existing methods and our LexFront lexer
are interchangeable, which is what lets the downstream classifier stay fixed.
"""
from collections.abc import Callable

from nltk.tokenize import TweetTokenizer, word_tokenize

from lexer.lexfront import LexFront

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


def get_tokenizer(name: str) -> Tokenizer:
    return TOKENIZERS[name]
