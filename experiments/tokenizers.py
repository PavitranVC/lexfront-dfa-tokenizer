"""All tokenizers behind one interface: `tokenize(text) -> list[str]`.

Compiler concept: the *scanner* stage. Three existing methods and our LexFront lexer
are interchangeable, which is what lets the downstream classifier stay fixed.
Subword tokenizers (P3: BPE and the LexFront + BPE hybrid; P4: WordPiece and Unigram) must be trained first, so they are built by a factory,
`build_tokenizer(name, train_texts)`, that receives the dataset's train texts only.
"""
from collections.abc import Callable

from nltk.tokenize import TweetTokenizer, word_tokenize

from lexer.lexfront import LexFront
from lexer.normalize import RULES, NormalizeConfig
from subword import unigram, wordpiece
from subword.bpe import BpeConfig, BpeTokenizer, LexFrontBpe, record, text_hash

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


# Trained subword tokenizers (P3), not in TOKENIZERS: {name: (hybrid?, vocabulary size)}. ':v<N>' names the
# vocabulary-size sensitivity grid. Pure BPE pre-tokenizes with Whitespace, the hybrid with WhitespaceSplit
# (its input is one word-like LexFront token per piece; see docs/DECISIONS.md, P3).
BPE_VOCAB_SIZE = 8000
SUBWORD_TOKENIZERS: dict[str, tuple[bool, int]] = {
    "bpe": (False, BPE_VOCAB_SIZE), "lexfront_bpe": (True, BPE_VOCAB_SIZE),
    "bpe:v32000": (False, 32000), "lexfront_bpe:v32000": (True, 32000)}
PRIMARY_SUBWORD = ("bpe", "lexfront_bpe")
# Library-trained subword tokenizers of P4, one vocabulary size (8000) and no variants: {name: (class, config)}.
# Both pre-tokenize with Whitespace, like the pure BPE.
LIBRARY_SUBWORD = {"wordpiece": (wordpiece.WordPieceTokenizer, wordpiece.WordPieceConfig(vocab_size=BPE_VOCAB_SIZE)),
                   "unigram": (unigram.UnigramTokenizer, unigram.UnigramConfig(vocab_size=BPE_VOCAB_SIZE))}
SUBWORD2 = tuple(LIBRARY_SUBWORD)
_RECORDS = {"wordpiece": wordpiece.record, "unigram": unigram.record}
_TRAINED: dict[tuple[str, str], Tokenizer] = {}


def bpe_config(name: str) -> BpeConfig:
    hybrid, vocab_size = SUBWORD_TOKENIZERS[name]
    return BpeConfig(vocab_size=vocab_size, pre_tokenizer="whitespace_split" if hybrid else "whitespace")


def subword_record(name: str, train_texts) -> dict:
    """{"subword": config, hybrid pass-through and train-text hash} for a trained tokenizer, {} otherwise."""
    if name in LIBRARY_SUBWORD:
        return {"subword": _RECORDS[name](LIBRARY_SUBWORD[name][1], train_texts)}
    if name not in SUBWORD_TOKENIZERS:
        return {}
    return {"subword": record(bpe_config(name), SUBWORD_TOKENIZERS[name][0], train_texts)}


def build_tokenizer(name: str, train_texts) -> Tokenizer:
    """The tokenizer `name`; a subword tokenizer is trained on `train_texts` (the train split only, never test).

    Trained tokenizers are memoized per (name, train-text hash), so NB, LR and the intrinsic metrics share one.
    """
    if name not in SUBWORD_TOKENIZERS and name not in LIBRARY_SUBWORD:
        return get_tokenizer(name)
    train_texts = list(train_texts)            # an iterator would be consumed by the hash below, leaving nothing to train on
    key = (name, text_hash(train_texts))
    if key not in _TRAINED:
        _TRAINED.clear()                       # one trained tokenizer at a time keeps memory flat on big datasets
        if name in LIBRARY_SUBWORD:
            cls, config = LIBRARY_SUBWORD[name]
        else:
            cls, config = (LexFrontBpe if SUBWORD_TOKENIZERS[name][0] else BpeTokenizer), bpe_config(name)
        _TRAINED[key] = cls(train_texts, config)
    return _TRAINED[key]
