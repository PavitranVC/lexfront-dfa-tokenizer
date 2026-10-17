"""Byte-pair encoding (BPE) subword tokenizers, trained with Hugging Face `tokenizers` (Rust, CPU).

NLP concept: BPE starts from single characters and repeatedly merges the most frequent adjacent pair into a new
symbol until the vocabulary has `vocab_size` entries. The learned merges are an ordered rewrite-rule list: encoding a
word applies them in rank order, so words split into vocabulary pieces (out-of-vocabulary tokens almost disappear, at
the price of more tokens per word). Pieces that continue a word carry the marker `##`, so "##ing" (inside a word)
and "ing" (a word of its own) stay different features.

Compiler concept: the hybrid is a two-stage front end. LexFront's scanner decides the token *class* first;
structural tokens (placeholders, hashtags, emoticons, emoji, punctuation, error tokens) are emitted unchanged, and only
word-like lexemes are handed to the subword stage, like a lexer that sends identifiers to a second-level analyzer.

Determinism: the trainer gives the `##`-marked alphabet its ids in hash-map order, and ties between equally frequent
pairs are broken by id, so two trainings on the same text could merge in a different order. `train_bpe` passes that
alphabet first, sorted, as special tokens (fixing every id) and drops them from the added-token list afterwards.
"""
import hashlib
import json
from dataclasses import asdict, dataclass

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers import __version__ as TOKENIZERS_VERSION

from lexer.lexfront import LexFront
from lexer.normalize import WORD_LIKE

UNK = "[UNK]"
PREFIX = "##"
PRE_TOKENIZERS = {"whitespace": pre_tokenizers.Whitespace, "whitespace_split": pre_tokenizers.WhitespaceSplit}
# LexFront classes the hybrid never splits (placeholders <URL> <USER> <NUM> come from URL, MENTION, NUMBER).
STRUCTURAL = ("URL", "MENTION", "NUMBER", "HASHTAG", "EMOTICON", "EMOJI", "PUNCT", "UNKNOWN")


@dataclass(frozen=True)
class BpeConfig:
    """Everything that defines a trained BPE apart from its training text."""
    vocab_size: int = 8000
    pre_tokenizer: str = "whitespace"
    continuing_subword_prefix: str = PREFIX
    unk_token: str = UNK


def text_hash(texts) -> str:
    """SHA-256 (16 hex) of the training texts, for the run fingerprint."""
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode()).hexdigest()[:16]


def sorted_alphabet(texts, pre_tokenizer, config: BpeConfig) -> list[str]:
    """[UNK], every character, then marker + every character seen after the first position of a pre-token.

    This is the initial alphabet the trainer builds itself; listing it sorted fixes the ids (see module docstring).
    """
    chars, inner = set(), set()
    for text in texts:
        for piece, _ in pre_tokenizer.pre_tokenize_str(text):
            chars.update(piece)
            inner.update(piece[1:])
    return [config.unk_token, *sorted(chars), *(config.continuing_subword_prefix + c for c in sorted(inner))]


def drop_added_tokens(tokenizer: Tokenizer) -> Tokenizer:
    """The same model without the seeded special tokens (they must not be matched as literals in input text)."""
    spec = json.loads(tokenizer.to_str())
    spec["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(spec))


def train_bpe(texts, config: BpeConfig = BpeConfig()) -> Tokenizer:
    """Train a BPE tokenizer on `texts` (already lowercased by the caller); deterministic for a given input."""
    texts = list(texts)
    pre = PRE_TOKENIZERS[config.pre_tokenizer]()
    tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token,
                                     continuing_subword_prefix=config.continuing_subword_prefix))
    tokenizer.pre_tokenizer = pre
    trainer = trainers.BpeTrainer(vocab_size=config.vocab_size, show_progress=False,
                                  special_tokens=sorted_alphabet(texts, pre, config),
                                  continuing_subword_prefix=config.continuing_subword_prefix)
    tokenizer.train_from_iterator(texts, trainer)
    return drop_added_tokens(tokenizer)


class BpeTokenizer:
    """Plain BPE over lowercased text: `tokenize(text) -> list[str]` of subword pieces."""

    def __init__(self, train_texts, config: BpeConfig = BpeConfig()):
        self.config = config
        self.model = train_bpe([t.lower() for t in train_texts], config)

    def tokenize(self, text: str) -> list[str]:
        return self.model.encode(text.lower()).tokens

    __call__ = tokenize


class LexFrontBpe:
    """LexFront first; structural tokens pass through whole, word-like tokens are split by a BPE trained on the
    word-like tokens of the train texts."""

    def __init__(self, train_texts, config: BpeConfig = BpeConfig(pre_tokenizer="whitespace_split")):
        self.config = config
        self.lexfront = LexFront()
        self._pieces: dict[str, list[str]] = {}
        self.model = train_bpe([" ".join(self.word_forms(t.lower())) for t in train_texts], config)

    def word_forms(self, text: str) -> list[str]:
        """Normalized forms of the word-like tokens of `text` (the hybrid BPE's training text)."""
        return [f for cls, _, forms in self.lexfront.analyze(text) if cls in WORD_LIKE for f in forms]

    def pieces(self, word: str) -> list[str]:
        """BPE pieces of one word-like form (memoized: words repeat a lot)."""
        if word not in self._pieces:
            self._pieces[word] = self.model.encode(word).tokens
        return self._pieces[word]

    def clear_cache(self) -> None:
        """Forget the memoized pieces (so a timing starts cold, like the tokenizers that have no cache)."""
        self._pieces.clear()

    def tokenize(self, text: str) -> list[str]:
        out = []
        for cls, _, forms in self.lexfront.analyze(text):
            for form in forms:
                out.extend(self.pieces(form) if cls in WORD_LIKE else [form])
        return out

    __call__ = tokenize


def record(config: BpeConfig, hybrid: bool, train_texts) -> dict:
    """The `subword` entry of a run JSON and fingerprint: config, hybrid pass-through, library version, text hash."""
    return {"model": "bpe", **asdict(config), "hybrid": hybrid,
            "pass_through_classes": list(STRUCTURAL) if hybrid else [],
            "train_text": "train split, lowercased", "train_text_hash": text_hash(t.lower() for t in train_texts),
            "tokenizers": TOKENIZERS_VERSION}
