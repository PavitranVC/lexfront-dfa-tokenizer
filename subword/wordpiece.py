"""WordPiece subword tokenizer, trained with Hugging Face `tokenizers` (Rust, CPU).

NLP concept: WordPiece starts, like BPE, from single characters and grows the vocabulary by merging pairs, but it
picks the pair that most increases the likelihood of the training text (frequency of the pair divided by the
frequencies of its two parts) instead of the most frequent pair. Encoding is greedy longest-match from the left: the
longest vocabulary piece that starts the word, then the longest that continues it (marked `##`), and a word that
cannot be covered becomes `[UNK]` as a whole. BPE applies its merge list instead; both end with a fixed vocabulary.

Determinism: as for BPE, the trainer assigns ids to the `##`-marked alphabet in hash-map order, so two trainings on the
same text can differ. `train_wordpiece` seeds the alphabet, sorted, as special tokens (the fix of `subword/bpe.py`).
"""
from dataclasses import asdict, dataclass

from tokenizers import Tokenizer, models, trainers
from tokenizers import __version__ as TOKENIZERS_VERSION

from .bpe import PRE_TOKENIZERS, PREFIX, UNK, drop_added_tokens, sorted_alphabet, text_hash


@dataclass(frozen=True)
class WordPieceConfig:
    """Everything that defines a trained WordPiece apart from its training text."""
    vocab_size: int = 8000
    pre_tokenizer: str = "whitespace"
    continuing_subword_prefix: str = PREFIX
    unk_token: str = UNK


def train_wordpiece(texts, config: WordPieceConfig = WordPieceConfig()) -> Tokenizer:
    """Train a WordPiece tokenizer on `texts` (already lowercased by the caller); deterministic for a given input."""
    texts = list(texts)
    pre = PRE_TOKENIZERS[config.pre_tokenizer]()
    tokenizer = Tokenizer(models.WordPiece(unk_token=config.unk_token,
                                           continuing_subword_prefix=config.continuing_subword_prefix))
    tokenizer.pre_tokenizer = pre
    trainer = trainers.WordPieceTrainer(vocab_size=config.vocab_size, show_progress=False,
                                        special_tokens=sorted_alphabet(texts, pre, config),
                                        continuing_subword_prefix=config.continuing_subword_prefix)
    tokenizer.train_from_iterator(texts, trainer)
    return drop_added_tokens(tokenizer)


class WordPieceTokenizer:
    """WordPiece over lowercased text: `tokenize(text) -> list[str]` of subword pieces."""

    def __init__(self, train_texts, config: WordPieceConfig = WordPieceConfig()):
        self.config = config
        self.model = train_wordpiece([t.lower() for t in train_texts], config)

    def tokenize(self, text: str) -> list[str]:
        return self.model.encode(text.lower()).tokens

    __call__ = tokenize


def record(config: WordPieceConfig, train_texts) -> dict:
    """The `subword` entry of a run JSON and fingerprint: config, library version, hash of the training text."""
    return {"model": "wordpiece", **asdict(config), "hybrid": False,
            "train_text": "train split, lowercased", "train_text_hash": text_hash(t.lower() for t in train_texts),
            "tokenizers": TOKENIZERS_VERSION}
