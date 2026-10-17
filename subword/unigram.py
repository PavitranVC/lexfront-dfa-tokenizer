"""Unigram language-model subword tokenizer, trained with Hugging Face `tokenizers` (Rust, CPU).

NLP concept: Unigram works top-down. It starts from a large set of candidate pieces, gives each a probability by EM
(expectation-maximization) on the training text, and repeatedly drops the pieces whose removal costs the least
likelihood until `vocab_size` pieces remain. A word is encoded by the Viterbi path: the split into pieces with the
highest total log-probability. BPE and WordPiece grow a vocabulary by merges; Unigram prunes one. There is no `##`
marker: a piece is the same string inside and at the start of a word.

Determinism: the set of pieces the trainer keeps is stable, but two things vary between trainings. Its EM scores
differ in the last digits (floating-point summation order). And the characters that pruning dropped but the trainer
must keep are put back at the bottom with scores `lowest + k * 0.0001`, with k following the iteration order of a hash
set, so which character gets which score is random. `canonical_unigram` rebuilds the model: it reassigns the scores of
that bottom block to its characters in code-point order, rounds every score to `score_decimals` and sorts the pieces,
so two trainings on the same text serialize identically (checked by tests and by repeated full-data trainings, see
docs/DECISIONS.md).
"""
import json
from dataclasses import asdict, dataclass

from tokenizers import Tokenizer, models, trainers
from tokenizers import __version__ as TOKENIZERS_VERSION

from .bpe import PRE_TOKENIZERS, UNK, text_hash

UNK_ID = 0                      # `canonical_unigram` puts the unknown token first
RESTORE_STEP = 1e-4             # the trainer's score step between the characters it puts back after pruning


@dataclass(frozen=True)
class UnigramConfig:
    """Everything that defines a trained Unigram apart from its training text (trainer defaults, pinned)."""
    vocab_size: int = 8000
    pre_tokenizer: str = "whitespace"
    unk_token: str = UNK
    shrinking_factor: float = 0.75
    max_piece_length: int = 16
    n_sub_iterations: int = 2
    score_decimals: int = 3


def _restored_block_size(ascending: list[tuple[str, float]]) -> int:
    """Length of the run at the bottom of the ascending-score list whose scores step by RESTORE_STEP (or tie)."""
    size = 1
    while size < len(ascending):
        gap = ascending[size][1] - ascending[size - 1][1]
        if gap > 1e-9 and abs(gap - RESTORE_STEP) > 1e-6:
            break
        size += 1
    return size


def _reassign_restored_characters(pieces: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Give the single characters of the bottom block the block's scores again, in code-point order."""
    ascending = sorted(pieces, key=lambda item: item[1])
    block = ascending[:_restored_block_size(ascending)]
    chars = sorted(piece for piece, _ in block if len(piece) == 1)
    scores = iter(sorted(score for piece, score in block if len(piece) == 1))
    new_score = dict(zip(chars, scores))
    return [(piece, new_score.get(piece, score)) for piece, score in pieces]


def canonical_unigram(tokenizer: Tokenizer, config: UnigramConfig) -> Tokenizer:
    """The same Unigram in a fixed form (see module docstring): `[UNK]` first, then pieces by (-score, piece)."""
    vocab = json.loads(tokenizer.to_str())["model"]["vocab"]
    unk_score = next(score for piece, score in vocab if piece == config.unk_token)
    pieces = _reassign_restored_characters([(piece, score) for piece, score in vocab if piece != config.unk_token])
    scored = sorted(((piece, round(score, config.score_decimals)) for piece, score in pieces),
                    key=lambda item: (-item[1], item[0]))
    canonical = Tokenizer(models.Unigram([(config.unk_token, round(unk_score, config.score_decimals)), *scored],
                                         unk_id=UNK_ID))
    canonical.pre_tokenizer = tokenizer.pre_tokenizer
    return canonical


def train_unigram(texts, config: UnigramConfig = UnigramConfig()) -> Tokenizer:
    """Train a Unigram tokenizer on `texts` (already lowercased by the caller); deterministic for a given input."""
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = PRE_TOKENIZERS[config.pre_tokenizer]()
    trainer = trainers.UnigramTrainer(vocab_size=config.vocab_size, show_progress=False, unk_token=config.unk_token,
                                      special_tokens=[config.unk_token], shrinking_factor=config.shrinking_factor,
                                      max_piece_length=config.max_piece_length,
                                      n_sub_iterations=config.n_sub_iterations)
    tokenizer.train_from_iterator(list(texts), trainer)
    return canonical_unigram(tokenizer, config)


class UnigramTokenizer:
    """Unigram over lowercased text: `tokenize(text) -> list[str]` of subword pieces."""

    def __init__(self, train_texts, config: UnigramConfig = UnigramConfig()):
        self.config = config
        self.model = train_unigram([t.lower() for t in train_texts], config)

    def tokenize(self, text: str) -> list[str]:
        """Pieces of the best split; an unseen span is named `[UNK]` (the library would return its surface string)."""
        encoding = self.model.encode(text.lower())
        return [self.config.unk_token if i == UNK_ID else piece for piece, i in zip(encoding.tokens, encoding.ids)]

    __call__ = tokenize


def record(config: UnigramConfig, train_texts) -> dict:
    """The `subword` entry of a run JSON and fingerprint: config, library version, hash of the training text."""
    return {"model": "unigram", **asdict(config), "hybrid": False,
            "train_text": "train split, lowercased", "train_text_hash": text_hash(t.lower() for t in train_texts),
            "tokenizers": TOKENIZERS_VERSION}
