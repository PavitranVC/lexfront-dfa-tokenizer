"""Extrinsic (accuracy, macro-F1) and intrinsic (OOV, fertility, vocab, throughput) metrics."""
import time
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy and macro-averaged F1."""
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def tokenize_all(tokenizer, texts) -> list[list[str]]:
    return [tokenizer(t) for t in texts]


def intrinsic_metrics(tokenizer, train_texts, test_texts, throughput_docs: int = 1000,
                      repeats: int = 3) -> dict:
    """Tokenizer quality numbers, measured on already-lowercased texts.

    oov_rate_tokens : share of test token occurrences never seen in train
    oov_rate_types  : share of distinct test tokens never seen in train
    fertility       : test tokens per whitespace-separated word of the test texts
    vocab_size      : distinct train tokens
    throughput      : tokens/s and docs/s on the first `throughput_docs` test texts (best of `repeats`)
    """
    train_vocab = Counter(tok for toks in tokenize_all(tokenizer, train_texts) for tok in toks)
    test_counts = Counter(tok for toks in tokenize_all(tokenizer, test_texts) for tok in toks)
    n_tokens = sum(test_counts.values())
    oov = sum(c for tok, c in test_counts.items() if tok not in train_vocab)
    oov_types = sum(1 for tok in test_counts if tok not in train_vocab)
    n_words = sum(len(t.split()) for t in test_texts)
    sample = test_texts[:throughput_docs]
    best = min(_time_once(tokenizer, sample) for _ in range(repeats))
    sample_tokens = sum(len(tokenizer(t)) for t in sample)
    return {"oov_rate_tokens": oov / n_tokens, "oov_rate_types": oov_types / len(test_counts),
            "fertility": n_tokens / n_words, "vocab_size": len(train_vocab),
            "n_test_tokens": n_tokens, "n_test_whitespace_words": n_words,
            "throughput_tokens_per_s": sample_tokens / best, "throughput_docs_per_s": len(sample) / best,
            "throughput_docs": len(sample)}


def _time_once(tokenizer, texts) -> float:
    """Seconds to tokenize `texts`; a tokenizer with a memo cache (`clear_cache`) is timed cold."""
    if hasattr(tokenizer, "clear_cache"):
        tokenizer.clear_cache()
    start = time.perf_counter()
    for t in texts:
        tokenizer(t)
    return time.perf_counter() - start
