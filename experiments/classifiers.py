"""TF-IDF features and the three classifiers (majority, Naive Bayes, Logistic Regression)."""
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

VECTORIZER_PARAMS = {"lowercase": True, "min_df": 2, "ngram_range": [1, 1], "token_pattern": None}
# Sensitivity grid (P0): the same settings with TF-IDF case folding switched off, so LexFront's own lowercase rule
# is active (baselines stay case-sensitive). Stored as an override of VECTORIZER_PARAMS, never as a second copy.
CASE_SENSITIVE_OVERRIDES = {"lowercase": False}
CLASSIFIER_NAMES = ("majority", "nb", "lr")


def vectorizer_params(overrides: dict | None = None) -> dict:
    """VECTORIZER_PARAMS with `overrides` applied (the settings a run records in its JSON and fingerprint)."""
    unknown = set(overrides or {}) - set(VECTORIZER_PARAMS)
    if unknown:
        raise ValueError(f"unknown vectorizer settings {sorted(unknown)}")
    return {**VECTORIZER_PARAMS, **(overrides or {})}


def vectorizer_kwargs(params: dict | None = None) -> dict:
    """Vectorizer settings as TfidfVectorizer arguments (JSON stores ngram_range as a list, sklearn wants a tuple)."""
    params = VECTORIZER_PARAMS if params is None else params
    return {k: tuple(v) if k == "ngram_range" else v for k, v in params.items()}


def make_vectorizer(tokenizer, params: dict | None = None) -> TfidfVectorizer:
    """TF-IDF over the tokens produced by `tokenizer` (identical settings for every tokenizer).

    `params` defaults to VECTORIZER_PARAMS, the single source of the settings, so the fingerprint and the JSON record
    what really ran; the sensitivity grid passes `vectorizer_params(CASE_SENSITIVE_OVERRIDES)`.
    """
    return TfidfVectorizer(tokenizer=tokenizer, **vectorizer_kwargs(params))


def make_classifier(name: str, seed: int = 0):
    """Majority baseline, Multinomial NB (the existing method) or Logistic Regression."""
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "nb":
        return MultinomialNB()
    if name == "lr":
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError(f"unknown classifier {name!r}")


def classifier_params(name: str, seed: int = 0) -> dict:
    """Plain-dict hyper-parameters for the result JSON."""
    return make_classifier(name, seed).get_params()
