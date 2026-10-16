"""TF-IDF features and the three classifiers (majority, Naive Bayes, Logistic Regression)."""
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

VECTORIZER_PARAMS = {"lowercase": True, "min_df": 2, "ngram_range": [1, 1], "token_pattern": None}
CLASSIFIER_NAMES = ("majority", "nb", "lr")


def vectorizer_kwargs() -> dict:
    """VECTORIZER_PARAMS as TfidfVectorizer arguments (JSON stores ngram_range as a list, sklearn wants a tuple)."""
    return {k: tuple(v) if k == "ngram_range" else v for k, v in VECTORIZER_PARAMS.items()}


def make_vectorizer(tokenizer) -> TfidfVectorizer:
    """TF-IDF over the tokens produced by `tokenizer` (identical settings for every tokenizer).

    VECTORIZER_PARAMS is the single source of the settings, so the fingerprint and the JSON record what really ran.
    """
    return TfidfVectorizer(tokenizer=tokenizer, **vectorizer_kwargs())


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
