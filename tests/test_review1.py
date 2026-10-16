from experiments import observations, review1
from tests.test_report import make_results


def test_lexfront_table_columns():
    df = review1.lexfront_table("u r sooo cool :)")
    assert list(df.columns) == ["class", "lexeme", "normalized"]
    assert df["class"].tolist() == ["WORD", "WORD", "ELONGATED", "WORD", "EMOTICON"]


def test_compare_tokenizers_shape():
    df = review1.compare_tokenizers()
    assert 6 <= len(df) <= 8 and list(df.columns)[1:] == ["whitespace", "nltk", "tweet", "lexfront"]


def test_scanner_demo_has_unknown_and_agrees():
    df, agree = review1.scanner_demo()
    assert "UNKNOWN" in set(df["class"]) and agree


def test_symbol_table_demo():
    df = review1.symbol_table_demo()
    assert {"token", "id", "frequency", "class"} <= set(df.columns) and len(df) > 5


def test_observations_come_from_json(tmp_path):
    res = make_results(tmp_path)
    text = observation_text = observations.observation("existing_vs_proposed", res)
    assert "ag_news" in text and "points" in text
    assert "best accuracy" in observations.observation("module1", res)
    assert "not available" in observations.observation("dfa", res)
