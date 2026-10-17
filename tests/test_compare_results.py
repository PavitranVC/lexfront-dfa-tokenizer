import json

from experiments.compare_results import compare, strip_markdown


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if isinstance(data, (dict, list)) else data)


def test_timing_is_ignored_and_metrics_are_not(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    write(old, "a.json", {"timestamp_utc": "t1", "timing_seconds": {"fit": 1.0}, "metrics": {"accuracy": 0.5}})
    write(new, "a.json", {"timestamp_utc": "t2", "timing_seconds": {"fit": 2.0}, "metrics": {"accuracy": 0.5}})
    assert compare(old, new) == ([], [])
    write(new, "a.json", {"timestamp_utc": "t2", "timing_seconds": {"fit": 2.0}, "metrics": {"accuracy": 0.6}})
    assert compare(old, new)[0] == ["changed: a.json"]


def test_missing_and_added_files(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    write(old, "gone.json", {})
    write(new, "extra.json", {})
    assert compare(old, new) == (["missing in new: gone.json"], ["extra.json"])


def test_markdown_throughput_column_dropped():
    table = "| a | throughput (t/s) |\n|---|---|\n| x | 1,000 |\n"
    assert strip_markdown(table) == strip_markdown(table.replace("1,000", "2,000")) != table


def test_additive_mode_allows_new_keys_only(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    write(old, "a.json", {"tokenizers": {"nltk": {"oov": 1}}})
    write(new, "a.json", {"tokenizers": {"nltk": {"oov": 1}, "unigram": {"oov": 2}}})
    assert compare(old, new)[0] == ["changed: a.json"]
    assert compare(old, new, additive=True) == ([], [])
    write(new, "a.json", {"tokenizers": {"nltk": {"oov": 3}, "unigram": {"oov": 2}}})
    assert compare(old, new, additive=True)[0] == ["changed: a.json"]
