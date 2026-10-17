"""Tests for the large-scale Sentiment140 pipeline (split, audit, noise, ladder, resume, statistics, report, export).

The tweets below are generated test fixtures; no reported number comes from them. Nothing here touches the network.
"""
import json
import subprocess
import zipfile

import numpy as np
import pytest

from experiments import large, run_large


def toy_raw(n_users=60, per_user=12, seed=0):
    rng = np.random.default_rng(seed)
    good, bad = ["wonderful", "happy", "love", "great", "awesome"], ["terrible", "sad", "hate", "awful", "boring"]
    filler = ["today", "really", "work", "tonight", "morning", "friend", "movie", "coffee"]
    texts, users, labels = [], [], []
    for u in range(n_users):
        for k in range(per_user):
            y = int(rng.integers(0, 2))
            words = rng.choice(good if y else bad, 2).tolist() + rng.choice(filler, 3).tolist()
            texts.append(" ".join(words) + f" @user{u} #{k % 3}")
            users.append(f"user{u}")
            labels.append(y)
    return large.Raw(texts, users, labels, ["negative", "positive"], {"hf_id": "toy", "repo_sha": "abc", "n_rows": len(texts)})


# ---- split ---------------------------------------------------------------------------------------------------------
def test_user_split_holds_out_whole_users_deterministically():
    users = [f"u{i % 40}" for i in range(400)]
    mask = large.user_test_mask(users, 0.25, seed=1)
    assert mask == large.user_test_mask(users, 0.25, seed=1) and mask != large.user_test_mask(users, 0.25, seed=2)
    test_users = {u for u, m in zip(users, mask) if m}
    train_users = {u for u, m in zip(users, mask) if not m}
    assert len(test_users) == 10 and not test_users & train_users


def test_build_dataset_splits_by_user_cleans_and_reports():
    raw = toy_raw()
    raw.texts.append("   ")                       # an empty text, to see the cleaning pass run after the split
    raw.users.append("user0")
    raw.labels.append(1)
    ds, report = large.build_dataset(raw)
    assert report["split"]["users_in_both_splits"] == 0 and report["split"]["repo_sha"] == "abc"
    assert report["cleaning"]["removed"]["empty"]["train"] + report["cleaning"]["removed"]["empty"]["test"] == 1
    assert len(ds.train_texts) + len(ds.test_texts) < len(raw.texts) and ds.test_texts
    small, _ = large.build_dataset(raw, 50, 20)
    assert (len(small.train_texts), len(small.test_texts)) == (50, 20)


# ---- audit ---------------------------------------------------------------------------------------------------------
def test_audit_counts_features_per_class():
    texts = ["so goooood :) @a http://x.co #t", "plain text", "&quot;hi&quot; :(", "plain too"]
    a = large.audit(texts, [1, 0, 0, 1])
    assert a["all"]["n"] == 4 and a["positive"]["n"] == 2 and a["negative"]["n"] == 2
    assert a["all"]["emoticon"] == 0.5 and a["all"]["url"] == 0.25 and a["all"]["elongation"] == 0.25
    assert a["negative"]["html_entity"] == 0.5 and a["positive"]["html_entity"] == 0.0


# ---- noise ---------------------------------------------------------------------------------------------------------
def test_noise_is_deterministic_and_off_at_zero():
    text = "this wonderful movie was absolutely fantastic yesterday"
    assert large.corrupt(text, 0.0, 3) == text
    assert large.corrupt(text, 1.0, 3) == large.corrupt(text, 1.0, 3)
    assert large.corrupt(text, 1.0, 3) != large.corrupt(text, 1.0, 4)           # depends on the tweet's position only
    assert large.corrupt(text, 1.0, 3) != text


def test_noise_leaves_structures_and_short_words_alone():
    text = "see @someone at http://example.com #wonderful :) ok no up"
    assert large.corrupt(text, 1.0, 0) == text                                    # nothing eligible
    mixed = "love @friend " + "amazing " * 20
    out = large.corrupt(mixed, 1.0, 0).split(" ")
    assert out[1] == "@friend" and out[2:-1] != ["amazing"] * 20


def test_noise_level_orders_the_amount_of_change():
    texts = ["this wonderful movie was absolutely fantastic yesterday morning"] * 200
    changed = {lv: sum(large.corrupt(t, p, i) != t for i, t in enumerate(texts)) for lv, p in large.NOISE_LEVELS.items()}
    assert changed["clean"] == 0 < changed["light"] < changed["heavy"]
    assert set(large.noisy_tests(texts[:3])) == set(large.NOISE_LEVELS)


def test_every_corruption_operation_returns_a_changed_or_valid_word():
    import random
    for seed in range(60):
        out = large._corrupt_word("fantastic", random.Random(seed))
        assert out.isalpha() and 7 <= len(out) <= 12 and out[0] == "f"


# ---- ladder ---------------------------------------------------------------------------------------------------------
def test_subsets_are_deterministic_distinct_and_sorted():
    a, b = large.subset_indices(1000, 100, 0), large.subset_indices(1000, 100, 0)
    assert a == b == sorted(set(a)) and len(a) == 100
    assert a != large.subset_indices(1000, 100, 1) and a != large.subset_indices(1000, 200, 0)[:100]


# ---- end to end ---------------------------------------------------------------------------------------------------
LADDER = (("60", 60, 2, ("default", "tuned")), ("full", None, 1, ("default", "tuned")), ("huge", 10 ** 9, 1, ("default",)))


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    base = tmp_path_factory.mktemp("large")
    ds, report = large.build_dataset(toy_raw())
    work, out = base / "work", base / "out"
    logs = []
    computed = large.run_grid(ds, ("whitespace", "nltk", "tweet", "lexfront"), LADDER, work, log=logs.append)
    out.mkdir()
    (out / "split.json").write_text(json.dumps(report))
    (out / "audit.json").write_text(json.dumps(large.audit(ds.train_texts, ds.train_labels, ds.label_names)))
    large.consolidate(work, out)
    run_large.write_outputs(out, n_boot=40)
    return {"ds": ds, "work": work, "out": out, "computed": computed, "logs": logs}


def test_grid_runs_every_cell_and_skips_sizes_larger_than_the_data(finished):
    runs = json.loads((finished["out"] / "runs.json").read_text())["runs"]
    # 4 tokenizers x (60: 2 repeats x 2 settings + full: 1 x 2 settings); the "huge" size is skipped
    assert finished["computed"] == len(runs) == 4 * (4 + 2)
    assert any("skip size huge" in line for line in finished["logs"])
    r = next(x for x in runs if x["size"] == "full" and x["setting"] == "tuned" and x["tokenizer"] == "lexfront")
    for clf in ("nb", "lr"):
        c = r["classifiers"][clf]
        assert set(c["metrics"]) == set(large.NOISE_LEVELS) and c["tuning"]["selected"] in c["tuning"]["scores"]
        assert c["metrics"]["clean"]["n_correct"] <= c["metrics"]["clean"]["n"]
    assert (finished["out"] / "predictions" / f"{r['run_id']}.json.gz").exists()
    assert not list((finished["out"] / "predictions").glob("*__r1.json.gz"))      # only full-train runs keep predictions


def test_stored_predictions_reproduce_the_stored_metrics(finished):
    runs = json.loads((finished["out"] / "runs.json").read_text())["runs"]
    r = next(x for x in runs if x["size"] == "full" and x["setting"] == "default" and x["tokenizer"] == "nltk")
    y_true, preds = large.load_predictions(finished["out"] / "predictions" / f"{r['run_id']}.json.gz")
    assert list(y_true) == finished["ds"].test_labels
    for clf, by_level in preds.items():
        for level, y_pred in by_level.items():
            assert abs(float((y_pred == y_true).mean()) - r["classifiers"][clf]["metrics"][level]["accuracy"]) < 1e-12


def test_resume_skips_finished_runs_and_recomputes_changed_data(finished):
    ds, work = finished["ds"], finished["work"]
    stamp = {p.name: p.stat().st_mtime_ns for p in (work / "runs").glob("*.json")}
    again = large.run_grid(ds, ("whitespace",), LADDER, work, log=lambda *_: None)
    assert again == 0 and stamp == {p.name: p.stat().st_mtime_ns for p in (work / "runs").glob("*.json")}
    lost = work / "runs" / "whitespace__default__60__r1.json"                   # a session died before this one was written
    lost.unlink()
    assert large.run_grid(ds, ("whitespace",), LADDER, work, log=lambda *_: None) == 1 and lost.exists()
    original = ds.train_texts[0]
    ds.train_texts[0] = "different text entirely"                                # changed data -> new fingerprint -> recompute
    try:
        changed = large.run_grid(ds, ("whitespace",), (LADDER[1],), work, log=lambda *_: None)
    finally:
        ds.train_texts[0] = original
    assert changed == 2


def test_statistics_and_hypotheses_are_complete(finished):
    s = json.loads((finished["out"] / "stats.json").read_text())
    block = s["full"]["default"]["lr"]
    assert block["n_test"] == len(finished["ds"].test_labels)
    pair = block["pairs"]["lexfront_minus_nltk"]["clean"]["accuracy"]
    assert pair["margin"] in ("within", "beyond", "unclear") and len(pair["ci"]) == 2
    assert set(block["noise_drop"]["lexfront_vs_nltk"]) == {"light", "heavy"}
    sixty = [g for g in s["gaps"] if g["size"] == "60" and g["setting"] == "default" and g["classifier"] == "lr"
             and g["level"] == "clean" and g["versus"] == "nltk"]
    assert sixty and sixty[0]["n"] == 2 and sixty[0]["min"] <= sixty[0]["mean"] <= sixty[0]["max"]
    assert set(s["hypotheses"]) == {"H1", "H2", "H3"}
    assert all(isinstance(h["observed"], bool) for h in s["hypotheses"].values())


def test_size_labels_sort_numerically_with_full_last():
    labels = ["full", "100000", "1000", "1000000", "10000"]
    assert sorted(labels, key=run_large.size_key) == ["1000", "10000", "100000", "1000000", "full"]


def test_margin_verdict_rules():
    assert run_large.margin_verdict([-0.002, 0.003]) == "within"
    assert run_large.margin_verdict([0.006, 0.01]) == "beyond"
    assert run_large.margin_verdict([-0.02, -0.006]) == "beyond"
    assert run_large.margin_verdict([-0.001, 0.02]) == "unclear"


def test_chunked_bootstrap_matches_a_single_draw_of_the_same_stream():
    from experiments import stats
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = np.where(rng.random(500) < 0.8, y, 1 - y)
    chunked = run_large.bootstrap_samples(y, {"a": p}, 300, seed=5, chunk=100)["a"]["accuracy"]
    first = stats.metric_samples(y, p, stats.bootstrap_indices(500, 100, np.random.default_rng(5)), 2)["accuracy"]
    assert len(chunked) == 300 and np.allclose(chunked[:100], first)


def test_report_is_generated_reproducible_and_states_the_hypotheses(finished):
    out = finished["out"]
    report = (out / "REPORT.md").read_text()
    for part in ("## 1. Data", "Size ladder, default", "Size ladder, tuned", "Full training set", "Practical numbers",
                 "Hypotheses of the protocol", "**H1**", "**H2**", "**H3**", "## Limits"):
        assert part in report
    assert "users present in both splits: 0" in report
    assert report.index("Size ladder, default") < report.index("Size ladder, tuned")             # default first
    assert report.index("| 60 |") < report.index("| full |")                                      # numeric sizes, then full
    assert report.index("Full training set, default, LR") < report.index("Full training set, default, NB")
    before = {p.name: p.read_bytes() for p in (out / "stats.json", out / "REPORT.md")}
    run_large.write_outputs(out, n_boot=40)                                      # --stats-only path: same files, byte for byte
    assert before == {p.name: p.read_bytes() for p in (out / "stats.json", out / "REPORT.md")}


def test_stats_only_cli_needs_no_data(finished, monkeypatch):
    monkeypatch.setattr(large, "load_raw", lambda: (_ for _ in ()).throw(AssertionError("stats-only must not load data")))
    run_large.main(["--out", str(finished["out"]), "--stats-only", "--n-boot", "40"])


# ---- export ---------------------------------------------------------------------------------------------------------
def test_zip_contains_the_result_files(finished, tmp_path):
    path = run_large.make_zip(finished["out"], tmp_path / "r.zip")
    names = zipfile.ZipFile(path).namelist()
    assert any(n.endswith("runs.json") for n in names) and any(n.endswith("REPORT.md") for n in names)


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def test_push_goes_to_a_new_branch_and_never_to_main(tmp_path):
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    repo = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(bare), str(repo))
    (repo / "README.md").write_text("x\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    git(repo, "push", "-q", "origin", "HEAD:main")
    out = repo / "results" / "large"
    out.mkdir(parents=True)
    (out / "runs.json").write_text("{}\n")
    with pytest.raises(ValueError):
        run_large.push_results(repo, out, "main", "secret-token")
    assert run_large.push_results(repo, out, "results/large-test", "secret-token") == "pushed branch results/large-test"
    assert "results/large/runs.json" in git(bare, "ls-tree", "-r", "--name-only", "results/large-test")
    assert "results/large" not in git(bare, "ls-tree", "-r", "--name-only", "main")


def test_push_failure_does_not_leak_the_token(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(tmp_path / "missing.git"))
    (repo / "o").mkdir()
    (repo / "o" / "f.json").write_text("{}\n")
    with pytest.raises(RuntimeError) as err:
        run_large.push_results(repo, repo / "o", "results/x", "secret-token")
    assert "secret-token" not in str(err.value) and "push" in str(err.value)
