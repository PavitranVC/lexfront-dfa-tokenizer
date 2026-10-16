"""Tests for the fixes from docs/AUDIT.md (audit #1); each test names its finding id."""
import json
from pathlib import Path

import pytest

from experiments import observations, pipeline, report, run_module2
from experiments.dfa_report import dfa_rules
from experiments.pipeline import result_path, run_cell
from lexer.dfa.compiler import compile_rules
from lexer.lexfront import LexFront
from lexer.normalize import NormalizeConfig
from lexer.ply_lexer import PlyLexer
from tests.test_experiments import toy_dataset

LEARN = Path(__file__).resolve().parent.parent / "docs" / "LEARN.md"
QUIET = {"log": lambda *_: None}


# ---- M1: the result cache cannot hide stale results ---------------------------------------
def test_record_carries_fingerprint(tmp_path):
    ds = toy_dataset()
    rec = run_cell(ds, "whitespace", ["nb"], tmp_path, **QUIET)[0]
    assert rec["fingerprint"] == pipeline.fingerprint(ds, "whitespace", "nb", 0)
    assert set(rec["fingerprint"]) == {"config", "code"}


def test_stale_config_is_recomputed(tmp_path):
    ds = toy_dataset()
    run_cell(ds, "whitespace", ["nb"], tmp_path, **QUIET)
    smaller = pipeline.fingerprint(toy_dataset(), "whitespace", "nb", 0)
    ds.train_texts, ds.train_labels = ds.train_texts[:-2], ds.train_labels[:-2]   # a different data subset
    assert pipeline.fingerprint(ds, "whitespace", "nb", 0)["config"] != smaller["config"]
    logged = []
    run_cell(ds, "whitespace", ["nb"], tmp_path, log=logged.append)
    assert len(logged) == 1                                       # recomputed, not read back


def test_stale_code_is_recomputed(tmp_path):
    ds = toy_dataset()
    run_cell(ds, "whitespace", ["nb"], tmp_path, **QUIET)
    path = result_path(tmp_path, ds.name, "whitespace", "nb")
    rec = json.loads(path.read_text())
    rec["fingerprint"]["code"] = "0" * 16                         # as if made by older code
    rec["metrics"]["accuracy"] = -1.0
    path.write_text(json.dumps(rec))
    fresh = run_cell(ds, "whitespace", ["nb"], tmp_path, **QUIET)[0]
    assert fresh["metrics"]["accuracy"] >= 0 and fresh["fingerprint"]["code"] == pipeline.code_fingerprint()


def test_record_without_fingerprint_is_recomputed(tmp_path):
    ds = toy_dataset()
    run_cell(ds, "whitespace", ["nb"], tmp_path, **QUIET)
    path = result_path(tmp_path, ds.name, "whitespace", "nb")
    rec = json.loads(path.read_text())
    del rec["fingerprint"]
    path.write_text(json.dumps(rec))
    assert not pipeline.is_fresh(path, pipeline.fingerprint(ds, "whitespace", "nb", 0))


def test_force_recomputes_matching_cache(tmp_path):
    ds = toy_dataset()
    run_cell(ds, "whitespace", ["nb"], tmp_path, **QUIET)
    logged = []
    run_cell(ds, "whitespace", ["nb"], tmp_path, force=True, log=logged.append)
    assert len(logged) == 1


@pytest.mark.parametrize("base", ["results", "results/quick"])
def test_committed_runs_carry_fingerprint(base):
    """Every committed run JSON has a fingerprint, so a later code change makes it stale (and recomputed)."""
    paths = sorted(Path(base).glob("module*/*__*__*.json"))
    assert len(paths) == 33
    assert all(set(json.loads(p.read_text()).get("fingerprint", {})) == {"config", "code"} for p in paths)


# ---- M2: per-dataset attribution sentence is generated from JSON ------------------------
def test_attribution_sentence_signs():
    assert "0.26 points **below**" in observations._attribution_sentence("tt", -0.26, 1.32, -1.58)
    assert "almost all of it comes from the classifier" in observations._attribution_sentence("ge", 17.54, 17.52, 0.02)
    assert "almost all" not in observations._attribution_sentence("x", 2.0, 1.0, 1.0)


def test_attribution_on_committed_results():
    text = observations.observation("attribution", "results")
    assert "**tweet_topic**: proposed is 0.26 points **below** existing" in text
    assert "**go_emotions**: proposed is 17.58 points above existing; almost all" in text


# ---- m1: intrinsic train size cannot silently change the test subset ---------------------
def test_intrinsic_max_train_with_max_test_is_rejected():
    with pytest.raises(SystemExit):
        run_module2.main(["--only", "intrinsic", "--intrinsic-max-train", "100", "--max-test", "50"])


# ---- m2: slang lookup respects the lowercase switch ---------------------------------------
def test_slang_is_case_sensitive_when_lowercase_off():
    lf = LexFront(NormalizeConfig(lowercase=False))
    assert lf.tokenize("OMG U") == ["OMG", "U"]
    assert lf.tokenize("omg u") == ["oh", "my", "god", "you"]
    assert LexFront().tokenize("OMG U") == ["oh", "my", "god", "you"]


# ---- m3 / m4: documented scanner edge cases ------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("www.", [("ELONGATED", "www"), ("PUNCT", ".")]),
    ("heyyy'll", [("CONTRACTION", "heyyy'll")]),
    ("\U0001F1EE\U0001F1F3", [("EMOJI", "\U0001F1EE"), ("EMOJI", "\U0001F1F3")]),     # flag = 2 tokens
    ("\U0001F468‍\U0001F469", [("EMOJI", "\U0001F468"), ("UNKNOWN", "‍"), ("EMOJI", "\U0001F469")]),
])
def test_scanner_edge_cases(text, expected):
    assert [(t.cls, t.lexeme) for t in PlyLexer().scan(text)] == expected


def test_elongated_www_collapses_to_ww():
    assert LexFront().tokenize("www.") == ["ww", "."]


@pytest.mark.parametrize("space", [" ", "　", " "])
def test_ply_skips_unicode_whitespace(space):
    assert [t.lexeme for t in PlyLexer().scan(f"hi{space}there")] == ["hi", "there"]


# ---- m8: only a missing file becomes a note -----------------------------------------------
def test_render_does_not_hide_bugs(tmp_path, monkeypatch):
    assert "not available" in report.render_all_tables(tmp_path)["dfa_states"]
    monkeypatch.setitem(report.TABLES, "dfa_states", lambda _: {}["missing key"])
    with pytest.raises(KeyError):
        report.render_all_tables(tmp_path)


def test_render_all_tables_writes_nothing(tmp_path):
    report.render_all_tables(tmp_path)
    assert not (tmp_path / "tables").exists()


# ---- m12: numbers hand-typed in docs/LEARN.md match the compiled automaton ---------------
def _trace(scanner, text):
    state, path = scanner.start, [scanner.start]
    for ch in text:
        state = scanner.table[state][scanner.lookup(ch)]
        path.append(state)
    return path


def test_learn_md_dfa_numbers():
    learn = LEARN.read_text()
    combined = compile_rules(dfa_rules())
    mention = compile_rules(dfa_rules(["MENTION"]))
    assert f"MENTION has {mention.nfa.n_states} states" in learn
    assert f"automaton has {len(combined.atoms)} atoms" in learn
    assert f"four-class automaton, {combined.min_dfa.n_states} minimized states" in learn
    scanner = combined.scanner()
    q = _trace(scanner, "#hi")
    assert f"q{q[0]} → q{q[1]} (`#`) → q{q[2]} (`h`) →\nq{q[3]} (`i`)" in learn
    assert " → ".join(f"q{i}" for i in _trace(scanner, ":-)")) in learn
    assert f"`@bob` ends in q{_trace(scanner, '@bob')[-1]}" in learn


# ---- m13: a missing cell is not drawn as a 0 bar ------------------------------------------
def test_existing_vs_proposed_figure_skips_missing_cell(tmp_path, monkeypatch):
    from experiments import figures
    from tests.test_report import make_results
    res = make_results(tmp_path)
    (res / "module2" / "ag_news__lexfront__lr.json").unlink()          # proposed cell missing
    drawn = {}
    monkeypatch.setattr(figures, "_save", lambda fig, _: drawn.setdefault("bars", [len(ax.patches) for ax in fig.axes]))
    figures.figure_existing_vs_proposed(res, tmp_path / "x.png")
    assert drawn["bars"] == [1, 1]                                    # only the existing bar on each panel


# ---- FIXAUDIT: BLAS/OpenMP thread count is pinned (lbfgs summation order) -------------------
class _ThreadProbe:
    """Stand-in classifier that records the native thread pools' sizes seen inside fit/predict."""

    def __init__(self):
        self.seen = []

    def _probe(self):
        from threadpoolctl import threadpool_info
        self.seen += [pool["num_threads"] for pool in threadpool_info()]

    def fit(self, x, y):
        self._probe()
        self.label = y[0]
        return self

    def predict(self, x):
        self._probe()
        return [self.label] * x.shape[0]

    def get_params(self):
        return {}


def _pool_sizes() -> list[int]:
    from threadpoolctl import threadpool_info
    return [pool["num_threads"] for pool in threadpool_info()]


def test_fit_predict_and_vectorize_run_single_threaded(tmp_path, monkeypatch):
    """A2-m4: the outer limit of 2 makes an unpinned run visible even on a 1-CPU host (it would report 2, not 1)."""
    from threadpoolctl import threadpool_limits
    probe, seen_by_tokenizer = _ThreadProbe(), []

    def probing_tokenizer(text):
        seen_by_tokenizer.extend(_pool_sizes())
        return text.split()

    monkeypatch.setattr(pipeline.clf_mod, "make_classifier", lambda name, seed=0: probe)
    monkeypatch.setattr(pipeline, "get_tokenizer", lambda name: probing_tokenizer)
    with threadpool_limits(limits=2):
        assert set(_pool_sizes()) == {2}                          # the control: unpinned code would see 2
        rec = run_cell(toy_dataset(), "whitespace", ["nb"], tmp_path, **QUIET)[0]
    assert probe.seen and set(probe.seen) == {pipeline.BLAS_THREADS} == {1}
    assert seen_by_tokenizer and set(seen_by_tokenizer) == {1}
    assert rec["environment"]["blas_threads"] == 1


# ---- AUDIT_2: fingerprint covers data content and numeric library versions ------------------
def test_vectorizer_built_from_params():
    """A2-m1: VECTORIZER_PARAMS is the single source of the TF-IDF settings."""
    params = pipeline.clf_mod.make_vectorizer(str.split).get_params()
    assert {k: params[k] for k in pipeline.clf_mod.VECTORIZER_PARAMS} == pipeline.clf_mod.vectorizer_kwargs()


def test_changed_text_same_sizes_changes_fingerprint():
    """A2-m2: same split sizes and labels, different texts -> different config fingerprint."""
    ds, upper = toy_dataset(), toy_dataset()
    upper.train_texts = [t.upper() for t in upper.train_texts]
    assert ds.summary() == upper.summary()
    assert pipeline.config_fingerprint(ds, "whitespace", "nb", 0) != pipeline.config_fingerprint(upper, "whitespace", "nb", 0)


@pytest.mark.parametrize("library", ["numpy", "scipy", "threadpoolctl", "sklearn", "nltk"])
def test_library_version_changes_fingerprint(monkeypatch, library):
    """A2-m3: a numpy/scipy/threadpoolctl upgrade invalidates the cache like an sklearn upgrade does."""
    ds = toy_dataset()
    before = pipeline.config_fingerprint(ds, "whitespace", "lr", 0)
    versions = dict(pipeline.library_versions(), **{library: "0.0.0-test"})
    monkeypatch.setattr(pipeline, "library_versions", lambda: versions)
    assert pipeline.config_fingerprint(ds, "whitespace", "lr", 0) != before


@pytest.mark.parametrize("base", ["results", "results/quick"])
def test_committed_runs_record_blas_threads(base):
    paths = sorted(Path(base).glob("module*/*__*__*.json"))
    assert paths and all(json.loads(p.read_text())["environment"].get("blas_threads") == 1 for p in paths)


def test_setup_check_warns_only_for_upgraded_loaded_packages():
    """A2-M2: an imported package whose version changed on disk needs a restart; an unimported one does not."""
    from experiments import setup_check
    before = setup_check.snapshot()
    assert before["numpy"]["loaded"] and setup_check.needs_restart(before) == []
    before["numpy"]["installed"] = "0.0.0-old"             # as if pip had replaced the imported numpy
    before["pandas"] = {"installed": "0.0.0-old", "loaded": False}
    logged = []
    assert setup_check.report_after_install(before, log=logged.append)
    assert setup_check.needs_restart(before) == ["numpy"] and "WARNING" in logged[-1] and "numpy" in logged[-1]
    assert logged[0].startswith("versions in use: numpy ")
