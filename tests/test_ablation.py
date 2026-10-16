"""Per-rule normalization ablation (P2): switches, variant names, fingerprints, stats, table (tiny inputs only)."""
import pytest

from experiments import pipeline, report, run_ablation
from experiments.observations import observation
from experiments.tokenizers import LEXFRONT_VARIANTS, TOKENIZERS, get_tokenizer, normalize_config
from lexer.normalize import RULES, NormalizeConfig
from tests.test_experiments import toy_dataset

QUIET = {"log": lambda *_: None}
TEXT = "OMG u r sooooo GOOD https://t.co/x @bob_1 scored 42"


def test_each_switch_changes_some_tokenizer_output():
    full = get_tokenizer("lexfront")(TEXT)
    for rule in RULES:
        assert get_tokenizer(f"lexfront:no_{rule}")(TEXT) != full, rule


def test_all_off_keeps_the_lexemes():
    assert get_tokenizer("lexfront:all_off")(TEXT) == ["OMG", "u", "r", "sooooo", "GOOD", "https://t.co/x",
                                                       "@bob_1", "scored", "42"]
    assert NormalizeConfig.all_off() == NormalizeConfig(**{r: False for r in RULES})


def test_variants_are_not_in_the_main_tokenizer_list():
    assert not set(LEXFRONT_VARIANTS) & set(TOKENIZERS)
    assert normalize_config("nltk") is None and normalize_config("lexfront") == NormalizeConfig()


def test_variants_have_distinct_fingerprints_and_file_names(tmp_path):
    ds = toy_dataset()
    names = ["lexfront", *LEXFRONT_VARIANTS]
    assert len({pipeline.config_fingerprint(ds, n, "lr", 0) for n in names}) == len(names)
    assert len({pipeline.result_path(tmp_path, "toy", n, "lr") for n in names}) == len(names)
    assert ":" not in pipeline.result_path(tmp_path, "toy", "lexfront:no_url", "lr").name
    assert pipeline.normalize_record("nltk") == {}


def test_ablation_runs_stats_table_and_observation(tmp_path, monkeypatch):
    def toy_load(name, *_args, **_kw):
        ds = toy_dataset()
        ds.name = name
        return ds
    monkeypatch.setattr(run_ablation.data, "load", toy_load)
    out = tmp_path / "ablation"
    records = run_ablation.run_ablation(out, ["ag_news"], **QUIET)
    assert len(records) == sum(len(s.variants) for s in run_ablation.SETTINGS)
    assert {r["normalize"]["url"] for r in records} == {True, False}
    pipeline.run_cell(toy_load("ag_news"), "lexfront", ["lr"], tmp_path / "module2", **QUIET)
    run_ablation.write_stats(out, n_boot=50, seed=1)
    abl = report.load_ablation(tmp_path)
    main = abl["datasets"]["ag_news"]["main"]
    assert main["reference_vs_grid"]["identical_predictions"] is True
    assert abl["datasets"]["ag_news"]["case"]["reference_vs_grid"] is None     # no sensitivity grid in this fixture
    assert "minus_reference" not in main["variants"]["full"] and "mcnemar_accuracy" in main["variants"]["all_off"]
    assert run_ablation.compute(out, n_boot=50, seed=1) == abl
    table = report.render_all_tables(tmp_path)["ablation_table"]
    assert table.count("\n") == 1 + sum(len(s.variants) for s in run_ablation.SETTINGS)
    assert "rule fires, docs train / test" in table
    fires = main["variants"]["no_url"]["rule_fires"]
    assert fires["train"]["docs"] > 0 and main["variants"]["no_number"]["rule_fires"]["train"]["tokens"] == 0
    text = observation("ablation", tmp_path)
    # the toy texts have no numbers or slang and are lowercase already: those three variants are the full system
    assert "**ag_news** (5 variants vs full)" in text and "no multiplicity correction" in text
    assert "Rule never fires on this data (train and test), so not a test of the rule: no_number, no_slang, " \
           "no_lowercase (case kept)" in text
    assert "10 difference CIs of the compared variants" in text and "if the intervals were independent" in text
    assert "all_off (TF-IDF still lowercases)" in table


def test_count_firing_counts_documents_and_tokens():
    configs = {"no_url": normalize_config("lexfront:no_url"), "no_number": normalize_config("lexfront:no_number")}
    counts = run_ablation.count_firing(["see https://a.co and https://b.co", "no link 42", "plain"], configs)
    assert counts["no_url"] == {"docs": 1, "tokens": 2, "n_docs": 3}
    assert counts["no_number"] == {"docs": 1, "tokens": 1, "n_docs": 3}


def test_train_only_firing_is_named_and_still_compared():
    from experiments.observations import _split_variants
    diff = {m: {"diff": 0.0, "ci": [0.0, 0.0], "ci_excludes_zero": False} for m in ("accuracy", "macro_f1")}
    variant = {"minus_reference": diff, "rule_fires": {"train": {"docs": 6, "tokens": 6}, "test": {"docs": 0, "tokens": 0}}}
    compared, never, train_only = _split_variants({"main": {"variants": {"full": {}, "no_mention": variant}}})
    assert [v for v, _ in compared] == ["no_mention"] and never == [] and train_only == [("no_mention", 6)]


def test_missing_ablation_is_reported_not_raised(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_ablation.compute(tmp_path / "ablation")
    assert observation("ablation", tmp_path).startswith("_not available")
