"""Markdown tables built *only* from the JSON files under a results directory.

Layout read here:  <results>/module1/<dataset>__<tokenizer>__<clf>.json
                   <results>/module2/<dataset>__lexfront__<clf>.json
                   <results>/module2/intrinsic__<dataset>.json
                   <results>/module2/dfa_states.json
                   <results>/stats/bootstrap.json                      (python -m experiments.run_stats)
                   <results>/sensitivity_case/module{1,2}/*.json        (python -m experiments.run_sensitivity)
                   <results>/sensitivity_case/stats/bootstrap.json      (python -m experiments.run_stats
                                                                         --results results/sensitivity_case)
                   <results>/ablation/stats/bootstrap.json              (python -m experiments.run_ablation)
                   <results>/bpe/stats/bootstrap.json                   (python -m experiments.run_bpe)
"""
import json
from pathlib import Path

import pandas as pd

DATASET_ORDER = ["tweet_topic", "go_emotions", "ag_news"]
TOKENIZER_ORDER = ["whitespace", "nltk", "tweet", "lexfront"]
SUBWORD_ORDER = ["bpe", "lexfront_bpe"]        # P3; after the four fixed tokenizers in tables (figures show four)
RUN_COLUMNS = ["module", "dataset", "tokenizer", "classifier", "accuracy", "macro_f1", "n_train", "n_test",
               "n_labels", "hf_id", "config", "split_info"]
CLASSIFIER_LABEL = {"majority": "Majority", "nb": "Naive Bayes", "lr": "Logistic Reg."}


def load_runs(results_dir) -> pd.DataFrame:
    """One row per classification run found in module1/ and module2/."""
    rows = []
    for sub in ("module1", "module2"):
        for path in sorted((Path(results_dir) / sub).glob("*__*__*.json")):
            r = json.loads(path.read_text())
            rows.append({"module": sub, "dataset": r["dataset"]["name"], "tokenizer": r["tokenizer"],
                         "classifier": r["classifier"]["name"], **r["metrics"],
                         "n_train": r["dataset"]["n_train"], "n_test": r["dataset"]["n_test"],
                         "n_labels": r["dataset"]["n_labels"], "hf_id": r["dataset"]["hf_id"],
                         "config": r["dataset"]["config"], "split_info": r["dataset"]["split_info"]})
    return pd.DataFrame(rows, columns=RUN_COLUMNS) if not rows else pd.DataFrame(rows)


def load_intrinsic(results_dir) -> pd.DataFrame:
    rows = []
    for path in sorted((Path(results_dir) / "module2").glob("intrinsic__*.json")):
        r = json.loads(path.read_text())
        for tok, m in r["tokenizers"].items():
            rows.append({"dataset": r["dataset"]["name"], "tokenizer": tok, **m,
                         "n_train": r["dataset"]["n_train"], "n_test": r["dataset"]["n_test"]})
    if not rows:
        raise FileNotFoundError(f"no intrinsic__*.json under {Path(results_dir) / 'module2'}")
    return pd.DataFrame(rows)


def load_dfa(results_dir) -> dict:
    return json.loads((Path(results_dir) / "module2" / "dfa_states.json").read_text())


def to_markdown(df: pd.DataFrame) -> str:
    """Plain GitHub-flavoured markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _pct(x) -> str:
    return f"{100 * x:.2f}"


def _order(series, order):
    return series.map({n: i for i, n in enumerate(order)}).fillna(len(order))


def _sorted(df):
    return df.assign(_d=_order(df["dataset"], DATASET_ORDER),
                     _t=_order(df["tokenizer"], TOKENIZER_ORDER + SUBWORD_ORDER)
                     ).sort_values(["_d", "_t"]).drop(columns=["_d", "_t"])


def load_datasets(results_dir) -> list[dict]:
    return json.loads((Path(results_dir) / "datasets.json").read_text())["datasets"]


def _subsample_text(sub) -> str:
    if not sub or (sub.get("max_train") is None and sub.get("max_test") is None):
        return "none (full data)"
    return f"train<={sub.get('max_train')}, test<={sub.get('max_test')}, seed {sub.get('seed')}"


def table_datasets(results_dir) -> str:
    """Dataset overview from datasets.json (written by `data.write_summary`)."""
    rows = []
    for r in sorted(load_datasets(results_dir), key=lambda r: DATASET_ORDER.index(r["name"])):
        counts = r["test_class_counts"]
        rows.append({"dataset": r["name"], "source": r["hf_id"] + (f" ({r['config']})" if r["config"] else ""),
                     "train split": r["split_info"].get("train"), "test split": r["split_info"].get("test"),
                     "train docs": r["n_train"], "test docs": r["n_test"], "labels": r["n_labels"],
                     "largest test class %": _pct(max(counts.values()) / sum(counts.values())),
                     "subsample": _subsample_text(r["split_info"].get("subsample"))})
    return to_markdown(pd.DataFrame(rows))


def table_module1(results_dir) -> str:
    """Accuracy / macro-F1 (in %) of the three existing tokenizers with the three classifiers."""
    runs = load_runs(results_dir)
    runs = runs[runs["module"] == "module1"]
    rows = []
    for (ds, tok), g in _sorted(runs).groupby(["dataset", "tokenizer"], sort=False):
        row = {"dataset": ds, "tokenizer": tok}
        for clf, label in CLASSIFIER_LABEL.items():
            hit = g[g["classifier"] == clf]
            row[f"{label} acc / F1"] = (f"{_pct(hit.accuracy.iloc[0])} / {_pct(hit.macro_f1.iloc[0])}"
                                        if len(hit) else "-")
        rows.append(row)
    return to_markdown(pd.DataFrame(rows))


def _cell(runs, dataset, tokenizer, clf):
    hit = runs[(runs.dataset == dataset) & (runs.tokenizer == tokenizer) & (runs.classifier == clf)]
    return hit.iloc[0] if len(hit) else None


def table_existing_vs_proposed(results_dir) -> str:
    """Existing (NLTK + NB) against proposed (LexFront + LR), plus the two mixed cells."""
    runs = load_runs(results_dir)
    rows = []
    for ds in [d for d in DATASET_ORDER if d in set(runs.dataset)]:
        existing, proposed = _cell(runs, ds, "nltk", "nb"), _cell(runs, ds, "lexfront", "lr")
        lf_nb, nltk_lr = _cell(runs, ds, "lexfront", "nb"), _cell(runs, ds, "nltk", "lr")
        if existing is None or proposed is None:
            continue
        rows.append({"dataset": ds,
                     "existing NLTK+NB acc / F1": f"{_pct(existing.accuracy)} / {_pct(existing.macro_f1)}",
                     "proposed LexFront+LR acc / F1": f"{_pct(proposed.accuracy)} / {_pct(proposed.macro_f1)}",
                     "delta acc (pts)": f"{100 * (proposed.accuracy - existing.accuracy):+.2f}",
                     "delta F1 (pts)": f"{100 * (proposed.macro_f1 - existing.macro_f1):+.2f}",
                     "LexFront+NB acc / F1": (f"{_pct(lf_nb.accuracy)} / {_pct(lf_nb.macro_f1)}"
                                              if lf_nb is not None else "-"),
                     "NLTK+LR acc / F1": (f"{_pct(nltk_lr.accuracy)} / {_pct(nltk_lr.macro_f1)}"
                                          if nltk_lr is not None else "-")})
    return to_markdown(pd.DataFrame(rows))


def table_intrinsic(results_dir) -> str:
    df = _sorted(load_intrinsic(results_dir))
    out = pd.DataFrame({"dataset": df.dataset, "tokenizer": df.tokenizer,
                        "OOV rate % (tokens)": df.oov_rate_tokens.map(_pct),
                        "OOV rate % (types)": df.oov_rate_types.map(_pct),
                        "fertility": df.fertility.map("{:.3f}".format),
                        "vocab size": df.vocab_size,
                        "throughput (tokens/s)": df.throughput_tokens_per_s.map("{:,.0f}".format)})
    return to_markdown(out)


def table_dfa(results_dir) -> str:
    d = load_dfa(results_dir)
    rows = [{"automaton": n, "alphabet atoms": s["alphabet_atoms"], "NFA states": s["nfa_states"],
             "DFA states": s["dfa_states"], "minimized DFA states": s["min_dfa_states"]}
            for n, s in d["per_class"].items()]
    c = d["combined"]
    rows.append({"automaton": "combined (" + "+".join(d["classes"]) + ")",
                 "alphabet atoms": c["alphabet_atoms"], "NFA states": c["nfa_states"],
                 "DFA states": c["dfa_states"], "minimized DFA states": c["min_dfa_states"]})
    e = d["elongated_expanded_lowercase"]
    rows.append({"automaton": "ELONGATED, 26 spelled-out branches (extra, not in LexFront)",
                 "alphabet atoms": e["alphabet_atoms"], "NFA states": e["nfa_states"],
                 "DFA states": e["dfa_states"], "minimized DFA states": e["min_dfa_states"]})
    return to_markdown(pd.DataFrame(rows))


SENSITIVITY_SUBDIR = "sensitivity_case"


def load_sensitivity_runs(results_dir) -> pd.DataFrame:
    """Runs of the case-sensitive TF-IDF grid; FileNotFoundError if that grid was not run here."""
    runs = load_runs(Path(results_dir) / SENSITIVITY_SUBDIR)
    if runs.empty:
        raise FileNotFoundError(f"no run JSONs under {Path(results_dir) / SENSITIVITY_SUBDIR}")
    return runs


def table_sensitivity_case(results_dir) -> str:
    """NB / LR with TF-IDF `lowercase=False` for all four tokenizers, and the LR change against the main grid."""
    sens, main = load_sensitivity_runs(results_dir), load_runs(results_dir)
    rows = []
    for (ds, tok), g in _sorted(sens).groupby(["dataset", "tokenizer"], sort=False):
        row = {"dataset": ds, "tokenizer": tok}
        for clf in ("nb", "lr"):
            hit = g[g["classifier"] == clf]
            row[f"{CLASSIFIER_LABEL[clf]} acc / F1 (case kept)"] = (
                f"{_pct(hit.accuracy.iloc[0])} / {_pct(hit.macro_f1.iloc[0])}" if len(hit) else "-")
        lr, base = _cell(sens, ds, tok, "lr"), _cell(main, ds, tok, "lr")
        both = lr is not None and base is not None
        row["LR acc change vs main (pts)"] = f"{100 * (lr.accuracy - base.accuracy):+.2f}" if both else "-"
        row["LR F1 change vs main (pts)"] = f"{100 * (lr.macro_f1 - base.macro_f1):+.2f}" if both else "-"
        rows.append(row)
    return to_markdown(pd.DataFrame(rows))


def load_bootstrap(results_dir) -> dict:
    return json.loads((Path(results_dir) / "stats" / "bootstrap.json").read_text())


def _ci_text(value, ci) -> str:
    """'+1.23 [-0.45, +2.34]' in points (signed) for a difference."""
    return f"{100 * value:+.2f} [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"


def _run_order(run: str) -> tuple:
    """Sort key for '<tokenizer>__<classifier>': tokenizer order, then majority, NB, LR."""
    tok, clf = run.split("__")
    rank = TOKENIZER_ORDER.index(tok) if tok in TOKENIZER_ORDER else len(TOKENIZER_ORDER)
    return rank, list(CLASSIFIER_LABEL).index(clf) if clf in CLASSIFIER_LABEL else len(CLASSIFIER_LABEL)


def p_value_text(p: float) -> str:
    """'= 0.773' style text; '< 0.001' below three decimals (a printed 0.000 would read as exactly zero)."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def table_ci_runs(results_dir) -> str:
    """Accuracy and macro-F1 (%) with 95% bootstrap CIs for every NB and LR run of the main grid."""
    boot = load_bootstrap(results_dir)
    rows = []
    for ds, d in boot["datasets"].items():
        for run in sorted(d["runs"], key=_run_order):
            tok, clf = run.split("__")
            if clf == "majority":
                continue
            r = d["runs"][run]
            rows.append({"dataset": ds, "tokenizer": tok, "classifier": CLASSIFIER_LABEL[clf],
                         "accuracy % [95% CI]": f"{_pct(r['accuracy']['value'])} [{_pct(r['accuracy']['ci'][0])}, "
                                                f"{_pct(r['accuracy']['ci'][1])}]",
                         "macro-F1 % [95% CI]": f"{_pct(r['macro_f1']['value'])} [{_pct(r['macro_f1']['ci'][0])}, "
                                                f"{_pct(r['macro_f1']['ci'][1])}]"})
    return to_markdown(pd.DataFrame(rows))


def load_sensitivity_bootstrap(results_dir) -> dict:
    """Bootstrap CIs of the case-sensitive grid (same code, seed and B as the main grid)."""
    return load_bootstrap(Path(results_dir) / SENSITIVITY_SUBDIR)


def table_sensitivity_ci(results_dir) -> str:
    """Paired differences (points, 95% CI) in the case-sensitive grid next to the same pair in the main grid."""
    sens, main = load_sensitivity_bootstrap(results_dir), load_bootstrap(results_dir)
    rows = []
    for ds, d in sens["datasets"].items():
        for name, p in d["pairs"].items():
            m = main["datasets"].get(ds, {}).get("pairs", {}).get(name)
            mc = p.get("mcnemar_accuracy")
            rows.append({"dataset": ds, "difference": p["label"],
                         "acc pts [95% CI], case kept": _ci_text(p["accuracy"]["diff"], p["accuracy"]["ci"]),
                         "acc pts [95% CI], main": _ci_text(m["accuracy"]["diff"], m["accuracy"]["ci"]) if m else "-",
                         "F1 pts [95% CI], case kept": _ci_text(p["macro_f1"]["diff"], p["macro_f1"]["ci"]),
                         "F1 pts [95% CI], main": _ci_text(m["macro_f1"]["diff"], m["macro_f1"]["ci"]) if m else "-",
                         "McNemar p (acc), case kept": p_value_text(mc["p_value"]).lstrip("= ") if mc else "-"})
    return to_markdown(pd.DataFrame(rows))


def table_ci(results_dir) -> str:
    """Paired bootstrap 95% CIs of the differences (points) and the exact McNemar p-value where computed."""
    boot = load_bootstrap(results_dir)
    rows = []
    for ds, d in boot["datasets"].items():
        for p in d["pairs"].values():
            mc = p.get("mcnemar_accuracy")
            rows.append({"dataset": ds, "difference": p["label"],
                         "accuracy pts [95% CI]": _ci_text(p["accuracy"]["diff"], p["accuracy"]["ci"]),
                         "macro-F1 pts [95% CI]": _ci_text(p["macro_f1"]["diff"], p["macro_f1"]["ci"]),
                         "CI excludes 0 (acc / F1)": (f"{'yes' if p['accuracy']['ci_excludes_zero'] else 'no'} / "
                                                      f"{'yes' if p['macro_f1']['ci_excludes_zero'] else 'no'}"),
                         "McNemar p (acc)": p_value_text(mc["p_value"]).lstrip("= ") if mc else "-"})
    return to_markdown(pd.DataFrame(rows))


ABLATION_SUBDIR = "ablation"


def load_ablation(results_dir) -> dict:
    """Ablation statistics (python -m experiments.run_ablation): per dataset, per setting, per variant."""
    return load_bootstrap(Path(results_dir) / ABLATION_SUBDIR)


def _fires_text(fires: dict | None) -> str:
    """'12 / 3' documents (train / test) where the removed rule changes a token; '-' if not counted."""
    return f"{fires['train']['docs']} / {fires['test']['docs']}" if fires else "-"


def _ablation_row(ds: str, setting: dict, variant: str, v: dict) -> dict:
    """One table row: the variant's scores and, unless it is the reference, its paired difference to the reference."""
    diff, mc = v.get("minus_reference"), v.get("mcnemar_accuracy")
    excl = (f"{'yes' if diff['accuracy']['ci_excludes_zero'] else 'no'} / "
            f"{'yes' if diff['macro_f1']['ci_excludes_zero'] else 'no'}") if diff else "-"
    label = variant + (" (TF-IDF still lowercases)" if variant == "all_off" else "")
    return {"dataset": ds, "setting": setting["label"], "variant": label + (" (reference)" if not diff else ""),
            "rule fires, docs train / test": _fires_text(v.get("rule_fires")) if diff else "-",
            "TF-IDF features": v["n_features"],
            "acc / F1 %": f"{_pct(v['accuracy']['value'])} / {_pct(v['macro_f1']['value'])}",
            "acc minus full, pts [95% CI]": _ci_text(diff["accuracy"]["diff"], diff["accuracy"]["ci"]) if diff else "-",
            "F1 minus full, pts [95% CI]": _ci_text(diff["macro_f1"]["diff"], diff["macro_f1"]["ci"]) if diff else "-",
            "CI excludes 0 (acc / F1)": excl,
            "predictions changed": v["n_predictions_changed"] if diff else "-",
            "McNemar p (acc)": p_value_text(mc["p_value"]).lstrip("= ") if mc else "-"}


def table_ablation(results_dir) -> str:
    """Per-rule ablation: every LexFront+LR variant against the full LexFront of the same setting."""
    abl = load_ablation(results_dir)
    rows = [_ablation_row(ds, setting, variant, v)
            for ds, settings in abl["datasets"].items() for setting in settings.values()
            for variant, v in setting["variants"].items()]
    return to_markdown(pd.DataFrame(rows))


BPE_SUBDIR = "bpe"
BPE_SENSITIVITY_SUBDIR = "bpe_v32000"          # same code, vocabulary 32000 (python -m experiments.run_bpe --vocab-size)


def load_bpe(results_dir, subdir: str = BPE_SUBDIR) -> dict:
    """Statistics of the subword grid (python -m experiments.run_bpe): runs, pairs and BPE configs per dataset."""
    return load_bootstrap(Path(results_dir) / subdir)


def _score_text(run: dict | None) -> str:
    return f"{_pct(run['accuracy']['value'])} / {_pct(run['macro_f1']['value'])}" if run else "-"


def table_bpe_runs(results_dir, subdir: str = BPE_SUBDIR) -> str:
    """Accuracy / macro-F1 (%) of BPE and LexFront + BPE with NB and LR, next to NLTK+LR and LexFront+LR."""
    boot = load_bpe(results_dir, subdir)
    rows = []
    for ds, d in boot["datasets"].items():
        bpe, hybrid = (k.removesuffix("__lr") for k in (d["pairs"]["hybrid_vs_bpe_lr"]["a"],
                                                         d["pairs"]["hybrid_vs_bpe_lr"]["b"]))
        rows.append({"dataset": ds, "BPE vocab": d["subword"][bpe]["vocab_size"],
                     "BPE+NB acc / F1": _score_text(d["runs"].get(f"{bpe}__nb")),
                     "BPE+LR acc / F1": _score_text(d["runs"].get(f"{bpe}__lr")),
                     "LexFront+BPE+NB acc / F1": _score_text(d["runs"].get(f"{hybrid}__nb")),
                     "LexFront+BPE+LR acc / F1": _score_text(d["runs"].get(f"{hybrid}__lr")),
                     "NLTK+LR acc / F1": _score_text(d["runs"].get("nltk__lr")),
                     "LexFront+LR acc / F1": _score_text(d["runs"].get("lexfront__lr"))})
    return to_markdown(pd.DataFrame(rows))


def table_bpe(results_dir, subdir: str = BPE_SUBDIR) -> str:
    """Paired bootstrap 95% CIs (points) of the four subword differences, McNemar p for BPE+LR vs NLTK+LR."""
    boot = load_bpe(results_dir, subdir)
    rows = []
    for ds, d in boot["datasets"].items():
        for p in d["pairs"].values():
            mc = p.get("mcnemar_accuracy")
            rows.append({"dataset": ds, "difference": p["label"],
                         "accuracy pts [95% CI]": _ci_text(p["accuracy"]["diff"], p["accuracy"]["ci"]),
                         "macro-F1 pts [95% CI]": _ci_text(p["macro_f1"]["diff"], p["macro_f1"]["ci"]),
                         "CI excludes 0 (acc / F1)": (f"{'yes' if p['accuracy']['ci_excludes_zero'] else 'no'} / "
                                                      f"{'yes' if p['macro_f1']['ci_excludes_zero'] else 'no'}"),
                         "McNemar p (acc)": p_value_text(mc["p_value"]).lstrip("= ") if mc else "-"})
    return to_markdown(pd.DataFrame(rows))


TABLES = {"datasets": table_datasets, "module1_results": table_module1,
          "existing_vs_proposed": table_existing_vs_proposed, "intrinsic_metrics": table_intrinsic,
          "dfa_states": table_dfa, "ci_table": table_ci, "ci_runs": table_ci_runs,
          "sensitivity_case": table_sensitivity_case, "sensitivity_ci": table_sensitivity_ci,
          "ablation_table": table_ablation, "bpe_runs": table_bpe_runs, "bpe_table": table_bpe,
          "bpe_v32000_runs": lambda r: table_bpe_runs(r, BPE_SENSITIVITY_SUBDIR),
          "bpe_v32000_table": lambda r: table_bpe(r, BPE_SENSITIVITY_SUBDIR)}


def render_all_tables(results_dir) -> dict[str, str]:
    """Build every table in memory (nothing is written); {name: markdown}.

    Only a missing input file is turned into a note; any other error is a bug and is raised.
    """
    tables = {}
    for name, fn in TABLES.items():
        try:
            tables[name] = fn(results_dir)
        except FileNotFoundError as err:   # input JSON not produced yet
            tables[name] = f"_not available: {err!r}_"
    return tables


def build_all_tables(results_dir, out_dir=None) -> dict[str, str]:
    """Build every table; write `<out_dir>/<name>.md` for those whose inputs exist; return {name: markdown}."""
    out_dir = Path(out_dir or Path(results_dir) / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = render_all_tables(results_dir)
    for name, text in tables.items():
        if not text.startswith("_not available"):
            (out_dir / f"{name}.md").write_text(text + "\n")
    return tables
