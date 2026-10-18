"""Phase A command line: run (resumable), statistics, generated report.

    python -m experiments.run_boost [--dataset sentiment140] [--out results/boost_sentiment140] [--work ../lexfront_boost_work]
                                    [--quick] [--tokenizers ...]
    python -m experiments.run_boost --stats-only --dataset sentiment140 --out results/boost_sentiment140

Protocol and hypotheses: docs/BOOST_PROTOCOL.md. `--stats-only` rebuilds stats.json and REPORT.md from the committed files only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import boost, large, stats

N_BOOT, SEED, MARGIN = 2000, 20261017, 0.005
REFERENCE = {"sentiment140": "results/large"}       # committed full-data predictions of the same split, when they exist
LOADERS = {"sentiment140": lambda quick, seed: _sentiment140(quick, seed)}
ALL = (*boost.WORD_TOKENIZERS, boost.CHAR)
STAGE_LABEL = {"ref_default": "committed default", "anchor": "default (re-run)", "ref_tuned": "committed tuned", "wide": "wide-tuned"}
CLASSIFIER_ORDER = ("lr", "nb")
PRIMARY_METRIC = {"civil_comments": "macro_f1"}        # everything else: accuracy (docs/D_PROTOCOL.md says why)


def _sentiment140(quick: bool, seed: int):
    ds, report = large.build_dataset(large.load_raw(), 30_000 if quick else None, 4_000 if quick else None, seed)
    return ds, report


def verdict(ci, margin: float = MARGIN) -> str:
    """Where a difference CI lies relative to the practical margin."""
    if ci[0] > margin:
        return "beyond (up)"
    if ci[1] < -margin:
        return "beyond (down)"
    return "within" if ci[0] > -margin and ci[1] < margin else "unclear"


def load_units(out_dir, ref_dir=None):
    """({(tokenizer, stage, classifier): predictions}, y_true, runs); references are keyed stage ref_default / ref_tuned."""
    out = Path(out_dir)
    runs = json.loads((out / "runs.json").read_text())["runs"]
    preds, y_true = {}, None
    for r in runs:
        y, by_clf = large.load_predictions(out / "predictions" / f"{r['run_id']}.json.gz")
        y_true = y if y_true is None else y_true
        if not np.array_equal(y, y_true):
            raise ValueError("test labels differ between units")
        for clf, by_level in by_clf.items():
            preds[(r["tokenizer"], r["stage"], clf)] = by_level["clean"]
    if ref_dir and Path(ref_dir, "predictions").exists():
        for tok in boost.WORD_TOKENIZERS:
            for setting in ("default", "tuned"):
                y, by_clf = large.load_predictions(Path(ref_dir) / "predictions" / f"{tok}__{setting}__full__r0.json.gz")
                if not np.array_equal(y, y_true):
                    raise ValueError("reference test labels differ from this run")
                for clf, by_level in by_clf.items():
                    preds[(tok, f"ref_{setting}", clf)] = by_level["clean"]
    return preds, y_true, runs


def bootstrap(y_true, preds: dict, n_boot: int, seed: int, n_classes: int, chunk: int = 250) -> dict:
    rng, out, done = np.random.default_rng(seed), {k: {m: [] for m in stats.METRICS} for k in preds}, 0
    while done < n_boot:
        idx = stats.bootstrap_indices(len(y_true), min(chunk, n_boot - done), rng)
        for k, y_pred in preds.items():
            for m, v in stats.metric_samples(y_true, y_pred, idx, n_classes).items():
                out[k][m].append(v)
        done += len(idx)
    return {k: {m: np.concatenate(v) for m, v in by.items()} for k, by in out.items()}


def pair_definitions(have) -> list:
    """(name, classifier, key_b, key_a): the comparison is key_b minus key_a. Only pairs whose two sides exist."""
    defs = []
    for clf in CLASSIFIER_ORDER:
        for stage in ("anchor", "wide"):
            for other in ("nltk", "tweet", "whitespace"):
                defs.append((f"{stage}: lexfront minus {other}", clf, ("lexfront", stage, clf), (other, stage, clf)))
        for tok in boost.WORD_TOKENIZERS:
            defs.append((f"gain of wide over committed tuned: {tok}", clf, (tok, "wide", clf), (tok, "ref_tuned", clf)))
            defs.append((f"gain of wide over default: {tok}", clf, (tok, "wide", clf), (tok, "anchor", clf)))
        for tok in boost.WORD_TOKENIZERS:
            defs.append((f"wide: char minus {tok}", clf, (boost.CHAR, "wide", clf), (tok, "wide", clf)))
    return [d for d in defs if d[2] in have and d[3] in have]


def hypotheses(s: dict) -> dict:
    pairs = {(p["name"], p["classifier"]): p for p in s["pairs"]}
    metric = s["config"]["primary_metric"]
    pick = lambda name, clf="lr": pairs.get((name, clf))                       # noqa: E731
    out = {}
    gains = [pick(f"gain of wide over committed tuned: {t}") for t in boost.WORD_TOKENIZERS]
    if all(gains):
        v = [g[metric]["verdict"] for g in gains]
        out["HA1"] = {"rule": "wide-tuned minus committed tuned, LR, each tokenizer 'beyond (up)'", "verdicts": dict(zip(boost.WORD_TOKENIZERS, v)),
                      "observed": all(x == "beyond (up)" for x in v), "opposite": any(x == "beyond (down)" for x in v)}
    a, b = pick("wide: lexfront minus nltk"), pick("wide: lexfront minus tweet")
    if a and b:
        v = [a[metric]["verdict"], b[metric]["verdict"]]
        out["HA2"] = {"rule": "wide-tuned LR, lexfront minus nltk and minus tweet both 'within'", "verdicts": {"nltk": v[0], "tweet": v[1]},
                      "observed": v == ["within", "within"], "opposite": any(x.startswith("beyond") for x in v)}
    acc = lambda stage: [s["accuracy"][f"{t}|{stage}|lr"][metric] for t in boost.WORD_TOKENIZERS if f"{t}|{stage}|lr" in s["accuracy"]]  # noqa: E731
    wide, anchor = acc("wide"), acc("anchor")
    if len(wide) == 4 and len(anchor) == 4:
        out["HA3"] = {"rule": "spread (max - min over the 4 tokenizers, LR) smaller wide-tuned than default", "wide_spread": max(wide) - min(wide),
                      "default_spread": max(anchor) - min(anchor), "observed": max(wide) - min(wide) < max(anchor) - min(anchor)}
    c = pick("wide: char minus lexfront")
    if c:
        v = c[metric]["verdict"]
        out["HA4"] = {"rule": "char minus wide-tuned lexfront (LR) is 'within' or 'beyond (up)'", "verdict": v,
                      "observed": v in ("within", "beyond (up)"), "opposite": v == "beyond (down)"}
    g = pick("gain of wide over committed tuned: lexfront", "nb")
    if g:
        v = g[metric]["verdict"]
        out["HA5"] = {"rule": "wide-tuned NB lexfront minus committed tuned NB lexfront is 'beyond (up)'", "verdict": v,
                      "observed": v == "beyond (up)", "opposite": v == "beyond (down)"}
    return out


def compute(out_dir, ref_dir=None, n_boot: int = N_BOOT, seed: int = SEED, primary: str = "accuracy") -> dict:
    preds, y_true, runs = load_units(out_dir, ref_dir)
    n_classes = int(max(y_true.max(), max(int(p.max()) for p in preds.values()))) + 1
    names = {k: "|".join(k) for k in preds}
    samples = bootstrap(y_true, {names[k]: p for k, p in preds.items()}, n_boot, seed, n_classes)
    point = {names[k]: stats.point_estimates(y_true, p, n_classes) for k, p in preds.items()}
    accuracy = {n: {**{m: point[n][m] for m in stats.METRICS}, "value": point[n]["accuracy"],
                    "ci": stats.percentile_ci(samples[n]["accuracy"])} for n in point}
    pairs = []
    for name, clf, kb, ka in pair_definitions(set(preds)):
        row = {"name": name, "classifier": clf}
        for m in ("accuracy", "macro_f1"):
            ci = stats.percentile_ci(samples[names[kb]][m] - samples[names[ka]][m])
            row[m] = {"diff": stats._clean(point[names[kb]][m] - point[names[ka]][m]), "ci": ci, "ci_excludes_zero": stats.excludes_zero(ci),
                      "verdict": verdict(ci)}
        pairs.append(row)
    counts = np.bincount(y_true, minlength=n_classes)
    anchor_matches = {f"{t}|{c}": bool(np.array_equal(preds[(t, "anchor", c)], preds[(t, "ref_default", c)]))
                      for t in boost.WORD_TOKENIZERS for c in CLASSIFIER_ORDER if (t, "anchor", c) in preds and (t, "ref_default", c) in preds}
    anchor_diff = {k: int((preds[(k.split("|")[0], "anchor", k.split("|")[1])] != preds[(k.split("|")[0], "ref_default", k.split("|")[1])]).sum())
                   for k in anchor_matches}
    s = {"config": {"method": "percentile bootstrap over test examples, paired, chunked", "n_boot": n_boot, "seed": seed, "margin": MARGIN,
                    "n_classes": n_classes, "primary_metric": primary, "not_covered": "training randomness; selection noise on the validation slice; one run per cell"},
         "n_test": int(len(y_true)), "test_class_counts": counts.tolist(), "majority_accuracy": float(counts.max() / counts.sum()),
         "accuracy": accuracy, "pairs": pairs, "anchor_matches_committed": anchor_matches, "anchor_n_different_predictions": anchor_diff}
    s["hypotheses"] = hypotheses(s)
    return s


# ---- report ---------------------------------------------------------------------------------------------------------
def _pts(x) -> str:
    return f"{100 * x:.2f}"


def _sgn(x) -> str:
    return f"{100 * x:+.2f}"


def render_report(out_dir, title: str) -> str:
    out = Path(out_dir)
    s = json.loads((out / "stats.json").read_text())
    runs = json.loads((out / "runs.json").read_text())["runs"]
    split = json.loads((out / "split.json").read_text()) if (out / "split.json").exists() else None
    L = [f"# Phase A: wider settings and character n-grams, {title}", "",
         "Generated by `python -m experiments.run_boost` from the JSON next to this file; do not edit. Protocol and hypotheses were fixed before the run "
         "in `docs/BOOST_PROTOCOL.md`. Accuracy in percent on the clean test set; `[a, b]` is the 95% bootstrap CI. Verdicts compare a difference CI with the "
         f"practical margin of +-{100 * MARGIN:.1f} points (`within`, `beyond (up/down)`, `unclear`). Primary metric for the hypotheses: {s['config']['primary_metric'].replace('_', '-')}. One training run per cell; no multiplicity correction; "
         "reasons for any difference are hypotheses, not findings.", "", "## 1. Data", ""]
    if split:
        c, sp = split["cleaning"], split["split"]
        L.append(f"{sp.get('hf_id')} @ `{sp.get('repo_sha')}`; after cleaning train {c['n_after']['train']}, test {c['n_after']['test']}; "
                 f"users in both splits: {sp.get('users_in_both_splits', 'n/a')}.")
    L.append(f"Test examples {s['n_test']}; class counts {s['test_class_counts']}; majority-class accuracy {_pts(s['majority_accuracy'])}.")
    L.append("")
    n = 2
    for clf in CLASSIFIER_ORDER:
        L += [f"## {n}. Accuracy, {clf.upper()}", "", "| tokenizer | " + " | ".join(STAGE_LABEL.values()) + " | wide-tuned macro-F1 |",
              "|---|" + "---|" * (len(STAGE_LABEL) + 1)]
        for t in ALL:
            cells = []
            for stage in STAGE_LABEL:
                a = s["accuracy"].get(f"{t}|{stage}|{clf}")
                cells.append("-" if a is None else f"{_pts(a['value'])} [{_pts(a['ci'][0])}, {_pts(a['ci'][1])}]")
            w = s["accuracy"].get(f"{t}|wide|{clf}")
            L.append(f"| {t} | " + " | ".join(cells) + f" | {'-' if w is None else _pts(w['macro_f1'])} |")
        L.append("")
        n += 1
    L += [f"## {n}. Selected settings (chosen on the validation slice of train only)", "",
          "| tokenizer | classifier | vectorizer | C or alpha | validation accuracy | features | peak RSS GB |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        if r["stage"] != "wide":
            continue
        for clf, rec in r["classifiers"].items():
            sel = rec["tuning"]["selected"]
            L.append(f"| {r['tokenizer']} | {clf} | {sel['vectorizer']} | {sel['value']} | {_pts(sel['val_accuracy'])} | "
                     f"{rec['vectorizer']['n_features']} | {r['peak_rss_gb']:.1f} |")
    L += ["", f"## {n + 1}. Paired accuracy differences (points [CI], verdict)", "", "| comparison | classifier | accuracy | macro-F1 |", "|---|---|---|---|"]
    for p in s["pairs"]:
        cells = [f"{_sgn(p[m]['diff'])} [{_sgn(p[m]['ci'][0])}, {_sgn(p[m]['ci'][1])}] {p[m]['verdict']}" for m in ("accuracy", "macro_f1")]
        L.append(f"| {p['name']} | {p['classifier']} | " + " | ".join(cells) + " |")
    if s["anchor_matches_committed"]:
        L += ["", f"Anchor: the re-run default fits reproduce the committed predictions exactly for {sum(s['anchor_matches_committed'].values())} of "
              f"{len(s['anchor_matches_committed'])} tokenizer/classifier pairs."]
        odd = {k: v for k, v in s["anchor_n_different_predictions"].items() if v}
        if odd:
            L += ["", "Test predictions that differ from the committed ones (of " + str(s["n_test"]) + "): "
                  + ", ".join(f"{k} {v}" for k, v in odd.items()) + "."]
    L += ["", f"## {n + 2}. Hypotheses of the protocol", ""]
    for name, h in s["hypotheses"].items():
        verdict_text = "observed" if h["observed"] else ("opposite direction observed" if h.get("opposite") else "not observed")
        detail = {"HA1": lambda: ", ".join(f"{k} {v}" for k, v in h["verdicts"].items()),
                  "HA2": lambda: ", ".join(f"vs {k}: {v}" for k, v in h["verdicts"].items()),
                  "HA3": lambda: f"spread {_pts(h['wide_spread'])} (wide-tuned) vs {_pts(h['default_spread'])} (default)",
                  "HA4": lambda: f"char minus lexfront: {h['verdict']}", "HA5": lambda: f"NB lexfront: {h['verdict']}"}[name]()
        L.append(f"* **{name}** ({h['rule']}): {verdict_text}; {detail}.")
    L += ["", "## Limits", "", "One dataset, one language, one training run per cell; the bootstrap covers test-set sampling only; validation-slice selection "
          "noise is not covered. No noise test, no other classifier families, no transformer. Differences say nothing about why they occur."]
    return "\n".join(L) + "\n"


def write_outputs(out_dir, ref_dir=None, title="Sentiment140", n_boot: int = N_BOOT, seed: int = SEED, primary: str = "accuracy") -> list[Path]:
    out = Path(out_dir)
    (out / "stats.json").write_text(json.dumps(compute(out, ref_dir, n_boot, seed, primary), indent=1) + "\n")
    (out / "REPORT.md").write_text(render_report(out, title))
    return [out / "stats.json", out / "REPORT.md"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="sentiment140", choices=list(LOADERS))
    p.add_argument("--out", default=None)
    p.add_argument("--work", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tokenizers", nargs="+", default=list(ALL), choices=list(ALL))
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stats-only", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out or (f"results/quick/boost_{a.dataset}" if a.quick else f"results/boost_{a.dataset}"))
    work = Path(a.work or f"../lexfront_boost_work_{a.dataset}" + ("_quick" if a.quick else ""))
    ref = None if a.quick else REFERENCE.get(a.dataset)
    if not a.stats_only:
        ds, report = LOADERS[a.dataset](a.quick, a.seed)
        out.mkdir(parents=True, exist_ok=True)
        (out / "split.json").write_text(json.dumps(report, indent=1) + "\n")
        print(f"train {len(ds.train_texts)}  test {len(ds.test_texts)}")
        print(f"units computed in this call: {boost.run_units(ds, a.tokenizers, work, a.seed, a.force)}")
        print("wrote", large.consolidate(work, out))
    for path in write_outputs(out, ref, a.dataset, a.n_boot or (200 if a.quick else N_BOOT), primary=PRIMARY_METRIC.get(a.dataset, "accuracy")):
        print("wrote", path)


if __name__ == "__main__":
    main()
