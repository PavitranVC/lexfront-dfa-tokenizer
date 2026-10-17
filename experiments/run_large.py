"""Large-scale Sentiment140 test: run (resumable), statistics, generated report, export.

    python -m experiments.run_large [--out results/large] [--work ../lexfront_large_work] [--quick] [--tokenizers ...]
    python -m experiments.run_large --stats-only     # statistics + REPORT.md from the committed files, no data, no fitting

Protocol and hypotheses: docs/LARGE_PROTOCOL.md. `--work` holds one JSON per fit set so a dead Colab session resumes; `--out`
holds the consolidated results that get committed (runs.json, predictions of the full-train runs, stats.json, REPORT.md).
"""
import argparse
import json
import os
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import large, stats

N_BOOT = 2000
SEED = 20261017
MARGIN = 0.005                      # practical margin, +-0.5 accuracy points, fixed in docs/LARGE_PROTOCOL.md
PRIMARY = ("default", "lr", "clean")  # setting, classifier, noise level the hypotheses refer to
SETTING_ORDER = ("default", "tuned")
CLASSIFIER_ORDER = ("lr", "nb")        # Logistic Regression first: it is the primary classifier


def size_key(label: str) -> float:
    """Sort key of a training-size label: numeric sizes ascending, 'full' last."""
    return int(label) if label.isdigit() else float("inf")


def margin_verdict(ci, margin: float = MARGIN) -> str:
    """'within' the practical margin, 'beyond' it (whole CI outside), or 'unclear'."""
    if ci[0] > -margin and ci[1] < margin:
        return "within"
    if ci[1] < -margin or ci[0] > margin:
        return "beyond"
    return "unclear"


# ---- statistics -------------------------------------------------------------------------------------------------------
def bootstrap_samples(y_true, preds: dict, n_boot: int, seed: int, chunk: int = 250) -> dict:
    """{name: {metric: (n_boot,) samples}} on shared resamples, drawn chunk by chunk so memory stays at chunk x n."""
    rng = np.random.default_rng(seed)
    n, out, done = len(y_true), {name: {m: [] for m in stats.METRICS} for name in preds}, 0
    while done < n_boot:
        idx = stats.bootstrap_indices(n, min(chunk, n_boot - done), rng)
        for name, y_pred in preds.items():
            for metric, values in stats.metric_samples(y_true, y_pred, idx, 2).items():
                out[name][metric].append(values)
        done += len(idx)
    return {name: {m: np.concatenate(v) for m, v in by.items()} for name, by in out.items()}


def _diff(point_b, point_a, samples_b, samples_a) -> dict:
    ci = stats.percentile_ci(samples_b - samples_a)
    return {"diff": stats._clean(point_b - point_a), "ci": ci, "ci_excludes_zero": stats.excludes_zero(ci),
            "margin": margin_verdict(ci)}


def full_data_stats(out_dir, runs, n_boot: int, seed: int) -> dict:
    """Bootstrap CIs and paired differences of the full-train runs (the only ones with stored predictions)."""
    result = {}
    full = {(r["tokenizer"], r["setting"]): r for r in runs if r["size"] == "full"}
    for setting in sorted({s for _, s in full}):
        for clf in large.CLASSIFIERS:
            names = [t for t in large.TOKENIZERS if (t, setting) in full]
            if not names:
                continue
            loaded = {t: large.load_predictions(Path(out_dir) / "predictions" / f"{full[(t, setting)]['run_id']}.json.gz")
                      for t in names}
            y_true = loaded[names[0]][0]
            preds = {f"{t}|{lv}": loaded[t][1][clf][lv] for t in names for lv in large.NOISE_LEVELS}
            samples = bootstrap_samples(y_true, preds, n_boot, seed)
            point = {k: stats.point_estimates(y_true, p, 2) for k, p in preds.items()}
            block = {"n_test": int(len(y_true)), "runs": {}, "pairs": {}, "noise_drop": {}}
            for t in names:
                block["runs"][t] = {lv: {m: {"value": point[f"{t}|{lv}"][m], "ci": stats.percentile_ci(samples[f"{t}|{lv}"][m])}
                                         for m in stats.METRICS} for lv in large.NOISE_LEVELS}
            for other in ("nltk", "tweet"):
                if "lexfront" in names and other in names:
                    block["pairs"][f"lexfront_minus_{other}"] = {
                        lv: {m: _diff(point[f"lexfront|{lv}"][m], point[f"{other}|{lv}"][m],
                                      samples[f"lexfront|{lv}"][m], samples[f"{other}|{lv}"][m]) for m in stats.METRICS}
                        for lv in large.NOISE_LEVELS}
                    # (LexFront's change clean -> level) minus (the other's change): positive = LexFront loses less accuracy
                    block["noise_drop"][f"lexfront_vs_{other}"] = {
                        lv: _diff(point[f"lexfront|{lv}"]["accuracy"] - point[f"lexfront|clean"]["accuracy"],
                                  point[f"{other}|{lv}"]["accuracy"] - point[f"{other}|clean"]["accuracy"],
                                  samples[f"lexfront|{lv}"]["accuracy"] - samples["lexfront|clean"]["accuracy"],
                                  samples[f"{other}|{lv}"]["accuracy"] - samples[f"{other}|clean"]["accuracy"])
                        for lv in large.NOISE_LEVELS if lv != "clean"}
            result.setdefault(setting, {})[clf] = block
    return result


def ladder_stats(runs) -> dict:
    """Mean / SD of accuracy over the random subsets of each size, and the per-subset LexFront-minus-other gap."""
    labels = sorted({r["size"] for r in runs}, key=size_key)
    by = {}
    for r in runs:
        by[(r["tokenizer"], r["setting"], r["size"], r["repeat"])] = r
    ladder, gaps = [], []
    keys = sorted({(r["setting"], r["size"]) for r in runs}, key=lambda k: (SETTING_ORDER.index(k[0]), size_key(k[1])))
    for setting, size in keys:
        reps = sorted({r["repeat"] for r in runs if r["setting"] == setting and r["size"] == size})
        for clf in large.CLASSIFIERS:
            for lv in large.NOISE_LEVELS:
                acc = {t: [by[(t, setting, size, k)]["classifiers"][clf]["metrics"][lv]["accuracy"] for k in reps
                           if (t, setting, size, k) in by] for t in large.TOKENIZERS}
                for t, values in acc.items():
                    if values:
                        ladder.append({"setting": setting, "classifier": clf, "level": lv, "size": size, "tokenizer": t,
                                       "n": len(values), "mean": float(np.mean(values)),
                                       "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
                for other in ("nltk", "tweet"):
                    pairs = [a - b for a, b in zip(acc["lexfront"], acc[other])] if acc["lexfront"] and acc[other] else []
                    if pairs:
                        gaps.append({"setting": setting, "classifier": clf, "level": lv, "size": size,
                                     "versus": other, "n": len(pairs), "mean": float(np.mean(pairs)),
                                     "sd": float(np.std(pairs, ddof=1)) if len(pairs) > 1 else None,
                                     "min": float(min(pairs)), "max": float(max(pairs))})
    return {"ladder": ladder, "gaps": gaps, "sizes": labels}


def hypotheses(summary: dict) -> dict:
    """H1-H3 of docs/LARGE_PROTOCOL.md evaluated with the rules fixed there; each entry says what was compared."""
    out = {}
    setting, clf, level = PRIMARY
    gaps = [g for g in summary["gaps"] if (g["setting"], g["classifier"], g["level"], g["versus"]) == (setting, clf, level, "nltk")]
    gaps.sort(key=lambda g: size_key(g["size"]))
    if len(gaps) >= 2:
        first, last = gaps[0], gaps[-1]
        out["H1"] = {"rule": "|mean gap LexFront-NLTK| (LR, default, clean) at the largest size < at the smallest size",
                     "smallest": {"size": first["size"], "gap": first["mean"]}, "largest": {"size": last["size"], "gap": last["mean"]},
                     "observed": abs(last["mean"]) < abs(first["mean"])}
    for setting_name, by_clf in summary["full"].items():
        block = by_clf.get("lr", {})
        drop = block.get("noise_drop", {}).get("lexfront_vs_nltk", {}).get("heavy")
        if setting_name == "default" and drop:
            out["H2"] = {"rule": "(LexFront heavy - clean accuracy) - (NLTK heavy - clean), LR, default, full data: CI excludes 0 and is positive",
                         **drop, "observed": drop["ci_excludes_zero"] and drop["diff"] > 0,
                         "opposite": drop["ci_excludes_zero"] and drop["diff"] < 0}
    d, t = (summary["full"].get(s, {}).get("lr", {}).get("pairs", {}).get("lexfront_minus_nltk", {}).get("clean", {}).get("accuracy")
            for s in ("default", "tuned"))
    if d and t:
        out["H3"] = {"rule": "|LexFront-NLTK accuracy difference| (LR, clean, full data) smaller with tuned than with default settings",
                     "default": d["diff"], "tuned": t["diff"], "observed": abs(t["diff"]) < abs(d["diff"])}
    return out


def compute(out_dir, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    runs = json.loads((Path(out_dir) / "runs.json").read_text())["runs"]
    summary = {"config": {"method": "percentile bootstrap over test tweets, paired, chunked", "n_boot": n_boot, "seed": seed,
                          "margin": MARGIN, "noise_levels": large.NOISE_LEVELS, "pairs": ["lexfront_minus_nltk", "lexfront_minus_tweet"],
                          "not_covered": "training randomness beyond the random subsets; one training run per cell"},
               **ladder_stats(runs), "full": full_data_stats(out_dir, runs, n_boot, seed)}
    summary["hypotheses"] = hypotheses(summary)
    return summary


# ---- report -----------------------------------------------------------------------------------------------------------
def _pts(x) -> str:
    return f"{100 * x:.2f}"


def _sgn(x) -> str:
    return f"{100 * x:+.2f}"


def _ci(d: dict) -> str:
    return f"{_sgn(d['diff'])} [{_sgn(d['ci'][0])}, {_sgn(d['ci'][1])}] {d['margin']}"


def render_report(out_dir) -> str:
    out = Path(out_dir)
    s = json.loads((out / "stats.json").read_text())
    runs = json.loads((out / "runs.json").read_text())["runs"]
    split = json.loads((out / "split.json").read_text()) if (out / "split.json").exists() else None
    audit = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else None
    L = ["# Large-scale test: Sentiment140", "",
         "Generated by `python -m experiments.run_large` from the JSON next to this file; do not edit. Protocol and hypotheses were "
         "fixed before the run in `docs/LARGE_PROTOCOL.md`. Accuracy in percent. `within` / `beyond` / `unclear` = where the 95% "
         f"bootstrap CI of a difference lies relative to the practical margin of +-{100 * MARGIN:.1f} points (`within`: the whole CI is inside "
         "the margin; `beyond`: the whole CI is outside it on one side; `unclear`: the CI crosses a margin boundary, even if it excludes 0). One training run per "
         "cell; no multiplicity correction; reasons for any difference are hypotheses, not findings.", ""]
    if split:
        c, sp = split["cleaning"], split["split"]
        L += ["## 1. Data", "",
              f"{sp['hf_id']} @ `{sp['repo_sha']}`, {sp['n_rows_before_split']} rows. Split by user: {sp['n_test_users']} of {sp['n_users']} users "
              f"({100 * sp['test_user_fraction']:.0f}%) held out for test; users present in both splits: {sp['users_in_both_splits']}. "
              f"After cleaning (experiments/clean.py): train {c['n_before']['train']} > {c['n_after']['train']}, test "
              f"{c['n_before']['test']} > {c['n_after']['test']}; removed (train+test): empty {c['removed']['empty']['train']}+{c['removed']['empty']['test']}, "
              f"conflicting {c['removed']['conflicting']['train']}+{c['removed']['conflicting']['test']}, duplicate "
              f"{c['removed']['duplicate']['train']}+{c['removed']['duplicate']['test']}, test texts also in train {c['removed']['test_in_train']}.", ""]
    if audit:
        feats = audit["features"]
        L += ["Share of tweets containing each feature (simple regexes), training split:", "",
              "| class | n | mean chars | " + " | ".join(feats) + " |", "|---|---|---|" + "---|" * len(feats)]
        for key in ("all", "negative", "positive"):
            a = audit[key]
            L.append(f"| {key} | {a['n']} | {a['mean_chars']:.1f} | " + " | ".join(f"{100 * a[f]:.2f}%" for f in feats) + " |")
        L.append("")
    sizes = s["sizes"]
    settings = [x for x in SETTING_ORDER if any(r["setting"] == x for r in runs)]
    for number, setting in enumerate(settings, start=2):
        L += [f"## {number}. Size ladder, {setting} setting (clean test text; mean over random subsets, SD in brackets)", ""]
        for clf in CLASSIFIER_ORDER:
            L += [f"**{clf.upper()}**", "", "| train size | " + " | ".join(large.TOKENIZERS) + " | LexFront - NLTK (mean, SD, min..max) |",
                  "|---|" + "---|" * (len(large.TOKENIZERS) + 1)]
            for z in sizes:
                cells = []
                for t in large.TOKENIZERS:
                    e = next((x for x in s["ladder"] if (x["setting"], x["classifier"], x["level"], x["size"], x["tokenizer"]) == (setting, clf, "clean", z, t)), None)
                    cells.append("-" if e is None else _pts(e["mean"]) + ("" if e["sd"] is None else f" ({100 * e['sd']:.2f})"))
                g = next((x for x in s["gaps"] if (x["setting"], x["classifier"], x["level"], x["size"], x["versus"]) == (setting, clf, "clean", z, "nltk")), None)
                gap = "-" if g is None else f"{_sgn(g['mean'])}, " + ("n/a" if g["sd"] is None else f"{100 * g['sd']:.2f}") + f", {_sgn(g['min'])}..{_sgn(g['max'])}"
                L.append(f"| {z} | " + " | ".join(cells) + f" | {gap} |")
            L.append("")
    n = len(settings) + 2
    for setting in [x for x in SETTING_ORDER if x in s["full"]]:
        for clf in [c for c in CLASSIFIER_ORDER if c in s["full"][setting]]:
            b = s["full"][setting][clf]
            L += [f"## {n}. Full training set, {setting}, {clf.upper()} ({b['n_test']} test tweets)", "",
                  "| tokenizer | clean | light noise | heavy noise |", "|---|---|---|---|"]
            for t, byl in b["runs"].items():
                L.append(f"| {t} | " + " | ".join(f"{_pts(byl[lv]['accuracy']['value'])} [{_pts(byl[lv]['accuracy']['ci'][0])}, {_pts(byl[lv]['accuracy']['ci'][1])}]"
                                                   for lv in large.NOISE_LEVELS) + " |")
            L += ["", "Paired accuracy differences in points [95% CI] and margin verdict:", "",
                  "| comparison | clean | light noise | heavy noise |", "|---|---|---|---|"]
            for name, byl in b["pairs"].items():
                L.append(f"| {name.replace('_', ' ')} | " + " | ".join(_ci(byl[lv]["accuracy"]) for lv in large.NOISE_LEVELS) + " |")
            for name, byl in b["noise_drop"].items():
                L.append(f"| accuracy lost to noise, {name.replace('_', ' ')} (positive = LexFront loses less) | - | "
                         + " | ".join(_ci(byl[lv]) for lv in ("light", "heavy")) + " |")
            L.append("")
            n += 1
    L += [f"## {n}. Practical numbers", "",
          "| tokenizer | tokenization docs/s (test tweets) | train vectorize s (full, default) | LR fit s (full, default) | features | peak RSS GB (max of its runs) |",
          "|---|---|---|---|---|---|"]
    for t in large.TOKENIZERS:
        mine = [r for r in runs if r["tokenizer"] == t]
        full = next((r for r in mine if r["size"] == "full" and r["setting"] == "default"), None)
        if not mine:
            continue
        L.append(f"| {t} | {mine[0]['throughput_test_docs']['docs_per_s']:.0f} | "
                 + (f"{full['classifiers']['lr']['timing_seconds']['tokenize_and_vectorize_train']:.0f} | {full['classifiers']['lr']['timing_seconds']['classifier_fit']:.0f} | "
                    f"{full['classifiers']['lr']['vectorizer']['n_features']} | " if full else "- | - | - | ")
                 + f"{max(r['peak_rss_gb'] for r in mine):.2f} |")
    L += ["", f"## {n + 1}. Hypotheses of the protocol", ""]
    for name, h in s["hypotheses"].items():
        verdict = "observed" if h.get("observed") else ("opposite direction observed" if h.get("opposite") else "not observed")
        detail = {"H1": lambda: f"gap {_sgn(h['smallest']['gap'])} at {h['smallest']['size']} vs {_sgn(h['largest']['gap'])} at {h['largest']['size']}",
                  "H2": lambda: f"difference in accuracy lost {_ci(h)}",
                  "H3": lambda: f"gap {_sgn(h['default'])} (default) vs {_sgn(h['tuned'])} (tuned)"}[name]()
        L.append(f"* **{name}** ({h['rule']}): {verdict}; {detail}.")
    if not s["hypotheses"]:
        L.append("* Not enough runs to evaluate the hypotheses.")
    L += ["", "## Limits", "",
          "One dataset, one source (tweets), one language; labels derived from emoticons by the dataset's authors; synthetic noise; "
          "no transformers. Subsets of one size overlap, so their spread understates variation across independent training sets. "
          "The bootstrap covers test-set sampling only. Differences measured here say nothing about why they occur."]
    return "\n".join(L) + "\n"


def write_outputs(out_dir, n_boot: int = N_BOOT, seed: int = SEED) -> list[Path]:
    out = Path(out_dir)
    (out / "stats.json").write_text(json.dumps(compute(out, n_boot, seed), indent=1) + "\n")
    (out / "REPORT.md").write_text(render_report(out))
    return [out / "stats.json", out / "REPORT.md"]


# ---- export (Colab) -----------------------------------------------------------------------------------------------------
def make_zip(out_dir, zip_path) -> Path:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(out_dir).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(Path(out_dir).parent))
    return zip_path


def push_results(repo_dir, out_dir, branch: str, token: str, remote: str = "origin") -> str:
    """Commit `out_dir` to a NEW branch and push it (never main). The token reaches git only through GIT_ASKPASS, so it is not
    on a command line or in the remote URL; any git output is scrubbed of it before it is returned."""
    if branch in ("main", "master"):
        raise ValueError("refusing to push results to the default branch; use a new branch name")
    repo, rel = Path(repo_dir), os.path.relpath(out_dir, repo_dir)
    git = lambda *a, **kw: subprocess.run(["git", "-c", "user.name=colab", "-c", "user.email=colab@localhost", *a], cwd=repo,  # noqa: E731
                                          capture_output=True, text=True, **kw)
    with tempfile.TemporaryDirectory() as tmp:
        askpass = Path(tmp) / "askpass.sh"
        askpass.write_text('#!/bin/sh\ncase "$1" in *sername*) echo x-access-token;; *) echo "$LEXFRONT_TOKEN";; esac\n')
        askpass.chmod(askpass.stat().st_mode | stat.S_IEXEC)
        env = {**os.environ, "GIT_ASKPASS": str(askpass), "LEXFRONT_TOKEN": token, "GIT_TERMINAL_PROMPT": "0"}
        for args, extra in ((("checkout", "-q", "-B", branch), {}), (("add", "-f", rel), {}),
                            (("commit", "-q", "-m", f"Large-scale Sentiment140 results ({rel})"), {}),
                            (("push", "-u", remote, branch), {"env": env})):
            done = git(*args, **extra)
            if done.returncode != 0:
                message = (done.stdout + done.stderr).replace(token, "***") if token else done.stdout + done.stderr
                raise RuntimeError(f"git {args[0]} failed:\n{message}")
    return f"pushed branch {branch}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="consolidated results (default results/large, results/quick/large with --quick)")
    p.add_argument("--work", default=None, help="per-run files for resuming (default ../lexfront_large_work[_quick])")
    p.add_argument("--quick", action="store_true", help="small sizes and subsample, for a dry run (CI, first look)")
    p.add_argument("--tokenizers", nargs="+", default=list(large.TOKENIZERS), choices=list(large.TOKENIZERS))
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stats-only", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out or ("results/quick/large" if a.quick else "results/large"))
    work = Path(a.work or ("../lexfront_large_work_quick" if a.quick else "../lexfront_large_work"))
    n_boot = a.n_boot or (200 if a.quick else N_BOOT)
    if not a.stats_only:
        raw = large.load_raw()
        ds, report = large.build_dataset(raw, 30_000 if a.quick else None, 4_000 if a.quick else None, a.seed)
        out.mkdir(parents=True, exist_ok=True)
        (out / "split.json").write_text(json.dumps(report, indent=1) + "\n")
        (out / "audit.json").write_text(json.dumps(large.audit(ds.train_texts, ds.train_labels, ds.label_names), indent=1) + "\n")
        print(f"train {len(ds.train_texts)}  test {len(ds.test_texts)}")
        done = large.run_grid(ds, a.tokenizers, large.QUICK_LADDER if a.quick else large.LADDER, work, a.seed, a.force)
        print(f"fit sets computed in this call: {done}")
        print("wrote", large.consolidate(work, out))
    for path in write_outputs(out, n_boot):
        print("wrote", path)


if __name__ == "__main__":
    main()
