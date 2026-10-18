"""Self-check for a results folder written by `run_boost` (phase A and later datasets). Exit 1 on any FAIL.

    python -m experiments.verify_boost --out results/boost_sentiment140 [--ref results/large] [--protocol docs/BOOST_PROTOCOL.md]
                                       [--base origin/main] [--texts docs/FILE.md[::heading prefix] ...]

(a) metrics reproduce from stored predictions; (b) every number in the given texts appears in REPORT.md; (c) the protocol commit precedes the
results commit; (d) compare_results --additive shows 0 differences against --base; (e) one test set for every unit; (f) no user overlap, leakage
rule recorded; (g) every accuracy above the majority baseline; (h) anchors reproduce the committed predictions exactly; (i) selection used
validation scores of train only and the stored winner is the argmax.
"""
import argparse
import inspect
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import boost, large, run_boost
from .metrics import classification_metrics

NUMBER = re.compile(r"(?<![\w.])\d+\.\d{2}(?![\w.])")


def check_metrics(runs, out) -> list[str]:
    bad = []
    for r in runs:
        y, by_clf = large.load_predictions(out / "predictions" / f"{r['run_id']}.json.gz")
        for clf, rec in r["classifiers"].items():
            m = classification_metrics(y, by_clf[clf]["clean"])
            stored = rec["metrics"]["clean"]
            if abs(m["accuracy"] - stored["accuracy"]) > 1e-12 or abs(m["macro_f1"] - stored["macro_f1"]) > 1e-12 \
                    or int((by_clf[clf]["clean"] == y).sum()) != stored["n_correct"]:
                bad.append(f"{r['run_id']}/{clf}")
    return bad


def numbers_in(texts: list[str], report: str) -> list[str]:
    known = set(NUMBER.findall(report))
    missing = []
    for spec in texts:
        path, _, heading = spec.partition("::")
        body = Path(path).read_text()
        if heading:
            start = body.index(heading)
            end = body.find("\n## ", start + 1)
            body = body[start:end if end > 0 else None]
        missing += [f"{path}: {n}" for n in NUMBER.findall(body) if n not in known]
    return missing


def first_commit(path: str) -> str | None:
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--reverse", "--", path], capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--ref")
    p.add_argument("--protocol")
    p.add_argument("--base")
    p.add_argument("--texts", nargs="*", default=[])
    a = p.parse_args(argv)
    out = Path(a.out)
    s = json.loads((out / "stats.json").read_text())
    report = (out / "REPORT.md").read_text()
    preds, y_true, runs = run_boost.load_units(out, a.ref)
    results = {}
    bad = check_metrics(runs, out)
    results["a metrics reproduce from stored predictions"] = (not bad, f"{len(runs)} units" if not bad else f"mismatch {bad}")
    missing = numbers_in(a.texts, report) if a.texts else []
    results["b every number in the texts is in REPORT.md"] = (not missing, f"checked {len(a.texts)} text(s)" if not missing else f"missing {missing[:8]}")
    if a.protocol:
        proto, res = first_commit(a.protocol), first_commit(str(out / "REPORT.md"))
        ok = proto is not None and (res is None or subprocess.run(["git", "merge-base", "--is-ancestor", proto, res]).returncode == 0
                                    and proto != res)
        results["c protocol commit precedes the results commit"] = (ok, f"protocol {str(proto)[:8]} results {str(res)[:8] if res else 'not committed yet'}")
    if a.base:
        with tempfile.TemporaryDirectory() as tmp:
            tar = subprocess.run(["git", "archive", a.base, "results"], capture_output=True)
            subprocess.run(["tar", "-x", "-C", tmp], input=tar.stdout)
            done = subprocess.run([sys.executable, "-m", "experiments.compare_results", "--additive", f"{tmp}/results", "results"], capture_output=True, text=True)
        results["d compare_results --additive: 0 differences"] = (done.returncode == 0, done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr[-200:])
    n_test = {len(y) for y in [y_true]} | {r["n_test"] for r in runs}
    same = len(n_test) == 1 and s["n_test"] == next(iter(n_test)) and all(len(v) == len(y_true) for v in preds.values())
    results["e one test set (size and labels) for every unit"] = (same, f"n_test {sorted(n_test)}, {len(preds)} prediction sets")
    split = json.loads((out / "split.json").read_text())
    users = split["split"].get("users_in_both_splits")
    leak_rule = "test_in_train" in split["cleaning"]["removed"]
    results["f no user overlap, duplicate/leakage rule applied"] = ((users in (0, None)) and leak_rule, f"users in both splits {users}; test_in_train removed {split['cleaning']['removed'].get('test_in_train')}")
    low = [k for k, v in s["accuracy"].items() if v["value"] <= s["majority_accuracy"]]
    results["g every accuracy above the majority baseline"] = (not low, f"majority {100 * s['majority_accuracy']:.2f}" + (f"; not above: {low}" if low else ""))
    if s["anchor_matches_committed"]:
        results["h anchors reproduce committed predictions exactly"] = (all(s["anchor_matches_committed"].values()), f"{sum(s['anchor_matches_committed'].values())}/{len(s['anchor_matches_committed'])}")
    else:
        results["h anchors reproduce committed predictions exactly"] = (a.ref is None, "no committed reference for this dataset" if a.ref is None else "NOT RUN")
    sel_bad = []
    for r in runs:
        for clf, rec in r["classifiers"].items():
            if r["stage"] != "wide":
                continue
            t = rec["tuning"]
            if t["selected"] != boost.pick_best(t["scores"]) or t["n_fit"] + t["n_val"] > r["n_train"] or any(set(x) != {"vectorizer", "value", "val_accuracy"} for x in t["scores"]):
                sel_bad.append(f"{r['run_id']}/{clf}")
    no_test_param = not any("test" in name for name in inspect.signature(boost.search).parameters)
    results["i selection on validation slice of train only"] = (not sel_bad and no_test_param, "winner = argmax of stored validation scores; search() takes no test data" if not sel_bad else str(sel_bad))
    for k, (ok, note) in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  ({k[0]}) {k[2:]}: {note}")
    return 0 if all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
