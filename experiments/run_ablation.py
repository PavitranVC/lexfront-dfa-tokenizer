"""Per-rule normalization ablation (P2): LexFront + LR with one normalization rule switched off at a time.

    python -m experiments.run_ablation [--out results/ablation] [--max-train N] [--max-test N] [--force]
    python -m experiments.run_ablation --stats-only [--out results/ablation]

Concept: an *ablation* removes one component and measures the change against the full system, with everything else
held fixed (lexer, TF-IDF settings, classifier, seed, data). Two settings (protocol: docs/versions/v4-ablation.md):

* main : TF-IDF `lowercase=True`, as the main grid. Variants full, no_url, no_mention, no_number, no_elongated,
         no_slang, all_off. No no_lowercase here: TF-IDF lowercases the text before LexFront sees it.
* case : TF-IDF `lowercase=False`, as the sensitivity grid. Variants full_cs, no_lowercase, all_off_cs.

Runs go to <out>/<setting>/ in the usual layout (run JSON + predictions/ sidecar). The statistics (same bootstrap code,
B and seed as `run_stats`) go to <out>/stats/bootstrap.json: every variant minus the full variant of its setting,
paired over the same resampled test indices, plus an exact McNemar test on accuracy. How often each removed rule
*fires* (train and test documents and lexer tokens whose normalized form differs from full LexFront) is counted once
per setting and dataset into <out>/<setting>/firing__<dataset>.json and copied into the stats: a variant whose rule
never fires is the full system itself, so its "comparison" is not a test of the rule.
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lexer.lexfront import LexFront
from lexer.normalize import RULES, normalize_token

from . import data, stats
from .classifiers import CASE_SENSITIVE_OVERRIDES, vectorizer_params
from .pipeline import code_fingerprint, load_predictions, result_path, run_cell
from .tokenizers import normalize_config
from .report import DATASET_ORDER, SENSITIVITY_SUBDIR
from .run_stats import N_BOOT, SEED, check_point_estimates, load_dataset_predictions

ABLATION_DIR = "results/ablation"
CLASSIFIER = "lr"


@dataclass
class Setting:
    """One ablation setting: TF-IDF overrides, the reference variant and {variant label: tokenizer name}."""
    name: str
    label: str
    vec_overrides: dict
    reference: str
    variants: dict = field(default_factory=dict)
    grid_subdir: str = ""      # where the matching full-LexFront run of the main or sensitivity grid lives


SETTINGS = (
    Setting("main", "TF-IDF lowercase (main grid)", {}, "full",
            {"full": "lexfront", **{f"no_{r}": f"lexfront:no_{r}" for r in RULES if r != "lowercase"},
             "all_off": "lexfront:all_off"}, "module2"),
    Setting("case", "case kept (TF-IDF lowercase=False)", CASE_SENSITIVE_OVERRIDES, "full_cs",
            {"full_cs": "lexfront", "no_lowercase": "lexfront:no_lowercase", "all_off_cs": "lexfront:all_off"},
            f"{SENSITIVITY_SUBDIR}/module2"),
)


def setting_texts(setting: Setting, texts) -> list[str]:
    """The text LexFront sees in this setting: TF-IDF lowercases it first unless the setting keeps case."""
    return list(texts) if setting.vec_overrides.get("lowercase") is False else [t.lower() for t in texts]


def count_firing(texts, configs: dict) -> dict:
    """{variant: {"docs", "tokens", "n_docs"}}: documents whose token list, and lexer tokens whose normalized form,
    differ from full LexFront when the variant's rules are switched off.

    Compiler concept: the semantic action of a token class only runs when the scanner emits that class; counting
    where the action's output changes tells whether a switched-off rule is exercised by the input at all. One scan per
    document (the lexer does not depend on the rules), then every configuration normalizes the same tokens.
    """
    scanner = LexFront()
    full = normalize_config("lexfront")
    counts = {v: {"docs": 0, "tokens": 0, "n_docs": 0} for v in configs}
    for text in texts:
        lexed = [(cls, lexeme) for cls, lexeme, _ in scanner.analyze(text)]
        reference = [normalize_token(cls, lexeme, full) for cls, lexeme in lexed]
        for variant, config in configs.items():
            forms = [normalize_token(cls, lexeme, config) for cls, lexeme in lexed]
            c = counts[variant]
            c["n_docs"] += 1
            c["tokens"] += sum(a != b for a, b in zip(forms, reference))
            c["docs"] += sum(forms, []) != sum(reference, [])
    return counts


def firing_path(out_dir, setting: Setting, dataset: str) -> Path:
    return Path(out_dir) / setting.name / f"firing__{dataset}.json"


def write_firing(out_dir, setting: Setting, ds, force: bool = False) -> Path:
    """Rule-firing counts per split for every non-reference variant; reused while data and code are unchanged."""
    path = firing_path(out_dir, setting, ds.name)
    stamp = {"data_hash": ds.content_hash(), "code": code_fingerprint()}
    if not force and path.exists() and all(json.loads(path.read_text()).get(k) == v for k, v in stamp.items()):
        return path
    configs = {v: normalize_config(tok) for v, tok in setting.variants.items() if v != setting.reference}
    record = {**stamp, "text": "lowercased (as TF-IDF does)" if setting.vec_overrides.get("lowercase") is not False
              else "as stored (case kept)",
              "splits": {split: count_firing(setting_texts(setting, texts), configs)
                         for split, texts in (("train", ds.train_texts), ("test", ds.test_texts))}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def variant_firing(out_dir, setting: Setting, dataset: str, variant: str) -> dict | None:
    """{"train": {...}, "test": {...}} firing counts of one variant; None if they were not counted here."""
    path = firing_path(out_dir, setting, dataset)
    if not path.exists():
        return None
    splits = json.loads(path.read_text())["splits"]
    return {split: counts[variant] for split, counts in splits.items()}


def run_ablation(out_dir=ABLATION_DIR, dataset_names=data.DATASET_NAMES, max_train=None, max_test=None,
                 seed: int = 0, force: bool = False, log=print) -> list[dict]:
    """Every variant of every setting, LR only; each dataset is loaded once."""
    records = []
    for dname in dataset_names:
        ds = data.load(dname, max_train, max_test, seed)
        log(f"== ablation {dname}: train={len(ds.train_texts)} test={len(ds.test_texts)}")
        for setting in SETTINGS:
            params = vectorizer_params(setting.vec_overrides)
            for tokenizer in setting.variants.values():
                records += run_cell(ds, tokenizer, [CLASSIFIER], Path(out_dir) / setting.name, seed, force, log,
                                    params)
            log("wrote", write_firing(out_dir, setting, ds, force))
    return records


def setting_paths(out_dir, setting: Setting, dataset: str) -> dict[str, Path]:
    """{variant label: run JSON} of one setting and dataset; empty if any variant is missing."""
    paths = {v: result_path(Path(out_dir) / setting.name, dataset, tok, CLASSIFIER) for v, tok in setting.variants.items()}
    return paths if all(p.exists() for p in paths.values()) else {}


def grid_check(out_dir, setting: Setting, dataset: str, y_true, y_ref) -> dict | None:
    """Does the full variant predict exactly what the full-LexFront+LR run of the main/sensitivity grid predicts?"""
    grid_run = result_path(Path(out_dir).parent / setting.grid_subdir, dataset, "lexfront", CLASSIFIER)
    if not grid_run.exists():
        return None
    gold, pred = load_predictions(grid_run)
    return {"run_json": grid_run.as_posix(),
            "identical_predictions": bool(np.array_equal(gold, y_true) and np.array_equal(pred, y_ref))}


def setting_stats(out_dir, setting: Setting, dataset: str, n_boot: int, seed: int) -> dict | None:
    """Per-variant CIs and paired differences (variant minus reference) for one setting and dataset."""
    paths = setting_paths(out_dir, setting, dataset)
    if not paths:
        return None
    y_true, preds, k = load_dataset_predictions(paths)
    idx = stats.bootstrap_indices(len(y_true), n_boot, np.random.default_rng(seed))
    ref = preds[setting.reference]
    variants = {}
    for v, path in paths.items():
        variants[v] = {"tokenizer": setting.variants[v], "run_json": path.as_posix(),
                       "n_features": json.loads(path.read_text())["vectorizer"]["n_features"],
                       **stats.run_summary(y_true, preds[v], idx, k)}
        check_point_estimates(variants[v], path)
        if v != setting.reference:
            variants[v]["minus_reference"] = stats.paired_difference(y_true, ref, preds[v], idx, k)
            variants[v]["mcnemar_accuracy"] = stats.mcnemar_exact(y_true, ref, preds[v])
            variants[v]["n_predictions_changed"] = int(np.sum(preds[v] != ref))
            variants[v]["rule_fires"] = variant_firing(out_dir, setting, dataset, v)
    return {"label": setting.label, "reference": setting.reference, "n_test": int(len(y_true)), "n_labels": k,
            "reference_vs_grid": grid_check(out_dir, setting, dataset, y_true, ref), "variants": variants}


def compute(out_dir=ABLATION_DIR, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """The whole ablation stats record; datasets without a complete setting are left out."""
    config = {"results_dir": str(out_dir), "method": "percentile bootstrap over test examples",
              "ci_level": stats.CI_LEVEL, "n_boot": n_boot, "seed": seed, "classifier": CLASSIFIER,
              "rng": "numpy.random.default_rng(seed), one fresh generator per (setting, dataset); all variants share "
                     "its resampled indices (paired)",
              "difference": "variant minus the setting's reference (full LexFront), in fractions (x100 for points)",
              "mcnemar": "exact two-sided binomial test on discordant pairs, accuracy, every variant vs reference",
              "multiplicity": "none: every interval is a separate 95% interval",
              "rule_fires": "per split: documents whose token list (docs) and lexer tokens whose normalized form "
                            "(tokens) differ from full LexFront, on the text LexFront sees in the setting; 0 in train "
                            "and test means the variant is the full system (not a test of the rule)",
              "not_covered": "training randomness: one training run per cell (seed 0)", "numpy": np.__version__,
              "settings": {s.name: {"label": s.label, "vectorizer_overrides": s.vec_overrides,
                                    "reference": s.reference, "variants": s.variants} for s in SETTINGS}}
    datasets = {}
    for ds in DATASET_ORDER:
        found = {s.name: setting_stats(out_dir, s, ds, n_boot, seed) for s in SETTINGS}
        found = {name: d for name, d in found.items() if d is not None}
        if found:
            datasets[ds] = found
    if not datasets:
        raise FileNotFoundError(f"no complete ablation setting under {out_dir}")
    return {"config": config, "datasets": datasets}


def write_stats(out_dir=ABLATION_DIR, n_boot: int = N_BOOT, seed: int = SEED) -> Path:
    out = Path(out_dir) / "stats" / "bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compute(out_dir, n_boot, seed), indent=2) + "\n")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=ABLATION_DIR)
    p.add_argument("--datasets", nargs="+", default=list(data.DATASET_NAMES))
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stats-only", action="store_true", help="only recompute <out>/stats from the stored predictions")
    a = p.parse_args(argv)
    if not a.stats_only:
        run_ablation(a.out, a.datasets, a.max_train, a.max_test, a.seed, a.force)
    print("wrote", write_stats(a.out))


if __name__ == "__main__":
    main()
