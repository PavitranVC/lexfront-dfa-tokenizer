"""Observations written *from the JSON results only* -- every number is computed here.

Each function returns a markdown bullet list. They state arithmetic facts (best, spread,
signed differences, and whether a bootstrap CI excludes 0). They do not explain *why*; explanations are
labelled hypotheses in the docs. Every cell is one training run (seed 0); the CIs cover test-set sampling only.
"""
import pandas as pd

from .report import (DATASET_ORDER, TOKENIZER_ORDER, load_bootstrap, load_datasets, load_dfa, load_intrinsic, load_runs,
                     load_sensitivity_runs, p_value_text)

CAVEAT = ("_One training run per cell (seed 0). The 95% confidence intervals (table `ci_table`, file "
          "`stats/bootstrap.json`) are percentile bootstraps over the test examples: they show how much a score "
          "moves with the draw of the test set, not with training randomness, which one run per cell cannot show "
          "(the NB and LR fits are deterministic; the full grid uses no subsampling, QUICK mode one fixed-seed "
          "subsample). A gap whose CI includes 0 should not be read as a real difference._")

SENSITIVITY_CAVEAT = ("_Robustness check, not a like-for-like comparison: with TF-IDF case folding off, LexFront "
                      "lowercases its own WORD tokens while the baselines keep case. One training run per cell, no "
                      "confidence intervals for this grid._")


def _p(x) -> str:
    return f"{100 * x:.2f}%"


def _get(runs, ds, tok, clf):
    hit = runs[(runs.dataset == ds) & (runs.tokenizer == tok) & (runs.classifier == clf)]
    return hit.iloc[0] if len(hit) else None


def _datasets(df):
    return [d for d in DATASET_ORDER if d in set(df.dataset)]


def obs_datasets(results_dir) -> str:
    out = []
    for r in sorted(load_datasets(results_dir), key=lambda r: DATASET_ORDER.index(r["name"])):
        counts = r["test_class_counts"]
        top = max(counts.values()) / sum(counts.values())
        out.append(f"- **{r['name']}**: {r['n_train']} train / {r['n_test']} test documents, {r['n_labels']} labels; "
                   f"the largest test class holds {_p(top)} of the test set.")
    return "\n".join(out)


def obs_module1(results_dir) -> str:
    runs = load_runs(results_dir)
    runs = runs[(runs.module == "module1")]
    out = []
    for ds in _datasets(runs):
        sub = runs[(runs.dataset == ds) & (runs.classifier != "majority")]
        best = sub.sort_values("accuracy", ascending=False).iloc[0]
        lr = sub[sub.classifier == "lr"].accuracy
        nb = sub[sub.classifier == "nb"].accuracy
        line = (f"- **{ds}**: best accuracy {_p(best.accuracy)} ({best.tokenizer} + {best.classifier.upper()}). ")
        if len(lr) and len(nb):
            line += (f"Across the three tokenizers LR is "
                     f"{100 * (lr.mean() - nb.mean()):+.2f} points vs NB on average; the tokenizer spread "
                     f"is {100 * (lr.max() - lr.min()):.2f} points under LR and "
                     f"{100 * (nb.max() - nb.min()):.2f} points under NB.")
        out.append(line)
    return "\n".join(out + ["", CAVEAT])


def obs_existing_vs_proposed(results_dir) -> str:
    runs = load_runs(results_dir)
    out = []
    for ds in _datasets(runs):
        ex, pr = _get(runs, ds, "nltk", "nb"), _get(runs, ds, "lexfront", "lr")
        nl, ln = _get(runs, ds, "nltk", "lr"), _get(runs, ds, "lexfront", "nb")
        if ex is None or pr is None:
            continue
        d_acc, d_f1 = 100 * (pr.accuracy - ex.accuracy), 100 * (pr.macro_f1 - ex.macro_f1)
        line = (f"- **{ds}**: proposed (LexFront+LR) vs existing (NLTK+NB): accuracy {d_acc:+.2f} points, "
                f"macro-F1 {d_f1:+.2f} points.")
        if nl is not None and ln is not None:
            line += (f" Split of the accuracy change: switching classifier with NLTK held fixed "
                     f"(NB to LR) {100 * (nl.accuracy - ex.accuracy):+.2f}; switching tokenizer with LR held fixed "
                     f"(NLTK to LexFront) {100 * (pr.accuracy - nl.accuracy):+.2f}; switching tokenizer with NB "
                     f"held fixed {100 * (ln.accuracy - ex.accuracy):+.2f}.")
        out.append(line)
    return "\n".join(out + ["", CAVEAT])


def _change_split(runs, ds):
    """(total, classifier part, tokenizer part) of proposed - existing accuracy, in points; None if a cell is missing.

    total = (NLTK+LR - NLTK+NB) + (LexFront+LR - NLTK+LR), so the two parts add up exactly.
    """
    cells = [_get(runs, ds, t, c) for t, c in [("nltk", "nb"), ("nltk", "lr"), ("lexfront", "lr")]]
    if any(c is None for c in cells):
        return None
    ex, nl, pr = (100 * c.accuracy for c in cells)
    return pr - ex, nl - ex, pr - nl


def _attribution_sentence(ds, total, clf_part, tok_part) -> str:
    """One arithmetic sentence per dataset: where the proposed-vs-existing change comes from."""
    split = f"classifier switch NB to LR {clf_part:+.2f}, tokenizer switch NLTK to LexFront under LR {tok_part:+.2f}"
    if total < 0:
        return f"- **{ds}**: proposed is {-total:.2f} points **below** existing ({split})."
    if abs(tok_part) <= 0.1 * abs(total):
        return f"- **{ds}**: proposed is {total:.2f} points above existing; almost all of it comes from the classifier ({split})."
    return f"- **{ds}**: proposed is {total:.2f} points above existing ({split})."


def obs_attribution(results_dir) -> str:
    """Per dataset: is proposed above or below existing, and how much is classifier vs tokenizer (accuracy)."""
    runs = load_runs(results_dir)
    out = []
    for ds in _datasets(runs):
        parts = _change_split(runs, ds)
        if parts is not None:
            out.append(_attribution_sentence(ds, *parts))
    return "\n".join(out + ["", CAVEAT])


def obs_intrinsic(results_dir) -> str:
    df = load_intrinsic(results_dir)
    out = []
    for ds in _datasets(df):
        sub = df[df.dataset == ds].set_index("tokenizer")
        low_oov = sub.oov_rate_tokens.idxmin()
        small_vocab = sub.vocab_size.idxmin()
        parts = [f"lowest OOV rate {_p(sub.oov_rate_tokens.min())} ({low_oov})",
                 f"smallest vocabulary {int(sub.vocab_size.min())} ({small_vocab})"]
        if "lexfront" in sub.index:
            lf = sub.loc["lexfront"]
            parts.append(f"LexFront: OOV {_p(lf.oov_rate_tokens)}, fertility {lf.fertility:.3f}, "
                         f"vocabulary {int(lf.vocab_size)}")
            if "nltk" in sub.index:
                ratio = lf.throughput_tokens_per_s / sub.loc["nltk"].throughput_tokens_per_s
                parts.append(f"LexFront throughput is {ratio:.2f}x NLTK's (tokens/s, this machine)")
        out.append(f"- **{ds}**: " + "; ".join(parts) + ".")
    return "\n".join(out)


def obs_dfa(results_dir) -> str:
    d = load_dfa(results_dir)
    c, e = d["combined"], d["elongated_expanded_lowercase"]
    lines = [f"- Combined automaton for {', '.join(d['classes'])}: {c['nfa_states']} NFA states -> "
             f"{c['dfa_states']} DFA states -> {c['min_dfa_states']} after minimization "
             f"({c['alphabet_atoms']} alphabet atoms).",
             "- Per class (NFA / DFA / minimized): " + "; ".join(
                 f"{n} {s['nfa_states']}/{s['dfa_states']}/{s['min_dfa_states']}" for n, s in d["per_class"].items()) + ".",
             f"- ELONGATED is written with a back-reference (`{d['elongated_backreference_pattern']}`) and is not "
             f"accepted by our regex parser, so it is not in the DFA engine. Spelled out as 26 alternatives "
             f"(lowercase only) it compiles to {e['nfa_states']} NFA / {e['dfa_states']} DFA / "
             f"{e['min_dfa_states']} minimized states."]
    return "\n".join(lines)


def _ci_phrase(d: dict) -> str:
    """'+0.09 points (95% CI -0.50 to +0.70, includes 0)' for one metric of a paired difference."""
    verdict = "excludes 0" if d["ci_excludes_zero"] else "includes 0"
    return f"{100 * d['diff']:+.2f} points (95% CI {100 * d['ci'][0]:+.2f} to {100 * d['ci'][1]:+.2f}, {verdict})"


def _pair_sentence(ds: str, pair: dict) -> str:
    line = (f"- **{ds}**, {pair['label']}: accuracy {_ci_phrase(pair['accuracy'])}; "
            f"macro-F1 {_ci_phrase(pair['macro_f1'])}.")
    mc = pair.get("mcnemar_accuracy")
    if mc:
        line += (f" Exact McNemar test on accuracy: p {p_value_text(mc['p_value'])} (discordant test examples: right only "
                 f"with {pair['a'].replace('__', '+')} {mc['a_right_b_wrong']}, right only with "
                 f"{pair['b'].replace('__', '+')} {mc['a_wrong_b_right']}).")
    return line


def obs_ci(results_dir) -> str:
    """Per dataset and pair: the paired difference, its bootstrap CI and whether the CI excludes 0."""
    boot = load_bootstrap(results_dir)
    cfg = boot["config"]
    out = [_pair_sentence(ds, pair) for ds, d in boot["datasets"].items() for pair in d["pairs"].values()]
    n_excl = sum(p[m]["ci_excludes_zero"] for d in boot["datasets"].values()
                 for name, p in d["pairs"].items() if name == "tokenizer_only_lr" for m in ("accuracy", "macro_f1"))
    n_all = 2 * sum("tokenizer_only_lr" in d["pairs"] for d in boot["datasets"].values())
    out.append(f"- Tokenizer-only comparison (LexFront+LR vs NLTK+LR): {n_excl} of {n_all} difference CIs "
               f"(accuracy and macro-F1, all datasets) exclude 0.")
    out.append(f"- Settings: {cfg['method']}, B = {cfg['n_boot']}, seed {cfg['seed']}, "
               f"{100 * cfg['ci_level']:.0f}% percentile intervals.")
    return "\n".join(out + ["", CAVEAT])


def obs_sensitivity_case(results_dir) -> str:
    """Case-sensitive TF-IDF grid against the main grid: the tokenizer-only gap and each tokenizer's LR change."""
    sens, main = load_sensitivity_runs(results_dir), load_runs(results_dir)
    out = []
    for ds in _datasets(sens):
        parts = []
        lf_s, nl_s = _get(sens, ds, "lexfront", "lr"), _get(sens, ds, "nltk", "lr")
        lf_m, nl_m = _get(main, ds, "lexfront", "lr"), _get(main, ds, "nltk", "lr")
        if all(c is not None for c in (lf_s, nl_s, lf_m, nl_m)):
            parts.append(f"LexFront+LR minus NLTK+LR accuracy is {100 * (lf_s.accuracy - nl_s.accuracy):+.2f} "
                         f"points with case kept vs {100 * (lf_m.accuracy - nl_m.accuracy):+.2f} in the main grid")
        changes = []
        for tok in TOKENIZER_ORDER:
            s, m = _get(sens, ds, tok, "lr"), _get(main, ds, tok, "lr")
            if s is not None and m is not None:
                changes.append(f"{tok} {100 * (s.accuracy - m.accuracy):+.2f}")
        if changes:
            parts.append("LR accuracy change from switching TF-IDF case folding off: " + ", ".join(changes))
        out.append(f"- **{ds}**: " + "; ".join(parts) + ".")
    return "\n".join(out + ["", SENSITIVITY_CAVEAT])


OBSERVATIONS = {"datasets": obs_datasets, "module1": obs_module1, "existing_vs_proposed": obs_existing_vs_proposed,
                "intrinsic": obs_intrinsic, "dfa": obs_dfa, "attribution": obs_attribution, "ci": obs_ci, "sensitivity_case": obs_sensitivity_case}


def observation(name: str, results_dir) -> str:
    """Observation markdown for one section, or a note if its JSON is not there (other errors are raised)."""
    try:
        return OBSERVATIONS[name](results_dir)
    except FileNotFoundError as err:
        return f"_not available: {err!r}_"
