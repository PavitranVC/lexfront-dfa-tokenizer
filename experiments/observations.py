"""Observations written *from the JSON results only* -- every number is computed here.

Each function returns a markdown bullet list. They state arithmetic facts (best, spread,
signed differences). They do not explain *why*; explanations are labelled hypotheses in the
docs. All runs use one seed and no confidence intervals, so small gaps are not claims.
"""
import pandas as pd

from .report import DATASET_ORDER, load_datasets, load_dfa, load_intrinsic, load_runs

CAVEAT = ("_Single run per cell (seed 0), no confidence intervals: gaps of about one point or less "
          "should not be read as real differences._")


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
                   f"the largest test class holds {_p(top)} of the test set (the accuracy a majority "
                   f"baseline would reach if train and test share their largest class).")
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


OBSERVATIONS = {"datasets": obs_datasets, "module1": obs_module1, "existing_vs_proposed": obs_existing_vs_proposed,
                "intrinsic": obs_intrinsic, "dfa": obs_dfa}


def observation(name: str, results_dir) -> str:
    """Observation markdown for one section, or a note if its JSON is not there."""
    try:
        return OBSERVATIONS[name](results_dir)
    except (FileNotFoundError, KeyError, IndexError, ValueError, AttributeError) as err:
        return f"_not available: {err!r}_"
