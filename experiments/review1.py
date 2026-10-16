"""Section functions behind the Review-1 notebook (the notebook only calls these).

QUICK_MODE runs everything on a fixed-seed subsample and writes to `results/quick/`;
the full runs live in `results/`. Tables, figures and observations are always rebuilt from
the JSON files of the chosen directory.
"""
from pathlib import Path

import pandas as pd

from lexer.dfa.compiler import compile_rules
from lexer.lexfront import LexFront
from lexer.ply_lexer import PlyLexer
from lexer.symbol_table import SymbolTable
from lexer.token_spec import DFA_CLASSES, get_spec

from . import data, report
from .classifiers import CLASSIFIER_NAMES
from .dfa_report import build_report, dfa_rules
from .figures import build_all_figures
from .observations import observation
from .pipeline import run_grid
from .run_ablation import run_ablation, write_stats as write_ablation_stats
from .run_stats import write as write_bootstrap
from .run_module2 import run_intrinsic
from .tokenizers import MODULE1_TOKENIZERS, TOKENIZERS

QUICK = {"max_train": 2000, "max_test": 500}
FULL = {"max_train": None, "max_test": None}

EXAMPLE_TWEETS = [
    "OMG u wont believe this!!! https://t.co/abc123 @mike_93 :)",
    "sooooo tired of mondays #monday #ugh \U0001F629\U0001F629",
    "can't wait for the game tonight, we're gonna win 3-0 <3",
    "idk tbh... this movie was gr8 but the ending, meh :-(",
    "@jenny thx 4 the 1,200 likes lolll #blessed \U0001F64F",
    "Breaking: stocks fall 2.5% as ppl worry abt rates www.news-site.com/markets",
    "I luuuuv it §§ but ur price is too high ;P",
    "She's like \"nooooo way\" and I'm like yessss!!!",
]


def results_dir(quick: bool) -> str:
    return "results/quick" if quick else "results"


def sizes(quick: bool) -> dict:
    return QUICK if quick else FULL


# ---- section 1: datasets ----------------------------------------------------------------
def run_datasets_section(quick: bool, log=print):
    """Load the three datasets and save `datasets.json` for the chosen results directory."""
    data.write_summary(results_dir(quick), seed=0, log=log, **sizes(quick))


# ---- section 2 / 4: experiments -------------------------------------------------------------
def run_module1_section(quick: bool, force: bool = False, log=print):
    """Module 1 grid; cached JSONs are reused only if their fingerprint matches, `force` recomputes all."""
    run_grid(data.DATASET_NAMES, MODULE1_TOKENIZERS, CLASSIFIER_NAMES, f"{results_dir(quick)}/module1",
             seed=0, force=force, log=log, **sizes(quick))


def run_module2_experiments(quick: bool, force: bool = False, log=print):
    out = f"{results_dir(quick)}/module2"
    build_report(out)
    run_intrinsic(data.DATASET_NAMES, out, sizes(quick)["max_train"], sizes(quick)["max_test"], log=log)
    run_grid(data.DATASET_NAMES, ["lexfront"], ["nb", "lr"], out, seed=0, force=force, log=log, **sizes(quick))


def run_stats_section(quick: bool, log=print):
    """Bootstrap CIs and paired tests from the stored predictions of the chosen grid (no refitting)."""
    log("wrote", write_bootstrap(results_dir(quick)))


def run_ablation_section(quick: bool, force: bool = False, log=print):
    """Per-rule normalization ablation (LexFront + LR, one rule off at a time) and its paired bootstrap CIs."""
    out = f"{results_dir(quick)}/ablation"
    run_ablation(out, data.DATASET_NAMES, sizes(quick)["max_train"], sizes(quick)["max_test"], force=force, log=log)
    log("wrote", write_ablation_stats(out))


def build_outputs(quick: bool) -> dict:
    """Rebuild tables and figures from the JSON of the chosen results directory."""
    base = results_dir(quick)
    return {"tables": report.build_all_tables(base), "figures": build_all_figures(base), "dir": base}


# ---- section 3: Module 2 demos ---------------------------------------------------------------
def lexfront_table(text: str, lexfront: LexFront | None = None) -> pd.DataFrame:
    """class / lexeme / normalized form for every LexFront token of `text`."""
    lexfront = lexfront or LexFront()
    return pd.DataFrame([{"class": c, "lexeme": lex, "normalized": " ".join(forms)}
                         for c, lex, forms in lexfront.analyze(text)])


def compare_tokenizers(texts=EXAMPLE_TWEETS) -> pd.DataFrame:
    """Side-by-side tokens of the four tokenizers (input lowercased like the vectorizer does)."""
    rows = []
    for i, text in enumerate(texts, 1):
        row = {"#": i}
        row.update({name: " ".join(f"[{t}]" for t in tok(text.lower())) for name, tok in TOKENIZERS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def symbol_table_demo(texts=EXAMPLE_TWEETS, top: int = 12) -> pd.DataFrame:
    """Symbol table after LexFront has scanned the example tweets."""
    table = SymbolTable()
    lf = LexFront(symbols=table)
    for t in texts:
        lf.tokenize(t)
    return pd.DataFrame(table.rows(top))


def scanner_demo(text: str = "#mood :-) @bob_1 café § <3 w8") -> tuple[pd.DataFrame, bool]:
    """Scan with our table-driven DFA scanner (4 regular classes) and check PLY agrees."""
    scanner = compile_rules(dfa_rules()).scanner()
    tokens = scanner.scan(text)
    ply_tokens = PlyLexer([get_spec(n) for n in DFA_CLASSES]).scan(text)
    return pd.DataFrame(tokens, columns=["class", "lexeme", "position"]), tokens == ply_tokens


def markdown(name_or_text: str) -> str:
    return report.TABLES[name_or_text]


def full_results_available() -> bool:
    return Path("results/module1").exists() and any(Path("results/module1").glob("*.json"))
