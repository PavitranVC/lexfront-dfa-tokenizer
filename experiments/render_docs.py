"""Fill generated blocks inside markdown files from the JSON results.

A block looks like
    <!-- BEGIN table:module1_results -->  ...  <!-- END table:module1_results -->
    <!-- BEGIN obs:module1 -->            ...  <!-- END obs:module1 -->
Everything between the markers is replaced; text outside is never touched.

    python -m experiments.render_docs [--results results] [files ...]
"""
import argparse
import re
from pathlib import Path

from .observations import observation
from .report import build_all_tables

DEFAULT_FILES = ["results/REPORT_NOTES.md", "docs/modules/module1.md", "docs/modules/module2.md", "README.md"]
BLOCK = re.compile(r"(<!-- BEGIN (table|obs):(\w+) -->)\n.*?\n(<!-- END \2:\3 -->)", re.DOTALL)


def render_text(text: str, tables: dict, results_dir: str) -> str:
    def fill(m):
        kind, name = m.group(2), m.group(3)
        body = tables.get(name, f"_unknown table {name}_") if kind == "table" else observation(name, results_dir)
        if kind == "obs":
            body = f"**DRAFT observations (generated from the JSON files):**\n\n{body}"
        return f"{m.group(1)}\n{body}\n{m.group(4)}"
    return BLOCK.sub(fill, text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("files", nargs="*", default=DEFAULT_FILES)
    a = p.parse_args(argv)
    tables = build_all_tables(a.results)
    for f in a.files:
        path = Path(f)
        if path.exists():
            path.write_text(render_text(path.read_text(), tables, a.results))
            print("rendered", f)


if __name__ == "__main__":
    main()
