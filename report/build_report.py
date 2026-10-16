"""Build report/Review1_Report.md (and .docx) from the template.

* Result tables are copied verbatim from results/tables/*.md (generated from the JSON files in results/).
* {{key}} / {{a,b}} citations are replaced by IEEE numbers in order of first appearance.
* The tokenizer-comparison table is read from the committed notebook output.
* Run from the repository root:  python report/build_report.py
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from refs import REFS  # noqa: E402

EXAMPLE_ROWS = [1, 2, 3, 4, 7]          # which of the eight notebook tweets to show
TOKENIZERS = ["whitespace", "nltk", "tweet", "lexfront"]
TOKENIZER_LABELS = {"whitespace": "whitespace", "nltk": "NLTK", "tweet": "TweetTokenizer", "lexfront": "LexFront"}
EXAMPLE_TWEETS = {   # copied from experiments/review1.py (EXAMPLE_TWEETS); checked below
    1: "OMG u wont believe this!!! https://t.co/abc123 @mike_93 :)",
    2: "sooooo tired of mondays #monday #ugh \U0001F629\U0001F629",
    3: "can't wait for the game tonight, we're gonna win 3-0 <3",
    4: "idk tbh... this movie was gr8 but the ending, meh :-(",
    7: "I luuuuv it §§ but ur price is too high ;P",
}


def table_from_file(name: str, widths: list[int]) -> str:
    lines = (ROOT / "results" / "tables" / f"{name}.md").read_text().strip().splitlines()
    n = len(lines[0].strip("|").split("|"))
    assert n == len(widths), (name, n, len(widths))
    lines[1] = "|" + "|".join("-" * w for w in widths) + "|"
    return "\n".join(lines)


def examples_table() -> str:
    review1 = (ROOT / "experiments" / "review1.py").read_text()
    for i, text in EXAMPLE_TWEETS.items():           # make sure the inputs match the repo code
        literal = json.dumps(text, ensure_ascii=False)[1:-1]
        assert text.replace("\U0001F629", "\\U0001F629") in review1 or literal in review1, text
    nb = json.loads((ROOT / "notebooks" / "LexFront_Review1.ipynb").read_text())
    page = "".join(nb["cells"][12]["outputs"][0]["data"]["text/html"])
    rows = re.findall(r"<tr>\s*<th>\d+</th>(.*?)</tr>", page, re.S)
    cells = [[html.unescape(c).strip() for c in re.findall(r"<td>(.*?)</td>", r, re.S)] for r in rows]
    out = ["| Ex. | Input | Tokenizer | Tokens |", "|---|---------|--------|----------------------------|"]
    for idx in EXAMPLE_ROWS:
        row = cells[idx - 1]
        assert row[0] == str(idx)
        for j, tok in enumerate(TOKENIZERS):
            first = j == 0
            inp = "`" + EXAMPLE_TWEETS[idx].replace("|", "\\|") + "`" if first else ""
            toks = row[1 + j].replace("|", "\\|")
            out.append(f"| {idx if first else ''} | {inp} | {TOKENIZER_LABELS[tok]} | `{toks}` |")
    return "\n".join(out)


def cite(text: str) -> tuple[str, list[str]]:
    order: list[str] = []

    def number(key: str) -> int:
        if key not in REFS:
            raise KeyError(f"unknown reference key: {key}")
        if key not in order:
            order.append(key)
        return order.index(key) + 1

    def sub(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        return ", ".join(f"[{number(k)}]" for k in keys)

    text = re.sub(r"\{\{([a-z0-9_,\s]+)\}\}", sub, text)
    return text, order


def build() -> Path:
    text = (HERE / "Review1_Report.template.md").read_text()
    def table_sub(m):
        name, w = m.group(1), [int(x) for x in m.group(2).split(",")]
        return table_from_file(name, w)
    text = re.sub(r"@@TABLE:(\w+):([\d,]+)@@", table_sub, text)
    text = text.replace("@@EXAMPLES@@", examples_table())
    text, order = cite(text)
    unused = sorted(set(REFS) - set(order))
    assert not unused, f"references never cited: {unused}"
    def linked(entry: str) -> str:
        head, sep, url = entry.rpartition("Available: ")
        return f"{head}{sep}<{url}>" if sep else entry

    refs = "\n\n".join(f"[{i}] {linked(REFS[k])}" for i, k in enumerate(order, 1))
    text = text.replace("@@REFERENCES@@", refs)
    out = HERE / "Review1_Report.md"
    out.write_text(text)
    print(f"wrote {out}  ({len(order)} references)")
    return out


if __name__ == "__main__":
    build()
    if "--docx" in sys.argv:
        subprocess.run([sys.executable, str(HERE / "make_docx.py")], check=True)
