"""Compare two results trees (for example a worktree of main against a branch) and list every difference that is
not timing: `python -m experiments.compare_results [--additive] OLD_RESULTS NEW_RESULTS [subdir ...]`.

JSON files are compared as data with the volatile keys removed (timestamps, timings, throughput); gzip sidecars
(per-example predictions) are compared as decoded JSON; markdown tables are compared without throughput columns; other files byte for byte. Files that exist
only in NEW are reported as added (not a difference); files missing from NEW are differences. Exit code 1 on any
difference, which makes this the check for "this change leaves every existing metric untouched".
"""
import gzip
import json
import sys
from pathlib import Path

VOLATILE = {"timestamp_utc", "timing_seconds", "throughput_tokens_per_s", "throughput_docs_per_s"}


def load(path: Path):
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return path.read_bytes()


def strip_markdown(text: str) -> str:
    """Drop the columns whose header mentions throughput from every markdown table (they are timing)."""
    drop, out = set(), []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = line.split("|")
            if "throughput" in line.lower() and not drop:
                drop = {i for i, c in enumerate(cells) if "throughput" in c.lower()}
            line = "|".join(c for i, c in enumerate(cells) if i not in drop)
        else:
            drop = set()
        out.append(line)
    return "\n".join(out)


def strip(value):
    if isinstance(value, dict):
        # "code" is the hash of the source files: it changes whenever any file that can affect a result changes, even
        # if no result does; the metrics, predictions and the config hash are what must stay identical
        return {k: strip(v) for k, v in value.items() if k not in VOLATILE and not (k == "code" and isinstance(v, str))}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def contains(old, new) -> bool:
    """True if `new` has everything `old` has, equal: extra keys in dicts of `new` are allowed (lists must be equal)."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and contains(v, new[k]) for k, v in old.items())
    return old == new


def compare(old: Path, new: Path, subdirs=(), additive: bool = False) -> tuple[list[str], list[str]]:
    """(differences, added files) between two results trees, restricted to `subdirs` of the old tree if given.
    With `additive`, JSON files of `new` may have extra dictionary keys (a new tokenizer's entry in an existing file)."""
    roots = [old / s for s in subdirs] if subdirs else [old]
    differences, seen = [], set()
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(old)
            seen.add(rel)
            other = new / rel
            if not other.exists():
                differences.append(f"missing in new: {rel}")
            elif path.suffix in {".json", ".gz"}:
                same = contains(strip(load(path)), strip(load(other))) if additive else strip(load(path)) == strip(load(other))
                if not same:
                    differences.append(f"changed: {rel}")
            elif path.suffix == ".md":
                if strip_markdown(path.read_text()) != strip_markdown(other.read_text()):
                    differences.append(f"changed: {rel}")
            elif path.read_bytes() != other.read_bytes():
                differences.append(f"changed: {rel}")
    added = sorted(str(p.relative_to(new)) for s in (subdirs or ["."]) for p in (new / s).rglob("*")
                   if p.is_file() and p.relative_to(new) not in seen)
    return differences, added


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--additive"]
    differences, added = compare(Path(args[0]), Path(args[1]), args[2:], additive="--additive" in sys.argv)
    print(f"{len(differences)} differences, {len(added)} added files")
    for line in differences:
        print(" ", line)
    sys.exit(1 if differences else 0)
