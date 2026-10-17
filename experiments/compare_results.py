"""Compare two results trees (for example a worktree of main against a branch) and list every difference that is
not timing: `python -m experiments.compare_results OLD_RESULTS NEW_RESULTS [subdir ...]`.

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
        return {k: strip(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def compare(old: Path, new: Path, subdirs=()) -> tuple[list[str], list[str]]:
    """(differences, added files) between two results trees, restricted to `subdirs` of the old tree if given."""
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
                if strip(load(path)) != strip(load(other)):
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
    differences, added = compare(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
    print(f"{len(differences)} differences, {len(added)} added files")
    for line in differences:
        print(" ", line)
    sys.exit(1 if differences else 0)
