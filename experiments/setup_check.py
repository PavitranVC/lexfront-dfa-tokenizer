"""Notebook setup check: which library versions are in use after `%pip install`, and is a restart needed?

A module that was already imported keeps running its old code after pip replaces it on disk, so the kernel would
mix two versions (like linking object files built against two versions of one library). This module compares the
installed versions before and after the install and says when the runtime must be restarted.
Standard library only: it runs before the project's dependencies are installed.
"""
import importlib
import importlib.metadata
import sys

# pip distribution name -> import name
WATCHED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
           "matplotlib": "matplotlib", "threadpoolctl": "threadpoolctl", "nltk": "nltk"}
SHOWN = ("numpy", "pandas", "scikit-learn")


def installed_version(dist: str) -> str | None:
    """Version of `dist` on disk now (None if not installed)."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def snapshot() -> dict:
    """Before the install: installed version of each watched package and whether the kernel already imported it."""
    return {dist: {"installed": installed_version(dist), "loaded": module in sys.modules}
            for dist, module in WATCHED.items()}


def needs_restart(before: dict) -> list[str]:
    """Packages that were imported before the install and whose installed version has changed since."""
    importlib.invalidate_caches()
    return [dist for dist, state in before.items()
            if state["loaded"] and installed_version(dist) != state["installed"]]


def version_in_use(dist: str) -> str | None:
    """The version the kernel runs: the imported module's, if already imported, else the one on disk."""
    module = sys.modules.get(WATCHED[dist])
    return getattr(module, "__version__", None) if module else installed_version(dist)


def report_after_install(before: dict, log=print) -> bool:
    """Print the versions in use; print a visible warning and return True if a restart is needed."""
    log("versions in use: " + ", ".join(f"{d} {version_in_use(d)}" for d in SHOWN))
    stale = needs_restart(before)
    if stale:
        changes = ", ".join(f"{d} {before[d]['installed']} -> {installed_version(d)}" for d in stale)
        log("!" * 78 + f"\nWARNING: pip changed packages that were already imported ({changes}).\n"
            "Restart the runtime (Colab: Runtime > Restart session) and run this cell again.\n" + "!" * 78)
    return bool(stale)
