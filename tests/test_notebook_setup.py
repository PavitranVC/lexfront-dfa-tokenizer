"""The notebook's setup cell must not run old code under a newer notebook (stale Colab clone with local changes)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_notebook", ROOT / "notebooks" / "make_notebook.py")
make_notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_notebook)


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def commit_files(repo, files, message):
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def run_setup(workdir, origin):
    """Run the setup cell up to (not including) the pip install, the way a notebook cell would, in `workdir`."""
    source = make_notebook.SETUP.split("from experiments import setup_check")[0]
    env = {**os.environ, "LEXFRONT_REPO_URL": str(origin)}
    return subprocess.run([sys.executable, "-c", source], cwd=workdir, capture_output=True, text=True, env=env)


def make_origin(tmp_path):
    origin = tmp_path / "remote" / "lexfront-dfa-tokenizer"      # `git clone` names the folder after the URL
    origin.mkdir(parents=True)
    git(origin, "init", "-q", "-b", "main")
    commit_files(origin, {"requirements.txt": "x\n", "lexer/token_spec.py": "v = 1\n",
                          "results/quick/m.json": "{}\n"}, "v1")
    return origin


def test_stale_clone_with_local_changes_is_updated(tmp_path):
    origin = make_origin(tmp_path)
    colab = tmp_path / "content"
    colab.mkdir()
    first = run_setup(colab, origin)                       # fresh runtime: clones
    assert first.returncode == 0, first.stderr
    clone = colab / "lexfront-dfa-tokenizer"
    (clone / "results/quick/m.json").write_text('{"a": "this run rewrote a tracked result file"}\n')
    commit_files(origin, {"lexer/token_spec.py": "v = 2\n", "results/quick/m.json": '{"b": 1}\n'}, "v2")
    second = run_setup(colab, origin)                      # later session, same runtime: must reach v2
    assert second.returncode == 0, second.stderr
    assert (clone / "lexer/token_spec.py").read_text() == "v = 2\n"
    assert "set aside" in second.stdout and git(clone, "stash", "list").strip()


def test_stale_clone_that_cannot_update_fails_loudly(tmp_path):
    origin = make_origin(tmp_path)
    colab = tmp_path / "content"
    colab.mkdir()
    assert run_setup(colab, origin).returncode == 0
    clone = colab / "lexfront-dfa-tokenizer"
    commit_files(clone, {"lexer/local.py": "x = 1\n"}, "local commit")       # diverges from origin
    commit_files(origin, {"lexer/token_spec.py": "v = 2\n"}, "v2")
    result = run_setup(colab, origin)
    assert result.returncode != 0 and "could not update the existing clone" in result.stderr


NLTK_STUB = """
import os, sys, types
class _Data:
    found = False
    def find(self, name):
        if not _Data.found:
            raise LookupError(name)
nltk = types.ModuleType("nltk")
nltk.data = _Data()
pathsec = types.ModuleType("nltk.pathsec")
pathsec.ALLOW_PROXIED_FETCH = False
def download(package, quiet=True):
    # what NLTK's SSRF guard does: it sees any *_proxy variable (urllib.request.getproxies) and refuses without the opt-in;
    # nltk.download prints the refusal and returns False
    proxy = any(k.lower().endswith("_proxy") and v for k, v in os.environ.items())
    optin = pathsec.ALLOW_PROXIED_FETCH or os.environ.get("NLTK_ALLOW_PROXIED_URLOPEN") == "1"
    if proxy and not optin:
        return False
    _Data.found = True
    return True
nltk.download = download
nltk.pathsec = pathsec
sys.modules["nltk"] = nltk
sys.modules["nltk.pathsec"] = pathsec
"""


def run_nltk_block(env_extra):
    source = make_notebook.SETUP.split("import nltk\n", 1)[1]
    code = ("import os\n" + NLTK_STUB + "import nltk\n" + source +
            "\nprint('opt-in left set:', 'NLTK_ALLOW_PROXIED_URLOPEN' in os.environ or nltk.pathsec.ALLOW_PROXIED_FETCH)\n")
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy") and k != "NLTK_ALLOW_PROXIED_URLOPEN"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**env, **env_extra})


@pytest.mark.parametrize("variable", ["HTTPS_PROXY", "http_proxy", "ALL_PROXY"])
def test_nltk_download_works_behind_a_proxy_and_opt_in_is_scoped(variable):
    result = run_nltk_block({variable: "http://proxy.invalid:3128"})
    assert result.returncode == 0, result.stderr
    assert "retrying once" in result.stdout and "opt-in left set: False" in result.stdout


def test_nltk_download_without_proxy_does_not_use_the_opt_in():
    result = run_nltk_block({})
    assert result.returncode == 0, result.stderr
    assert "retrying" not in result.stdout and "opt-in left set: False" in result.stdout
