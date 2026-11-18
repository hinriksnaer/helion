"""Git-fixture helpers shared by the checkout and review tests.

Separate from `support.py` so the tracing commit, which does not touch git,
does not depend on the checkout modules.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from agentic.core import git

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _run(repo: Path) -> Callable[..., None]:
    """Run a git command in ``repo``, failing the test if it fails."""

    def run(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )

    return run


def _branch(repo: Path) -> str:
    return git.git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo).strip()


def _git_repo(tmp_path: Path) -> Path:
    """A two-commit repository, for the tools a reviewer is given."""
    repo = tmp_path / "r"
    repo.mkdir()

    def run(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )

    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    run("add", "a.txt")
    run("commit", "-q", "-m", "first")
    (repo / "a.txt").write_text("two\n")
    run("commit", "-q", "-am", "second")
    return repo


def _diverged_repo(tmp_path: Path) -> Path:
    """``upstream`` and the working branch each moved on after a common base."""
    repo = _git_repo(tmp_path)
    run = _run(repo)
    here = _branch(repo)
    run("branch", "upstream")
    run("checkout", "-q", "upstream")
    (repo / "a.txt").write_text("upstream\n")
    run("commit", "-qam", "upstream change")
    run("checkout", "-q", here)
    (repo / "b.txt").write_text("local\n")
    run("add", "b.txt")
    run("commit", "-qm", "local change")
    return repo
