"""Running git, and hashing a tree -- nothing about what a run is *about*.

This is the plumbing every git-aware workflow shares. What a run is about lives
in ``core.target`` (the VCS-shaped "one change per task") and what an agent may
ask a revision for is a shell command away.

Three properties are deliberate, and they are why this is not a handful of
inline ``subprocess.run`` calls:

* ``GIT_OPTIONAL_LOCKS=0``, so a *read* never takes the index lock: ``git status``
  rewrites ``.git/index`` as a side effect, and a run has several agents reading
  at once while the user may be running ``git add``.
* ``--end-of-options`` wherever a caller hands in a revision, so a revision cannot
  smuggle a flag into git.
* :class:`GitError` carries git's own stderr: a run that silently sees an empty
  diff is indistinguishable from a clean change.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

#: Diff flags that keep the text readable: no external drivers, no colour, no
#: rename detection churn, and enough context to read a hunk.
DIFF_FLAGS = (
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    # A user's `diff.noprefix` (or `--no-prefix`) removes the a/ b/ headers that
    # `git apply` needs; naming the prefixes defeats the config.
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "-U5",
)

#: The largest file whose bytes go into the tree digest. A bigger untracked file
#: contributes its size instead of its content: hashing a gigabyte of build
#: output to answer "did anything write here" is not worth it, and a changed size
#: is already a changed tree.
_TREE_MAX_FILE = 2_000_000


class GitError(RuntimeError):
    """A git command failed; the message carries git's own stderr."""


def git_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment every git command here runs with.

    ``GIT_OPTIONAL_LOCKS=0`` keeps a *read* from taking the index lock: ``git
    status`` rewrites ``.git/index`` as a side effect, so two agents running
    concurrently would race the user's own ``git add``. A function rather than an
    inline literal because it is the part worth pinning in a test.
    """
    env = dict(os.environ if base is None else base)
    # A git hook (pre-commit, post-checkout, ...) exports these to point git at
    # the caller's repository/index. Inheriting them would make our worktree
    # commands operate on the caller's checkout: a `git reset --mixed base` in a
    # scratch worktree moved the user's branch.
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(name, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def _run(
    args: tuple[str, ...],
    *,
    cwd: Path | None = None,
    stdin: bytes | None = None,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    """One git command, with the environment above and nothing decided yet.

    Every wrapper here differs only in what it does with the result, so this is
    the only place a subprocess is started.
    """
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=text,
        errors="replace" if text else None,
        check=False,
        env=git_env(),
    )


def git(*args: str, cwd: Path | None = None, ok: tuple[int, ...] = (0,)) -> str:
    """Run git with ``args`` and return stdout.

    Raises :class:`GitError` with git's stderr, because a run that silently sees
    an empty diff is indistinguishable from a clean change. ``ok`` names the exit
    codes that are not failures: ``git diff --no-index`` exits 1 when the files
    differ, which is the answer we asked for.
    """
    done = _run(args, cwd=cwd)
    if done.returncode not in ok:
        raise GitError(
            f"git {' '.join(args)} failed ({done.returncode}): "
            f"{(done.stderr or done.stdout).strip()}"
        )
    return done.stdout


def git_completed(
    *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run git and hand back the whole result, for callers that read its code.

    ``git ls-files --others`` answers with its exit status as well as its output,
    and the caller decides whether that status means anything.
    """
    return _run(args, cwd=cwd)


def git_bytes(*args: str, cwd: Path | None = None, stdin: bytes | None = None) -> bytes:
    """Run git with ``args`` on bytes in and out: a binary patch is not text."""
    done = _run(args, cwd=cwd, stdin=stdin, text=False)
    if done.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({done.returncode}): "
            f"{done.stderr.decode(errors='replace').strip()}"
        )
    return done.stdout


def repo_root(cwd: Path | None = None) -> Path:
    return Path(git("rev-parse", "--show-toplevel", cwd=cwd).strip())


def branch_exists(branch: str, cwd: Path | None = None) -> bool:
    """Whether ``branch`` exists locally. A branch is a claim, so this asks git."""
    done = git_completed(
        "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", cwd=cwd
    )
    return done.returncode == 0


def head_sha(cwd: Path | None = None) -> str:
    return git("rev-parse", "HEAD", cwd=cwd).strip()


#: Untracked directories that are an environment, not a change. A virtualenv is
#: usually gitignored, but not always (this repository's root `.gitignore` lists
#: `venv`, not `.venv`), and then it is diffed as thousands of new files, copied
#: into every reader's snapshot and hashed after every turn. It is linked into a
#: workspace instead (``worktree.LINKED``).
ENVIRONMENT_PATHS = (".venv",)


def untracked_files(
    cwd: Path | None = None, *, ignore: tuple[Path, ...] = ()
) -> list[str]:
    """Files git would add, honouring .gitignore and skipping ``ignore``.

    ``ignore`` is for output the run itself writes inside the repository. With the
    agentic ``.gitignore`` in place git already drops them, but a checkout without
    one would otherwise list its own previous log as part of the change -- and,
    because the diff is part of the prompt, miss the cache on every run.
    """
    out = git("ls-files", "--others", "--exclude-standard", "-z", cwd=cwd)
    ignored = tuple(Path(p).resolve() for p in ignore)
    root = (cwd or Path()).resolve()
    names: list[str] = []
    # `-z` is what keeps a non-ASCII name from coming back quoted, which would
    # then be copied to a path that does not exist.
    for line in out.split("\0"):
        if not line:
            continue
        if line.split("/", 1)[0] in ENVIRONMENT_PATHS:
            continue
        if ignored:
            full = root / line
            if any(full == skip or skip in full.parents for skip in ignored):
                continue
        names.append(line)
    return names


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_digest(cwd: Path | None = None, *, ignore: tuple[Path, ...] = ()) -> str:
    """A digest of the working tree, for proving nothing wrote to it.

    The read-only guarantee used to be structural: an agent had no tool that
    could write and no command that could. That cost more than it was worth --
    `git` and shell redirection are what a model actually types, and every
    refusal spent a retry until the run died -- so the guarantee is now checked
    instead: this digest is taken before and after, and a run that changed the
    tree reports "not reviewed" rather than a verdict about a tree it edited.

    Tracked changes come from ``git diff HEAD``; untracked files are hashed
    individually, because this repository's own ``agentic/`` was untracked while
    it was being reviewed and ``git status`` alone cannot see an edit to it.
    """
    tracked = _hash(git("diff", *DIFF_FLAGS, "HEAD", cwd=cwd).encode())
    entries = [f"head {head_sha(cwd=cwd)}", f"tracked {tracked}"]
    # A run writes its own log inside the repository. That is output, not the
    # change under review, so it is excluded by path -- the log exists before the
    # digest is taken and grows while the run proceeds.
    ignored = tuple(Path(p).resolve() for p in ignore)
    for path in untracked_files(cwd=cwd):
        full = (cwd or Path()).resolve() / path
        if any(full == skip or skip in full.parents for skip in ignored):
            continue
        try:
            size = full.stat().st_size
            data = full.read_bytes() if size <= _TREE_MAX_FILE else b""
        except OSError:
            entries.append(f"untracked {path} unreadable")
            continue
        entries.append(f"untracked {path} {size} {_hash(data)}")
    return _hash("\n".join(entries).encode())
