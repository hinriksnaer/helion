"""Worktrees: where every agent with a workspace does its work.

No agent works in the caller's checkout. Both kinds of workspace are git
worktrees in a temporary directory *outside* the checkout:

* a **reader's** tree (:func:`snapshot` or a revision checkout): a copy of a
  checkout as it is now -- its ``HEAD``, its uncommitted changes and its
  untracked files, uncommitted as they are in the source -- or, when the task
  names a revision, a checkout of that revision. Either is private to one
  worker. The worker may run anything there; what it must not do is change it,
  and that is checked after every turn (:meth:`Workspace.changed`). A turn that
  changed it gets no verdict, and the workspace is rebuilt before that worker's
  next turn.

Private reader trees are what let independent readers run concurrently: nothing
one reviewer's shell writes can reach another reviewer's tree, and a verdict is
about a tree that was confirmed unchanged when the turn ended, so it can be
recorded at once rather than at the end of the run.

A **writer's** tree is the third kind: it is on a branch the run owns, cut from
a base revision, and it is *supposed* to change. The branch lives in the
caller's repository (worktrees share the object database and the refs), so
removing the worktree at the end leaves the branch behind: that is the run's
result, and the caller asks git for it rather than for a patch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import TYPE_CHECKING

from . import git

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Ignored paths a reader still needs, linked into a snapshot from its source.
#: ``.venv`` holds the installed library source a reviewer reads to settle how a
#: dependency behaves. It is a link, so the file tools (which refuse a path
#: whose real location is outside the workspace) do not follow it; the shell does.
#: What a workspace links from its source instead of copying: the environment.
LINKED = git.ENVIRONMENT_PATHS


@dataclass(frozen=True)
class Tree:
    """The tree a turn works in, as an agent's deps.

    A tool a workflow adds has to act on the workspace the *run* was given, not
    on the one that existed when the agent was built: the agent is built once and
    answers a task about a commit, a working tree, or a writer's branch. The root
    travels as the run's deps for that reason, and ``Workspace.working_dir()``
    (async) is the harness's own way to ask the same question of a sandbox.
    """

    root: Path


@contextmanager
def worktree(repo: Path, revision: str) -> Iterator[Path]:
    """A detached worktree of ``repo`` at ``revision``, removed on exit."""
    parent = Path(tempfile.mkdtemp(prefix="agentic-worktree-"))
    path = parent / "tree"
    common = _common_dir(repo)
    _add(repo, path, revision)
    try:
        yield path
    finally:
        _remove(common, path)
        shutil.rmtree(parent, ignore_errors=True)


@dataclass
class Workspace:
    """A reader's private tree, and what it was made from."""

    path: Path
    source: Path
    #: The repository's shared git directory, which outlives the worktree the
    #: snapshot was made from.
    common: Path
    ignore: tuple[Path, ...] = ()
    baseline: str = ""
    #: The revision a checkout was made at, or ``None`` for a copy of a working
    #: tree. A rebuild repeats the same checkout rather than copying the source,
    #: which may have moved since.
    revision: str | None = None
    #: The branch a writer's worktree was made on, and ``None`` for a reader's
    #: tree. A writer's tree is expected to change, so it is not checked.
    branch: str | None = None

    @property
    def writable(self) -> bool:
        return self.branch is not None

    def digest(self) -> str:
        return git.tree_digest(self.path)

    def changed(self) -> bool:
        """Whether anything wrote to the workspace since it was (re)built."""
        return self.digest() != self.baseline

    def rebuild(self) -> None:
        """Make the workspace again from ``source``, at the same path.

        Readers only: a writer's tree holds its work, and rebuilding it would
        throw that away.

        A snapshot of a working tree is read again, so a rebuild mid-run reflects
        the source as it is then. A checkout of a revision is made at that same
        revision (a workflow that needs one fixed tree snapshots a writer's
        worktree or a commit, which do not move).
        """
        _remove(self.common, self.path)
        if self.revision is None:
            _fill(self)
        else:
            _add(self.source, self.path, self.revision)
            _link(self)


@contextmanager
def reader_workspace(
    source: Path, *, revision: str | None = None, ignore: tuple[Path, ...] = ()
) -> Iterator[Workspace]:
    """The tree one reader is given: the revision under review, or the source now.

    A task that names a revision -- which is what a commit or a range is -- gets
    a **checkout of that revision**. That is the tree the diff in its prompt ends
    at, so what the agent reads and runs is the change under review; the caller's
    checkout is usually ahead of it by every later commit in the stack. It is also
    cheaper (no patch to apply, no untracked file to copy).

    A task with no revision is about a working tree, and only a copy of it can
    answer what the tree holds, so that case keeps :func:`snapshot`.

    Both kinds carry the ignored paths a reader needs (:data:`LINKED`) and both
    are checked for writes: a reviewer with a shell can edit a checkout of a
    revision just as easily as a snapshot, and its verdict must not be recorded
    against a tree it changed.
    """
    if revision:
        with worktree(source, revision) as path:
            workspace = Workspace(
                path=path,
                source=source,
                common=_common_dir(source),
                revision=revision,
            )
            _link(workspace)
            yield workspace
        return
    with snapshot(source, ignore=ignore) as workspace:
        yield workspace


@contextmanager
def snapshot(source: Path, *, ignore: tuple[Path, ...] = ()) -> Iterator[Workspace]:
    """A private copy of ``source`` as it is now, removed on exit.

    ``ignore`` names untracked paths that are not part of the tree (a run's own
    log directory); they are not copied.
    """
    parent = Path(tempfile.mkdtemp(prefix="agentic-snapshot-"))
    workspace = Workspace(
        path=parent / "tree", source=source, common=_common_dir(source), ignore=ignore
    )
    try:
        _fill(workspace)
        yield workspace
    finally:
        _remove(workspace.common, workspace.path)
        shutil.rmtree(parent, ignore_errors=True)


@contextmanager
def writer_workspace(
    source: Path, *, branch: str, base: str | None = None
) -> Iterator[Workspace]:
    """A worktree on a new branch of ``source``, cut from ``base`` (default HEAD).

    The branch is made in the caller's repository, so it outlives the worktree:
    when the run ends, the branch is what the caller checks out, reviews or
    deletes. Nothing here pushes, and a branch that already exists is an error
    rather than a force-reset -- an existing branch is another run's claim.
    """
    parent = Path(tempfile.mkdtemp(prefix="agentic-worktree-"))
    path = parent / "tree"
    common = _common_dir(source)
    _add_branch(source, path, branch, base)
    workspace = Workspace(
        path=path, source=source, common=common, branch=branch, revision=base
    )
    try:
        _link(workspace)
        yield workspace
    finally:
        _remove(common, path)
        shutil.rmtree(parent, ignore_errors=True)


def _add_branch(repo: Path, path: Path, branch: str, base: str | None) -> None:
    git.git(
        "worktree", "add", "--quiet", "-b", branch, str(path), "--end-of-options",
        base or "HEAD", cwd=repo,
    )  # fmt: skip


def _add(repo: Path, path: Path, revision: str) -> None:
    git.git(
        "worktree", "add", "--detach", "--quiet", str(path), "--end-of-options",
        revision, cwd=repo,
    )  # fmt: skip


def _common_dir(repo: Path) -> Path:
    return Path(
        git.git(
            "rev-parse", "--path-format=absolute", "--git-common-dir", cwd=repo
        ).strip()
    )


#: The prefixes this package gives the directories it creates in the temporary
#: directory. A run removes its own; these are what is left when it does not.
_WORKSPACE_PREFIXES = ("agentic-worktree-", "agentic-snapshot-")

#: How old a leftover workspace must be before it is swept. A turn is bounded by
#: minutes, so a directory nothing has written to for hours belongs to a run that
#: was killed, crashed, or lost its machine.
STALE_HOURS = 6.0


def sweep(
    *, root: Path | None = None, older_than_hours: float = STALE_HOURS
) -> list[Path]:
    """Remove the workspace directories a run that died left behind.

    `kill -9`, a crashed interpreter or a reboot leaves a tree in the temporary
    directory -- roughly a hundred megabytes each, plus a worktree entry in the
    repository it was made from -- and nothing else ever comes back for it. Only
    this package's own prefixes are touched, and only directories nothing has
    written to in hours, so a live run of any length is left alone.
    """
    where = root if root is not None else Path(tempfile.gettempdir())
    cutoff = time.time() - older_than_hours * 3600
    gone: list[Path] = []
    try:
        entries = list(where.iterdir())
    except OSError:  # an unreadable temporary directory is not this run's problem
        return gone
    for entry in entries:
        if not entry.name.startswith(_WORKSPACE_PREFIXES) or not entry.is_dir():
            continue
        try:
            if entry.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        gone.append(entry)
    return gone


def _remove(common: Path, path: Path) -> None:
    """Remove the worktree at ``path``, through the shared git directory.

    Not through the worktree it was made from: the source may already be gone by
    the time the run ends.
    """
    reap_jobs(path)
    try:
        git.git(
            "--git-dir", str(common), "worktree", "remove", "--force", str(path),
            cwd=common,
        )  # fmt: skip
    except git.GitError:
        shutil.rmtree(path, ignore_errors=True)
        git.git("--git-dir", str(common), "worktree", "prune", cwd=common)


#: Where the harness's shell tool keeps one directory per job, with a ``handle``
#: file naming the process and its group. A command runs detached from the run
#: by design ("the process is not tied to the run or the event loop"), so the one
#: place that knows a job exists must also be the place that ends it: a cancelled
#: turn otherwise leaves a scan crawling at a full core for an hour, writing into
#: a log in a directory this function is about to delete.
_JOBS = Path(".pydantic-ai-harness") / "shell"


def reap_jobs(path: Path) -> None:
    """Stop the shell jobs still running inside ``path``.

    A command runs detached from the run by design, so the one place that knows a
    job exists must be the place that ends it: the workspace teardown calls this
    when the tree is removed, and a *reader's* turn calls it when that turn ends.
    A reader has no work that outlives its turn -- every reader turn starts from
    its own checkout of the same tree -- and a script that outlived its call cost
    one scan 271 s of a core, so there it is reaped early rather than later.
    """
    base = path / _JOBS
    if not base.is_dir():
        return
    for handle in base.glob("*/handle"):
        try:
            # `<supervisor pid> <process group or ->`; the group is the job's own
            # session, so killing it takes the command and everything it started.
            pid, group = handle.read_text(errors="replace").split()[:2]
            target = int(group) if group != "-" else int(pid)
        except (OSError, ValueError):
            continue
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(target, sig)
            except (OSError, ValueError):
                break


def _fill(workspace: Workspace) -> None:
    """Build the snapshot: ``HEAD``, then the uncommitted and untracked work."""
    source, path = workspace.source, workspace.path
    _add(source, path, git.head_sha(source))
    # Staged and unstaged changes both land unstaged: a reader asks what the
    # tree holds, not what the next commit would.
    # `DIFF_FLAGS` names the a/ b/ prefixes and disables external diff drivers:
    # a user's `diff.noprefix`, `color.diff` or `diff.external` would otherwise
    # produce a patch `git apply` cannot read.
    patch = git.git_bytes("diff", *git.DIFF_FLAGS, "--binary", "HEAD", cwd=source)
    if patch.strip():
        git.git_bytes(
            "apply", "--binary", "--whitespace=nowarn", "-", cwd=path, stdin=patch
        )
    for name in git.untracked_files(source, ignore=workspace.ignore):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target, follow_symlinks=False)
    _link(workspace)


def _link(workspace: Workspace) -> None:
    """Link the ignored paths a reader needs and take the baseline digest."""
    for name in LINKED:
        if (workspace.source / name).exists() and not _lexists(workspace.path / name):
            (workspace.path / name).symlink_to(workspace.source / name)
    workspace.baseline = workspace.digest()


def _lexists(path: Path) -> bool:
    return path.is_symlink() or path.exists()
