"""Workspaces: a reader's private snapshot, and what a turn that changes it gets.

Real repositories in ``tmp_path``, real worktrees, no model: the turn itself is
replaced by a function that does to the workspace what an agent's shell might.
"""

# The repository's type checker is pyrefly (`pyrefly check agentic`). Editors
# using pyright instead report `Config(...)` and `ProviderConfig(...)` as missing
# the fields that have defaults, because pydantic builds `__init__` and a checker
# without its plugin cannot see that. Scoped to this file, which builds them.
# pyright: reportCallIssue=false
from __future__ import annotations

import asyncio
import subprocess
from typing import TYPE_CHECKING
from typing import Any

import pytest

from agentic.core import git
from agentic.core import worktree
from agentic.core.config import Config
from agentic.core.config import ModelRef

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from agentic.core.lifecycle import Run
    from agentic.core.step import Task
    from agentic.core.step import Worker


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    (repo / "b.txt").write_text("bee\n")
    (repo / ".gitignore").write_text("ignored/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _worktrees(repo: Path) -> int:
    return _git(repo, "worktree", "list", "--porcelain").count("worktree ")


class TestSnapshot:
    def test_it_holds_the_tree_as_it_is_now(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / "a.txt").write_text("two\n")  # unstaged
        (repo / "b.txt").write_text("staged\n")
        _git(repo, "add", "b.txt")
        (repo / "new").mkdir()
        (repo / "new" / "c.txt").write_text("untracked\n")
        (repo / "ignored").mkdir()
        (repo / "ignored" / "junk").write_text("x\n")
        (repo / "log").mkdir()
        (repo / "log" / "out").write_text("run output\n")
        before = _git(repo, "status", "--porcelain")

        with worktree.snapshot(repo, ignore=(repo / "log",)) as ws:
            assert ws.path.resolve() != repo.resolve()
            assert (ws.path / "a.txt").read_text() == "two\n"
            assert (ws.path / "b.txt").read_text() == "staged\n"
            assert (ws.path / "new" / "c.txt").read_text() == "untracked\n"
            # Not part of the tree: ignored files and the run's own output.
            assert not (ws.path / "ignored").exists()
            assert not (ws.path / "log").exists()
            # Uncommitted there as here, so `git diff HEAD` shows the change.
            assert "a.txt" in _git(ws.path, "diff", "HEAD", "--name-only")
            assert _git(ws.path, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
            assert not ws.changed()
            path = ws.path

        assert not path.exists()
        assert _worktrees(repo) == 1
        assert _git(repo, "status", "--porcelain") == before

    def test_a_write_is_seen_and_a_rebuild_undoes_it(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        with worktree.snapshot(repo) as ws:
            (ws.path / "a.txt").write_text("edited by an agent\n")
            assert ws.changed()
            ws.rebuild()
            assert (ws.path / "a.txt").read_text() == "one\n"
            assert not ws.changed()
            (ws.path / "stray.txt").write_text("new file\n")
            assert ws.changed()
        assert (repo / "a.txt").read_text() == "one\n"
        assert not (repo / "stray.txt").exists()

    def test_the_environment_is_linked_not_copied(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / ".venv").mkdir()
        (repo / ".venv" / ".gitignore").write_text("*\n")
        with worktree.snapshot(repo) as ws:
            assert (ws.path / ".venv").is_symlink()
            assert (ws.path / ".venv").resolve() == (repo / ".venv").resolve()

    def test_an_untracked_environment_is_linked_not_copied(
        self, tmp_path: Path
    ) -> None:
        # Only gitignored `.venv` was covered, and this repository's root
        # .gitignore lists `venv`, not `.venv`: the untracked case copied a
        # multi-gigabyte environment into every reader's snapshot and then hashed
        # it after every turn. `ENVIRONMENT_PATHS` is what keeps it out of the
        # untracked-file list, and the link is what puts it back.
        repo = _repo(tmp_path)
        (repo / ".venv" / "bin").mkdir(parents=True)
        (repo / ".venv" / "bin" / "python").write_text("")
        (repo / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
        with worktree.snapshot(repo) as ws:
            assert (ws.path / ".venv").is_symlink()
            assert ws.digest()  # hashing the snapshot must not walk the venv


def _open_run(repo: Path) -> Run:
    from agentic.core.lifecycle import Run
    from agentic.core.lifecycle import RunSummary
    from agentic.core.report import RunLog

    return Run(
        config=Config(providers={}, roles={}),
        repo=repo,
        log=RunLog.create(repo, command="t"),
        summary=RunSummary(),
        timeout=60.0,
        usage_limits=None,
    )


class TestTurnInASnapshot:
    """A reader's answer is recorded at once -- unless it changed its snapshot."""

    def _worker_and_task(self, run: Run, repo: Path) -> tuple[Any, Any]:
        pytest.importorskip("pydantic_ai_harness")
        from agentic.core.step import Task
        from agentic.core.step import Worker
        from agentic.workflows.review import workflow as agent_mod

        # The run makes the reader's tree itself, from the task it is given.
        step = agent_mod.reviewer_step(skills_dirs=())
        worker = Worker(
            index=0,
            name="r",
            ref=ModelRef(provider="p", model="m"),
            step=step,
            definition=step.definition(),
            model=None,  # type: ignore[arg-type]  # the turn is replaced
            digest="def",
        )
        return worker, Task(name="u", prompt="review this", inputs={"change": "d"})

    def _fake_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        effect: Callable[[], object] | None = None,
    ) -> None:
        from agentic.core import lifecycle
        from agentic.core import steps as steps_mod
        from agentic.core.turn import TurnResult

        async def run_turn(*_: object, **__: object) -> TurnResult:
            if effect is not None:
                effect()
            return TurnResult(
                result="",
                elapsed=1.0,
                usage=None,
                raw=b"[]",
                steps=steps_mod.StepRecorder(),
                event="ok",
            )

        monkeypatch.setattr(lifecycle, "run_turn", run_turn)

    def test_a_clean_turn_leaves_its_evidence_when_it_ends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        repo = _repo(tmp_path)
        run = _open_run(repo)
        with run.workspaces:
            worker, task = self._worker_and_task(run, repo)
            self._fake_turn(monkeypatch)
            outcome = asyncio.run(run.turn(worker, task))
        assert not outcome.failed
        for path in (outcome.outcome_path, outcome.messages_path, outcome.steps_path):
            assert path is not None and path.is_file()

    def test_a_reader_turn_reaps_the_jobs_it_leaves_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader has nothing that should outlive its turn.

        A shell job that outlived its call kept a core busy for 271 s of one
        scan; a reader's tree is rebuilt from the same checkout every turn, so
        nothing done in it is worth keeping alive.
        """
        from agentic.core import worktree as worktree_mod

        reaped: list[str] = []
        monkeypatch.setattr(
            worktree_mod, "reap_jobs", lambda path: reaped.append(str(path))
        )
        repo = _repo(tmp_path)
        run = _open_run(repo)
        with run.workspaces:
            worker, task = self._worker_and_task(run, repo)
            self._fake_turn(monkeypatch)
            asyncio.run(run.turn(worker, task))
            # Reaped when the turn ended, not when the workspace closed.
            assert reaped == [str(run.workspace_for(worker, task).path)]
        assert len(reaped) == 2  # and again by the teardown

    def test_a_writer_turn_keeps_its_jobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Its tree holds the work: a build it backgrounded must survive the turn.
        from agentic.core import worktree as worktree_mod
        from agentic.core.step import Task
        from agentic.core.step import Worker
        from agentic.workflows.improve import pipeline as pipeline_mod
        from agentic.workflows.improve.state import PREFIX

        reaped: list[str] = []
        monkeypatch.setattr(
            worktree_mod, "reap_jobs", lambda path: reaped.append(str(path))
        )
        repo = _repo(tmp_path)
        run = _open_run(repo)
        step = pipeline_mod._step("write", skills_dirs=())
        writer = Worker(
            index=0,
            name="writer",
            ref=ModelRef(provider="p", model="m"),
            step=step,
            definition=step.definition(),
            model=None,  # type: ignore[arg-type]  # the turn is replaced
            digest="def",
        )
        task = Task(
            name="write",
            prompt="make it",
            inputs={},
            branch=f"{PREFIX}keeps-jobs",
            base=git.head_sha(cwd=repo),
        )
        self._fake_turn(monkeypatch)
        with run.workspaces:
            asyncio.run(run.turn(writer, task))
            assert reaped == [], "a writer's backgrounded job must survive its turn"

    def test_a_turn_that_changed_its_snapshot_gets_no_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        repo = _repo(tmp_path)
        run = _open_run(repo)
        with run.workspaces:
            worker, task = self._worker_and_task(run, repo)
            workspace = run.workspace_for(worker, task)
            target = workspace.path / "a.txt"
            self._fake_turn(monkeypatch, lambda: target.write_text("reformatted\n"))
            outcome = asyncio.run(run.turn(worker, task))
            # Rebuilt for the worker's next turn.
            assert target.read_text() == "one\n"
            assert not workspace.changed()
        assert outcome.failed
        assert "changed its workspace" in str(outcome.error)
        # What the agent said is still in the evidence.
        assert outcome.outcome_path is not None and outcome.outcome_path.is_file()
        assert (repo / "a.txt").read_text() == "one\n"


class TestReaderWorkspace:
    """A reader gets the tree its change ends at, not the caller's checkout."""

    def test_a_revision_is_checked_out_rather_than_copied(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / "a.txt").write_text("two\n")
        _git(repo, "commit", "-qam", "second")
        tip = _git(repo, "rev-parse", "HEAD").strip()
        (repo / "a.txt").write_text("three, uncommitted\n")

        with worktree.reader_workspace(repo, revision=tip) as workspace:
            # The commit under review, not the caller's uncommitted work.
            assert (workspace.path / "a.txt").read_text() == "two\n"
            assert _git(workspace.path, "rev-parse", "HEAD").strip() == tip
            assert not workspace.changed()
            # A checkout of a revision is writable like any worktree, so a write
            # is detected the same way: a verdict must not be recorded against a
            # tree the reviewer edited.
            (workspace.path / "a.txt").write_text("scratch\n")
            assert workspace.changed()
            workspace.rebuild()
            assert (workspace.path / "a.txt").read_text() == "two\n"
            assert _git(workspace.path, "rev-parse", "HEAD").strip() == tip
            assert not workspace.changed()

    def test_a_revision_workspace_links_the_shared_environment(
        self, tmp_path: Path
    ) -> None:
        repo = _repo(tmp_path)
        (repo / ".venv" / "lib").mkdir(parents=True)
        (repo / ".venv" / "lib" / "marker.txt").write_text("installed\n")
        (repo / ".gitignore").write_text("ignored/\n.venv\n")
        _git(repo, "add", ".gitignore")
        _git(repo, "commit", "-qm", "ignore the venv")
        tip = _git(repo, "rev-parse", "HEAD").strip()

        with worktree.reader_workspace(repo, revision=tip) as workspace:
            linked = workspace.path / ".venv" / "lib" / "marker.txt"
            assert linked.is_file()
            assert linked.read_text() == "installed\n"
            assert not workspace.changed()

    def test_no_revision_is_a_copy_of_the_working_tree(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / "a.txt").write_text("uncommitted\n")
        with worktree.reader_workspace(repo) as workspace:
            assert (workspace.path / "a.txt").read_text() == "uncommitted\n"
            (workspace.path / "a.txt").write_text("edited\n")
            assert workspace.changed()


class TestWorkspaceForATask:
    """The tree is what the task is about: one agent, a workspace per task."""

    class Definition:
        """Just enough of a `Definition` for a worker to be built."""

        names: tuple[str, ...] = ()

        def digest(self) -> str:
            return "d"

    def _worker(self) -> Worker[str]:
        from agentic.core.step import Step
        from agentic.core.step import Worker

        return Worker(
            index=0,
            name="r",
            ref=ModelRef(provider="p", model="m"),
            step=Step(
                kind="test",
                instructions="",
                output_type=str,
            ),
            model=None,  # type: ignore[arg-type]  # never asked for one
            definition=self.Definition(),  # type: ignore[arg-type]  # a stand-in
            digest="d",
        )

    def _task(self, revision: str | None) -> Task:
        from agentic.core.step import Task

        return Task(
            name="u", prompt="review this", inputs={"change": "d"}, revision=revision
        )

    def test_a_task_with_a_revision_gets_that_checkout(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / "a.txt").write_text("two\n")
        _git(repo, "commit", "-qam", "second")
        tip = _git(repo, "rev-parse", "HEAD").strip()
        (repo / "a.txt").write_text("three, uncommitted\n")

        run = _open_run(repo)
        worker = self._worker()
        task = self._task(tip)
        with run.workspaces:
            workspace = run.workspace_for(worker, task)
            # The commit under review, not the caller's uncommitted work.
            assert (workspace.path / "a.txt").read_text() == "two\n"
            assert _git(workspace.path, "rev-parse", "HEAD").strip() == tip
            # One workspace per (worker, task): a writer keeps its own across
            # turns, and a reader is never rebuilt under a running turn.
            assert run.workspace_for(worker, task) is workspace

    def test_a_task_without_a_revision_reads_a_copy(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / "a.txt").write_text("uncommitted\n")
        run = _open_run(repo)
        with run.workspaces:
            workspace = run.workspace_for(self._worker(), self._task(None))
            assert (workspace.path / "a.txt").read_text() == "uncommitted\n"


class TestLeftoverShellJobs:
    def test_a_job_left_in_the_workspace_is_stopped_with_it(
        self, tmp_path: Path
    ) -> None:
        """A command runs detached from the run, by the harness's design.

        Nothing else ends it: a cancelled turn left a scan crawling at a full
        core for an hour, writing into a log in the directory the teardown was
        about to delete. The workspace's job directory names the process and its
        group, so the teardown that deletes the tree also stops what is running
        in it.
        """
        job = subprocess.Popen(
            ["sleep", "120"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            directory = tmp_path / worktree._JOBS / "job1"
            directory.mkdir(parents=True)
            # The harness's handle file: `<supervisor pid> <group or ->`.
            (directory / "handle").write_text(f"{job.pid} {job.pid}\n")

            worktree.reap_jobs(tmp_path)

            job.wait(timeout=10)
            assert job.returncode is not None
        finally:
            if job.poll() is None:  # pragma: no cover - only on a failed reap
                job.kill()
                pytest.fail("the job was still running after the workspace went")

    def test_a_workspace_without_jobs_is_untouched(self, tmp_path: Path) -> None:
        worktree.reap_jobs(tmp_path)  # no `.pydantic-ai-harness` at all
        (tmp_path / "file.txt").write_text("keep me\n")
        worktree.reap_jobs(tmp_path)
        assert (tmp_path / "file.txt").read_text() == "keep me\n"


class TestStaleWorkspaces:
    def test_only_old_directories_with_our_prefixes_are_swept(
        self, tmp_path: Path
    ) -> None:
        """A killed run leaves its tree behind, and nothing else comes back for it.

        `kill -9`, a crashed interpreter or a reboot leaves ~100 MB in the
        temporary directory plus a worktree entry in the repository it came from.
        A live run of any length is left alone: the sweep is by age, not by name.
        """
        import os
        import time

        stale = tmp_path / "agentic-worktree-stale"
        fresh = tmp_path / "agentic-worktree-live"
        other = tmp_path / "someone-elses-dir"
        for directory in (stale, fresh, other):
            directory.mkdir()
        old = time.time() - (worktree.STALE_HOURS + 1) * 3600
        for directory in (stale, other):
            os.utime(directory, (old, old))

        gone = worktree.sweep(older_than_hours=worktree.STALE_HOURS, root=tmp_path)

        assert gone == [stale]
        assert not stale.exists()
        assert fresh.exists()  # a live run, whatever its age in names
        assert other.exists()  # not ours to remove
