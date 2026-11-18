# pyright: reportCallIssue=false
# `Config(...)` and friends take pydantic fields with defaults; a checker without
# pydantic's plugin reads every `Field(_default, ...)` as required. The repo's own
# type gate does not cover `agentic/`; silencing only the call check keeps real
# type errors visible.

"""The improve workflow: the steps, the gate, the branch, and one delegation.

Four steps answer prose and one (the review) answers a contract; what is tested
here is the part a person would otherwise have to remember: that a step which
does not answer stops the run, that the branch name comes from the chooser, that
`scan` and `choose` are different models, that a writer's branch outlives its
worktree, and that `Run.step` behaves. The work itself is a model's.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING
from typing import Any

import pytest

from agentic.core import git
from agentic.core.lifecycle import Run
from agentic.core.lifecycle import RunSummary
from agentic.core.lifecycle import StepFailed
from agentic.core.report import RunLog
from agentic.core.step import Task
from agentic.tests.git_support import _git_repo
from agentic.workflows.improve import pipeline as pipeline_mod
from agentic.workflows.improve import record as record_mod
from agentic.workflows.improve.state import PREFIX
from agentic.workflows.improve.state import ImproveSettings
from agentic.workflows.improve.state import Round
from agentic.workflows.improve.state import Summary
from agentic.workflows.improve.state import Workers
from agentic.workflows.improve.state import slug

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from agentic.core.config import Config
    from agentic.core.config import ProviderConfig
    from agentic.core.step import Outcome
    from agentic.core.step import Worker
    from agentic.core.turn import TurnResult
    from agentic.workflows.improve.state import ImproveOptions


class TestTheBranch:
    def test_the_branch_is_the_choosers_first_line(self) -> None:
        choice = "Drop the second Settings validation!\n\nIt is real because ..."
        assert f"{PREFIX}{slug(choice)}" == (
            "agentic/improve/drop-the-second-settings-validation"
        )

    def test_a_short_or_odd_name_still_yields_a_branch(self) -> None:
        assert slug("") == "change"
        assert slug("a/b:c") == "a-b-c"
        assert len(slug("x" * 200)) <= 48

    def test_an_existing_branch_is_a_claim(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path)
        branch = f"{PREFIX}repeated-validation"
        assert not git.branch_exists(branch, cwd=repo)
        git.git("branch", branch, cwd=repo)
        assert git.branch_exists(branch, cwd=repo)

    def test_in_flight_branches_are_only_this_namespace(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path)
        git.git("branch", f"{PREFIX}one", cwd=repo)
        git.git("branch", "someone-else/two", cwd=repo)
        assert pipeline_mod._in_flight(repo) == [f"{PREFIX}one"]


class TestTheGate:
    def test_a_delivered_branch_passes(self) -> None:
        summary = Summary(branch=f"{PREFIX}x", commits=["abc fix"])
        assert summary.delivered
        assert summary.result_code() == 0

    def test_nothing_worth_doing_is_success(self) -> None:
        assert Summary(exhausted=True, stop="nothing to do").result_code() == 0

    def test_a_failed_step_is_incomplete(self) -> None:
        assert Summary(stop="the scan did not answer").result_code() == 3

    def test_a_refused_branch_is_incomplete(self) -> None:
        summary = Summary(branch=f"{PREFIX}x", stop="review refused")
        assert not summary.delivered
        assert summary.result_code() == 3


class TestThePipelineIsAPipeline:
    """Level 3: ordinary async code, no graph engine and no bespoke DSL."""

    def test_the_run_is_a_coroutine_function(self) -> None:
        assert inspect.iscoroutinefunction(pipeline_mod.run_improve)

    def test_nothing_in_the_workflow_imports_a_graph_engine(self) -> None:
        for module in (pipeline_mod, record_mod):
            assert "pydantic_graph" not in inspect.getsource(module)

    def test_the_steps_run_in_order(self) -> None:
        source = inspect.getsource(pipeline_mod._hand_off)
        order = [
            source.index('name="scan"'),
            source.index('name="choose"'),
            source.index('name="plan"'),
            source.index('name="write"'),
            source.index("_review("),
        ]
        assert order == sorted(order)


class TestTheSteps:
    def test_four_steps_answer_prose(self) -> None:
        # The next model reads them; only the review's answer is acted on by code.
        for name in ("scan", "choose", "plan", "write"):
            step = pipeline_mod._step(name, skills_dirs=())
            assert step.output_type is str
            assert step.instructions.strip()

    def test_only_the_writer_commits(self) -> None:
        writer = pipeline_mod._step("write", skills_dirs=())
        # The writer commits with its own shell: no tool of ours is bound to it,
        # and the branch is the result the run keeps.
        assert writer.kind == "improve_write"

    def test_every_step_is_the_harness_agent_with_its_own_policy(self) -> None:
        pytest.importorskip("pydantic_ai")
        pytest.importorskip("pydantic_ai_harness")

        digests = {
            pipeline_mod._step(name, skills_dirs=()).definition().digest()
            for name in ("scan", "choose", "plan", "write")
        }
        assert len(digests) == 4
        assert "Coder" in pipeline_mod._step("scan", skills_dirs=()).definition().names


class TestTheStep:
    """`Run.step`: the plumbing a pipeline should not repeat, tested once."""

    def test_a_failed_answer_stops_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _git_repo(tmp_path)
        run, worker = _writer(repo)
        _fake_turn(monkeypatch, "did it", error="the provider fell over")
        with run.workspaces, pytest.raises(StepFailed) as raised:
            asyncio.run(run.step(worker, name="write", prompt="make the change"))
        assert raised.value.name == "writer"
        assert "provider fell over" in str(raised.value)

    def test_continuing_reuses_the_task_and_the_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _git_repo(tmp_path)
        run, worker = _writer(repo)
        seen: list[dict[str, object]] = []

        async def fake(*args: object, **kwargs: object) -> TurnResult:
            seen.append(kwargs)
            return _turn("did it")

        monkeypatch.setattr("agentic.core.lifecycle.run_turn", fake)
        with run.workspaces:
            first = asyncio.run(run.step(worker, name="write", prompt="make it"))
            assert first.task is not None and first.tree is not None
            asyncio.run(
                run.step(worker, name="write", continue_from=first, prompt="fix it")
            )
        assert seen[1]["message_history"] == first.history
        assert first.task is not None


class TestAFailedTurnSays:
    """What the record keeps of a turn that died.

    `str(exc)` of an SDK transport error is "Connection error.", which cannot say
    whether the endpoint or this side dropped the stream -- so the exception and
    what raised it are kept, and the run stops on one attempt exactly as before.
    """

    def test_the_record_says_what_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _git_repo(tmp_path)
        run, worker = _writer(repo)
        base = git.head_sha(cwd=repo)
        cut = OSError("peer closed the connection")
        cut.__cause__ = RuntimeError("Connection error.")
        _fake_turn(monkeypatch, "", error="Connection error.", failure=cut)
        with run.workspaces, pytest.raises(StepFailed) as raised:
            asyncio.run(
                run.step(
                    worker,
                    name="write",
                    prompt="make it",
                    branch=f"{PREFIX}says-what-raised",
                    base=base,
                )
            )
        assert str(raised.value) == (
            "writer: OSError: peer closed the connection"
            " <- RuntimeError: Connection error."
        )

    def test_the_turn_is_run_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No retry mechanic: a failed turn stops the run, and says so.
        repo = _git_repo(tmp_path)
        run, worker = _writer(repo)
        base = git.head_sha(cwd=repo)
        attempted: list[str] = []

        async def fake(*args: object, **kwargs: object) -> TurnResult:
            attempted.append(str(kwargs["steps_path"]))
            return _turn("", error="Connection error.", failure=OSError("cut"))

        monkeypatch.setattr("agentic.core.lifecycle.run_turn", fake)
        with run.workspaces, pytest.raises(StepFailed):
            asyncio.run(
                run.step(
                    worker,
                    name="write",
                    prompt="make it",
                    branch=f"{PREFIX}runs-once",
                    base=base,
                )
            )
        assert len(attempted) == 1
        assert "OSError: cut" in "\n".join(run.log.timeline)


class TestAStackIsReviewedCommitByCommit:
    """A branch may carry a stack, and the unit of review is one commit.

    That is the unit a maintainer approves and what `stack-pr` submits, so the
    verdict has to be attributable to a commit -- and a commit whose sha did not
    move keeps the verdict it already earned.
    """

    def _stack_repo(self, tmp_path: Path) -> tuple[Path, str]:
        repo = _git_repo(tmp_path)
        base = git.head_sha(cwd=repo)
        run_git = _git_runner(repo)
        for name in ("second", "third"):
            (repo / "a.txt").write_text(f"{name}\n")
            run_git("commit", "-q", "-am", name)
        return repo, base

    def _workers(self) -> Workers:
        # The review workers are only built when absent; a sentinel is enough
        # because `review_range` is replaced below.
        return Workers(
            scan=None,
            choose=None,
            plan=None,
            write=None,
            reviewers=[object()],
            summarizer=object(),
        )

    def test_the_stack_is_one_review_per_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentic.workflows.review.workflow import Verdict

        repo, base = self._stack_repo(tmp_path)
        calls: list[tuple[str, str]] = []

        async def fake_review_range(
            *args: object, **kwargs: object
        ) -> tuple[list[object], Outcome[Any]]:
            calls.append((str(kwargs["base"]), str(kwargs["head"])))
            return [], _outcome(Verdict(blocking=False, text="clean"))

        monkeypatch.setattr(
            "agentic.workflows.review.workflow.review_range", fake_review_range
        )
        run = _run(repo)
        with run.workspaces:
            verdicts = asyncio.run(
                pipeline_mod._review(
                    run, _config(), ImproveSettings(), self._workers(), base, repo
                )
            )
        assert len(verdicts) == 2, verdicts
        assert [head for _, head in calls] == [v.commit for v in verdicts]
        # The first commit is reviewed against base, the second against the first.
        assert calls[0][0] == base
        assert calls[1][0] == verdicts[0].commit
        assert not pipeline_mod._blocking(verdicts)

    def test_an_unmoved_commit_keeps_its_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentic.workflows.review.workflow import Verdict

        repo, base = self._stack_repo(tmp_path)
        calls: list[str] = []

        async def fake_review_range(
            *args: object, **kwargs: object
        ) -> tuple[list[object], Outcome[Any]]:
            calls.append(str(kwargs["head"]))
            return [], _outcome(Verdict(blocking=False, text="clean"))

        monkeypatch.setattr(
            "agentic.workflows.review.workflow.review_range", fake_review_range
        )
        run = _run(repo)
        with run.workspaces:
            first = asyncio.run(
                pipeline_mod._review(
                    run, _config(), ImproveSettings(), self._workers(), base, repo
                )
            )
            again = asyncio.run(
                pipeline_mod._review(
                    run,
                    _config(),
                    ImproveSettings(),
                    self._workers(),
                    base,
                    repo,
                    seen=first,
                )
            )
        assert len(calls) == 2, "the second pass re-reviewed an unchanged commit"
        assert [v.commit for v in again] == [v.commit for v in first]
        assert "kept its verdict" in "\n".join(run.log.timeline)

    def test_a_blocked_commit_blocks_the_stack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentic.workflows.review.workflow import Verdict

        repo, base = self._stack_repo(tmp_path)
        verdicts = iter(
            [
                Verdict(blocking=False, text="clean"),
                Verdict(blocking=True, text="a blocker in the second commit"),
            ]
        )

        async def fake_review_range(
            *args: object, **kwargs: object
        ) -> tuple[list[object], Outcome[Any]]:
            return [], _outcome(next(verdicts))

        monkeypatch.setattr(
            "agentic.workflows.review.workflow.review_range", fake_review_range
        )
        run = _run(repo)
        with run.workspaces:
            reviewed = asyncio.run(
                pipeline_mod._review(
                    run, _config(), ImproveSettings(), self._workers(), base, repo
                )
            )
        assert pipeline_mod._blocking(reviewed)
        text = pipeline_mod._text(reviewed)
        assert "a blocker in the second commit" in text
        # Each section names its commit, so the writer knows which one to amend.
        for item in reviewed:
            assert item.commit in text


class TestTheWriterTree:
    def test_the_branch_outlives_the_worktree(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path)
        base = git.head_sha(cwd=repo)
        run, worker = _writer(repo, base)
        task = Task(
            name="write",
            prompt="make the change",
            inputs={"x": "y"},
            branch=f"{PREFIX}drop-it",
            base=base,
        )
        with run.workspaces:
            workspace = run.workspace_for(worker, task)
            assert workspace.writable
            assert workspace.path != repo
            assert git.head_sha(cwd=workspace.path) == base
        # The branch is the result and the claim, so it survives the worktree.
        assert git.branch_exists(f"{PREFIX}drop-it", cwd=repo)
        assert not workspace.path.exists()


class TestResolveBase:
    def test_an_explicit_base_is_honoured(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path)
        first = git.git("rev-list", "--max-parents=0", "HEAD", cwd=repo).strip()
        assert pipeline_mod.resolve_base(repo, first) == first

    def test_without_upstream_it_falls_back_to_head(self, tmp_path: Path) -> None:
        repo = _git_repo(tmp_path)
        assert pipeline_mod.resolve_base(repo, None) == git.head_sha(cwd=repo)


class TestRecords:
    def test_the_record_carries_the_prose_and_the_branch(self) -> None:
        summary = Summary(
            scan="1. drop it -- because",
            choice="Drop it\n\nIt is real.",
            plan="Goal: drop it",
            branch=f"{PREFIX}drop-it",
            base="abc",
            commits=["abc123 drop it"],
            models={"scan": "gw/a", "choose": "gw/b"},
            rounds=[Round(label="attempt-1", changed="dropped it", verdict="clean")],
        )
        record = pipeline_mod.REPORT.record(summary)
        assert record["branch"] == f"{PREFIX}drop-it"
        assert record["delivered"] is True
        assert record["choice"].startswith("Drop it")
        text = pipeline_mod.REPORT.describe(summary).text
        assert f"{PREFIX}drop-it" in text


class TestEndToEnd:
    def test_a_run_records_and_gates_on_what_happened(self, tmp_path: Path) -> None:
        """The wiring, on the `test` provider: no network, no credential."""
        pytest.importorskip("pydantic_ai")
        from agentic.core.config import ProviderConfig

        repo = _git_repo(tmp_path)
        config = _config(
            ProviderConfig(kind="test"),
            workflows={"improve": {"scan": "one", "choose": "two"}},
        )
        summary = asyncio.run(pipeline_mod.run_improve(config, _options(), repo))
        assert summary.log_dir is not None
        assert summary.exit_code() in (0, 3)
        assert (summary.log_dir / "run.json").is_file()
        assert (summary.log_dir / "index.md").is_file()
        # The scan reads the tree the branch is cut from, not the caller's
        # checkout: a plan naming a file only HEAD has is unbuildable.
        from agentic.core import git

        base = pipeline_mod.resolve_base(repo, None)
        assert base == git.head_sha(cwd=repo)
        assert f"scan: reading {base}" in (summary.log_dir / "run.log").read_text()


# -- helpers -----------------------------------------------------------------


def _config(
    provider: ProviderConfig | None = None, *, workflows: dict[str, Any] | None = None
) -> Config:
    from agentic.core.config import Config
    from agentic.core.config import ProviderConfig as Provider

    return Config(
        providers={"p": provider or Provider(kind="test")},
        roles={"one": "p/a", "two": "p/b"},
        default_role="one",
        workflows=workflows or {},
    )


def _options() -> ImproveOptions:
    from agentic.workflows.improve.state import ImproveOptions

    return ImproveOptions()


def _run(repo: Path) -> Run:
    return Run(
        config=_config(),
        repo=repo,
        log=RunLog.create(repo, command="t"),
        summary=RunSummary(),
        timeout=60.0,
        usage_limits=None,
    )


def _writer(repo: Path, base: str | None = None) -> tuple[Run, Worker[str]]:
    run = _run(repo)
    (worker,) = run.build_workers(
        [
            (
                "writer",
                _config().resolve("one"),
                pipeline_mod._step("write", skills_dirs=()),
            )
        ]
    )
    return run, worker


def _git_runner(repo: Path) -> Callable[..., None]:
    import subprocess

    def run(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )

    return run


def _outcome(result: object) -> Outcome[object]:
    from agentic.core.step import Outcome as Value

    return Value(name="t", model="p/m", result=result, elapsed_seconds=0.1)


def _turn(
    answer: str,
    *,
    error: str | None = None,
    failure: Exception | None = None,
) -> TurnResult:
    from agentic.core import steps as steps_mod
    from agentic.core.turn import TurnResult

    return TurnResult(
        result=answer,
        elapsed=1.0,
        usage=None,
        raw=b"[]",
        steps=steps_mod.StepRecorder(),
        event=error or "ok",
        history=[],
        failure=failure,
    )


def _fake_turn(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    *,
    error: str | None = None,
    failure: Exception | None = None,
) -> None:
    async def fake(*args: object, **kwargs: object) -> TurnResult:
        return _turn(answer, error=error, failure=failure)

    monkeypatch.setattr("agentic.core.lifecycle.run_turn", fake)


class TestTheSettings:
    def test_the_loop_is_bounded_by_a_ceiling_not_a_policy(self) -> None:
        assert ImproveSettings().max_rounds >= 1
