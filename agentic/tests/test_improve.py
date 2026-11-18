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
from agentic.workflows.improve.state import slug

if TYPE_CHECKING:
    from pathlib import Path

    from agentic.core.config import Config
    from agentic.core.config import ProviderConfig
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

    def test_only_the_writer_commits_and_is_not_reusable(self) -> None:
        writer = pipeline_mod._step("write", skills_dirs=(), reusable=False)
        # The writer commits with its own shell: no tool of ours is bound to it,
        # and its answer is a side effect, so it is never served from a record.
        assert writer.kind == "improve_write"
        assert writer.reusable is False

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
        summary=RunSummary(environment="env"),
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
                pipeline_mod._step("write", skills_dirs=(), reusable=False),
            )
        ]
    )
    return run, worker


def _turn(answer: str, *, error: str | None = None) -> TurnResult:
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
    )


def _fake_turn(
    monkeypatch: pytest.MonkeyPatch, answer: str, *, error: str | None = None
) -> None:
    async def fake(*args: object, **kwargs: object) -> TurnResult:
        return _turn(answer, error=error)

    monkeypatch.setattr("agentic.core.lifecycle.run_turn", fake)


class TestTheSettings:
    def test_the_loop_is_bounded_by_a_ceiling_not_a_policy(self) -> None:
        assert ImproveSettings().max_rounds >= 1
