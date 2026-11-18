# pyright: reportCallIssue=false
# `Config(...)`/`ProviderConfig(...)` take pydantic fields with defaults; a checker
# without pydantic's plugin reads every `Field(default, ...)` as required. The
# repo's type gate does not cover `agentic/`; only the call check is silenced.

"""The review workflow: what it asks, who answers, and what the gate says.

The reviewers do the reading -- they have the shell and the tree -- and a third
model reads their answers and decides. What is tested is that shape: the prompt
says *what* to review, the summary is the only typed answer, and the exit code
comes from it.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from agentic.tests.git_support import _git_repo
from agentic.tests.support import _REPO_ROOT
from agentic.workflows.review import workflow as review_mod
from agentic.workflows.review.workflow import ReviewOptions
from agentic.workflows.review.workflow import Summary
from agentic.workflows.review.workflow import Verdict
from agentic.workflows.review.workflow import summarize_prompt
from agentic.workflows.review.workflow import what_to_review

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import TypedDict

    from pydantic_ai.workspaces import LocalWorkspaceBackend

    from agentic.core.config import Config
    from agentic.core.worktree import Tree

    class TreeKwargs(TypedDict):
        workspace: LocalWorkspaceBackend
        deps: Tree


class TestWhatToReview:
    def test_no_arguments_means_the_uncommitted_tree(self) -> None:
        what = what_to_review(ReviewOptions())
        assert "uncommitted" in what
        assert "git status" in what

    def test_head_means_the_last_commit(self) -> None:
        assert "last commit" in what_to_review(ReviewOptions(head=True))

    def test_revisions_are_handed_over_as_given(self) -> None:
        what = what_to_review(ReviewOptions(revisions=["origin/main...HEAD"]))
        assert "origin/main...HEAD" in what
        assert "merge base" in what


class TestThePrompts:
    def test_a_reviewer_is_told_what_and_how_but_not_what_to_think(self) -> None:
        prompt = review_mod.review_prompt("the last commit (`git show HEAD`)")
        assert "the last commit" in prompt
        # It reads the change itself: no host-side diff is handed over.
        assert "git show" in prompt
        assert "blockers first" in prompt

    def test_the_operator_text_is_passed_verbatim(self) -> None:
        prompt = review_mod.review_prompt("the change", target="the new parser")
        assert "the new parser" in prompt

    def test_the_summariser_sees_both_reviews_and_the_failures(self) -> None:
        first = review_mod.Outcome(
            name="a",
            model="p/a",
            result="one blocker: x",
            elapsed_seconds=0,
        )
        second = review_mod.Outcome(
            name="b",
            model="p/b",
            result="",
            elapsed_seconds=0,
            error="the provider fell over",
        )
        prompt = summarize_prompt("the change", [first, second])
        assert "one blocker: x" in prompt
        assert "FAILED" in prompt
        assert "not a pass" in prompt


class TestTheVerdict:
    def test_an_empty_summary_is_thin_not_failed(self) -> None:
        # A summariser that answered nothing still answered; a turn that failed
        # is `outcome.failed`, which is what the gate reads.
        assert Verdict(blocking=False).error is None

    def test_a_clean_verdict_says_so(self) -> None:
        verdict = Verdict(blocking=False, text="nothing blocking")
        assert verdict.error is None
        assert verdict.summary_line().startswith("clean:")

    def test_a_blocking_verdict_says_so(self) -> None:
        assert Verdict(blocking=True, text="x").summary_line().startswith("blocked:")


class TestTheGate:
    def _summary(self, *, blocking: bool, failed_reviewer: bool = False) -> Summary:
        reviews = [
            review_mod.Outcome(
                name="a",
                model="p/a",
                result="ok",
                elapsed_seconds=0.0,
                error="boom" if failed_reviewer else None,
            )
        ]
        return Summary(
            reviews=reviews,
            verdict=Verdict(blocking=blocking, text="why"),
        )

    def test_a_clean_verdict_is_exit_0(self) -> None:
        assert self._summary(blocking=False).result_code() == 0

    def test_a_blocking_verdict_is_exit_2(self) -> None:
        assert self._summary(blocking=True).result_code() == 2

    def test_a_failed_reviewer_is_not_a_pass_and_not_a_blocker(self) -> None:
        summary = self._summary(blocking=False, failed_reviewer=True)
        assert summary.result_code() == 3
        assert not summary.passed()


class TestTheAgents:
    def test_reviewers_answer_prose_and_the_summariser_a_verdict(self) -> None:
        pytest.importorskip("pydantic_ai")
        pytest.importorskip("pydantic_ai_harness")

        reviewer = review_mod.reviewer_step(skills_dirs=())
        summarizer = review_mod.summarizer_step(skills_dirs=())
        assert reviewer.output_type is str
        assert summarizer.output_type is Verdict
        # Both are the harness's agent, with their own policy.
        assert "Coder" in reviewer.definition().names
        assert reviewer.definition().digest() != summarizer.definition().digest()

    def test_the_policy_is_appended_through_coder(self) -> None:
        """`Coder(instructions=...)`, not a capability beside it.

        The harness appends a `Coder`'s `instructions` to the text it writes for
        its own tools. Adding the policy as a separate capability instead put it
        first in the system prompt, ahead of the conventions those tools follow.
        """
        pytest.importorskip("pydantic_ai")
        pytest.importorskip("pydantic_ai_harness")
        from pydantic_ai.models.function import AgentInfo
        from pydantic_ai.models.function import FunctionModel

        from agentic.core import stack

        seen: dict[str, str] = {}

        async def stream(
            _messages: list[object], info: AgentInfo
        ) -> AsyncIterator[str]:
            seen["instructions"] = info.instructions or ""
            yield "reviewed"

        step = review_mod.reviewer_step(skills_dirs=())
        agent = stack.build_agent(
            FunctionModel(stream_function=stream), step.definition(), str
        )
        agent.run_sync("review it", **_tree(_REPO_ROOT))

        text = seen["instructions"]
        assert review_mod.REVIEW_INSTRUCTIONS[:60] in text
        assert text.index("You are a software engineering agent") < text.index(
            review_mod.REVIEW_INSTRUCTIONS[:60]
        )


class TestDryRun:
    def test_it_names_the_reviewers_and_the_summariser(self, tmp_path: Path) -> None:
        pytest.importorskip("pydantic_ai")
        repo = _git_repo(tmp_path)
        config = _config()
        text = review_mod.dry_run(config, ReviewOptions(head=True), repo)
        assert "reviewing:" in text
        assert "summarizer:" in text
        assert "capabilities: Coder" in text


class TestEndToEnd:
    def test_a_run_records_the_reviews_and_the_verdict(self, tmp_path: Path) -> None:
        pytest.importorskip("pydantic_ai")
        repo = _git_repo(tmp_path)
        summary = asyncio.run(
            review_mod.run_review(_config(), ReviewOptions(head=True), repo)
        )
        assert summary.log_dir is not None
        assert summary.exit_code() in (0, 2, 3)
        record = json.loads((summary.log_dir / "run.json").read_text())
        assert "reviews" in record
        assert "verdict" in record
        assert (summary.log_dir / "index.md").is_file()


# -- helpers -----------------------------------------------------------------


def _config() -> Config:
    from agentic.core.config import Config
    from agentic.core.config import ProviderConfig

    return Config(
        providers={"p": ProviderConfig(kind="test")},
        roles={"one": "p/a", "two": "p/b", "three": "p/c"},
        default_role="one",
        workflows={"review": {"agents": ["one", "two"], "summarizer": "three"}},
    )


def _tree(root: Path) -> TreeKwargs:
    from agentic.core import stack
    from agentic.core.worktree import Tree

    return {"workspace": stack.backend(root), "deps": Tree(root)}
