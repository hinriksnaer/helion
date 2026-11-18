"""The policy files describe *this* harness, and these tests check that they do.

The defect class this guards against is the one that kept happening: an
instruction file, a README line or the example config describing a tool, an
answer field or a settings key that the code no longer has. It is invisible to
every other test -- the code is right, the prose is stale -- and it costs a model
turn per occurrence, because the model reads the prose and tries to do what it
says.

Four checks, each one cheap and each one for a real incident:

- every instruction file is read by a step (``rank.md`` and ``scope.md`` were
  read by nothing);
- every step has an instruction file, so a rename cannot leave a step without a
  policy;
- no instruction file names a tool its agent is not bound to (``write.md`` told
  the writer to commit with ``git_commit`` for two days);
- the shipped example config validates against the settings it documents
  (``max_diff_chars`` in ``[workflows.review]`` was rejected by the model).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

import pytest
import tomllib

from agentic.workflows.improve import pipeline as pipeline_mod
from agentic.workflows.improve.state import ImproveSettings
from agentic.workflows.review import workflow as review_mod
from agentic.workflows.review.workflow import ReviewSettings

if TYPE_CHECKING:
    from agentic.core.step import Step

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every tool a step's agent may name in prose. A backticked call to anything
#: else is a policy describing a tool that does not exist.
_KNOWN_TOOLS = frozenset(
    {
        "check_command",
        "delegate_task",
        "edit_file",
        "grep",
        "list_files",
        "load_capability",
        "read_file",
        "shell",
        "start_command",
        "write_file",
    }
)

#: `` `name(` `` -- a tool call, as opposed to a command or a file in backticks.
_CALL = re.compile(r"`([a-z_]+)\(")


def _steps() -> dict[str, Step[str]]:
    """Every improve step, by the instruction file it is given."""
    return {name: pipeline_mod._step(name, skills_dirs=()) for name in _STEP_NAMES}


_STEP_NAMES = ("scan", "choose", "plan", "write")


def test_every_instruction_file_is_read_by_a_step() -> None:
    directory = _REPO_ROOT / "agentic" / "workflows" / "improve" / "instructions"
    files = {path.stem for path in directory.glob("*.md")}
    assert files == set(_STEP_NAMES), (
        f"instruction files without a step: {sorted(files - set(_STEP_NAMES))}; "
        f"steps without a file: {sorted(set(_STEP_NAMES) - files)}"
    )


def test_no_step_names_a_tool_it_does_not_have() -> None:
    pytest.importorskip("pydantic_ai")
    pytest.importorskip("pydantic_ai_harness")
    from pydantic_ai.models.test import TestModel

    from agentic.core import stack

    for name, step in _steps().items():
        mentioned = set(_CALL.findall(step.instructions))
        assert mentioned <= _KNOWN_TOOLS, (
            f"{name}.md names {sorted(mentioned - _KNOWN_TOOLS)}, which no step has"
        )
        # And the ones it does name are really bound to its agent.
        agent = stack.build_agent(
            TestModel(), step.definition(), step.output_type, None
        )
        bound = _tool_names(agent)
        missing = mentioned - bound
        assert not missing, f"{name}.md names {sorted(missing)}, not bound on its agent"
        assert "delegate_task" in bound, "every step keeps the harness's delegation"


def _tool_names(agent: object) -> set[str]:
    """The function tools an agent offers, walked through its wrapped toolsets."""
    found: set[str] = set()

    def walk(toolset: object) -> None:
        tools = getattr(toolset, "tools", None)
        if isinstance(tools, dict):
            found.update(tools)
        for attribute in ("wrapped", "toolset", "toolsets", "inner"):
            inner = getattr(toolset, attribute, None)
            if inner is None:
                continue
            for item in inner if isinstance(inner, (list, tuple)) else (inner,):
                walk(item)

    for toolset in getattr(agent, "toolsets", []):
        walk(toolset)
    return found


def test_the_review_policies_are_read() -> None:
    # A policy that nothing loads is a policy nobody follows.
    assert review_mod.REVIEW_INSTRUCTIONS.strip()
    assert review_mod.SUMMARY_INSTRUCTIONS.strip()
    assert "{{CHECKLIST}}" not in review_mod.REVIEW_INSTRUCTIONS


def test_the_example_config_validates() -> None:
    """The shipped example must start a run, not fail one."""
    from agentic.core.config import Config

    path = _REPO_ROOT / "agentic" / "config.example.toml"
    document = tomllib.loads(path.read_text())
    config = Config.model_validate(document)
    for name, settings in (
        ("improve", ImproveSettings),
        ("review", ReviewSettings),
    ):
        config.workflow(name, settings)
