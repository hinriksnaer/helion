"""What every agent is: the harness's ``Coder`` with the step's own policy.

There is one baseline. Every step in every workflow runs the harness's complete
coding agent -- ``Coder`` -- in a workspace of its own, and adds only its policy
text, through the ``Coder(instructions=...)`` argument the harness documents: it
appends to the agent's own instructions, so a step never restates a capability
and never reorders what the harness says about its tools.

What ``Coder`` brings -- the file and shell tools, delegation, compaction, output
limits, the retry budget, ``RepoContext`` (AGENTS.md), ``RepairToolArguments``,
prompt caching -- is the harness's to improve, and a release that improves the
agent improves every workflow at once. Nothing here names a capability parameter:
a ``Shell(...)`` wrapper is a second copy of the harness's parameter list and
drifts from it. The one thing this module adds is what a *run* must supply, the
command environment.

The workspace is a run-time argument: ``Coder`` works in the run's workspace,
which the caller passes to ``agent.run(workspace=...)``, so nothing is composed
per tree. One agent answers every task, and a workspace that outlives a turn (a
writer's branch) needs no second agent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import TYPE_CHECKING
from typing import Any

from pydantic_ai.workspaces import LocalWorkspaceBackend

from .worktree import Tree

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.capabilities import CombinedCapability
    from pydantic_ai.models import Model

    from .config import ThinkingLevel
#: Variables a command keeps from the caller's environment. A local workspace
#: otherwise passes only ``PATH``, ``HOME``, ``LANG`` and ``LC_*``, which drops
#: the caller's GPU selection and the board's identity: an inherited
#: ``CUDA_VISIBLE_DEVICES=""`` means "no GPU" on a machine other people share,
#: and without it an agent's ``pytest`` takes a GPU nobody claimed.
#: ``LD_LIBRARY_PATH`` is here because a wheel built against the system C++ runtime
#: (a torch nightly in a nix devshell) imports only when the loader can find
#: ``libstdc++``. An empty value is forwarded as empty, because empty is a choice
#: and unset is a different one.
FORWARDED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "GPU_BOARD_DIR",
    "GPU_BOARD_SESSION",
    "LD_LIBRARY_PATH",
)

#: The marker ``scripts/autoreview.py`` refuses to run under. Set in every agent's
#: command environment: a reviewer's shell could otherwise start the incumbent
#: review with the caller's credentials, at the caller's expense, on the very
#: change it is reviewing -- and nothing about that is the reviewer's decision.
SESSION_MARKER = "HELION_AUTOREVIEW_SESSION"


def forwarded_env() -> dict[str, str]:
    """The caller's coordination variables, empty values included, plus the marker."""
    env = {name: os.environ[name] for name in FORWARDED_ENV if name in os.environ}
    env[SESSION_MARKER] = "1"
    return env


def command_env(root: Path) -> dict[str, str]:
    """The workspace's command environment: its own tree first, plus coordination.

    ``PYTHONPATH`` is what makes ``import helion`` (and any ``pytest`` an agent
    runs) the tree under work rather than the caller's editable install. Provider
    keys are not forwarded: a command must not spend one by accident.
    """
    return {**forwarded_env(), "PYTHONPATH": str(root)}


def backend(root: Path) -> LocalWorkspaceBackend:
    """The workspace one turn works in, as the run's ``workspace=`` argument.

    A *backend*, not the ``LocalWorkspace`` capability: a capability is bound to
    the agent and so to one directory, while a backend is per-run data. That is
    what lets one agent answer a task about a commit, a working tree and a
    writer's branch without being rebuilt between them.

    It carries the run's command environment (``PYTHONPATH``, the caller's GPU
    selection and board identity). Provider keys are not forwarded: a command
    must not spend one by accident.
    """
    return LocalWorkspaceBackend(root, env=command_env(root))


@dataclass(frozen=True)
class Definition:
    """One agent's composition: what the harness is given, and what a record names.

    ``capabilities`` is built by :meth:`agentic.core.step.Step.definition`, the
    one place a policy is attached. ``names`` is the short list the record carries
    -- what this package adds, not ``Coder``'s internals -- and :meth:`digest` is
    over the policy, so the record says which text produced an answer.
    """

    capabilities: CombinedCapability[Any]
    names: tuple[str, ...]
    instructions: str

    def digest(self) -> str:
        return definition_digest(
            instructions=self.instructions,
            material={"capabilities": list(self.names)},
        )


def build_agent(
    model: Model,
    definition: Definition,
    output_type: type[Any],
    thinking: ThinkingLevel | None = None,
) -> Agent[Any, Any]:
    """An agent that is ``definition``, answering with ``output_type``.

    ``thinking`` is the role's effort level from the config, passed as the
    harness's unified ``thinking`` setting so one line covers every provider.
    ``None`` sets nothing at all: the model keeps its own default, which is what
    every run did before the level could be configured.

    ``deps_type`` is the tree a run works in (:class:`agentic.core.worktree.Tree`),
    which is how a tool added by a workflow finds the workspace it was pointed at
    rather than being bound to one tree when the agent was built.
    """
    from pydantic_ai import Agent
    from pydantic_ai.settings import ModelSettings

    settings = ModelSettings()
    if thinking is not None:
        settings["thinking"] = thinking
    return Agent(
        model,
        output_type=output_type,
        deps_type=Tree,
        model_settings=settings,
        capabilities=[definition.capabilities],
    )


#: Class names that say nothing about the capability: a container, or a generic
#: carrier whose meaning is in its ``id``. For these the harness id is the name.
_GENERIC_CLASS_NAMES = frozenset({"Capability", "CombinedCapability", "Hooks"})


def capability_id(capability: AbstractCapability[Any]) -> str:
    """The name a composition records for one capability.

    The class name, which is what a reader recognises (``FileSystem``,
    ``ClearToolResults``), falling back to the harness ``id`` where the class is a
    generic carrier (``Capability(id='review_instructions')``). Two capabilities of
    the same class need distinct ids in the harness anyway, so a repeated name is
    a composition the harness would already have rejected.
    """
    name = type(capability).__name__
    if name in _GENERIC_CLASS_NAMES:
        return str(getattr(capability, "id", None) or name)
    return name


def definition_digest(*, instructions: str, material: dict[str, Any]) -> str:
    """A digest of what the agent *is*: its policy, composition and tools.

    Recorded on the run so a reader can tell whether two answers came from the
    same agent, which is what makes a run's evidence comparable.
    """
    payload = json.dumps(
        {"instructions": instructions, **material},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
