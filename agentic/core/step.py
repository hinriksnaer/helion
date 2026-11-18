"""What a workflow declares: a step, the worker that runs it, the task it is asked.

A **step** is one kind of agent turn: a policy, an answer type, and any tool the
harness has no equivalent for. A **worker** is a step bound to a model under a
name. A **task** is a prompt plus the digests of what it is about, which together
name the recorded answer.

Nothing here runs anything; :class:`agentic.core.lifecycle.Run` does.

The answer is a type, not a protocol: ``str`` for a step whose answer the next
model reads, or a pydantic model for a step whose answer the *run* reads (a gate,
a loop condition). ``Answer`` is the base those models share.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel

from . import artifact as artifact_store

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model

    from .artifact import Artifact
    from .config import ModelRef
    from .config import ThinkingLevel
    from .stack import Definition


class Answer(BaseModel):
    """The base for a step whose answer the *run* reads, rather than the next model.

    ``error`` set means "this turn produced no answer", which is never reused.
    A step that answers ``str`` needs none of this: prose is for the next model,
    and a failed turn is the failure itself.
    """

    error: str | None = None

    def summary(self) -> str:
        if self.error is not None:
            return f"no answer: {self.error}"
        return type(self).__name__


A = TypeVar("A")


#: What a model answered, in one line, for the log and the record.
def answer_summary(answer: object) -> str:
    if isinstance(answer, Answer):
        return answer.summary()
    text = str(answer).strip()
    return text.splitlines()[0][:120] if text else "(no answer)"


@dataclass(frozen=True)
class Step(Generic[A]):
    """One kind of agent turn.

    Attributes:
        kind: The record namespace (``improve_scan``, ``reviewer``), and part of
            the definition's name.
        instructions: The policy text. It is in the definition digest, so a
            record names the exact text its answer was produced under.
        output_type: ``str`` for prose the next model reads, or a pydantic model
            whose fields the run acts on.
        skills_dirs: Skill libraries offered to the agent, read from the run's
            workspace.
        reusable: Whether a recorded answer may be served instead of running the
            step. False for a step whose output is a side effect (a commit).
        schema_version: The answer shape's version; a record from another version
            is treated as absent, never as a pass.
    """

    kind: str
    instructions: str
    output_type: type[A]
    skills_dirs: tuple[str, ...] = ()
    reusable: bool = True
    schema_version: str = "1"

    def parse(self, data: dict[str, Any]) -> A:
        """A recorded answer, back in this step's shape."""
        if isinstance(self.output_type, type) and issubclass(
            self.output_type, BaseModel
        ):
            return self.output_type.model_validate(data)  # type: ignore[return-value]
        return data.get("text", "")  # type: ignore[return-value]

    def as_record(self, answer: object) -> dict[str, Any]:
        if isinstance(answer, BaseModel):
            return answer.model_dump(mode="json")
        return {"text": answer}

    def definition(self) -> Definition:
        """The agent's composition: ``Coder`` with this policy, and the recorder.

        The policy goes through ``Coder(instructions=...)``, which appends it to
        the instructions the harness writes for its own tools, so there is one
        instruction source and the harness's conventions come first.
        """
        from pydantic_ai.capabilities import CombinedCapability
        from pydantic_ai.capabilities import ProcessEventStream
        from pydantic_ai_harness import Coder
        from pydantic_ai_harness import Skills

        from .stack import Definition
        from .stack import capability_id
        from .steps import observe

        parts: list[Any] = [Coder(instructions=self.instructions)]
        # The recorder is a capability, so `run()` streams to it without an
        # `event_stream_handler` argument and a delegated sub-run inherits it,
        # and it is always present, so a record's digest does not depend on
        # whether tracing is on.
        parts.append(ProcessEventStream(observe, id="agentic_steps"))
        if self.skills_dirs:
            parts.append(Skills(list(self.skills_dirs), missing_directories="skip"))
        return Definition(
            capabilities=CombinedCapability(parts),
            names=tuple(capability_id(part) for part in parts),
            instructions=self.instructions,
        )

    def agent(
        self,
        model: Model,
        thinking: ThinkingLevel | None = None,
        definition: Definition | None = None,
    ) -> Agent[Any, Any]:
        """The agent for this step. Pass the definition you hold, if you hold one."""
        from . import stack

        return stack.build_agent(
            model, definition or self.definition(), self.output_type, thinking
        )


@dataclass
class Worker(Generic[A]):
    """A step bound to a model under a name."""

    index: int
    name: str
    ref: ModelRef
    step: Step[A]
    model: Model[Any]
    definition: Definition
    digest: str
    #: What the model's provider resolves to, for the artifact key.
    provider: str = ""
    #: The role's model settings, as the artifact key records them.
    settings: str = ""
    #: What a reader's private tree is taken from (``None``: the caller's checkout).
    source: Path | None = None
    #: The built agent. A workspace is a run-time argument, so one agent answers
    #: every task.
    agent: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class Task:
    """What a worker is asked.

    Attributes:
        name: The workflow's name for this unit of work, for a reader.
        prompt: The text the agent is given.
        inputs: Named digests of what the task is about. With the prompt, the
            model and the step they are the cache key.
        revision: The revision the reader's checkout is taken at, when the task is
            about a commit or a range.
        branch: For a task whose answer is a change: the branch the run owns, in
            a worktree cut from ``base``. ``None`` for a reader.
        base: Where ``branch`` is cut from; the caller's ``HEAD`` when unset.
    """

    name: str
    prompt: str
    inputs: dict[str, str]
    revision: str | None = None
    branch: str | None = None
    base: str | None = None

    def identity(self) -> str:
        return artifact_store.diff_digest(
            json.dumps(
                [self.name, self.prompt, self.inputs, self.revision, self.branch],
                sort_keys=True,
            )
        )

    def key(self, worker: Worker[Any]) -> str:
        """The artifact key of this question, asked of ``worker``."""
        return artifact_store.key_for(
            agent=worker.step.kind,
            model=worker.ref.label,
            provider=worker.provider,
            settings=worker.settings,
            prompt=self.prompt,
            inputs=self.inputs,
            schema_version=worker.step.schema_version,
        )


@dataclass
class Outcome(Generic[A]):
    """What one worker answered for one task, and where its evidence is."""

    name: str
    model: str
    result: A
    cached: bool
    elapsed_seconds: float
    stale: bool = False
    usage: dict[str, float] | None = None
    timing: dict[str, float] | None = None
    outcome_path: Path | None = None
    messages_path: Path | None = None
    steps_path: Path | None = None
    artifact: str | None = None
    #: The live message history, for continuing this turn. Empty when cached.
    history: list[ModelMessage] = field(default_factory=list, repr=False)
    #: The task this answered, so a later turn continues the same conversation
    #: and works in the same tree (a writer's refine round).
    task: Task | None = field(default=None, repr=False)
    #: The tree this turn worked in. ``None`` for a served record.
    tree: Path | None = field(default=None, repr=False)
    #: Why this turn produced no answer, or ``None``. A failed turn is a stopped
    #: run, never an empty success.
    error: str | None = None
    pending: Artifact | None = field(default=None, repr=False)

    @property
    def failed(self) -> bool:
        """Whether this turn produced no answer.

        Two ways to say so, because there are two kinds of step: prose steps
        report the failure on the outcome (there is no field to carry it), and a
        typed answer may carry its own ``error`` (a reviewer that did not review).
        """
        return self.error is not None or getattr(self.result, "error", None) is not None

    def as_record(self) -> dict[str, Any]:
        """The fields every workflow's run record carries for one outcome."""
        return {
            "name": self.name,
            "model": self.model,
            "cached": self.cached,
            "stale": self.stale,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "timing": self.timing,
            "usage": self.usage,
            "transcript": _name(self.messages_path),
            "steps": _name(self.steps_path),
            "outcome": _name(self.outcome_path),
            "artifact": self.artifact,
            "error": self.error,
        }


def _name(path: Path | None) -> str | None:
    return path.name if path is not None else None
