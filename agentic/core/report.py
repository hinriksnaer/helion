"""What a run writes down: its directory, its record, and what a person reads.

One module, because both halves answer the same question. ``RunLog`` is the
layout of a run directory and the files in it; ``Report`` turns a workflow's
:class:`Description` into ``run.json``, ``index.md`` and the terminal text. The
pipeline hands both to :meth:`agentic.core.lifecycle.Run.open`, which writes on
every exit path, and the entry point hands the report to
:func:`agentic.core.cli.main`, which prints it.

A run directory answers, file by file: ``index.md`` what happened, for a person;
``run.json`` the same as data; ``run.log`` a timestamped timeline;
``<n>-<m>-<name>.outcome.json`` the answer the pipeline gated on;
``.messages.json`` the whole conversation, completed from the event stream so the
tool results ``ClearToolResults`` replaced are back in it; and ``.steps.jsonl``
one line per tool call, which says how much of the wall clock the tools took.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from .lifecycle import RunSummary
    from .step import Outcome
    from .steps import StepRecorder

S = TypeVar("S", bound="RunSummary")


@dataclass(frozen=True)
class Description:
    """What a workflow says about a run of itself, in the shapes it is read.

    One structure rather than one callable per output: the fields ``run.json``
    holds, the ``index.md`` sections, the verdict line and the terminal text are
    four readings of the same thing, and returning them together is what keeps a
    workflow from filling in three and forgetting the fourth.
    """

    #: The workflow's own ``run.json`` fields, over ``RunSummary.run_record()``.
    fields: dict[str, Any]
    #: The ``index.md`` body, between the common header and the verdict.
    sections: list[str]
    #: One line for a run that finished: the result and the exit code.
    verdict: str
    #: What the terminal shows for a run that finished.
    text: str
    #: What a *stopped* run still shows: whatever the workflow already has
    #: (findings, the branch it created), or ``""`` for nothing.
    partial: str = ""


@dataclass(frozen=True)
class Report(Generic[S]):
    """How a run is written down: the workflow's description, and the common part.

    ``describe`` is called once per output, and builds strings -- nothing here
    caches, and nothing here may touch the network or the tree.
    """

    describe: Callable[[S], Description]

    def record(self, summary: S) -> dict[str, Any]:
        """The run as JSON: what ``--json`` prints and what ``run.json`` holds."""
        return {**summary.run_record(), **self.describe(summary).fields}

    def finish(self, summary: S, log: RunLog) -> None:
        """Write ``run.json`` and ``index.md``; called on every exit path."""
        described = self.describe(summary)
        verdict = (
            f"not finished -- the run stopped: {summary.error}"
            if summary.error is not None
            else described.verdict
        )
        log.write_json("run.json", self.record(summary))
        log.event(verdict)
        lines = [*header(summary), "", *described.sections]
        lines += ["", "## Verdict", "", verdict]
        log.write_index("\n".join(lines))

    def print(self, summary: S, *, as_json: bool) -> None:
        """Print the run for the terminal, or its record with ``as_json``."""
        described = self.describe(summary)
        if as_json:
            print(json.dumps(self.record(summary), indent=2, sort_keys=True))
        elif summary.error is not None:
            print(f"not finished -- the run stopped: {summary.error}")
            # The findings a run already made, and a branch it already created,
            # are why it ran: printing only the error hid both.
            if described.partial:
                print()
                print(described.partial)
        else:
            print(described.text)


def header(summary: RunSummary) -> list[str]:
    """Who ran, with what."""
    lines = ["## Agents", ""]
    lines += [f"- **{name}** -- `{label}`" for name, label in summary.models.items()]
    if not summary.models:
        lines.append("none built")
    lines += [
        "",
        "versions: "
        + ", ".join(f"`{k}={v}`" for k, v in sorted(summary.versions.items())),
    ]
    if summary.environment:
        lines.append(f"environment: `{summary.environment[:16]}`")
    lines += [
        f"definition {name}: `{digest[:16]}`"
        for name, digest in summary.definitions.items()
    ]
    if summary.capabilities:
        lines += ["", "## Capabilities", "", "```"]
        lines.append(", ".join(summary.capabilities))
        lines.append("```")
    return lines


def outcome_line(outcome: Outcome[Any]) -> list[str]:
    """One worker's answer for ``index.md``: flags, timing, summary, evidence."""
    flags = [
        flag
        for flag, on in (("cached", outcome.cached), ("stale", outcome.stale))
        if on
    ]
    marker = f" ({', '.join(flags)})" if flags else ""
    timing = outcome.timing or {}
    took = f"{outcome.elapsed_seconds:.0f}s"
    if timing:
        took += (
            f" (model {timing.get('model_seconds', 0):.0f}s,"
            f" tools {timing.get('tool_seconds', 0):.0f}s)"
        )
    lines = [
        (
            f"- **{outcome.name}**{marker} -- `{outcome.model}`, {took}: "
            f"{outcome.result.summary()}"
        )
    ]
    for label, path in (
        ("transcript", outcome.messages_path),
        ("steps", outcome.steps_path),
    ):
        if path is not None:
            lines.append(f"  - {label}: `{path.name}`")
    return lines


#: Where run logs live, relative to the repository root. Gitignored: evidence
#: about a local tree is machine-local state.
LOGS_DIR = Path("agentic/logs")


def slugify(text: str) -> str:
    """A filesystem-safe name from a caller-supplied label, bounded in length.

    The label comes from config (a reviewer's name), so it reaches a path. Only
    ``[A-Za-z0-9._-]`` survives, no separator does, and a leading dot is dropped
    so no component can be ``.`` or ``..``.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lstrip(".")
    return (slug or "agent")[:60]


def stem(task_index: int, agent_index: int, name: str) -> str:
    """The file stem of one agent's evidence for one task.

    Both indices are in it. A run over several tasks reuses the agent index for
    every task, so naming by agent alone made each task overwrite the last one's
    answer, transcript and steps, and ``run.json`` then pointed every task at the
    final task's files.
    """
    return f"{task_index}-{agent_index}-{slugify(name)}"


@dataclass
class RunLog:
    """One run's directory, written as the run proceeds."""

    directory: Path
    command: str = ""
    timeline: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, root: Path, *, command: str = "") -> RunLog:
        """Create a fresh directory for this run.

        Named ``<timestamp>-<pid>`` like the incumbent tool's `.logs/autoreview`,
        so a person looking for "the run I just did" finds it the same way. Two
        runs in the same second share a pid, so the name is made unique with a
        counter: a plain ``mkdir(exist_ok=True)`` let the second run write into
        the first run's directory, overwriting the evidence it existed to keep --
        and, because that directory was in the second run's ignore list, hiding
        the first run's log from the working-tree diff and faking a cache hit.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = root / LOGS_DIR / f"{stamp}-{os.getpid()}"
        directory = base
        suffix = 1
        while directory.exists():
            directory = base.with_name(f"{base.name}-{suffix}")
            suffix += 1
        directory.mkdir(parents=True)
        log = cls(directory=directory, command=command)
        log.event(f"run started: {command}".strip())
        return log

    # -- writing -----------------------------------------------------------

    def event(self, text: str) -> None:
        """Append one timestamped line to the timeline, flushed immediately.

        Written as it happens rather than at the end: a run that hangs or is
        killed should still say what it had reached.
        """
        line = f"{datetime.now().strftime('%H:%M:%S')} {text}"
        self.timeline.append(line)
        with (self.directory / "run.log").open("a") as handle:
            handle.write(line + "\n")

    def agent_log(self, task_index: int, index: int, name: str, text: str) -> Path:
        """Write one agent's own events for one task."""
        path = self.directory / f"{stem(task_index, index, name)}.log"
        path.write_text(text)
        return path

    def write_outcome(
        self, task_index: int, index: int, name: str, outcome: dict[str, Any]
    ) -> Path:
        """Write the workflow's answer, exactly as the gate read it."""
        path = self.directory / f"{stem(task_index, index, name)}.outcome.json"
        path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
        return path

    def write_messages(
        self, task_index: int, index: int, name: str, raw: bytes | None
    ) -> Path | None:
        """Write the full message history: tool calls and results included.

        ``raw`` is whatever the model harness produced (pydantic-ai serializes its
        own history). Absent history is recorded as absent, not as an empty file
        that looks like a reviewer that did nothing.
        """
        if raw is None:
            return None
        path = self.directory / f"{stem(task_index, index, name)}.messages.json"
        path.write_bytes(raw)
        return path

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def steps_path(self, task_index: int, index: int, name: str) -> Path:
        """Where a turn's steps are appended while it runs."""
        return self.directory / f"{stem(task_index, index, name)}.steps.jsonl"

    def write_steps(
        self, task_index: int, index: int, name: str, steps: StepRecorder
    ) -> Path:
        """Write a turn that did not append live (a fake turn, a direct call)."""
        return steps.write(self.steps_path(task_index, index, name))

    def write_text(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text)
        return path

    def write_index(self, body: str) -> Path:
        """Write the summary a person opens first."""
        header = f"# agentic {self.command or 'run'} -- {self.directory.name}\n\n"
        timeline = "\n".join(self.timeline)
        return self.write_text(
            "index.md",
            f"{header}```\n{timeline}\n```\n\n{body.rstrip()}\n",
        )


def output_paths(root: Path) -> tuple[Path, ...]:
    """Where a run writes inside the repository: its logs and its artifacts.

    A workflow hashes the tree before and after an agent turn to prove the agent
    did not change it, and builds a working-tree diff of the change under review.
    Neither should see the run's own output, and neither should see a *previous*
    run's: ignoring only the current run's directory made every review of an
    unchanged diff miss the cache, because the last run's log had become an
    untracked file in the next run's diff.

    Lives here rather than in a workflow because both of those are properties of
    the runtime, not of what a workflow is about.
    """
    from .artifact import STORE_DIR

    return (root / LOGS_DIR, root / STORE_DIR)
