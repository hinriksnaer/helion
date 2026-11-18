"""What one reviewer turn *did*, step by step, and where its time went.

A message history says what the model called and got back; it does not say how
long any of it took. ``ModelResponse.timestamp`` in pydantic-ai is stamped when
the request is *sent* (`models/openai/_chat_streaming.py` defaults it to
``now_utc()`` at stream construction), so consecutive message timestamps measure
"model streaming plus tool execution" glued together -- enough to see that a run
took 636 seconds, not enough to say which half. One reference run of a 1.2 KB
diff spent 11 minutes with no way to attribute them.

The agent loop emits ``FunctionToolCallEvent`` before a tool runs and
``FunctionToolResultEvent`` after it returns, in real time, so a handler around
those two events measures tool time exactly; the remainder of the turn is the
model's. That is the whole of this module: a recorder for those events, and the
JSONL file it leaves behind for a later tool (or a later run of an agent) to
read.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from dataclasses import dataclass
import json
import time
from typing import TYPE_CHECKING
from typing import Any

from . import trace

if TYPE_CHECKING:
    from collections.abc import AsyncIterable
    from collections.abc import Iterator
    from pathlib import Path

    from pydantic_ai import RunContext
    from pydantic_ai.messages import AgentStreamEvent

#: How much of a tool's arguments a step line keeps: this much at the start, and
#: `_ARGS_TAIL` at the end. The head is where the command is and the tail is where
#: a heredoc or a long path ends, so a clipped-as-JSON line is no longer readable
#: either way -- which is why a long call keeps both ends rather than only the
#: front. The full text is in the transcript next to it, except for a turn that
#: ended before its history did.
_ARGS_HEAD = 800
_ARGS_TAIL = 200

#: How much of a tool's *result* a step line keeps: one line's worth, whitespace
#: collapsed. The result is where friction shows -- `No such file or directory`,
#: `command not found`, an exit status inside a pipeline -- and a length alone
#: cannot say whether a delegate's six minutes produced anything.
_RESULT_HEAD = 240


@dataclass
class Step:
    """One tool call of one turn."""

    #: The harness's id for the call, which is what pairs its start with its end.
    id: str
    tool: str
    args: str
    started_ms: int
    finished_ms: int | None = None
    result_chars: int | None = None
    #: ``True``/``False`` once the call returned, ``None`` while it is in flight
    #: or when the run died before it came back. A step that never finished is not
    #: a step that succeeded -- the run that hit the retry limit recorded two
    #: unfinished calls as ``ok: true``, which is the opposite of what happened.
    ok: bool | None = None
    #: The start of what the call returned, whitespace-collapsed and bounded by
    #: `_RESULT_HEAD`. Enough to tell a finding from a failure; the whole result is
    #: in the transcript.
    result_head: str = ""
    #: A call a delegated sub-run made rather than the delegating agent itself.
    #: A sub-run is not a recorded turn of its own, so this is what keeps its
    #: calls auditable: without it a finding whose locator only appeared inside a
    #: delegate pointed at evidence the parent's steps file did not contain.
    sub: bool = False
    #: *Which* sub-run made the call: the delegate's name, which is the agent the
    #: model asked for (``"self"``, numbered ``"self #2"`` when one turn asks for
    #: it twice), or its task's opening words for a sub-run no ``delegate_task``
    #: call of ours started. Empty for the turn's own calls. ``sub`` above says
    #: whether a call was delegated and stays the flag a reader already knows; this
    #: is the part that tells two delegates of one turn apart, which a bare boolean
    #: cannot.
    delegate: str = ""

    @property
    def seconds(self) -> float | None:
        if self.finished_ms is None:
            return None
        return (self.finished_ms - self.started_ms) / 1000.0


class StepRecorder:
    """Records one turn's tool steps and the wall time they account for.

    Used as pydantic-ai's ``event_stream_handler``; ``elapsed`` is filled by the
    caller around ``run_sync`` so the model's share can be computed as
    ``elapsed - tool_seconds``.
    """

    def __init__(self, *, label: str = "", path: Path | None = None) -> None:
        #: Which worker this turn is, for the live view: two turns run at once and
        #: a line that does not say whose it is is useless.
        self.label = label
        #: Where the steps are appended as the turn runs, or ``None`` until the
        #: caller asks for a file. A long turn is readable while it runs, and a
        #: crashed turn keeps what it did up to then.
        self.path = path
        self._sink: Any = None
        self.steps: list[Step] = []
        self._open: dict[str, Step] = {}
        self._start = time.monotonic()
        self.elapsed_seconds: float | None = None
        #: Each tool's return part, by ``tool_call_id``. Kept because the event
        #: stream is the last place the real result exists: `ClearToolResults`
        #: clears the run's own history in place once it grows past a threshold,
        #: so a transcript read at the end of the turn shows
        #: `[tool result cleared]` where the answer was. See `core.turn.complete_history`.
        self.tool_returns: dict[str, Any] = {}
        #: The sub-runs of this turn, by the identity pydantic-ai gives one, so the
        #: several stream replays of a sub-run are one delegate and not several.
        #: The harness hands its handler the same `conversation_id` for every
        #: replay of one child run, and a fresh one for the next child.
        self._sub_runs: dict[str, str] = {}
        #: The run id of this turn's own agent run. The capability handler is
        #: bound to the agent, so a delegated sub-run's events reach it too; any
        #: run that is not the owner is a delegate's.
        self.owner: str | None = None
        #: Tool-call ids already recorded per delegate. A sub-run's events reach
        #: the handler more than once, and a replay is not a second call.
        self._seen: dict[str, set[str]] = {}
        #: Each open call's arguments as the model wrote them, by call id. The
        #: record keeps a clipped copy, and the live view cannot use that: a JSON
        #: argument cut short is not JSON, so a long command printed as
        #: `{"command": "cd \"…` instead of the command.
        self._asked_args: dict[str, object] = {}
        #: Every call's arguments as the model wrote them, for a turn that has no
        #: history to recover one from (`core.turn.recovered_history`). Unlike
        #: `_asked_args` these stay after the call returns.
        self.full_args: dict[str, str | dict[str, Any] | None] = {}

    def full_arguments(self, step: Step) -> str | dict[str, Any] | None:
        """The call's arguments as the model wrote them, not the clipped copy.

        Kept for a turn that ended without a history to recover them from
        (`core.turn.recovered_history`): a rebuilt transcript that carried the
        log's 800-character clip would not be the call the agent made.
        """
        return self.full_args.get(step.id, step.args)

    def delegate_name(self, ctx: RunContext[object] | None) -> str:
        """A name for the sub-run ``ctx`` belongs to, stable across its replays.

        The harness hands the handler the same ``conversation_id`` for every replay
        of one child run and a fresh one for the next child, so that is what a
        delegate is identified by. The name is its position in the turn, ``delegate
        1``, ``delegate 2``: the task is the row above the block, so the name only
        has to be short, stable and distinct.
        """
        key = str(getattr(ctx, "conversation_id", "") or "")
        if key not in self._sub_runs:
            self._sub_runs[key] = f"delegate {len(self._sub_runs) + 1}"
        return self._sub_runs[key]

    async def __call__(
        self, _ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]
    ) -> None:
        await self.record(_ctx, events)

    def _append(self, record: dict[str, Any]) -> None:
        """One event, on disk, now.

        Flushed per line: the point of appending rather than dumping at the end is
        that a person (or another process) can watch a long turn, and that a turn
        which dies leaves its steps behind. No ``fsync``: the file is evidence for
        a reader, not a write-ahead log.
        """
        if self.path is None:
            return
        if self._sink is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._sink = self.path.open("a")
        self._sink.write(json.dumps(record, sort_keys=True) + "\n")
        self._sink.flush()

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    async def record(
        self,
        _ctx: RunContext[object],
        events: AsyncIterable[AgentStreamEvent],
        *,
        delegate: str = "",
    ) -> None:
        """Record the events of this turn, or of a delegated sub-run of it."""
        # Imported here so this module itself pulls in nothing heavy: the run
        # record is read by tools that may not have the agentic extra installed.
        from pydantic_ai.messages import FunctionToolCallEvent
        from pydantic_ai.messages import FunctionToolResultEvent
        from pydantic_ai.messages import RetryPromptPart
        from pydantic_ai.messages import ToolReturnPart

        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                part = event.part
                seen = self._seen.setdefault(delegate, set())
                if delegate and part.tool_call_id in seen:
                    # A replayed sub-run: the same call, already recorded.
                    continue
                seen.add(part.tool_call_id)
                step = Step(
                    id=part.tool_call_id,
                    tool=part.tool_name,
                    args=_clip(part.args),
                    started_ms=self._ms(),
                    sub=bool(delegate),
                    delegate=delegate,
                )
                self.steps.append(step)
                self._open[part.tool_call_id] = step
                self._append(_call_record(step))
                self._asked_args[part.tool_call_id] = part.args
                self.full_args[part.tool_call_id] = part.args
                # The live view's "what is running now". A span does not carry
                # the arguments until the call returns, which is too late.
                trace.tool_call(
                    self.label,
                    part.tool_name,
                    part.args,
                    sub=delegate,
                    call_id=part.tool_call_id,
                )
            elif isinstance(event, FunctionToolResultEvent):
                part = event.part
                step = self._open.pop(part.tool_call_id, None)
                if step is None:
                    continue
                step.finished_ms = self._ms()
                # pydantic-ai 2.x answers a failed or retried call with a
                # `ToolReturnPart` whose `outcome` carries it; the legacy
                # `RetryPromptPart` is still used for some validation failures.
                step.ok = not isinstance(part, RetryPromptPart) and (
                    getattr(part, "outcome", "success") == "success"
                )
                # Both a return part and a retry prompt carry `.content`.
                step.result_chars = len(str(part.content))
                step.result_head = _head(str(part.content))
                self._append(_result_record(step))
                if isinstance(part, ToolReturnPart):
                    self.tool_returns[part.tool_call_id] = part
                trace.tool_result(
                    label=self.label,
                    tool=step.tool,
                    args=self._asked_args.pop(part.tool_call_id, step.args),
                    seconds=step.seconds or 0.0,
                    chars=step.result_chars,
                    ok=bool(step.ok),
                    sub=delegate,
                    call_id=part.tool_call_id,
                )

    def _ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)

    @property
    def tool_seconds(self) -> float:
        """Wall time spent in tools, with overlapping calls counted once.

        The agent loop runs the tool calls of one model response concurrently, so
        adding their durations up reports two seconds for two simultaneous
        one-second calls -- and ``model_seconds``, which is the remainder, then
        understates the model's share. The union of the call intervals is the
        answer this module exists to give.
        """
        intervals = sorted(
            (step.started_ms, step.finished_ms)
            for step in self.steps
            if step.finished_ms is not None
        )
        total_ms = 0
        current: list[int] | None = None
        for start, end in intervals:
            if current is None or start > current[1]:
                if current is not None:
                    total_ms += current[1] - current[0]
                current = [start, end]
            else:
                current[1] = max(current[1], end)
        if current is not None:
            total_ms += current[1] - current[0]
        return total_ms / 1000.0

    def summary(self) -> dict[str, float]:
        """Wall time, and how much of it was tools rather than the model."""
        elapsed = self.elapsed_seconds or 0.0
        tool = self.tool_seconds
        return {
            "elapsed_seconds": round(elapsed, 1),
            "tool_seconds": round(tool, 1),
            "model_seconds": round(max(elapsed - tool, 0.0), 1),
        }

    def write(self, path: Path) -> Path:
        """Write a turn that did not append while it ran, in the live file's format.

        The same two lines per call -- the call as it starts, its outcome as it
        returns -- so a record appended live and one written at the end read back
        the same way, and there is one writer rather than two formats nobody kept
        in step.
        """
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        self.path = path
        for step in self.steps:
            self._append(_call_record(step))
            if step.finished_ms is not None:
                self._append(_result_record(step))
        self.close()
        return path


#: The recorder of the turn running in this context. A turn runs in its own
#: asyncio task, and a task copies the context it was created in, so two
#: reviewers gathered on one event loop never see each other's recorder.
_ACTIVE: ContextVar[StepRecorder | None] = ContextVar(
    "agentic_turn_recorder", default=None
)


@contextmanager
def active(recorder: StepRecorder) -> Iterator[None]:
    """Make ``recorder`` the turn that delegated sub-runs record into."""
    token = _ACTIVE.set(recorder)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


async def observe(
    ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]
) -> None:
    """The handler every agent carries, as a capability (`ProcessEventStream`).

    The turn that is running owns the recorder, and a turn runs in its own
    asyncio task whose context a sub-run inherits -- so a delegate records into
    the turn that started it, and two reviewers gathered on one loop never share
    one. With no turn around it (a dry run, a caller driving the agent directly)
    the stream is drained: the agent loop waits on a generator nobody reads.
    """
    recorder = _ACTIVE.get()
    if recorder is None:
        async for _ in events:
            pass
        return
    if recorder.owner is None:
        recorder.owner = getattr(ctx, "run_id", None)
    owner = recorder.owner is None or getattr(ctx, "run_id", None) == recorder.owner
    await recorder.record(
        ctx, events, delegate="" if owner else recorder.delegate_name(ctx)
    )


def _clip(value: object) -> str:
    """A call's arguments, keeping both ends of a long one.

    A command's start says what it is and its end says what it is aimed at, so a
    heredoc or a piped tail survives the clip that would otherwise cut it off.
    """
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= _ARGS_HEAD + _ARGS_TAIL:
        return text
    omitted = len(text) - _ARGS_HEAD - _ARGS_TAIL
    return f"{text[:_ARGS_HEAD]} …[{omitted} chars omitted]… {text[-_ARGS_TAIL:]}"


def _head(text: str) -> str:
    """The start of a tool result, on one line, as far as `_RESULT_HEAD`.

    Whitespace is collapsed rather than left in: one record per call is a line in
    a JSONL file that people read, and a result's first line is what says whether
    the call worked.
    """
    collapsed = " ".join(text.split())
    return collapsed[:_RESULT_HEAD]


def _call_record(step: Step) -> dict[str, Any]:
    """A call as it starts: the one shape the live file and ``write`` both use."""
    return {"event": "call", **asdict(step), "seconds": step.seconds}


def _result_record(step: Step) -> dict[str, Any]:
    """How that call ended, merged onto the call line by its id."""
    return {
        "event": "result",
        "id": step.id,
        "finished_ms": step.finished_ms,
        "ok": step.ok,
        "result_chars": step.result_chars,
        "result_head": step.result_head,
        "seconds": step.seconds,
    }


def evidence(path: Path, *, limit: int = 40, head: int = 120) -> str:
    """What a turn ran, and how each call ended: the facts behind its prose.

    A review says what a model concluded; this says what it did. The model that
    weighs several reviews can then check a claim against the command that
    produced it instead of re-running every check itself -- which is what the
    summariser did when it was handed prose alone. Bounded to the last ``limit``
    calls, one line each, each result cut at ``head`` characters.
    """
    lines: list[str] = []
    for record in read_steps(path)[-limit:]:
        summary = trace.argument_summary(
            str(record.get("tool") or ""), str(record.get("args") or "")
        )
        failed = " (failed)" if record.get("ok") is False else ""
        result = str(record.get("result_head") or "").strip()
        detail = f" -> {result[:head]}" if result else ""
        lines.append(f"- `{summary}`{failed}{detail}")
    return "\n".join(lines)


def read_steps(path: Path) -> list[dict[str, object]]:
    """Read a steps file back: one record per call, in the order they started.

    The file has two lines per call -- the call as it starts, its outcome as it
    returns -- merged here on the call id. A line that is not JSON is skipped, not
    fatal: the file is appended to while something may be reading it.
    """
    steps: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return steps
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.pop("event", None)
        call_id = str(record.get("id") or "")
        if kind == "call":
            by_id[call_id] = record
            steps.append(record)
        elif kind == "result" and call_id in by_id:
            by_id[call_id].update(record)
    return steps
