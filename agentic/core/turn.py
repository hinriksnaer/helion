"""One agent turn: its answer, its cost, its timing and its transcript.

Every workflow asks the same question of a pydantic-ai run -- what did it answer,
what did it cost, how much of the wall clock was tools, and what actually
happened -- and each one answering it separately is how two workflows come to
disagree about what a run cost. So the mechanics live here and a workflow passes
what is its own: the prompt, the budget, and how to turn a failure into its own
contract's "not answered" value.

Nothing here knows what an answer means. ``TurnResult.result`` is whatever the
agent's ``output_type`` validated; ``error_result`` is the caller's way of saying
"this run produced no answer", which is a contract-shaped decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
import time
from typing import TYPE_CHECKING
from typing import Any

from . import steps
from . import trace
from .steps import StepRecorder

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.usage import RunUsage
    from pydantic_ai.usage import UsageLimits
    from pydantic_ai.workspaces import WorkspaceBackend

    from .worktree import Tree

#: The installed libraries an answer depends on. Recorded, and compared before a
#: recorded answer is reused: the harness is 0.x and its behaviour moves between
#: minors.
VERSION_PACKAGES = ("pydantic", "pydantic-ai-slim", "pydantic-ai-harness")


def versions() -> dict[str, str]:
    """The installed library versions an answer depends on."""
    found: dict[str, str] = {}
    for name in VERSION_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    return found


def usage_of(usage: RunUsage | None) -> dict[str, float] | None:
    """The counters a turn reported, or ``None`` when it reported nothing.

    Named field by field rather than by a list of ``getattr`` names: the fields
    are pydantic-ai's, the checker knows them, and a rename shows up here.
    ``output_reasoning_tokens`` is the interesting one -- one reference reviewer
    spent 93% of its output tokens thinking -- and it used to be dropped.

    ``cost`` is a ``Decimal``, which is neither an ``int`` nor a ``float``, so a
    numeric isinstance check would drop it every time, including on the one kind
    of provider that reports a price here.
    """
    if usage is None:
        return None
    counters: dict[str, float] = {}
    reported = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "output_reasoning_tokens": usage.output_reasoning_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "requests": usage.requests,
        "tool_calls": usage.tool_calls,
    }
    for name, value in reported.items():
        if isinstance(value, int | float):
            counters[name] = float(value)
    if usage.cost is not None:
        counters["cost"] = float(usage.cost)
    return counters or None


def failure_usage(exc: BaseException) -> RunUsage | None:
    """The usage a failed run reported, when it carried any.

    A cancellation carries the counters of the work it did before it stopped
    (``RunCancelled.usage``); the other failure types do not. An all-zero record
    is not a measurement, so it is reported as absent instead. When this returns
    ``None``, :func:`usage_from_messages` still recovers what the run did.
    """
    from pydantic_ai.exceptions import RunCancelled

    if not isinstance(exc, RunCancelled):
        return None
    usage = exc.usage
    if usage is None or not usage.requests:
        return None
    return usage


def usage_from_messages(messages: list[ModelMessage]) -> RunUsage | None:
    """The usage of a run that failed, summed from the responses it got.

    A turn that hits a token limit, or fails on a later request, ends with no
    ``RunResult`` -- so it recorded 0 tokens while every captured response
    carried its own usage. The counters are per request, and summing one per
    ``ModelResponse`` is what the run would have reported.
    """
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.usage import RunUsage

    total: RunUsage | None = None
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        usage = message.usage
        if usage is None:
            continue
        total = RunUsage() if total is None else total
        # A `RequestUsage` carries no request or tool-call count, so the request
        # this response answered is counted here or not at all.
        total.incr(usage)
        total.requests += 1
    return total if total is not None and total.requests else None


@dataclass
class TurnResult:
    """Everything one turn left behind.

    ``result`` is the caller's validated answer type; on a failure it is
    whatever ``error_result`` produced, so a crash is data about the run rather
    than an exception that discards the turns that did finish.
    """

    result: Any
    elapsed: float
    usage: dict[str, float] | None
    raw: bytes | None
    steps: StepRecorder
    event: str
    #: The live message history the run ended with, for continuing the
    #: conversation (a retry with verifier feedback). Not the transcript: the
    #: transcript restores cleared tool results, which would re-inflate the
    #: context a continuation sends.
    history: list[ModelMessage] = field(default_factory=list)


async def run_turn(
    agent: Agent[Any, Any],
    prompt: str,
    *,
    timeout: float,
    usage_limits: UsageLimits | None,
    message_history: list[ModelMessage] | None = None,
    workspace: WorkspaceBackend | None = None,
    deps: Tree | None = None,
    steps_path: Path | None = None,
    label: str = "",
    span_attributes: Mapping[str, str] | None = None,
) -> TurnResult:
    """Run one agent over one prompt. Never raises: a failure is a result.

    ``agent.run``, not ``run_sync``: turns are gathered on one event loop, which
    is the native way to run several at once and the only one that shares a
    cancellation scope.

    The transcript is taken from the run as it happens, not read off the finished
    result: the history a run ends with is not a transcript (``ClearToolResults``
    clears old tool results in place, see :func:`complete_history`) and a run
    that fails ends with no history at all. ``capture_run_messages`` is held open
    around the run for the failure paths that do not carry their own messages.

    ``message_history`` continues an earlier turn of the same agent, which is
    how a workflow hands verifier feedback back to an agent without starting it
    from nothing.

    ``workspace`` is the tree this turn works in, passed the way the harness
    intends a workspace to travel: the agent is bound once and told where to
    work per run. ``deps`` is the same tree as a tool can read it
    (:class:`agentic.core.worktree.Tree`), so a workflow's git tool acts on the
    tree the task is about rather than on the one it was built for.

    ``label`` and ``span_attributes`` are the live view's: the label tags every
    tool line with the worker that ran it, and the attributes go on the
    ``agentic.turn`` span that roots this turn's tree. Both are inert when
    tracing is off, which is the default.
    """
    from pydantic_ai import capture_run_messages

    recorder = StepRecorder(label=label, path=steps_path)
    started = time.monotonic()
    run: Any = None
    failure: Exception | None = None
    # A stop -- Ctrl-C, a cancelled task -- is a `BaseException` and not the
    # model failing. It is not turned into a result (the run must stop), but the
    # history it had built is written before it goes: the record is written by
    # the caller, which a cancelled turn never reaches.
    stopped: BaseException | None = None
    # The span wraps the run and nothing else: `trace.span` marks it failed only
    # when the exception propagates through it, and a caught-in-place failure
    # showed a ✔ for a failed turn. The exception is handled just outside.
    with capture_run_messages() as captured:
        try:
            with (
                trace.span("agentic.turn", span_attributes),
                # Which turn a delegated sub-run records into (core.steps). It is
                # bound inside the run so it belongs to this task's context: two
                # reviewers gathered on one loop must not share a recorder.
                steps.active(recorder),
            ):
                run = await agent.run(
                    prompt,
                    timeout=timeout,
                    usage_limits=usage_limits,
                    message_history=message_history,
                    workspace=workspace,
                    deps=deps,
                )
        except Exception as exc:  # a failed turn is a result, not a crash
            failure = exc
        except BaseException as exc:  # a stop is not a failed answer
            stopped = exc
        finally:
            # The steps are already on disk, one line per event; this only closes
            # the file. A turn that dies keeps what it did.
            recorder.close()

    recorder.elapsed_seconds = time.monotonic() - started
    if stopped is not None:
        if steps_path is not None:
            raw = transcript_bytes(history_for(stopped, captured, recorder), recorder)
            if raw is not None:
                messages_path(steps_path).write_bytes(raw)
        raise stopped
    if failure is not None:
        failed = history_for(failure, captured, recorder)
        return TurnResult(
            # A failed turn has no answer. Prose reports it as the empty string;
            # a typed answer is the outcome's `error` (see `Outcome.failed`).
            result="",
            elapsed=recorder.elapsed_seconds,
            usage=usage_of(failure_usage(failure) or usage_from_messages(failed)),
            raw=transcript_bytes(failed, recorder),
            steps=recorder,
            event=str(failure),
            history=list(failed),
        )
    # `usage` is a property in pydantic-ai 2.x, and calling it is deprecated.
    return TurnResult(
        result=run.output,
        elapsed=recorder.elapsed_seconds,
        usage=usage_of(run.usage),
        raw=transcript_bytes(run.all_messages(), recorder),
        steps=recorder,
        event="ok",
        history=run.all_messages(),
    )


def messages_path(steps_path: Path) -> Path:
    """A turn's transcript, next to the steps file it was recorded with."""
    return steps_path.with_name(
        steps_path.name.removesuffix(".steps.jsonl") + ".messages.json"
    )


def transcript_bytes(
    messages: list[ModelMessage], recorder: StepRecorder
) -> bytes | None:
    """The turn's message history, with the cleared tool results put back.

    ``None`` when the run left no history at all -- recorded as absent rather
    than as an empty transcript that looks like an agent that did nothing.
    """
    if not messages:
        return None
    return dump(complete_history(messages, recorder.tool_returns))


def dump(messages: Sequence[ModelMessage]) -> bytes:
    """Serialize a history the way pydantic-ai's own ``all_messages_json`` does."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.dump_json(list(messages))


def complete_history(
    messages: Sequence[ModelMessage], tool_returns: dict[str, Any]
) -> list[ModelMessage]:
    """Put the tool results back that the context manager took out.

    ``tool_returns`` maps ``tool_call_id`` to the return part the event stream
    saw, so a cleared result is replaced by the real one. Only parts whose id was
    recorded are touched: the output tool's result is the answer rather than a
    tool return, and a part nobody recorded is left exactly as the run had it.
    """
    if not tool_returns:
        return list(messages)

    from pydantic_ai.messages import ToolReturnPart

    completed: list[ModelMessage] = []
    for message in messages:
        parts = getattr(message, "parts", None)
        if parts is None:
            completed.append(message)
            continue
        restored = [
            tool_returns[part.tool_call_id]
            if isinstance(part, ToolReturnPart) and part.tool_call_id in tool_returns
            else part
            for part in parts
        ]
        completed.append(
            message
            if all(new is old for new, old in zip(restored, parts, strict=True))
            else replace(message, parts=restored)  # type: ignore[type-var]
        )
    return completed


def failure_messages(
    error: BaseException, captured: Sequence[ModelMessage]
) -> list[ModelMessage]:
    """The history a failed run had built, from whichever source kept it.

    A cancellation carries its own history and the usage that produced it, so it
    is asked first and ``captured`` -- the ``capture_run_messages()`` list the
    caller kept open around the run -- is the fallback for everything else. A
    failed run is the one where a transcript matters most: it explains why the
    review did not happen.
    """
    from pydantic_ai.exceptions import RunCancelled

    if isinstance(error, RunCancelled):
        return error.all_messages()
    return list(captured)


def history_for(
    error: BaseException, captured: Sequence[ModelMessage], recorder: StepRecorder
) -> list[ModelMessage]:
    """A failed or stopped turn's history, never empty when the turn did anything.

    A stop that arrives mid-request carries neither a ``RunCancelled`` nor a
    captured message, so :func:`failure_messages` has nothing to return and the
    turn kept its ``steps.jsonl`` and lost its transcript -- which is what
    happened to the twenty-minute scan whose transcript would have explained why
    it never answered. The recorder saw every call and every result, so the
    transcript is rebuilt from those instead of being reported as absent.
    """
    return failure_messages(error, captured) or recovered_history(recorder)


def recovered_history(recorder: StepRecorder) -> list[ModelMessage]:
    """A transcript assembled from the calls and results the recorder saw.

    One response per call and one request per result, in the order they ran: the
    same alternating shape a real history has, so the file reads with the same
    tools. It says what the turn did, not why -- the reasoning in flight is gone
    with the run -- and that is still the difference between a turn nobody can
    explain and one a reader can follow.
    """
    from pydantic_ai.messages import ModelRequest
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.messages import ToolReturnPart

    messages: list[ModelMessage] = []
    for step in recorder.steps:
        args: str | dict[str, Any] | None = recorder.full_arguments(step)
        messages.append(
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name=step.tool, args=args, tool_call_id=step.id)
                ]
            )
        )
        returned = recorder.tool_returns.get(step.id)
        if isinstance(returned, ToolReturnPart):
            messages.append(ModelRequest(parts=[returned]))
    return messages
