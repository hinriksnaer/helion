"""One turn: steps, usage, the transcript and the composed stack."""

from __future__ import annotations

import asyncio
import io
import json
from typing import TYPE_CHECKING
from typing import Any

import pytest

from agentic.core import steps as steps_mod
from agentic.core.turn import usage_of
from agentic.tests.support import TracerProviderFixture
from agentic.tests.support import _ctx
from agentic.tests.support import _usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pydantic_ai.messages import ModelMessage


class TestSteps:
    def test_a_step_records_its_duration(self) -> None:
        step = steps_mod.Step(
            id="c0", tool="read_file", args="p", started_ms=100, finished_ms=350
        )
        assert step.seconds == 0.25
        assert steps_mod.Step(id="c1", tool="t", args="a", started_ms=0).seconds is None

    def test_the_summary_splits_model_and_tool_time(self) -> None:
        recorder = steps_mod.StepRecorder()
        recorder.steps = [
            steps_mod.Step(
                id="c0", tool="read_file", args="p", started_ms=0, finished_ms=1000
            )
        ]
        recorder.elapsed_seconds = 10.0
        summary = recorder.summary()
        assert summary["tool_seconds"] == 1.0
        assert summary["model_seconds"] == 9.0

    def test_overlapping_tool_calls_are_counted_once(self) -> None:
        # pydantic-ai runs one response's tool calls concurrently. Summing their
        # durations reported 2s of tools for two simultaneous 1s calls, and
        # model_seconds is whatever is left over.
        recorder = steps_mod.StepRecorder()
        recorder.steps = [
            steps_mod.Step(id="a", tool="a", args="", started_ms=0, finished_ms=1000),
            steps_mod.Step(id="b", tool="b", args="", started_ms=500, finished_ms=1500),
            steps_mod.Step(
                id="c", tool="c", args="", started_ms=2000, finished_ms=2500
            ),
        ]
        recorder.elapsed_seconds = 10.0
        summary = recorder.summary()
        assert summary["tool_seconds"] == 2.0
        assert summary["model_seconds"] == 8.0

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        recorder = steps_mod.StepRecorder()
        recorder.steps = [
            steps_mod.Step(
                id="c1", tool="read_file", args="p", started_ms=0, finished_ms=500
            ),
            steps_mod.Step(
                id="c2", tool="read_file", args="q", started_ms=500, finished_ms=600
            ),
        ]
        path = recorder.write(tmp_path / "s.jsonl")
        steps = steps_mod.read_steps(path)
        assert [step["tool"] for step in steps] == ["read_file", "read_file"]
        assert steps[0]["seconds"] == 0.5

    def test_a_line_that_is_not_json_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "s.jsonl"
        path.write_text('not json\n{"event": "call", "id": "c1", "tool": "x"}\n')
        assert steps_mod.read_steps(path) == [{"id": "c1", "tool": "x"}]

    def test_a_missing_steps_file_reads_as_empty(self, tmp_path: Path) -> None:
        assert steps_mod.read_steps(tmp_path / "absent.jsonl") == []

    def test_a_delegated_sub_run_records_into_the_turn_that_started_it(
        self, tmp_path: Path
    ) -> None:
        """A delegate's calls are evidence, not just the text of its return.

        `SubAgents` passes its handler to the child agent, and the child records
        into the delegating turn's recorder; without this the delegate's work
        appeared nowhere but one `delegate_task` result.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import FunctionToolCallEvent
        from pydantic_ai.messages import FunctionToolResultEvent
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.messages import ToolReturnPart

        async def events() -> AsyncIterator[object]:
            yield FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name="read_file", args='{"path": "a"}', tool_call_id="c1"
                )
            )
            yield FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name="read_file", content="body", tool_call_id="c1"
                )
            )

        from types import SimpleNamespace

        recorder = steps_mod.StepRecorder(label="reviewer")
        parent = SimpleNamespace(run_id="turn", conversation_id="turn")
        child = SimpleNamespace(run_id="sub", conversation_id="sub-1")

        async def drive(turn: object) -> None:
            await steps_mod.observe(turn, events())  # type: ignore[arg-type]

        # Nothing bound: the stream is drained and nothing is recorded.
        asyncio.run(drive(parent))
        assert recorder.steps == []

        with steps_mod.active(recorder):
            # The turn's own run, then a delegate's: the same capability handler
            # sees both, and the run id is what tells them apart.
            asyncio.run(drive(parent))
            asyncio.run(drive(child))
        assert [step.tool for step in recorder.steps] == ["read_file", "read_file"]
        assert [step.sub for step in recorder.steps] == [False, True]
        assert recorder.steps[1].delegate == "delegate 1"

        # The flags survive the write: a later reader of steps.jsonl can tell
        # whose call it was.
        records = steps_mod.read_steps(recorder.write(tmp_path / "s.jsonl"))
        assert [record["sub"] for record in records] == [False, True]
        assert [record["delegate"] for record in records] == ["", "delegate 1"]

    def test_a_delegate_is_named_once_per_sub_run(self) -> None:
        """Which sub-run made a call, not just that a sub-run made it.

        The harness calls the handler several times for one sub-run, replaying its
        events; only `conversation_id` is stable across those calls and different
        for the next sub-run, so that is what names a delegate. Numbering every
        *call* instead made forty delegates three calls into a turn, each with two
        calls of its own, and pushed the live view off the terminal.
        """
        recorder = steps_mod.StepRecorder(label="reviewer")
        task = "You are reviewing a new Python package…"
        # A replay of the same sub-run is the same delegate, and the handler is
        # handed a fresh context object each time, so nothing here is identity-based.
        assert recorder.delegate_name(_ctx("conv-a", task)) == "delegate 1"
        assert recorder.delegate_name(_ctx("conv-a", task)) == "delegate 1"
        # The next sub-run is the next delegate, whatever its task says.
        assert recorder.delegate_name(_ctx("conv-b", task)) == "delegate 2"
        assert recorder.delegate_name(_ctx("conv-c", "")) == "delegate 3"

    def test_a_cancelled_turn_keeps_its_transcript(self, tmp_path: Path) -> None:
        """A stop is not a failed answer, and the record is written by the caller.

        `CancelledError` is a `BaseException`, so it never reached the failure
        path: a killed run kept its `steps.jsonl` and lost the history, which is
        the artifact that explains why it stopped.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import ToolReturnPart
        from pydantic_ai.models.function import AgentInfo
        from pydantic_ai.models.function import DeltaToolCall
        from pydantic_ai.models.function import FunctionModel

        from agentic.core import stack
        from agentic.core.turn import messages_path
        from agentic.core.turn import run_turn
        from agentic.core.worktree import Tree
        from agentic.tests.support import _REPO_ROOT
        from agentic.workflows.review.workflow import reviewer_step

        def answered(messages: list[ModelMessage]) -> bool:
            return any(
                isinstance(part, ToolReturnPart)
                for message in messages
                for part in getattr(message, "parts", [])
            )

        async def stream(
            messages: list[ModelMessage], _info: AgentInfo
        ) -> AsyncIterator[dict[int, DeltaToolCall]]:
            if answered(messages):  # the model answered, then the run was killed
                raise asyncio.CancelledError
            yield {
                0: DeltaToolCall(
                    name="read_file",
                    json_args='{"path": "pyproject.toml"}',
                    tool_call_id="c1",
                )
            }

        step = reviewer_step(skills_dirs=())
        steps_path = tmp_path / "0-0-reviewer.steps.jsonl"
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                run_turn(
                    stack.build_agent(
                        FunctionModel(stream_function=stream), step.definition(), str
                    ),
                    "review",
                    timeout=60,
                    usage_limits=None,
                    workspace=stack.backend(_REPO_ROOT),
                    deps=Tree(_REPO_ROOT),
                    steps_path=steps_path,
                )
            )
        # The steps stayed live, and the history is beside them under the name
        # the run record points at.
        assert steps_path.is_file()
        assert messages_path(steps_path).is_file()

    def test_a_long_argument_keeps_both_of_its_ends(self) -> None:
        """The end of a command is where a heredoc or a piped tail lives.

        A head-only clip turned a shell call into `{"command": "cd \"…` and hid
        the script a scan was running; the tail says what the command was aimed
        at, so both ends survive and the omitted count is written down.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import FunctionToolCallEvent
        from pydantic_ai.messages import ToolCallPart

        command = "cd /tmp/x && " + " && ".join(f"echo step{i}" for i in range(80))
        args = json.dumps({"command": command})
        assert len(args) > 1_000

        async def events() -> AsyncIterator[Any]:
            yield FunctionToolCallEvent(
                part=ToolCallPart(tool_name="shell", args=args, tool_call_id="c1")
            )

        recorder = steps_mod.StepRecorder(label="r")
        with steps_mod.active(recorder):
            asyncio.run(steps_mod.observe(None, events()))  # type: ignore[arg-type]

        step = recorder.steps[0]
        assert len(step.args) < len(args)
        assert "chars omitted" in step.args
        assert args.startswith(step.args[:800])
        assert step.args.endswith(args[-200:])
        # The whole of it is still what the run made, for a recovered transcript.
        assert recorder.full_arguments(step) == args

    def test_a_long_command_is_summarized_not_printed_as_json(self) -> None:
        """The summary is made from the whole argument, not from a clipped one.

        `trace.tool_call` cut the JSON to 160 characters *before*
        `argument_summary` parsed it, so a command whose JSON passed that length
        was not JSON any more and the line printed `{"command": "cd \"…`, escaped
        quotes and all, instead of the command.
        """
        from agentic.core import trace

        stream = io.StringIO()
        trace.reset()
        try:
            assert trace.configure(
                enabled=True, provider=TracerProviderFixture().provider(), stream=stream
            )
            args = json.dumps(
                {
                    "command": 'cd "$(git rev-parse --show-toplevel)" && cat x.py',
                    "note": "y" * 200,
                }
            )
            assert len(args) > 160
            trace.tool_call("deepseek", "run_command", args, call_id="c1")
        finally:
            trace.reset()
        out = stream.getvalue()
        assert 'cd "$(git rev-parse --show-toplevel)" && cat x.py' in out, out
        assert '{"command"' not in out, out


class TestTheStepRecorderIsACapability:
    def test_a_real_turn_records_its_tool_calls(self) -> None:
        """The recorder is bound as a capability, not passed per run.

        `run()` enables streaming itself when the capability is registered, which
        is what feeds the live view and `steps.jsonl`; without this test the
        wiring could be removed and every unit test would still pass.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import ToolReturnPart
        from pydantic_ai.models.function import AgentInfo
        from pydantic_ai.models.function import DeltaToolCall
        from pydantic_ai.models.function import FunctionModel

        from agentic.core import stack
        from agentic.core.turn import run_turn
        from agentic.core.worktree import Tree
        from agentic.tests.support import _REPO_ROOT
        from agentic.workflows.review.workflow import summarizer_step

        def answered(messages: list[ModelMessage]) -> bool:
            return any(
                isinstance(part, ToolReturnPart)
                for message in messages
                for part in getattr(message, "parts", [])
            )

        async def stream(
            messages: list[ModelMessage], _info: AgentInfo
        ) -> AsyncIterator[dict[int, DeltaToolCall]]:
            if answered(messages):
                yield {
                    0: DeltaToolCall(
                        name="final_result",
                        json_args='{"blocking": false, "text": "nothing blocking"}',
                        tool_call_id="c2",
                    )
                }
            else:
                yield {
                    0: DeltaToolCall(
                        name="read_file",
                        json_args='{"path": "pyproject.toml"}',
                        tool_call_id="c1",
                    )
                }

        step = summarizer_step(skills_dirs=())
        agent = stack.build_agent(
            FunctionModel(stream_function=stream), step.definition(), step.output_type
        )
        turn = asyncio.run(
            run_turn(
                agent,
                "decide",
                timeout=60,
                usage_limits=None,
                workspace=stack.backend(_REPO_ROOT),
                deps=Tree(_REPO_ROOT),
            )
        )
        assert [record.tool for record in turn.steps.steps] == ["read_file"]
        assert turn.steps.steps[0].ok is True
        assert turn.steps.steps[0].seconds is not None


class TestStepsAreWrittenLive:
    def test_a_call_is_on_disk_before_the_turn_ends(self, tmp_path: Path) -> None:
        """The file is appended per event, not dumped when the turn ends.

        A long turn is otherwise invisible until it finishes, and a turn that dies
        leaves nothing behind. `read_steps` merges the call and its outcome.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import FunctionToolCallEvent
        from pydantic_ai.messages import FunctionToolResultEvent
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.messages import ToolReturnPart

        path = tmp_path / "live.steps.jsonl"
        recorder = steps_mod.StepRecorder(label="r", path=path)

        async def call() -> AsyncIterator[Any]:
            yield FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name="read_file", args='{"path": "a"}', tool_call_id="c1"
                )
            )

        async def result() -> AsyncIterator[Any]:
            yield FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name="read_file", content="body", tool_call_id="c1"
                )
            )

        with steps_mod.active(recorder):
            asyncio.run(steps_mod.observe(None, call()))  # type: ignore[arg-type]
            # Before the outcome, and before the turn ends: it is on disk.
            assert path.is_file()
            first = steps_mod.read_steps(path)
            assert [record["tool"] for record in first] == ["read_file"]
            assert first[0]["ok"] is None

            asyncio.run(steps_mod.observe(None, result()))  # type: ignore[arg-type]
        recorder.close()

        merged = steps_mod.read_steps(path)
        assert len(merged) == 1
        assert merged[0]["ok"] is True
        assert merged[0]["result_chars"] == 4
        assert merged[0]["result_head"] == "body"
        assert merged[0]["seconds"] is not None

    def test_a_result_keeps_its_opening_text(self, tmp_path: Path) -> None:
        """A length alone cannot say whether a call worked or produced anything.

        A result's opening text is where friction shows -- `No such file or
        directory`, `command not found`, a scan that found nothing -- and the
        record kept only how many characters came back, so whether a delegate's
        six minutes produced a candidate could not be read out of the run.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import FunctionToolCallEvent
        from pydantic_ai.messages import FunctionToolResultEvent
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.messages import ToolReturnPart

        path = tmp_path / "failed.steps.jsonl"
        recorder = steps_mod.StepRecorder(label="r", path=path)
        content = "sh: line 1: nvidia-smi: command not found\n" + "x" * 10_000

        async def call() -> AsyncIterator[Any]:
            yield FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name="shell",
                    args='{"command": "nvidia-smi"}',
                    tool_call_id="c1",
                )
            )

        async def result() -> AsyncIterator[Any]:
            yield FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name="shell", content=content, tool_call_id="c1"
                )
            )

        with steps_mod.active(recorder):
            asyncio.run(steps_mod.observe(None, call()))  # type: ignore[arg-type]
            asyncio.run(steps_mod.observe(None, result()))  # type: ignore[arg-type]
        recorder.close()

        record = steps_mod.read_steps(path)[0]
        assert record["result_chars"] == len(content)
        head = record["result_head"]
        assert isinstance(head, str)
        assert head.startswith("sh: line 1: nvidia-smi")
        # One line, and bounded: the record is read by people and by JSONL tools.
        assert "\n" not in head
        assert len(head) <= 240


class TestUsage:
    def test_reasoning_tokens_are_recorded(self) -> None:
        # The one counter that says how much of the answer was thinking; it used
        # to be dropped, and one reference reviewer spent 93% of its output there.
        pytest.importorskip("pydantic_ai")
        counters = usage_of(_usage())
        assert counters is not None
        assert counters["output_reasoning_tokens"] == 30
        assert "cost" not in counters

    def test_a_reported_cost_is_recorded(self) -> None:
        # `RunUsage.cost` is a Decimal, and `isinstance(Decimal(...), int | float)`
        # is False: the numeric check dropped it on every provider that reports one.
        pytest.importorskip("pydantic_ai")
        from decimal import Decimal

        from pydantic_ai.usage import RunUsage as _RunUsage

        counters = usage_of(_RunUsage(requests=1, cost=Decimal("0.25")))
        assert counters is not None
        assert counters["cost"] == 0.25

    def test_no_usage_is_none(self) -> None:
        assert usage_of(None) is None


class TestTranscript:
    def test_a_cleared_tool_result_is_put_back(self) -> None:
        """`ClearToolResults` rewrites history in place; the transcript must not.

        One reference run kept 3 of 149 tool returns and replaced the rest with
        `[tool result cleared]`, so a reader could see that a tool was called and
        not what it said.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import ModelRequest
        from pydantic_ai.messages import ToolReturnPart

        from agentic.core.turn import complete_history

        cleared = ToolReturnPart(
            tool_name="read_file",
            content="[tool result cleared]",
            tool_call_id="c1",
        )
        real = ToolReturnPart(
            tool_name="read_file", content="the real text", tool_call_id="c1"
        )
        history: list[ModelMessage] = [ModelRequest(parts=[cleared])]

        completed = complete_history(history, {"c1": real})
        part = completed[0].parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.content == "the real text"
        # What was not recorded is left exactly as the run had it.
        assert complete_history(history, {}) == history

    def test_a_cancelled_run_keeps_its_messages(self) -> None:
        # The timeout case that left no transcript at all: an 1800 s, 299-step
        # run had no messages.json, because the exception discarded the history.
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.exceptions import RunTimedOut
        from pydantic_ai.messages import ModelRequest
        from pydantic_ai.messages import UserPromptPart

        from agentic.core.turn import failure_messages

        messages = [ModelRequest(parts=[UserPromptPart(content="review this")])]
        error = RunTimedOut("timed out", messages=messages)
        assert failure_messages(error, []) == messages

    def test_another_failure_uses_the_captured_history(self) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import ModelRequest
        from pydantic_ai.messages import UserPromptPart

        from agentic.core.turn import failure_messages

        messages = [ModelRequest(parts=[UserPromptPart(content="review this")])]
        assert failure_messages(RuntimeError("boom"), messages) == messages

    def test_a_stop_with_no_history_still_leaves_a_transcript(self) -> None:
        """The scan that kept its steps and lost its transcript.

        A `CancelledError` carries no history and `capture_run_messages` saw
        nothing, so the turn had a `steps.jsonl` and no `messages.json` -- the one
        artifact that explains why it never answered. The recorder saw every call
        and every result, so the transcript is rebuilt from those.
        """
        pytest.importorskip("pydantic_ai")
        from pydantic_ai.messages import FunctionToolCallEvent
        from pydantic_ai.messages import FunctionToolResultEvent
        from pydantic_ai.messages import ModelRequest
        from pydantic_ai.messages import ModelResponse
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.messages import ToolReturnPart

        from agentic.core.turn import dump
        from agentic.core.turn import history_for

        recorder = steps_mod.StepRecorder(label="deepseek")
        args = json.dumps({"command": "rg -n foo"})

        async def call() -> AsyncIterator[Any]:
            yield FunctionToolCallEvent(
                part=ToolCallPart(tool_name="shell", args=args, tool_call_id="c1")
            )

        async def result() -> AsyncIterator[Any]:
            yield FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name="shell", content="a hit", tool_call_id="c1"
                )
            )

        with steps_mod.active(recorder):
            asyncio.run(steps_mod.observe(None, call()))  # type: ignore[arg-type]
            asyncio.run(steps_mod.observe(None, result()))  # type: ignore[arg-type]

        history = history_for(asyncio.CancelledError(), [], recorder)
        assert [type(message) for message in history] == [ModelResponse, ModelRequest]
        part = history[0].parts[0]
        assert isinstance(part, ToolCallPart)
        # The whole argument, not the log's clipped copy.
        assert part.args == args
        # And it serializes like any other transcript.
        assert b"a hit" in dump(history)

        # A turn that did nothing still leaves nothing: absent, not empty.
        assert history_for(asyncio.CancelledError(), [], steps_mod.StepRecorder()) == []


class TestFailedTurnUsage:
    def test_a_failed_turn_reports_what_it_spent(self) -> None:
        """A turn that dies on a later request still recorded 0 tokens.

        Every captured response carries its own usage, and a `RequestUsage` has
        no request count of its own, so it is counted per response.
        """
        from pydantic_ai.messages import ModelResponse
        from pydantic_ai.messages import TextPart
        from pydantic_ai.usage import RequestUsage

        from agentic.core.turn import usage_from_messages

        response = ModelResponse(
            parts=[TextPart("hi")], usage=RequestUsage(input_tokens=10, output_tokens=2)
        )
        usage = usage_from_messages([response, response])
        assert usage is not None
        assert usage.requests == 2
        assert usage.input_tokens == 20
        assert usage.output_tokens == 4
        # A run that got no response at all still reports nothing.
        assert usage_from_messages([]) is None


class TestFailureText:
    """What the record says about a turn that died, not just its message."""

    def test_the_chain_is_kept_because_it_says_what_dropped(self) -> None:
        from agentic.core.turn import failure_text

        cut = OSError("peer closed the connection")
        cut.__cause__ = RuntimeError("Connection error.")
        assert failure_text(cut) == (
            "OSError: peer closed the connection <- RuntimeError: Connection error."
        )

    def test_an_exception_with_no_message_still_names_its_type(self) -> None:
        from agentic.core.turn import failure_text

        assert failure_text(TimeoutError()) == "TimeoutError"

    def test_a_cycle_does_not_spin(self) -> None:
        # `__context__` is set by `raise` inside `except`, so a loop is reachable.
        from agentic.core.turn import failure_text

        first = ValueError("a")
        second = ValueError("b")
        first.__context__ = second
        second.__context__ = first
        assert failure_text(first).count("<- ") == 3


class TestForwardedEnvironment:
    """A command keeps the caller's GPU selection and board identity."""

    def test_an_empty_cuda_visible_devices_is_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentic.core.stack import forwarded_env

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert forwarded_env()["CUDA_VISIBLE_DEVICES"] == ""

    def test_an_unset_variable_is_not_invented(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentic.core.stack import forwarded_env

        monkeypatch.delenv("GPU_BOARD_DIR", raising=False)
        assert "GPU_BOARD_DIR" not in forwarded_env()

    def test_an_agents_shell_cannot_start_a_review_of_its_own_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `scripts/autoreview.py` refuses to run under this marker, which is what
        # keeps a `--fix` session from nesting another review. A reviewer's shell
        # could otherwise start the incumbent with the caller's credentials, on
        # the change it is reviewing.
        from agentic.core.stack import SESSION_MARKER
        from agentic.core.stack import forwarded_env

        monkeypatch.delenv(SESSION_MARKER, raising=False)
        assert forwarded_env()[SESSION_MARKER] == "1"


class TestLiveViewWithTheRecorder:
    """The view's side of the step recorder: arguments and delegate rows."""

    def test_the_recorder_gives_the_view_the_unclipped_arguments(self) -> None:
        """The record clips arguments at 300 chars; the view must not be handed that.

        A JSON argument cut mid-string is not JSON, so `argument_summary` fell back
        to printing `{"command": "cd \"…` and the reader lost the command.
        """

        from pydantic_ai.messages import FunctionToolCallEvent
        from pydantic_ai.messages import ToolCallPart

        from agentic.core import trace

        command = "cd /tmp/x && " + " && ".join(f"echo step{i}" for i in range(80))
        args = json.dumps({"command": command})
        assert len(args) > steps_mod._ARGS_HEAD + steps_mod._ARGS_TAIL

        async def events() -> AsyncIterator[Any]:
            yield FunctionToolCallEvent(
                part=ToolCallPart(tool_name="run_command", args=args, tool_call_id="c1")
            )

        stream = io.StringIO()
        trace.stop()
        try:
            assert trace.configure(
                enabled=True, provider=TracerProviderFixture().provider(), stream=stream
            )
            recorder = steps_mod.StepRecorder(label="deepseek")
            asyncio.run(recorder(None, events()))  # type: ignore[arg-type]
        finally:
            trace.stop()
        written = stream.getvalue()
        assert "cd /tmp/x && echo step0" in written, written
        assert '{"command"' not in written, written
        # The record still keeps its own bounded copy, both ends of it, and the
        # live line carries the command once rather than twice.
        assert "chars omitted" in recorder.steps[0].args
        assert written.count("echo step0") == 1


class TestEvidence:
    """What a turn ran, as the facts a reviewer's prose can be held to."""

    def _write(self, path: Path, *, failed: bool = False) -> None:
        records = [
            {
                "event": "call",
                "id": "c1",
                "tool": "shell",
                "args": '{"command": "rg -n baddbmm helion"}',
                "started_ms": 10,
                "sub": False,
                "delegate": "",
            },
            {
                "event": "result",
                "id": "c1",
                "ok": not failed,
                "result_chars": 240,
                "result_head": "helion/_testing/cute_codegen.py:571: assert op_name",
                "seconds": 0.4,
            },
        ]
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    def test_it_lists_the_command_and_how_it_ended(self, tmp_path: Path) -> None:
        path = tmp_path / "0-0-reviewer.steps.jsonl"
        self._write(path)
        lines = steps_mod.evidence(path).splitlines()
        assert len(lines) == 1
        # The command, as the live view summarizes it, and the result's opening.
        assert "rg -n baddbmm helion" in lines[0]
        assert "cute_codegen.py:571" in lines[0]
        assert "failed" not in lines[0]

    def test_a_failed_call_is_marked(self, tmp_path: Path) -> None:
        path = tmp_path / "0-0-reviewer.steps.jsonl"
        self._write(path, failed=True)
        assert "(failed)" in steps_mod.evidence(path)

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert steps_mod.evidence(tmp_path / "nothing.steps.jsonl") == ""

    def test_the_summary_prompt_carries_what_each_reviewer_ran(
        self, tmp_path: Path
    ) -> None:
        """The gap this closes: a summariser given prose alone re-derives evidence.

        It spent 179 s re-running two reviewers' checks because the only way to
        weigh their claims was to reproduce them.
        """
        pytest.importorskip("pydantic_ai")
        from agentic.core.step import Outcome
        from agentic.workflows.review.workflow import summarize_prompt

        path = tmp_path / "0-0-reviewer.steps.jsonl"
        self._write(path)
        reviews = [
            Outcome[str](
                name="deepseek",
                model="some/model",
                result="Nothing blocks it.",
                elapsed_seconds=1.0,
                steps_path=path,
            )
        ]
        prompt = summarize_prompt("the change `abc...HEAD`", reviews)
        assert "What it ran:" in prompt
        assert "rg -n baddbmm helion" in prompt
        assert "cute_codegen.py:571" in prompt

        # A reviewer with no steps file says nothing extra.
        no_steps = [
            Outcome[str](
                name="glm",
                model="some/model",
                result="Nothing blocks it.",
                elapsed_seconds=0.0,
            )
        ]
        assert "What it ran:" not in summarize_prompt("the change", no_steps)
