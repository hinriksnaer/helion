"""The live view: the tree, its leaves and the append-only renderer."""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.core.trace import Slot


class TestDisplayHelpers:
    """What a leaf says, without a terminal in the way."""

    def test_the_interesting_argument_is_unwrapped(self) -> None:
        from agentic.core.trace import argument_summary

        assert (
            argument_summary("run_command", '{"command": "git log --oneline"}')
            == "git log --oneline"
        )
        assert (
            argument_summary("read_file", '{"path": "a/b.py", "limit": 80}') == "a/b.py"
        )
        assert argument_summary("git_show", '{"revision": "HEAD~1"}') == "HEAD~1"

    def test_an_unknown_tool_falls_back_to_its_arguments(self) -> None:
        from agentic.core.trace import argument_summary

        assert argument_summary("mystery", '{"x": 1, "y": "two"}') == "x=1 y=two"
        assert argument_summary("mystery", "not json") == "not json"

    def test_a_long_argument_is_clipped(self) -> None:
        from agentic.core.trace import argument_summary

        summary = argument_summary("run_command", '{"command": "' + "x" * 200 + '"}')
        assert len(summary) == 62
        assert summary.endswith("\N{HORIZONTAL ELLIPSIS}")

    def test_a_model_id_is_shown_by_its_last_segment(self) -> None:
        from agentic.core.trace import model_name

        assert (
            model_name("openai/rits/deepseek-ai/DeepSeek-V4.1-Flash")
            == "DeepSeek-V4.1-Flash"
        )
        assert model_name("") == ""


class TestLiveView:
    """The in-place view: a root, a task line, its workers, and their children."""

    class TTY(io.StringIO):
        """A stream that says it is a terminal, so `make` picks the view."""

        def isatty(self) -> bool:
            return True

    def _slot(
        self,
        *,
        worker: str = "deepseek",
        task: str = "[0] abc",
        status: str = "running",
        elapsed: float | None = None,
        steps: int = 0,
        task_index: int = 0,
    ) -> Slot:
        from agentic.core.trace import Slot

        return Slot(
            worker=worker,
            task=task,
            started=0.0,
            task_index=task_index,
            status=status,
            elapsed=elapsed,
            steps=steps,
        )

    def test_a_tall_block_is_cut_to_the_terminal(self) -> None:
        """A live view taller than the screen scrolls on every refresh.

        The root line carries the run's clock and turn count, so it stays; the
        rest is the newest lines -- what a person is watching -- and a row says
        how many were left out.
        """
        from agentic.core.trace import block

        slots = [
            self._slot(worker=f"w{n}", status="done", elapsed=1.0, task_index=n)
            for n in range(6)
        ]
        rows = block(slots, width=100, height=8)
        assert len(rows) == 8
        assert rows[0].startswith("\N{BLACK RIGHT-POINTING SMALL TRIANGLE}")
        assert "earlier line(s)" in rows[1]
        assert any("w5" in row for row in rows), "the newest turn must be visible"

    def test_a_finished_turn_drops_its_tool_lines_before_anything_is_cut(self) -> None:
        """The economy before the cut: commentary goes, the summary stays.

        A finished turn's tool lines were the running commentary; its worker line
        already carries how long it took and what it found, so they are the first
        thing to give up when the block does not fit.
        """
        from agentic.core.trace import Entry
        from agentic.core.trace import block

        slot = self._slot(status="done", elapsed=12.0)
        for index in range(3):
            slot.add(
                Entry(
                    kind="tool",
                    text=f"rg needle{index}",
                    state="ok",
                    seconds=0.5,
                    seq=index,
                )
            )
        full = block([slot], width=100)
        assert any("needle0" in row for row in full)
        tight = block([slot], width=100, height=4)
        assert len(tight) <= 4
        assert not any("needle0" in row for row in tight)
        assert any("deepseek" in row for row in tight)

    def test_a_run_is_a_root_over_its_tasks_and_workers(self) -> None:
        from agentic.core.trace import Entry
        from agentic.core.trace import block

        slot = self._slot(steps=12)
        slot.tool_seconds = 3.4
        slot.model = "openai/rits/deepseek-ai/DeepSeek-V4.1-Flash"
        slot.add(Entry(kind="chat", text="chat", state="ok", seconds=5.5, seq=1))
        slot.open_calls["c"] = Entry(kind="chat", text="chat", seq=2)

        rows = block([slot], width=100, frame=0)
        # root -> task -> worker -> children
        assert rows[0].startswith("\N{BLACK RIGHT-POINTING SMALL TRIANGLE}")
        assert "task 1" in rows[1] and "[0] abc" in rows[1]
        assert "deepseek" in rows[2] and "12 calls" in rows[2]
        # The model is named once, on the worker's line.
        assert "DeepSeek-V4.1-Flash" in rows[2]
        assert (
            rows[3]
            == "      \N{BOX DRAWINGS LIGHT VERTICAL}  \N{HEAVY CHECK MARK} chat  5.5s"
        )
        # Both leaves sit at the same level, in the order they happened.
        assert rows[4].startswith("      \N{BOX DRAWINGS LIGHT VERTICAL}  ")
        assert len(rows) == 5

    def test_the_root_counts_finished_turns(self) -> None:
        from agentic.core import trace
        from agentic.core.trace import block

        done = self._slot(status="done", elapsed=2.0)
        running = self._slot(worker="glm", status="running")
        try:
            trace.plan(4)
            rows = block([done, running], width=100)
            assert "1/4 turns" in rows[0]
        finally:
            trace.plan(0)

    def test_a_worker_line_carries_the_verdict(self) -> None:
        from agentic.core.trace import block
        from agentic.core.trace import worker_line

        slot = self._slot(status="done", elapsed=152.0)
        slot.findings = "1 blocker, 2 nits"
        line = worker_line(slot)
        assert line.startswith("      \N{HEAVY CHECK MARK}")
        assert "1 blocker, 2 nits" in line
        assert "152s" in line
        # One task, one worker: root, task, worker.
        assert len(block([slot], width=120)) == 3

    def test_the_block_fits_the_width(self) -> None:
        """Columns that do not fit wrap, and a wrapped line makes a live block jump."""
        from agentic.core.trace import Entry
        from agentic.core.trace import block

        slot = self._slot(
            task="[0] 0947598fa [scripts] Move the review checklist into scripts/",
        )
        slot.open_calls["c"] = Entry(
            kind="tool", text="run_command  {" + '"command": "' + "x" * 120
        )
        for width in (40, 60, 80, 120):
            rows = block([slot], width=width, frame=0)
            assert all(len(row) <= width for row in rows), (width, rows)
        # The long rows are the task and its child, and both are cut, not wrapped.
        narrow = block([slot], width=40, frame=0)
        assert narrow[1].endswith("\N{HORIZONTAL ELLIPSIS}")
        assert narrow[-1].endswith("\N{HORIZONTAL ELLIPSIS}")

    def test_history_is_bounded_and_counted(self) -> None:
        """A long review makes hundreds of calls; the block stays a fixed height."""
        from agentic.core.trace import HISTORY
        from agentic.core.trace import Entry
        from agentic.core.trace import block

        slot = self._slot()
        for step in range(10):
            slot.add(Entry(kind="tool", text=f"call{step}", state="ok", seconds=0.1))
        rows = block([slot], width=100, frame=0)
        assert any(f"… {10 - HISTORY} earlier" in row for row in rows)
        # root, task, worker, the counter, and the kept children
        assert len(rows) == HISTORY + 4
        assert "call9" in rows[-1]

    def test_the_counter_sits_where_the_dropped_children_were(self) -> None:
        """Not always at the top: a model request can still be open below it."""
        from agentic.core.trace import Entry
        from agentic.core.trace import block

        slot = self._slot()
        # Still open, and older than everything that got dropped.
        slot.open.append(Entry(kind="chat", text="chat  gw/model", seq=1))
        for step in range(2, 9):
            slot.add(Entry(kind="tool", text=f"call{step}", state="ok", seq=step))

        rows = block([slot], width=100, frame=0)
        chat_at = next(i for i, row in enumerate(rows) if "chat  gw/model" in row)
        counter_at = next(i for i, row in enumerate(rows) if "earlier" in row)
        assert chat_at < counter_at, rows
        assert "earlier calls" in rows[counter_at]
        # The kept children come after the counter, oldest first.
        assert "call6" in rows[counter_at + 1]
        assert "call8" in rows[counter_at + 3]

    def test_only_leaves_animate(self) -> None:
        """The task's spinner was `FRAMES[0]` forever, next to a moving leaf."""
        from agentic.core import trace
        from agentic.core.trace import block

        slot = self._slot()
        first = block([slot], width=100, frame=0)
        later = block([slot], width=100, frame=3)
        # The task line is a static mark whatever the frame...
        assert first[1] == later[1]
        assert first[1].startswith(f"   {trace.RUNNING} task 1")
        # ...and the worker's own glyph does move.
        assert first[2][6] in trace.FRAMES
        assert first[2][6] != later[2][6]

    def test_a_failed_turn_is_marked(self) -> None:
        from agentic.core import trace
        from agentic.core.trace import worker_line

        slot = self._slot(status="failed", elapsed=1.0)
        assert worker_line(slot).startswith(f"      {trace.GLYPH_FAILED}")

    def test_a_terminal_gets_the_view_and_a_pipe_gets_lines(self) -> None:
        from agentic.core import trace

        assert isinstance(trace.make(io.StringIO()), trace.LiveRenderer)
        assert isinstance(trace.make(self.TTY()), trace.TreeView)

    def test_children_reach_the_turn_that_ran_them(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        from agentic.core import trace

        view = trace.TreeView(self.TTY())
        provider = TracerProvider()
        provider.add_span_processor(view)
        tracer = provider.get_tracer("test")
        with (
            tracer.start_as_current_span(
                "agentic.turn",
                attributes={
                    "agentic.worker": "deepseek",
                    "agentic.task": "one",
                    "agentic.task_index": 0,
                },
            ),
            tracer.start_as_current_span(
                "agentic.turn",
                attributes={
                    "agentic.worker": "glm",
                    "agentic.task": "one",
                    "agentic.task_index": 0,
                },
            ),
        ):
            view.tool_call(
                "deepseek",
                "run_command",
                '{"command": "pytest"}',
                call_id="c1",
            )
            view.tool_result(
                label="deepseek",
                tool="run_command",
                args='{"command": "pytest"}',
                seconds=1.5,
                chars=90,
                ok=True,
                call_id="c1",
            )

        rows = view.lines()
        # One task (both workers are on index 0), then a worker line each, and
        # the tool call under the worker that ran it.
        assert len(rows) == 5
        assert "task 1" in rows[1]
        assert "deepseek" in rows[2]
        assert "run_command" in rows[3] and "1.5s" in rows[3]
        glm_line = next(row for row in rows if "glm" in row)
        assert "run_command" not in glm_line

    def test_a_delegated_call_nests_under_its_delegate(self) -> None:
        """A sub-run's calls belong to the delegate that made them, by name.

        The worker label stays the delegating turn's, because that is what the
        view looks a slot up by: marking the label instead dropped the line
        entirely, which is silent rather than wrong-looking. The delegate's own
        name is the other half: two sub-runs of one turn were one interleaved
        list under a bare `sub:` prefix, with no way to say which ran what.
        """
        from opentelemetry.sdk.trace import TracerProvider

        from agentic.core.trace import TreeView

        view = TreeView(io.StringIO())
        provider = TracerProvider()
        provider.add_span_processor(view)
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span(
            "agentic.turn",
            attributes={
                "agentic.worker": "deepseek",
                "agentic.task": "one",
                "agentic.task_index": 0,
            },
        ):
            view.tool_call(
                "deepseek", "read_file", '{"path": "a"}', call_id="a", sub="core"
            )
            view.tool_result(
                label="deepseek",
                tool="read_file",
                args='{"path": "a"}',
                seconds=0.2,
                chars=10,
                ok=True,
                call_id="a",
                sub="core",
            )
            view.tool_call(
                "deepseek",
                "read_file",
                '{"path": "b"}',
                call_id="b",
                sub="workflows",
            )
            view.tool_result(
                label="deepseek",
                tool="read_file",
                args='{"path": "b"}',
                seconds=0.3,
                chars=10,
                ok=True,
                call_id="b",
                sub="workflows",
            )

            rows = view.lines()

        core = next(row for row in rows if "core" in row)
        workflows = next(row for row in rows if "workflows" in row)
        # Each delegate has its own line and its own count, not the turn's.
        assert "core" in core and "1 calls" in core
        assert "workflows" in workflows and "1 calls" in workflows
        # Its calls are nested under it, nameless: the line above says whose.
        assert rows.index(workflows) > rows.index(core)
        assert not any("sub: " in row for row in rows)
        read_a = next(row for row in rows if "read_file  a" in row)
        assert read_a.startswith("      \u2502  \u2502  "), rows

        # The worker's own line counts the turn's own calls only.
        worker = next(row for row in rows if "deepseek" in row and "calls" in row)
        assert "0 calls" in worker and "2 delegate(s)" in worker

    def test_a_delegate_row_turns_until_its_call_returns(self) -> None:
        """A delegate is working until the `delegate_task` call that made it ends.

        Its own model requests are not in this view -- they are not tool calls and
        the sub-run's spans are not drawn -- so a row whose mark was "do I have a
        call in flight" read as finished for the whole minute it spent thinking.
        The delegation is the only thing that brackets the sub-run, so it is what
        the mark follows.
        """
        from opentelemetry.sdk.trace import TracerProvider

        from agentic.core import trace

        view = trace.TreeView(self.TTY())
        provider = TracerProvider()
        provider.add_span_processor(view)
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span(
            "agentic.turn",
            attributes={
                "agentic.worker": "deepseek",
                "agentic.task": "one",
                "agentic.task_index": 0,
            },
        ):
            view.tool_call(
                "deepseek",
                "delegate_task",
                '{"task": "t"}',
                call_id="d1",
            )
            # The delegate's first call: its row appears, already turning.
            view.tool_call(
                "deepseek", "read_file", '{"path": "a"}', call_id="a", sub="delegate 1"
            )
            view.tool_result(
                label="deepseek",
                tool="read_file",
                args='{"path": "a"}',
                seconds=0.2,
                chars=10,
                ok=True,
                call_id="a",
                sub="delegate 1",
            )
            line = next(row for row in view.lines() if "delegate 1" in row)
            assert any(frame in line for frame in trace.FRAMES), line

            # The call returning is what finishes the row.
            view.tool_result(
                label="deepseek",
                tool="delegate_task",
                args='{"task": "t"}',
                seconds=61.0,
                chars=10,
                ok=True,
                call_id="d1",
            )
            line = next(row for row in view.lines() if "delegate 1" in row)
            assert trace.GLYPH_OK in line, line
            assert not any(frame in line for frame in trace.FRAMES), line

    def test_a_delegate_row_says_how_long_it_has_been_going(self) -> None:
        """The clock, not the tool seconds: a delegate's calls are file reads.

        `11 calls · 0s in tools` was the whole row for a sub-run that had been
        working for minutes, which read as a defect and as nothing happening.
        """
        from agentic.core import trace

        now = time.monotonic()
        ledger = trace.Ledger(opened=now - 372, steps=11, tool_seconds=0.4)
        running = trace.delegate_line("delegate 1", ledger, now=now)
        assert "0.4s in tools" in running, running
        assert "6m12s" in running, running

        # Finished: how long it took, not how old it is.
        ledger.closed = ledger.opened + 65
        done = trace.delegate_line("delegate 1", ledger)
        assert "1m5s" in done, done

    def test_a_delegates_clock_spans_the_call_that_asked_for_it(self) -> None:
        """The clock is the delegation's, not the handler streams'.

        The harness opens and closes the event handler several times per sub-run,
        so a clock that restarted with each stream reported `1s` for a sub-run
        that had been working for five minutes. The parent's `delegate_task` call
        is made before the child starts and returns when it is done, so it is the
        one thing that brackets the whole delegation.
        """
        from agentic.core import trace

        slot = trace.Slot(calls={"d1": trace.Delegate(started=100.0)})
        ledger = slot.ledger("delegate 1")
        assert ledger.opened == 100.0
        assert ledger.closed is None
        assert "2m0s" in trace.delegate_line("delegate 1", ledger, now=220.0)

        # The call returning is what closes it, and the duration stops moving.
        ledger.closed = 400.0
        assert "5m0s" in trace.delegate_line("delegate 1", ledger, now=999.0)

    def test_the_renderer_brackets_a_delegate_with_the_delegating_call(self) -> None:
        """`delegate_task` starting and returning is what opens and closes a row."""
        from opentelemetry.sdk.trace import TracerProvider

        from agentic.core import trace

        view = trace.TreeView(self.TTY())
        provider = TracerProvider()
        provider.add_span_processor(view)
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span(
            "agentic.turn",
            attributes={
                "agentic.worker": "deepseek",
                "agentic.task": "one",
                "agentic.task_index": 0,
            },
        ):
            view.tool_call("deepseek", "delegate_task", '{"task": "t"}', call_id="d1")
            view.tool_call(
                "deepseek", "read_file", '{"path": "a"}', call_id="a", sub="delegate 1"
            )
            ledger = view._slots[next(iter(view._slots))].delegates["delegate 1"]
            assert ledger.closed is None, "a running row must keep turning"

            view.tool_result(
                label="deepseek",
                tool="delegate_task",
                args='{"task": "t"}',
                seconds=1.0,
                chars=10,
                ok=True,
                call_id="d1",
            )
            assert ledger.closed is not None

    def test_only_the_last_delegates_are_drawn(self) -> None:
        """A turn that delegates forty times is not forty blocks tall.

        The block has to stay a fixed height, or the in-place rewrite scrolls the
        view off the terminal and paints over itself; a review that delegates
        twenty-two times did exactly that. Delegates are bounded like a turn's
        other children: the last few, and a count of the rest.
        """
        from agentic.core.trace import Entry
        from agentic.core.trace import block

        slot = self._slot()
        for index in range(5):
            ledger = slot.ledger(f"self #{index + 1}")
            ledger.add(
                Entry(
                    kind="tool",
                    text=f"read_file  f{index}.py",
                    state="ok",
                    seconds=0.0,
                    seq=index + 1,
                )
            )

        rows = block([slot], width=100, frame=0)
        text = "\n".join(rows)
        assert "… 2 earlier delegates" in text
        assert "self #5" in text and "self #1" not in text
        # root, task, worker and the counter, then three delegates of two rows.
        assert len(rows) == 4 + 3 * 2

    def test_a_chat_child_attaches_by_parentage(self) -> None:
        """The bug the demo caught: pydantic-ai's `chat` has no worker attribute.

        Children used to be looked up by `agentic.worker`, which only our own
        turn span carries, so a model request never appeared under its turn.
        """
        from opentelemetry.sdk.trace import TracerProvider

        from agentic.core import trace

        view = trace.TreeView(self.TTY())
        provider = TracerProvider()
        provider.add_span_processor(view)
        tracer = provider.get_tracer("test")
        with (
            tracer.start_as_current_span(
                "agentic.turn",
                attributes={"agentic.worker": "deepseek", "agentic.task": "one"},
            ),
            tracer.start_as_current_span("invoke_agent agent"),
            tracer.start_as_current_span("chat gw/model"),
        ):
            pass

        rows = view.lines()
        assert "deepseek" in rows[2]
        # The agent run is the turn itself, so it is not drawn: the request sits
        # directly under the worker, at the one leaf level.
        assert not any("invoke_agent" in row for row in rows)
        assert "chat  0.0s" in rows[3]
        assert rows[3].startswith("      \N{BOX DRAWINGS LIGHT VERTICAL}  ")

    def test_a_finished_turn_gets_its_tick(self) -> None:
        """The bug the demo caught: the slot was keyed by `id(span)`.

        The SDK hands `on_start` and `on_end` different objects for one span, so
        the slot was opened and never closed: the line kept its spinner.
        """
        from opentelemetry.sdk.trace import TracerProvider

        from agentic.core import trace

        view = trace.TreeView(self.TTY())
        provider = TracerProvider()
        provider.add_span_processor(view)
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span(
            "agentic.turn",
            attributes={"agentic.worker": "deepseek", "agentic.task": "one"},
        ):
            view.tool_call(
                "deepseek",
                "run_command",
                '{"command": "pytest"}',
                call_id="c1",
            )
            assert view.lines()[2][6] in trace.FRAMES

        worker = view.lines()[2]
        assert worker.startswith("      \N{HEAVY CHECK MARK}")
        assert worker.endswith("s)")

    def test_the_view_stops_cleanly_without_a_terminal_start(self) -> None:
        from agentic.core import trace

        view = trace.TreeView(self.TTY())
        view.stop()  # never started: stopping must not raise
        view.start()
        view.stop()
