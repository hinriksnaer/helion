"""Live tracing of agent turns, over pydantic-ai's own OpenTelemetry spans.

The run record says what a turn did, but only after it ends: ``steps.jsonl`` is
written when ``agent.run`` returns, so a thirteen-minute turn is thirteen
minutes of silence and "is it still running?" is answered by ``ps``. pydantic-ai
already emits a span per agent run, per model request and per tool call (the
``gen_ai.*`` conventions), and a ``SpanProcessor`` sees each one *when it
starts*. That is the heartbeat, without inventing a format of our own.

Two things are ours, because a span does not carry them:

* :class:`LiveRenderer` prints the structural spans as a tree, nesting from the
  span's parent: our ``agentic.turn``, pydantic-ai's ``invoke_agent`` and
  ``chat``.
* :func:`tool_call` prints the tool calls, *with their arguments* -- which
  pydantic-ai attaches only when the call returns, far too late to see what is
  running. The step recorder calls it, so the live view names the command.

Instrumentation is off unless asked for (``--trace`` or
``HELION_AGENTIC_TRACE=1``), and it is deliberately outside the capability
composition: it changes what a run *shows*, never what an agent can see, so
turning it on does not invalidate a single recorded verdict.

Spans go to the terminal, and to an OTLP endpoint as well when the standard
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set -- so a self-hosted Jaeger or Tempo gets
the same tree with no flag of ours. Content capture is off: the prompt is the
change under review, and a trace is not the place for it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
import importlib
import json
import os
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import TextIO

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import SpanContext
from opentelemetry.trace import StatusCode

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping

    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace import Span
    from opentelemetry.sdk.trace import TracerProvider
    from rich.text import Text

#: Set to a truthy value to trace every workflow, not just the one that has a
#: ``--trace`` flag. ``--trace`` asks for the same thing for one run.
ENV_VAR = "HELION_AGENTIC_TRACE"

#: The standard OTLP variable. When it is set, spans are also exported there;
#: unset means the terminal is the only sink.
OTLP_ENV_VAR = "OTEL_EXPORTER_OTLP_ENDPOINT"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: What the root of the tree says: the command, when the run started, and how
#: many turns it will run in total (set once the pipeline has resolved targets).
_TITLE = ""
_STARTED = 0.0
_TOTAL = 0

# -- installation -------------------------------------------------------------

_LOCK = threading.Lock()
_PROVIDER: Any = None
_RENDERER: Any = None


def requested(explicit: bool | None = None) -> bool:
    """Whether to trace: the caller's flag if it has one, else the environment."""
    if explicit is not None:
        return explicit
    return os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY


def active() -> bool:
    """Whether tracing is installed in this process right now."""
    return _RENDERER is not None


def configure(
    *,
    enabled: bool | None = None,
    provider: TracerProvider | None = None,
    stream: TextIO | None = None,
    title: str = "",
) -> bool:
    """Install live tracing once per process; return whether it is on.

    ``provider`` and ``stream`` exist for the tests, which need a provider they
    can read spans back from and a stream they can assert on. A second call
    reuses what the first installed rather than stacking processors: a run calls
    this from :meth:`agentic.core.lifecycle.Run.open`, and a workflow that never
    does is traced only if the environment asks for it.

    ``Agent.instrument_all`` is called outside the lock: it is pydantic-ai's own
    global, and calling it while holding a lock this module also takes is how a
    re-entrant call deadlocks (``configure(False)`` used to call ``reset`` under
    the lock and hung the first test that ran).
    """
    global _PROVIDER, _RENDERER, _TITLE, _STARTED

    tracer_provider: TracerProvider | None = None
    detached: TracerProvider | None = None
    with _LOCK:
        if not requested(enabled):
            detached, renderer = _detach()
            if renderer is not None:
                renderer.stop()
        else:
            if _PROVIDER is None:
                if title:
                    _TITLE = title
                _STARTED = time.monotonic()
                _PROVIDER = provider if provider is not None else _build_provider()
                # A terminal gets the in-place view, anything else lines.
                _RENDERER = make(stream)
                _PROVIDER.add_span_processor(_RENDERER)
                _RENDERER.start()
            tracer_provider = _PROVIDER
    if tracer_provider is None:
        if detached is not None:
            _uninstrument()
        return False
    _instrument(tracer_provider)
    return True


def plan(total: int) -> None:
    """How many turns this run will run, for the root line's count."""
    global _TOTAL
    _TOTAL = total


def _draw(draw: Callable[[], None]) -> None:
    """Draw, and stop drawing if the terminal went away.

    A renderer writes to stderr; a closed pipe (`| head`) must cost the live
    view, not the turn it is watching.
    """
    global _RENDERER
    if _RENDERER is None:
        return
    try:
        draw()
    except Exception:
        _RENDERER = None


def turn_done(label: str, summary: str) -> None:
    """What a worker concluded, shown on its line once the turn is over."""
    renderer = _RENDERER
    if renderer is not None:
        _draw(lambda: renderer.turn_done(label, summary))


def summary() -> str:
    """Where a traced run's spans go, for the run record."""
    endpoint = os.environ.get(OTLP_ENV_VAR, "").strip()
    return "spans to stderr" + (f" and {endpoint}" if endpoint else "")


def stop() -> None:
    """Flush what is buffered and turn tracing off; safe to call when off.

    The OTLP exporter is a batch one, so a run that ends without a flush exports
    nothing at all -- the record is complete and the trace is empty, which is the
    worst way to fail. ``force_flush`` before ``shutdown`` is what makes the last
    spans arrive.
    """
    with _LOCK:
        provider, renderer = _detach()
    if renderer is not None:
        renderer.stop()
    if provider is None:
        return
    provider.force_flush()
    provider.shutdown()
    _uninstrument()


def reset() -> None:
    """Stop tracing, for the next test in a module to start from nothing."""
    with _LOCK:
        provider, renderer = _detach()
    if renderer is not None:
        renderer.stop()
    if provider is not None:
        _uninstrument()


def _detach() -> tuple[TracerProvider | None, Any]:
    """Give up the installed provider and renderer; the caller holds the lock."""
    global _PROVIDER, _RENDERER

    provider, renderer = _PROVIDER, _RENDERER
    _PROVIDER = _RENDERER = None
    return provider, renderer


def _instrument(provider: TracerProvider) -> None:
    from pydantic_ai import Agent
    from pydantic_ai import InstrumentationSettings

    # `include_content=False`: prompts and tool results are the change under
    # review, and the terminal or an OTLP backend is not where they belong.
    Agent.instrument_all(
        InstrumentationSettings(tracer_provider=provider, include_content=False)
    )


def _uninstrument() -> None:
    from pydantic_ai import Agent

    # `False`, not `None`: pydantic-ai resolves instrumentation with
    # `if not instrument`, so False is its documented "off" value while None is
    # only what the property returns before anything is set.
    Agent.instrument_all(False)


def _build_provider() -> TracerProvider:
    """A provider that renders to the terminal, and exports when asked to."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    endpoint = os.environ.get(OTLP_ENV_VAR, "").strip()
    if not endpoint:
        return provider
    try:
        # Resolved at run time because the exporter is deliberately *not* in the
        # `agentic` extra: an install that never sets the endpoint should not need
        # it, and the except below is where the message to install it lives.
        exporter = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        ).OTLPSpanExporter
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"{OTLP_ENV_VAR} is set but the OTLP exporter is not installed; "
            "pip install opentelemetry-exporter-otlp-proto-http, or unset it "
            "to trace to the terminal alone"
        ) from exc
    provider.add_span_processor(BatchSpanProcessor(exporter()))
    return provider


# -- what a turn uses ----------------------------------------------------------


@contextmanager
def span(name: str, attributes: Mapping[str, str] | None = None) -> Iterator[None]:
    """One span for this block, or nothing at all when tracing is off."""
    if _PROVIDER is None:
        yield
        return
    tracer = _PROVIDER.get_tracer("agentic")
    with tracer.start_as_current_span(name, attributes=dict(attributes or {})):
        yield


def tool_call(
    label: str, tool: str, args: object, *, call_id: str, sub: str = ""
) -> None:
    """What the turn is running now: the tool and its arguments.

    ``call_id`` is the harness's id for the call, which is what pairs this line
    with its :func:`tool_result`: one response's calls run concurrently, so
    position is not an identity. ``sub`` names the delegated sub-run that made the
    call, which is also how the view files it: the label stays the worker, so the
    line still attaches to the turn that started the delegate.
    """
    renderer = _RENDERER
    if renderer is None:
        return
    # The whole argument, not a clipped one: a JSON argument cut at some length is
    # not JSON, so a long command fell back to printing the raw `{"command": …}`.
    # `argument_summary` flattens and clips what it found.
    _draw(lambda: renderer.tool_call(label, tool, args, call_id=call_id, sub=sub))


def tool_result(
    *,
    label: str,
    tool: str,
    args: object,
    seconds: float,
    chars: int,
    ok: bool,
    call_id: str,
    sub: str = "",
) -> None:
    """The matching close: how long the call took and how much it said.

    ``args`` is repeated on purpose: one response's tool calls run concurrently,
    so a duration alone does not say which call it closed. ``sub`` is the delegate
    that made it, as in :func:`tool_call`.
    """
    renderer = _RENDERER
    if renderer is None:
        return
    _draw(
        lambda: renderer.tool_result(
            label=label,
            tool=tool,
            args=args,
            seconds=seconds,
            chars=chars,
            ok=ok,
            call_id=call_id,
            sub=sub,
        )
    )


GLYPH_OK = "\N{HEAVY CHECK MARK}"
GLYPH_FAILED = "\N{HEAVY BALLOT X}"


#: The span names that open a line in the live view. Everything else a turn
#: emits is a tool call, which arrives through :meth:`tool_call` instead -- with
#: the arguments, which a span does not carry until the call ends.
STRUCTURAL = frozenset({"agentic.turn", "invoke_agent", "chat"})

#: The argument a tool call is really about: a command's text, not the JSON
#: wrapper around it. Anything else is shown as `key=value` pairs.
_ARGUMENT = {
    "run_command": "command",
    "check_command": "command",
    "start_command": "command",
    "git_show": "revision",
    "git_diff": "revisions",
    "git_log": "revisions",
    "read_file": "path",
    "search_files": "pattern",
    "find_files": "pattern",
    "list_directory": "path",
    "delegate_task": "task",
}

#: How much of a leaf's text a row keeps: one glance, not a transcript. The
#: record (``steps.jsonl``) keeps 300.
_LEAF_CHARS = 62

#: What a turn is doing while no tool is running.
THINKING = "thinking"

#: The rotation shown while a slot is running: the one thing that says "alive"
#: without a repaint per event. Only leaves animate -- a spinning run or task
#: header says nothing new and draws the eye away from the worker that is
#: actually waiting.
FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: What an ancestor shows while something under it is running.
RUNNING = "\N{BLACK RIGHT-POINTING SMALL TRIANGLE}"

#: How many finished children a turn keeps. A long review makes hundreds of tool
#: calls; the block has to stay a fixed height, so old ones are dropped and
#: counted.
HISTORY = 3

#: How many delegated sub-run blocks a turn keeps. A model that delegates forty
#: times would otherwise draw forty blocks -- a hundred rows for one running turn,
#: which scrolls the live view off the bottom of the terminal and leaves the
#: in-place rewrite printing over itself.
DELEGATE_HISTORY = 3

#: The narrowest a task column is allowed to get before it is dropped, so a
#: command on an 80-column terminal still fits.
_MIN_TASK = 12

#: Indentation: a task under the root, a worker under its task, and the one
#: level of leaves -- model requests and the calls they asked for -- under that.
_TASK = "   "
_WORKER = "      "
_LEAF = "│  "
_ARROW = "\N{BLACK RIGHT-POINTING SMALL TRIANGLE}"


@dataclass
class Entry:
    """One child of a turn: a model request, or a tool call."""

    kind: str  # "chat" | "tool" | "agent"
    text: str
    depth: int = 0
    #: Creation order, so a parent that closes after its children stays where it
    #: opened instead of jumping to the bottom when it moves into the history.
    seq: int = 0
    state: str = "open"  # "open" | "ok" | "failed"
    seconds: float | None = None

    def glyph(self, frame: int) -> str:
        if self.state == "open":
            return FRAMES[frame % len(FRAMES)]
        return GLYPH_OK if self.state == "ok" else GLYPH_FAILED


@dataclass
class Ledger:
    """One agent's children in the live view: its requests and its tool calls.

    A turn keeps one of these for itself and one per delegated sub-run. That is
    the difference between "a call a delegate made" and "*which* delegate made
    it": with one shared ledger two sub-runs' calls interleaved in the same
    three rows, and a worker's counter could not say whose work it was counting.
    """

    #: Tool calls in flight, by the harness's call id: one response's calls run
    #: concurrently, so a single "current call" slot lost the others.
    open_calls: dict[str, Entry] = field(default_factory=dict)
    #: Children still running (a model request, an agent run).
    open: list[Entry] = field(default_factory=list)
    #: Finished children, oldest first; bounded to ``HISTORY``.
    history: list[Entry] = field(default_factory=list)
    dropped: int = 0
    #: Of those dropped children, how many were tool calls: a header counts
    #: steps, so the counter has to say which it means.
    dropped_steps: int = 0
    steps: int = 0
    tool_seconds: float = 0.0
    #: When a delegation started, and when the turn that asked for it got its
    #: answer. Only a delegate's ledger carries a clock: a turn's own is its slot's.
    opened: float = 0.0
    closed: float | None = None

    def add(self, entry: Entry) -> None:
        if entry in self.open:
            self.open.remove(entry)
        self.history.append(entry)
        while len(self.history) > HISTORY:
            self.dropped += 1
            if self.history.pop(0).kind == "tool":
                self.dropped_steps += 1

    def glyph(self, frame: int) -> str:
        """The row's mark: turning while this agent is still working.

        A delegate is working until the ``delegate_task`` call that started it
        returns -- its own model requests are not in this ledger -- so that call is
        what says whether the row is finished. An ancestor's mark is static because
        the leaf under it already says what is happening.
        """
        if self.closed is None or self.open or self.open_calls:
            return FRAMES[frame % len(FRAMES)]
        if any(entry.state == "failed" for entry in self.history):
            return GLYPH_FAILED
        return GLYPH_OK


@dataclass(eq=False)
class Delegate:
    """One ``delegate_task`` call: the row it started, and that row's clock.

    The call is the only thing that brackets a sub-run's whole lifetime -- it is
    made before the child starts and returns when the child is done -- so its two
    times are the row's. The harness's event handler for a sub-run opens and
    closes several times, which is why the stream is not a clock.
    """

    started: float
    closed: float | None = None
    ledger: Ledger | None = None


@dataclass
class Slot(Ledger):
    """One turn as the live view shows it: a header, its children, its delegates."""

    worker: str = ""
    task: str = ""
    started: float = 0.0
    task_index: int = 0
    #: The model this worker runs: shown once on its own line instead of on
    #: every model request under it.
    model: str = ""
    status: str = "running"
    elapsed: float | None = None
    #: What the reviewer concluded, once it has: "1 blocker, 2 nits".
    findings: str = ""
    #: A delegated sub-run's children, by the name the step recorder gave the
    #: delegate, in the order the delegates first appeared.
    delegates: dict[str, Ledger] = field(default_factory=dict)
    #: The turn's ``delegate_task`` calls, by the harness's call id; a delegate
    #: claims the oldest unclaimed one, which is what its clock is measured from.
    calls: dict[str, Delegate] = field(default_factory=dict)

    def ledger(self, delegate: str) -> Ledger:
        """Where this call's children go: a delegate's own, or the turn's own."""
        if not delegate:
            return self
        known = self.delegates.get(delegate)
        if known is not None:
            return known
        ledger = Ledger()
        claim = next((d for d in self.calls.values() if d.ledger is None), None)
        if claim is not None:
            claim.ledger = ledger
            ledger.opened = claim.started
            ledger.closed = claim.closed
        else:
            # A sub-run no `delegate_task` call of this turn started (a caller
            # driving the agent itself): its own first stream is all there is.
            ledger.opened = time.monotonic()
        self.delegates[delegate] = ledger
        return ledger


def argument_summary(tool: str, args: object) -> str:
    """A tool call as one readable phrase: the command, the path, the query.

    The view used to print the raw JSON, which put `{"command": …}` in front of
    the only part a reader wants and pushed the duration off the line.
    """
    # Flattened first: a real model writes literal newlines inside a JSON string,
    # which `json.loads` refuses, and one tree line cannot hold them anyway.
    text = " ".join(str(args).split())
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            field_name = _ARGUMENT.get(tool)
            value = payload.get(field_name) if field_name else None
            if isinstance(value, str):
                text = value
            else:
                text = " ".join(
                    f"{key}={value}"
                    for key, value in payload.items()
                    if value not in (None, "", [], {})
                )
    flat = " ".join(text.split())
    if len(flat) <= _LEAF_CHARS:
        return flat
    return flat[: _LEAF_CHARS - 1] + "…"


def model_name(label: str) -> str:
    """The last segment of `provider/model`: model ids are long and constant."""
    return label.rsplit("/", 1)[-1] if label else ""


def _tool_time(seconds: float) -> str:
    """Time in tools: a decimal below ten seconds, so a fast one is not "0s"."""
    return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds // 60:.0f}m{seconds % 60:0.0f}s"


def root_line(*, title: str = "") -> str:
    """The run itself: what it is doing."""
    return f"{_ARROW} {title or 'agentic run'}"


def task_line(group: list[Slot], index: int) -> str:
    """One task: the commit or unit every worker under it is reviewing."""
    elapsed = max((slot.elapsed or 0.0) for slot in group)
    suffix = (
        "" if any(slot.status == "running" for slot in group) else f"  {elapsed:.0f}s"
    )
    return f"{_TASK}{_state(group)} task {index + 1}  {group[0].task}{suffix}"


def worker_line(slot: Slot, *, frame: int = 0) -> str:
    """One worker on that task, with what it has found so far.

    The counters are the turn's own calls; a delegated sub-run counts under its
    own line, because a worker that spent ten minutes waiting on two delegates
    should not read as one that made three hundred calls itself.
    """
    delegates = f" · {len(slot.delegates)} delegate(s)" if slot.delegates else ""
    if slot.status == "running":
        glyph = FRAMES[frame % len(FRAMES)]
        tail = (
            f"{slot.steps} calls{delegates} · "
            f"{_tool_time(slot.tool_seconds)} in tools · {_clock(slot)}"
        )
    else:
        glyph = GLYPH_OK if slot.status == "done" else GLYPH_FAILED
        verdict = f"{slot.findings} · " if slot.findings else ""
        tail = f"{verdict}{slot.elapsed or 0.0:.0f}s{delegates}"
    model = f"  {model_name(slot.model)}" if slot.model else ""
    return f"{_WORKER}{glyph} {slot.worker}{model}  ({tail})"


def delegate_line(
    name: str, ledger: Ledger, *, frame: int = 0, now: float | None = None
) -> str:
    """One delegated sub-run: which delegate, how long it ran, what it did itself.

    The clock is the number that answers "is it doing anything": a delegate's tool
    seconds are milliseconds of file reads, so `0s in tools` was the whole story
    before this, for a sub-run that had been working for minutes.
    """
    tail = f"{ledger.steps} calls"
    if ledger.tool_seconds:
        tail += f" · {_tool_time(ledger.tool_seconds)} in tools"
    if ledger.opened:
        # A finished sub-run shows the time it took; a running one shows its age.
        clock = ledger.closed if ledger.closed is not None else now
        if clock is None:
            clock = time.monotonic()
        tail += f" · {_elapsed(max(clock - ledger.opened, 0.0))}"
    return f"{_WORKER}{_LEAF}{ledger.glyph(frame)} {name}  ({tail})"


def children(ledger: Ledger, *, frame: int = 0, base: str = _WORKER) -> list[str]:
    """The turn's children, as tree lines: its agent runs, requests and calls."""
    rows: list[str] = []
    entries = sorted(
        [*ledger.history, *ledger.open, *ledger.open_calls.values()],
        key=lambda e: e.seq,
    )
    counter = ""
    if ledger.dropped:
        what = "earlier calls" if ledger.dropped_steps == ledger.dropped else "earlier"
        counter = f"{base}{_LEAF}… {ledger.dropped} {what}"
    # The counter belongs where the dropped children were, which is not always
    # the top: a model request that is still open can be older than they are.
    first_kept = min((entry.seq for entry in ledger.history), default=0)
    placed = False
    for entry in entries:
        if counter and not placed and entry.seq >= first_kept:
            rows.append(counter)
            placed = True
        suffix = f"  {entry.seconds:.1f}s" if entry.seconds is not None else ""
        # Every leaf sits at the same level, in the order it happened: a model
        # request's children are the calls it asked for, and those always follow
        # it, so nesting them another level only made the request look like a
        # sibling of the worker.
        rows.append(f"{base}{_LEAF}{entry.glyph(frame)} {entry.text}{suffix}")
    if counter and not placed:
        rows.append(counter)
    return rows


def block(
    slots: list[Slot], *, width: int = 80, frame: int = 0, height: int | None = None
) -> list[str]:
    """The whole run: a root, one line per task, and its workers under them.

    Grouping by task is what makes a run over several tasks readable: eight
    reviewer turns otherwise appear as eight unrelated blocks with no indication
    of which task they belong to or how much of the run is left.

    ``height`` is the terminal's, when there is one, and a block taller than the
    screen scrolls the whole view on every refresh -- which leaves a screenful of
    half-drawn frames. Two economies apply, in that order, before anything is
    dropped: a finished turn's tool lines go (its worker line already carries the
    duration, the call count and the verdict), and if it is still too tall the
    block is cut to the newest lines with a row saying how many were left out.
    Without a ``height`` -- a file, a pipe, a test -- nothing is economised and
    every line is drawn.
    """
    rows = _block_rows(slots, width=width, frame=frame, collapse=False)
    if height is None or len(rows) <= max(height, 1):
        return rows
    rows = _block_rows(slots, width=width, frame=frame, collapse=True)
    if len(rows) <= max(height, 1):
        return rows
    return _clamp(rows, height=height, width=width)


def _block_rows(
    slots: list[Slot], *, width: int, frame: int, collapse: bool
) -> list[str]:
    """The block's lines, with a finished turn's commentary kept or dropped."""
    done = sum(1 for slot in slots if slot.status != "running")
    rows = [
        _fit(
            f"{root_line(title=_TITLE)}{_summarise(done=done, total=_TOTAL)}",
            width,
        )
    ]
    for index, group in _by_task(slots):
        rows.append(_fit(task_line(group, index), width))
        for slot in group:
            rows.append(_fit(worker_line(slot, frame=frame), width))
            live = slot.status == "running" or not collapse
            if live:
                for row in children(slot, frame=frame):
                    rows.append(_fit(row, width))
            # A delegate's calls are nested under its own line, one indent in: two
            # sub-runs on one turn are two blocks, not one interleaved list. Only
            # the last few are drawn -- a turn that delegates forty times must not
            # be forty blocks tall -- and the rest are counted, like children are.
            delegates = list(slot.delegates.items())
            dropped = max(len(delegates) - DELEGATE_HISTORY, 0)
            if dropped:
                rows.append(
                    _fit(f"{_WORKER}{_LEAF}… {dropped} earlier delegates", width)
                )
            for name, ledger in delegates[dropped:]:
                rows.append(_fit(delegate_line(name, ledger, frame=frame), width))
                if not live:
                    continue
                # One indent under the delegate's line: `children` adds the leaf.
                for row in children(ledger, frame=frame, base=f"{_WORKER}{_LEAF}"):
                    rows.append(_fit(row, width))
    return rows


def _clamp(rows: list[str], *, height: int, width: int) -> list[str]:
    """Cut the block to the terminal: the root line, then the newest lines.

    The root carries the run's clock and its turn count, so it survives; what is
    dropped is the oldest body line, with a row saying how many. A terminal with
    no room for even that keeps its last line, which is the running turn.
    """
    if height < 3:
        return rows[-max(height, 1) :]
    tail = rows[-(height - 2) :]
    omitted = len(rows) - 1 - len(tail)
    marker = _fit(f"{_WORKER}{_LEAF}… {omitted} earlier line(s)", width)
    return [rows[0], marker, *tail]


def _summarise(*, done: int, total: int) -> str:
    elapsed = _elapsed(time.monotonic() - _STARTED) if _STARTED else "0s"
    turns = f"{done}/{total} turns" if total else f"{done} turns"
    return f"   {elapsed} · {turns}"


def _by_task(slots: list[Slot]) -> list[tuple[int, list[Slot]]]:
    """Slots grouped by task, in the order the tasks first appeared."""
    groups: dict[int, list[Slot]] = {}
    for slot in slots:
        groups.setdefault(slot.task_index, []).append(slot)
    return sorted(groups.items())


def _state(group: list[Slot]) -> str:
    """An ancestor's mark: static, because the moving part is the leaf."""
    if any(slot.status == "running" for slot in group):
        return RUNNING
    if any(slot.status == "failed" for slot in group):
        return GLYPH_FAILED
    return GLYPH_OK


def _fit(text: str, width: int) -> str:
    """Cut to ``width`` with an ellipsis, and never leave a line longer."""
    if width <= 0 or len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _clock(slot: Slot) -> str:
    seconds = time.monotonic() - slot.started
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds // 60:.0f}m{seconds % 60:0.0f}s"


def _turn_label(span: ReadableSpan) -> tuple[str, str, int]:
    """The worker, the task, and the task's position, from a turn span."""
    attributes = span.attributes or {}
    worker = str(attributes.get("agentic.worker") or "?")
    task = str(attributes.get("agentic.task") or "")
    raw = attributes.get("agentic.task_index")
    index = int(raw) if isinstance(raw, int | str) else 0
    return worker, task, index


def _key(span: ReadableSpan) -> int:
    """A span's identity for this process.

    Not ``id(span)``: the SDK hands ``on_start`` and ``on_end`` *different*
    objects for one span, so an object-keyed slot is opened and never closed --
    the line keeps its spinner and the turn never gets its tick.
    """
    context = span.get_span_context()
    return context.span_id if context is not None else id(span)


def _parent_key(span: ReadableSpan) -> int | None:
    """The span id of this span's parent, when it has one.

    A child is attached to its turn through parentage, not through an attribute:
    ``agentic.worker`` is on our own turn span, and pydantic-ai's ``chat`` span
    does not carry it -- looking a worker up there attached nothing, which is
    why the first version's children never appeared.
    """
    parent = span.parent
    if parent is None:
        return None
    if isinstance(parent, SpanContext):
        return parent.span_id
    context = parent.get_span_context()
    return context.span_id if context is not None else None


def _is_structural(name: str) -> bool:
    return name.split(" ", 1)[0] in STRUCTURAL


def _failed(span: ReadableSpan) -> bool:
    # `StatusCode` is a plain Enum, so comparing its member to the int 2 is
    # False and every failed span would print a tick.
    return span.status.status_code == StatusCode.ERROR


class TreeView(SpanProcessor):
    """A turn's header plus its last few children, rewritten in place.

    ``Live`` is given a callable renderable, so rich re-reads the state at its
    own refresh rate: our side only mutates a dict, never touches the cursor, and
    a burst of tool calls cannot flicker the screen. The block is fitted to the
    terminal width and to a fixed child count, so a long review does not scroll
    and an 80-column terminal does not wrap.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        from rich.console import Console
        from rich.live import Live

        console = Console(file=stream or sys.stderr, highlight=False, soft_wrap=True)
        #: Turn spans in the order they opened, so lines never jump around.
        self._slots: dict[int, Slot] = {}
        self._order: list[int] = []
        #: The running turn of each worker, for the tool lines that follow it.
        self._by_worker: dict[str, int] = {}
        #: Open child entries by span, so the end of a span closes its own row.
        self._open: dict[int, tuple[Slot, Entry]] = {}
        #: span id -> its parent's span id, for the walk to the turn.
        self._parents: dict[int, int] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._console = console
        self._frame = 0
        self._started = False
        # `Live` evaluates its renderable while it is constructed -- it decides
        # the initial height before the first refresh -- so every attribute the
        # renderable reads has to exist before this line.
        self._live = Live(
            console=console, get_renderable=self._render, refresh_per_second=8
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._live.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        # Finished turns keep their block on screen once the run is over.
        self._live.stop()

    def shutdown(self) -> None:
        self.stop()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # -- SpanProcessor -----------------------------------------------------

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if not _is_structural(span.name):
            return
        kind = span.name.split(" ", 1)[0]
        with self._lock:
            parent = _parent_key(span)
            if parent is not None:
                self._parents[_key(span)] = parent
            if kind == "agentic.turn":
                worker, task, index = _turn_label(span)
                slot = Slot(
                    worker=worker,
                    task=task,
                    started=time.monotonic(),
                    task_index=index,
                    model=str((span.attributes or {}).get("agentic.model") or ""),
                )
                self._slots[_key(span)] = slot
                self._order.append(_key(span))
                self._by_worker[worker] = _key(span)
            else:
                found = self._owning_slot(span)
                if found is None:
                    return
                slot, depth = found
                if kind == "invoke_agent":
                    # The agent run wraps the whole turn, and the worker's line
                    # already says it: a row for it would nest every request one
                    # level deeper for nothing.
                    return
                self._seq += 1
                # One hop on the way to the turn is that invisible agent run, so
                # a model request draws at depth 0.
                # A request's span name is "chat <provider/model>", the same
                # string on every row; the worker's line says it once.
                entry = Entry(kind=kind, text="chat", seq=self._seq)
                slot.open.append(entry)
                self._open[_key(span)] = (slot, entry)

    def on_end(self, span: ReadableSpan) -> None:
        if not _is_structural(span.name):
            return
        kind = span.name.split(" ", 1)[0]
        with self._lock:
            state = "failed" if _failed(span) else "ok"
            seconds = _seconds(span)
            if kind == "agentic.turn":
                slot = self._slots.get(_key(span))
                if slot is None:
                    return
                slot.elapsed = seconds
                # "done", not "ok": a turn's own line has two terminal states,
                # while a child is only ever open, ok or failed.
                slot.status = "failed" if _failed(span) else "done"
                return
            opened = self._open.pop(_key(span), None)
            if opened is None:
                return
            slot, entry = opened
            entry.state = state
            entry.seconds = seconds
            slot.add(entry)

    # -- the step recorder's children --------------------------------------

    def tool_call(
        self, label: str, tool: str, args: object, *, call_id: str, sub: str = ""
    ) -> None:
        with self._lock:
            slot = self._slot_for(label)
            if slot is not None:
                if tool == "delegate_task":
                    # Made before the child starts and returning when it is done:
                    # the one thing that brackets a sub-run a replay cannot split.
                    slot.calls[call_id] = Delegate(time.monotonic())
                ledger = slot.ledger(sub)
                self._seq += 1
                ledger.open_calls[call_id] = Entry(
                    kind="tool",
                    text=_tool_text(tool, args),
                    seq=self._seq,
                )

    def tool_result(
        self,
        *,
        label: str,
        tool: str,
        args: object,
        seconds: float,
        chars: int,
        ok: bool,
        call_id: str,
        sub: str = "",
    ) -> None:
        with self._lock:
            slot = self._slot_for(label)
            if slot is None:
                return
            if tool == "delegate_task":
                call = slot.calls.get(call_id)
                if call is not None:
                    call.closed = time.monotonic()
                    if call.ledger is not None:
                        call.ledger.closed = call.closed
            ledger = slot.ledger(sub)
            ledger.steps += 1
            ledger.tool_seconds += seconds
            entry = ledger.open_calls.pop(call_id, None)
            if entry is None:
                self._seq += 1
                entry = Entry(kind="tool", text="", seq=self._seq)
            entry.state = "ok" if ok else "failed"
            # The arguments are repeated on purpose: one response's tool calls
            # run concurrently, so the close is matched by what it ran.
            entry.text = _tool_text(tool, args)
            entry.seconds = seconds
            ledger.add(entry)

    def turn_done(self, label: str, summary: str) -> None:
        with self._lock:
            slot = self._slot_for(label)
            if slot is not None:
                slot.findings = summary

    def _slot_for(self, worker: str) -> Slot | None:
        return self._slots.get(self._by_worker.get(worker, -1))

    def _owning_slot(self, span: ReadableSpan) -> tuple[Slot, int] | None:
        """The turn a child span belongs to, and how deep under it, by parentage.

        The chain is ``agentic.turn`` -> ``invoke_agent`` -> ``chat`` -> tools, so
        one hop is not enough: looking only at the immediate parent attached
        nothing, because a model request's parent is the agent run.
        """
        key = _parent_key(span)
        depth = 0
        while key is not None and depth < 8:
            slot = self._slots.get(key)
            if slot is not None:
                return slot, depth
            key = self._parent_of_key(key)
            depth += 1
        slot = self._slot_for(_worker_of(span))
        return (slot, 0) if slot is not None else None

    def _parent_of_key(self, key: int) -> int | None:
        """The parent of a span we have already seen, from its recorded chain."""
        return self._parents.get(key)

    # -- rendering ---------------------------------------------------------

    def lines(self) -> list[str]:
        """The block as plain text, fitted to the width.

        The renderable is this plus styling, and the tests assert on this: a
        terminal is not needed to know what the view says.
        """
        with self._lock:
            slots = [self._slots[key] for key in self._order]
            return block(slots, width=self._console.width, frame=self._frame)

    def _render(self) -> Text:
        from rich.text import Text

        with self._lock:
            self._frame += 1
            slots = [self._slots[key] for key in self._order]
            width = self._console.width
        # One block for the whole run: the root and its tasks are what make a
        # per-commit run legible, so this must not be drawn per turn.
        rows = block(slots, width=width, frame=self._frame, height=self._console.height)
        text = Text(no_wrap=True, overflow="ellipsis")
        for index, row in enumerate(rows):
            if index:
                text.append("\n")
            stripped = row.lstrip()
            lead = len(row) - len(stripped)
            glyph = stripped[0] if stripped else " "
            style = "dim" if row.startswith(_WORKER) else ""
            text.append(row[:lead])
            if glyph == GLYPH_OK:
                text.append(glyph, style="green")
            elif glyph == GLYPH_FAILED:
                text.append(glyph, style="red")
            elif glyph in FRAMES:
                text.append(glyph, style="cyan")
            elif index == 0:
                text.append(row[lead:], style="bold")
                continue
            else:
                text.append(glyph, style=style)
            text.append(row[lead + 1 :], style=style)
        return text


def _seconds(span: ReadableSpan) -> float:
    if span.end_time and span.start_time:
        return (span.end_time - span.start_time) / 1e9
    return 0.0


def _worker_of(span: ReadableSpan) -> str:
    attributes = span.attributes or {}
    return str(attributes.get("agentic.worker") or "?")


#: How far the append-only view will indent before it stops following parents.
#: A span tree deeper than this is not a run shape worth drawing.
_MAX_DEPTH = 32


class LiveRenderer(SpanProcessor):
    """Append-only lines, for a stream that is not a terminal.

    A file or a pipe has no cursor to move, so this prints the same events the
    view shows: a tool call with its arguments when it starts, its duration when
    it ends, and the turn's structure around them.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._open: dict[int, tuple[int, int]] = {}
        #: span id -> its parent's span id. Nesting has to come from a map of our
        #: own: ``ReadableSpan.parent`` is a ``SpanContext``, which has no
        #: ``parent``, so walking it stopped after one hop and every span deeper
        #: than a child -- a delegated sub-run's ``invoke_agent``, its chat, its
        #: tools -- rendered flat under the worker that started it.
        self._parents: dict[int, int] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Nothing to start: each line is printed when it happens."""

    def stop(self) -> None:
        """Nothing to stop: no cursor state is held between lines."""

    def shutdown(self) -> None:
        self.stop()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # -- SpanProcessor -----------------------------------------------------

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        parent = _parent_key(span)
        with self._lock:
            # Every span's parent, drawn or not: a structural span's chain to the
            # root can pass through a span this renderer does not print.
            if parent is not None:
                self._parents[_key(span)] = parent
            if not _is_structural(span.name):
                return
            depth = self._depth(span)
            self._open[_key(span)] = (depth, span.start_time or time.time_ns())
        line = f"{'  ' * depth}\N{BLACK RIGHT-POINTING SMALL TRIANGLE} {_name(span)}"
        # A closed pipe must cost the view, not the turn: OpenTelemetry does not
        # catch what a span processor raises.
        self._draw(line)

    def on_end(self, span: ReadableSpan) -> None:
        if not _is_structural(span.name):
            return
        with self._lock:
            depth, started = self._open.pop(
                _key(span), (self._depth(span), span.start_time or time.time_ns())
            )
        ended = span.end_time or time.time_ns()
        mark = GLYPH_FAILED if _failed(span) else GLYPH_OK
        line = f"{'  ' * depth}{mark} {_name(span)} {(ended - started) / 1e9:.1f}s"
        self._draw(line)

    def _depth(self, span: ReadableSpan) -> int:
        """How far under the root this span sits, by parentage we recorded."""
        depth = 0
        key = _parent_key(span)
        while key is not None and depth < _MAX_DEPTH:
            depth += 1
            key = self._parents.get(key)
        return depth

    # -- the step recorder's lines ----------------------------------------

    def tool_call(
        self, label: str, tool: str, args: str, *, call_id: str, sub: str = ""
    ) -> None:
        arrow = "\N{BLACK RIGHT-POINTING SMALL TRIANGLE}"
        self._write(f"{_sub_label(label, sub)} {arrow} {_tool_text(tool, args)}")

    def tool_result(
        self,
        *,
        label: str,
        tool: str,
        args: str,
        seconds: float,
        chars: int,
        ok: bool,
        call_id: str,
        sub: str = "",
    ) -> None:
        mark = GLYPH_OK if ok else GLYPH_FAILED
        self._write(
            f"{_sub_label(label, sub)} {mark} {_tool_text(tool, args)}  {seconds:.1f}s"
        )

    # -- printing ----------------------------------------------------------

    def turn_done(self, label: str, summary: str) -> None:
        """Nothing to do: the append view already printed the turn's structure."""

    def line(self, text: str) -> None:
        self._write(text)

    def _draw(self, text: str) -> None:
        """Write one line, and let a closed pipe cost the view rather than the turn.

        The renderer is a span processor, and OpenTelemetry does not catch what
        one raises: `| head` closing the pipe used to fail the turn it was watching.
        """
        try:
            self._write(text)
        except Exception:
            return

    def _write(self, text: str) -> None:
        with self._lock:
            stream = self._stream or sys.stderr
            stream.write(text.rstrip("\n") + "\n")
            stream.flush()


def make(stream: TextIO | None = None) -> TreeView | LiveRenderer:
    """The renderer for this stream: in place on a terminal, lines otherwise."""
    target = stream if stream is not None else sys.stderr
    isatty = getattr(target, "isatty", None)
    if callable(isatty) and isatty():
        return TreeView(target)
    return LiveRenderer(target)


def _tool_text(tool: str, args: object) -> str:
    """A tool line's text. Whose call it was is the line it is nested under."""
    return f"{tool}  {argument_summary(tool, args)}"


def _sub_label(label: str, sub: str) -> str:
    """The worker whose turn this line belongs to, and the delegate that ran it."""
    return f"{label} › {sub}" if sub else label


def _name(span: ReadableSpan) -> str:
    """A span's name, plus what labels it when it has attributes we set."""
    if span.name.split(" ", 1)[0] == "agentic.turn":
        worker, task, index = _turn_label(span)
        return f"{span.name}  [{index}] {worker}  {task}"
    return span.name
