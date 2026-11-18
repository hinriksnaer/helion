"""Helpers the workflow test modules share.

Not fixtures: plain functions and one span-provider helper, so a module can use
them without depending on a test class in another file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from opentelemetry.sdk.trace import TracerProvider
    from pydantic_ai import RunContext
    from pydantic_ai.usage import RunUsage


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ctx(conversation: str, prompt: str) -> RunContext[object]:
    """A child run's `RunContext`, as far as naming a delegate goes.

    Two fields: `conversation_id`, which the harness keeps stable across the
    replays of one sub-run and makes fresh for the next, and `prompt`, the task
    the delegating agent wrote. A real `RunContext` needs a model, a usage and a
    deps object it does not use here; a cast says which two fields it stands in
    for.
    """
    from types import SimpleNamespace

    return cast(
        "RunContext[object]",
        SimpleNamespace(conversation_id=conversation, prompt=prompt),
    )


async def _call(tool: str, args: str) -> AsyncIterator[Any]:
    """One tool-call event, as the agent loop emits it."""
    from pydantic_ai.messages import FunctionToolCallEvent
    from pydantic_ai.messages import ToolCallPart

    yield FunctionToolCallEvent(
        part=ToolCallPart(tool_name=tool, args=args, tool_call_id="d1")
    )


def _usage() -> RunUsage:
    """A pydantic-ai usage record, as one turn reports it."""
    from pydantic_ai.usage import RunUsage as _RunUsage

    return _RunUsage(
        input_tokens=100,
        output_tokens=40,
        output_reasoning_tokens=30,
        cache_read_tokens=60,
        cache_write_tokens=0,
        requests=1,
        tool_calls=2,
    )


class TracerProviderFixture:
    """A provider whose finished spans the test can read back."""

    def __init__(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        self.exporter = InMemorySpanExporter()
        self._provider = TracerProvider()
        self._provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def provider(self) -> TracerProvider:
        return self._provider
