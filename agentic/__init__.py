"""Project-specific agentic pipelines for Helion contributors.

An *experimental* home for agent automation that is meant to be shared: the
convention lives in ``agentic/README.md``, the runtime in ``agentic/core``, the
pipelines in ``agentic/workflows``, and the entry points in ``agentic/scripts``.

Nothing in ``agentic/`` is imported by ``helion/`` or by the shipped package. It
is a development tool, and it is disposable by design.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
