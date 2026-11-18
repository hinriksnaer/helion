"""What every entry point shares, importable without the ``agentic`` extra.

An entry point imports its workflow inside :func:`extra_required`, so a checkout
that never installed ``.[agentic]`` gets one line saying what to install and
exit status 4 ("could not start"), not a traceback. The flags every run takes
are added here too (:func:`add_run_arguments`), because an entry point parses
its command line before it knows whether the extra is there.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator

#: ``agentic.core.lifecycle.EXIT_CANNOT_START``, which cannot be imported before
#: the extra is known to be there; a test pins the two together.
CANNOT_START = 4

#: The top-level modules the extra installs.
_EXTRA = frozenset({"pydantic", "pydantic_ai", "pydantic_ai_harness", "opentelemetry"})


@contextmanager
def extra_required() -> Iterator[None]:
    """Turn a missing extra into a one-line error and exit status 4."""
    if sys.version_info < (3, 11):
        # `core/config.py` raises ImportError for a missing tomllib, which is not
        # a missing extra and used to surface as a traceback with exit 1. The
        # interpreter is the thing to name, and this is "could not start" (4).
        print(
            "error: agentic needs Python 3.11 or newer (tomllib, and pydantic-ai's "
            f"own floor); this is {sys.version.split()[0]}.\n"
            "       Run it with the project's environment python.",
            file=sys.stderr,
        )
        raise SystemExit(CANNOT_START) from None
    try:
        yield
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in _EXTRA:
            raise
        print(
            f"error: agentic needs its own extra ({exc.name} is missing).\n"
            "       pip install -e '.[agentic]'   # or run with that environment's python",
            file=sys.stderr,
        )
        raise SystemExit(CANNOT_START) from None


def parse_arguments(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """Parse a command line; a usage error exits 4, not argparse's 2.

    2 means "a blocker was found" to this tool's callers, so a typo that exited
    with argparse's status would look like a defect. ``--help`` still exits 0.
    """
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 2:
            raise SystemExit(CANNOT_START) from None
        raise


def add_run_arguments(
    parser: argparse.ArgumentParser, *, dry_run: str | None = None
) -> None:
    """The flags every run takes: ``core.lifecycle.RunOptions``, config, output.

    ``dry_run`` is the workflow's description of ``--dry-run``; a workflow
    without one gets no such flag.
    """
    parser.add_argument(
        "--config", default=None, metavar="PATH", help="Config file (TOML)."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Wall-clock budget per agent turn. Default: from the config.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore recorded answers and run every turn again.",
    )
    parser.add_argument(
        "--reuse-stale",
        action="store_true",
        help=(
            "Serve a recorded answer even though the agent changed since it was "
            "made (its instructions, composition, tool set or environment). Off "
            "by default: another agent's answer is not this agent's. Use it when "
            "the input is unchanged and only the agent was edited."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Print the live span tree (agent run, model request, every tool call "
            "with what it ran) while the agents work. HELION_AGENTIC_TRACE=1 does "
            "the same for every run; OTEL_EXPORTER_OTLP_ENDPOINT also exports the "
            "spans there (Jaeger, Tempo, ...)."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the run record as JSON."
    )
    if dry_run is not None:
        parser.add_argument("--dry-run", action="store_true", help=dry_run)


def run_options(args: argparse.Namespace) -> dict[str, Any]:
    """The ``RunOptions`` fields from flags :func:`add_run_arguments` added."""
    return {
        "no_cache": args.no_cache,
        "reuse_stale": args.reuse_stale,
        "timeout": args.timeout,
        "trace": args.trace,
    }
