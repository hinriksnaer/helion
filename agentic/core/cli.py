"""What every entry point does once its options are parsed.

Find the checkout, load the local config, describe the run (``--dry-run``) or
run the pipeline -- the one ``asyncio.run`` of the process -- then print the
report and exit with the run's status. Every way of not getting started (no
checkout, a bad config, a revision that does not exist, a role nobody
configured) is one line on stderr and exit status 4, never a traceback and
never mistaken for a finding.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING
from typing import TypeVar

from . import git
from .config import load_config
from .lifecycle import EXIT_CANNOT_START
from .lifecycle import EXIT_OK
from .lifecycle import RunStopped
from .lifecycle import RunSummary

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from collections.abc import Coroutine
    from pathlib import Path
    from typing import Any

    from .config import Config
    from .report import Report

S = TypeVar("S", bound=RunSummary)


def main(
    args: argparse.Namespace,
    *,
    report: Report[S],
    start: Callable[[Config, Path], Coroutine[Any, Any, S]],
    dry_run: Callable[[Config, Path], str] | None = None,
    hint: Callable[[Exception], str | None] | None = None,
) -> int:
    """Run ``start(config, repo)`` and report it; the process exit status.

    ``dry_run`` describes the run instead when ``args.dry_run`` is set; ``hint``
    may add a line of advice to a could-not-start error.
    """
    try:
        repo = git.repo_root()
        config = load_config(args.config, root=repo)
        if dry_run is not None and args.dry_run:
            print(dry_run(config, repo))
            return EXIT_OK
        summary = asyncio.run(start(config, repo))
    except RunStopped as stopped:
        # The run already wrote its record; report it and use its status (3),
        # rather than the "could not start" (4) the same exception means before
        # the first turn.
        report.print(stopped.summary, as_json=args.json)
        if stopped.summary.log_dir is not None:
            print(f"log: {stopped.summary.log_dir}", file=sys.stderr)
        return stopped.summary.exit_code()
    except (FileNotFoundError, ValueError, git.GitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        advice = hint(exc) if hint is not None else None
        if advice:
            print(f"hint: {advice}", file=sys.stderr)
        return EXIT_CANNOT_START
    report.print(summary, as_json=args.json)
    if summary.log_dir is not None:
        print(f"log: {summary.log_dir}", file=sys.stderr)
    return summary.exit_code()
