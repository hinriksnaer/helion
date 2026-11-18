#!/usr/bin/env python3
"""Review a change with two independent composed agents.

    agentic/scripts/autoreview.py                     # the uncommitted tree
    agentic/scripts/autoreview.py --head               # the last commit
    agentic/scripts/autoreview.py --dry-run            # what would run, and with what
    agentic/scripts/autoreview.py --reviewer glm       # one reviewer instead of two

This is not a replacement for ``scripts/autoreview.py``. That tool drives
coding-agent CLIs; this one drives composed pydantic-ai agents. They are
deliberately separate until this one has shown, on real stacks, that it finds
what the other finds. Running it changes nothing about the existing process.

Reviewers come from ``[workflows.review] agents`` in the local config, so an unadorned
run asks two independent models -- the same shape as the incumbent's claude +
codex, because one model agreeing with itself is not a review. They run
concurrently and their findings are never merged into an opaque verdict: what
blocks is computed from the contract, per reviewer.

Exit status, deliberately **not** the incumbent's:

    0   reviewed, nothing blocking (nits do not block)
    2   a blocker was found
    3   not fully reviewed: a reviewer failed, the working tree changed during
        the run, or there was nothing to review
    4   could not start: a config error, a revision that does not exist, a
        missing extra, a reviewer that could not be built

The incumbent returns 0 however many findings it printed and 2 only when the run
aborts; this tool's 2 means exactly "the change is blocked", so a gate must be
told which of the two it is reading. 4 exists for the same reason: a typo in a
local config must not look like a defect in the change under review.

Every run writes ``agentic/logs/<stamp>-<pid>/``: ``index.md`` first, then the
full message history and validated answer per reviewer, plus ``run.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Running this file directly puts `agentic/scripts` on sys.path, not the repo
# root the package is imported from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentic.scripts._entry import add_run_arguments  # noqa: E402
from agentic.scripts._entry import extra_required  # noqa: E402
from agentic.scripts._entry import parse_arguments  # noqa: E402
from agentic.scripts._entry import run_options  # noqa: E402

with extra_required():
    from agentic.core import cli
    from agentic.workflows.review import REPORT
    from agentic.workflows.review import ReviewOptions
    from agentic.workflows.review import dry_run
    from agentic.workflows.review import run_review


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "revisions",
        nargs="*",
        metavar="REVISION",
        help="Revisions to review, e.g. origin/main..HEAD. Default: the working tree.",
    )
    parser.add_argument(
        "--head", action="store_true", help="Review the last commit instead."
    )
    parser.add_argument(
        "--target",
        default="",
        metavar="TEXT",
        help=(
            "Free text describing what to review, passed to the reviewers "
            "verbatim, e.g. --target 'the agentic/ implementation in the diffs'. "
            "Use it when the change cannot be named as a revision range; it is "
            "part of the prompt, and so part of the cache key."
        ),
    )
    parser.add_argument(
        "--reviewer",
        action="append",
        default=None,
        metavar="ROLE|PROVIDER/MODEL",
        help=(
            "Review with this capability role (e.g. 'deepseek') or an explicit "
            "'provider/model'. Repeatable. Default: [workflows.review] agents from the "
            "local config."
        ),
    )
    add_run_arguments(
        parser,
        dry_run=(
            "Print the target, the reviewers and the capability composition, "
            "then exit. Needs no credentials; it is how a degraded agent is "
            "spotted."
        ),
    )
    return parse_arguments(parser, argv)


def _hint(exc: Exception) -> str | None:
    """The commonest mistake: writing what you want in words.

    Neither this tool nor the incumbent takes "current diffs" -- the working tree
    is what you get by passing no revision at all.
    """
    if "unknown revision" in str(exc) or "ambiguous argument" in str(exc):
        return (
            "pass no revision to review the working tree, or a git range such as "
            "origin/main..HEAD"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = ReviewOptions(
        revisions=list(args.revisions),
        head=args.head,
        reviewers=list(args.reviewer or ()),
        target=args.target,
        **run_options(args),
    )
    return cli.main(
        args,
        report=REPORT,
        start=lambda config, repo: run_review(config, options, repo),
        dry_run=lambda config, repo: dry_run(config, options, repo),
        hint=_hint,
    )


if __name__ == "__main__":
    sys.exit(main())
