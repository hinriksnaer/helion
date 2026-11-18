#!/usr/bin/env python3
"""Find one small improvement, make it, review it, and leave a branch.

    agentic/scripts/improve.py --dry-run            # who, what with, and the shape
    agentic/scripts/improve.py                      # scan, pick, plan, write, review
    agentic/scripts/improve.py --base origin/main   # cut the branch from a revision

Five steps, two of them model-free:

1. **scan**: one model proposes candidates, on a private copy of the tree;
2. **choose**: a *different* model investigates them and picks one, and says why;
3. **plan**: scopes the smallest change that carries the value;
4. **write**: makes it in a worktree on a branch the run owns and commits there --
   one commit per change, up to `max_commits`, when the work has parts;
5. **review**: this directory's review workflow judges the branch, and the
   reviewers send their blockers back to the writer in its own conversation
   until they find none. The reviewer decides; a round that adds no commit, or
   `max_rounds`, ends the loop.

What it leaves is a **branch**: `agentic/improve/<slug of the chosen name>`, cut
from the merge base with upstream. Nothing is
pushed, and a branch the reviewers refuse is **kept**: a commit is work, and
deleting it is a person's decision. The branch is also the claim -- a candidate
whose branch exists is skipped -- so delete a refused one to release its
candidate to a later run.

Exit status:

    0   a branch was delivered, or there was nothing worth doing
    3   no branch: a step did not answer, the writers' branch was already
        claimed, the reviewers still had blockers, or the run stopped partway
    4   could not start: a config error, a scan and choose role pointing at one
        model, a missing extra, a bad base revision

There is deliberately no 2: this workflow's answer is a branch, not a verdict,
so "blocked" is not a state of the run. A refused change is 3 and its findings
are in the record.

Every run writes `agentic/logs/<stamp>-<pid>/index.md` (the candidates, the
choice, the plan, each attempt, the graph) and `run.json`.
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
    from agentic.workflows.improve.pipeline import REPORT
    from agentic.workflows.improve.pipeline import ImproveOptions
    from agentic.workflows.improve.pipeline import dry_run
    from agentic.workflows.improve.pipeline import run_improve


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        default=None,
        metavar="REVISION",
        help=(
            "Cut the branch from here. Default: the merge base of HEAD and "
            "origin/main, which is the change this checkout actually makes."
        ),
    )
    add_run_arguments(
        parser,
        dry_run=(
            "Print the roles, the budgets, each step's capabilities and tool set, "
            "the facts the scan sees and the run's shape, then exit. Needs no "
            "credentials."
        ),
    )
    return parse_arguments(parser, argv)


def _hint(exc: Exception) -> str | None:
    """The commonest mistakes, in the words of what was asked for."""
    text = str(exc)
    if "scan` and `choose` both resolve to" in text:
        return (
            'set `[workflows.improve] choose = "<other-role>"`, or add a role that '
            "points at another model; `independent_choice = false` turns the check "
            "off for a single-model rehearsal"
        )
    if "unknown revision" in text or "ambiguous argument" in text:
        return "pass --base with a revision that exists, e.g. --base origin/main"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = ImproveOptions(base=args.base, **run_options(args))
    return cli.main(
        args,
        report=REPORT,
        start=lambda config, repo: run_improve(config, options, repo),
        dry_run=lambda config, repo: dry_run(config, options, repo),
        hint=_hint,
    )


if __name__ == "__main__":
    sys.exit(main())
