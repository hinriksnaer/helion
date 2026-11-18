"""The improve run: a sequence of delegations, one loop, and a branch.

This is the library's level 3 -- *programmatic agent hand-off*: application code
calls one agent after another, keeps one budget, and continues a conversation by
passing its history. The [multi-agent guide][guide] puts graphs at level 4, *"for
the most complex cases, a graph-based state machine"*; this run is a line with a
retry loop whose exit condition the reviewers decide, so it is an `async def`.

Four steps answer **prose**, because the next model reads it: the scan's list,
the chooser's reasoning, the plan, the writer's account. The one typed answer is
the review's, because the run acts on it -- it counts blockers to decide whether
the loop continues and whether the branch is delivered.

[guide]: https://pydantic.dev/docs/ai/guides/multi-agent-applications/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import git
from ...core.lifecycle import Run
from ...core.lifecycle import StepFailed
from ...core.step import Outcome
from ...core.step import Step
from ..review import workflow as review_pipeline
from .record import REPORT
from .record import dry_run
from .state import PREFIX
from .state import Delivered
from .state import ImproveOptions
from .state import ImproveSettings
from .state import Nothing
from .state import Rejected
from .state import Result
from .state import Round
from .state import Summary
from .state import Workers
from .state import independent
from .state import roles
from .state import slug

if TYPE_CHECKING:
    from ...core.config import Config
    from ..review.workflow import Verdict

__all__ = [
    "REPORT",
    "ImproveOptions",
    "ImproveSettings",
    "Summary",
    "dry_run",
    "resolve_base",
    "run_improve",
]

_INSTRUCTIONS = Path(__file__).resolve().parent / "instructions"


def _step(name: str, *, skills_dirs: tuple[str, ...]) -> Step[str]:
    """One step: `Coder` + its policy, answering with prose."""
    return Step(
        kind=f"improve_{name}",
        instructions=(_INSTRUCTIONS / f"{name}.md").read_text(),
        output_type=str,
        skills_dirs=skills_dirs,
    )


async def run_improve(config: Config, options: ImproveOptions, repo: Path) -> Summary:
    """Find something worth doing, make it, and hand back a branch -- or nothing."""
    settings = config.workflow("improve", ImproveSettings)
    summary = Summary()
    with Run.open(
        config,
        repo,
        summary,
        command="improve",
        settings=settings,
        options=options,
        report=REPORT,
    ) as run:
        # The one config error worth failing before a model is asked: two roles
        # pointing at one model makes the choosing step a self-review.
        problem = independent(config, settings)
        if problem:
            raise ValueError(problem)
        base = resolve_base(repo, options.base)
        summary.base = base
        workers = _workers(run, config, settings, base)
        run.log.event(f"base {base[:12]}; roles {roles(config, settings)}")
        try:
            result = await _hand_off(
                run, config, settings, workers, base, repo, summary
            )
        except StepFailed as failed:
            result = Rejected(f"{failed.name} did not answer: {failed.outcome.error}")
        _settle(run, summary, options, result)
    return summary


async def _hand_off(
    run: Run,
    config: Config,
    settings: ImproveSettings,
    workers: Workers,
    base: str,
    repo: Path,
    summary: Summary,
) -> Result:
    """Scan, choose, plan, write, then review until the reviewers agree.

    Every task names `base` as its revision, which is also the tree the writer's
    branch is cut from: what the scan reasons about is what `base..HEAD` will show
    a reviewer. Without it the scan reads the caller's checkout -- usually ahead of
    `base` by every unpushed commit -- and a plan can name a file the writer's tree
    does not have.
    """
    summary.scan = (
        await run.step(
            workers.scan,
            name="scan",
            prompt=_scan_prompt(base, repo, run.timeout),
            inputs={"base": base},
            revision=base,
        )
    ).result
    summary.choice = (
        await run.step(
            workers.choose,
            name="choose",
            prompt=_choose_prompt(summary.scan),
            revision=base,
        )
    ).result
    summary.plan = (
        await run.step(
            workers.plan,
            name="plan",
            prompt=_plan_prompt(summary.choice),
            revision=base,
        )
    ).result

    # The branch names what the chooser chose: its first line, slugged.
    branch = f"{PREFIX}{slug(summary.choice)}"
    if git.branch_exists(branch, cwd=repo):
        return Nothing(
            f"`{branch}` already exists: another run holds this candidate. Delete "
            "the branch to try it again."
        )
    summary.branch = branch
    write = await run.step(
        workers.write,
        name="write",
        prompt=_write_prompt(summary.plan, branch, base, settings.max_commits),
        inputs={"plan": summary.plan},
        branch=branch,
        base=base,
    )
    summary.rounds.append(Round(label="attempt-1", changed=write.result))
    assert write.tree is not None
    if not _commits(write.tree, base):
        return Rejected("the writer committed nothing on the branch", [write.result])

    verdict = await _review(run, config, settings, workers, base, write.tree)
    _record(summary, verdict)
    rounds = 0
    while _blocking(verdict) and rounds < settings.max_rounds:
        rounds += 1
        commits = _commits(write.tree, base)
        write = await run.step(
            workers.write,
            name="write",
            continue_from=write,
            prompt=_refine_prompt(verdict),
            label=f"refine-{rounds}",
        )
        summary.rounds.append(Round(label=f"refine-{rounds}", changed=write.result))
        assert write.tree is not None
        if _commits(write.tree, base) == commits:
            return Rejected(
                "the writer answered a review round without committing anything",
                [_text(verdict)],
            )
        verdict = await _review(
            run, config, settings, workers, base, write.tree, seen=verdict
        )
        _record(summary, verdict)
    if any(item.verdict.failed for item in verdict):
        return Rejected(
            "review could not judge the change: a step failed", [_text(verdict)]
        )
    if _blocking(verdict):
        return Rejected(
            f"review still blocks the change after {rounds} refine round(s)",
            [_text(verdict)],
        )
    return Delivered(branch, _commits(write.tree, base))


def _workers(run: Run, config: Config, settings: ImproveSettings, base: str) -> Workers:
    choosing = roles(config, settings)
    skills = tuple(config.skills_dirs)
    specs = [
        ("scan", _step("scan", skills_dirs=skills)),
        ("choose", _step("choose", skills_dirs=skills)),
        ("plan", _step("plan", skills_dirs=skills)),
        # The writer commits with its own shell: the policy says how, and that
        # the branch is the result. Nothing here needs a bespoke commit tool.
        ("writer", _step("write", skills_dirs=skills)),
    ]
    scan, choose, plan, write = run.build_workers(
        [(name, config.resolve(choosing[name]), step) for name, step in specs]
    )
    # Reviewers are built at the first review: they read a snapshot of the
    # writer's tree, which does not exist until the writer has worked.
    return Workers(scan=scan, choose=choose, plan=plan, write=write)


@dataclass
class CommitVerdict:
    """One commit of the branch, and what the reviewers said about it."""

    commit: str
    verdict: Outcome[Verdict]


async def _review(
    run: Run,
    config: Config,
    settings: ImproveSettings,
    workers: Workers,
    base: str,
    tree: Path,
    *,
    seen: list[CommitVerdict] | None = None,
) -> list[CommitVerdict]:
    """Review the branch **commit by commit**: the unit a stack is submitted in.

    A stack is reviewed one commit at a time, because that is what a maintainer
    approves and what `stack-pr` submits. A commit whose sha did not move between
    rounds keeps the verdict it already has: the writer amending the top commit
    does not re-open the review of the one below it.
    """
    if not workers.reviewers:
        workers.reviewers = review_pipeline.reviewers(
            run, config, roles=settings.reviewers, snapshot_of=tree
        )
        workers.summarizer = review_pipeline.summarizer(run, config, snapshot_of=tree)
    known = {item.commit: item.verdict for item in seen or ()}
    verdicts: list[CommitVerdict] = []
    for commit, parent in _stack(tree, base):
        if commit in known:
            run.log.event(f"  {commit}: kept its verdict from an earlier round")
            verdicts.append(CommitVerdict(commit=commit, verdict=known[commit]))
            continue
        _, verdict = await review_pipeline.review_range(
            run,
            config,
            repo=tree,
            base=parent,
            head=commit,
            name=f"review-{commit}",
            reviewers_=workers.reviewers,
            summarizer_=workers.summarizer,
        )
        verdicts.append(CommitVerdict(commit=commit, verdict=verdict))
    return verdicts


def _record(summary: Summary, verdict: list[CommitVerdict]) -> None:
    round_ = summary.rounds[-1]
    round_.blocking = _blocking(verdict)
    round_.verdict = _text(verdict)


def _blocking(verdict: list[CommitVerdict]) -> bool:
    return any(
        not item.verdict.failed and item.verdict.result.blocking for item in verdict
    )


def _text(verdict: list[CommitVerdict]) -> str:
    if not verdict:
        return "no commits to review"
    parts: list[str] = []
    for item in verdict:
        if item.verdict.failed:
            parts.append(f"{item.commit}: no verdict ({item.verdict.error})")
        else:
            parts.append(f"{item.commit}: {item.verdict.result.text.strip()}")
    return "\n\n".join(parts)


def _stack(tree: Path, base: str) -> list[tuple[str, str]]:
    """Every commit on the branch with its parent, oldest first.

    ``base`` is the parent of the first one, so a one-commit branch is one range.
    """
    try:
        listed = git.git(
            "log", "--reverse", "--format=%h %p", f"{base}..HEAD", cwd=tree
        )
    except git.GitError:
        return []
    stack: list[tuple[str, str]] = []
    parent = base
    for line in listed.splitlines():
        parts = line.split()
        if not parts:
            continue
        commit = parts[0]
        stack.append((commit, parent))
        parent = commit
    return stack


def _commits(tree: Path, base: str) -> list[str]:
    try:
        listed = git.git(
            "log", "--reverse", "--format=%h %s", f"{base}..HEAD", cwd=tree
        )
    except git.GitError:
        return []
    return [line for line in listed.splitlines() if line.strip()]


def _in_flight(repo: Path) -> list[str]:
    """The branches another run already holds. Prompt context, not a gate."""
    try:
        listed = git.git(
            "branch", "--list", f"{PREFIX}*", "--format=%(refname:short)", cwd=repo
        )
    except git.GitError:
        return []
    return [line.strip() for line in listed.splitlines() if line.strip()]


# -- the prompts -------------------------------------------------------------
# What each step is asked. They live next to the call that asks, and carry only
# what that model needs; the rules are in `instructions/*.md`.


def _scan_prompt(base: str, repo: Path, timeout: float) -> str:
    in_flight = _in_flight(repo)
    minutes = max(1, round(timeout / 60))
    return f"""\
This is a checkout of the repository at `{base[:12]}`. Find five small,
verifiable improvements in it.

Another model will investigate your list and choose one; a third will scope it
and a fourth will build it. So make the case for each one, and name the files.

This step has about {minutes} minutes of wall clock before it is stopped, and a
delegation to a copy of yourself has been taking five to ten of them. Start one
only when there is time left to use what it finds.

Already in flight in `{PREFIX}*`, so do not propose these again: \
{", ".join(in_flight) if in_flight else "(nothing)"}

Check the machine before you propose something only a GPU can show: you have a
shell, and `nvidia-smi -L`, `python -c "import torch, triton"` and
`scripts/gpu_board.py status` answer it. This machine is shared.
"""


#: A reader's tree is digested before and after the turn, so an edit left behind
#: discards the answer. Both steps that validate end up wanting to try something,
#: and the safe way to do it is a fact about their workspace rather than a
#: procedure, so it is stated once here and not repeated per policy.
_READER_TREE = """\
This is a private snapshot taken for your turn: it is hashed before and after, so
an edit you leave behind discards your answer. To try something, write a scratch
file under `/tmp` with the shell (`cat > /tmp/try.py <<'EOF'`), or import and
subclass from a scratch script -- the file tools cannot leave the tree, and
anything you put inside it counts as an edit."""


def _choose_prompt(scan: str) -> str:
    return f"""\
Another model scanned this checkout and proposed the candidates below. Investigate
them and choose the one this run works on. You decide: nothing scores you after.

{_READER_TREE}

Start your answer with one imperative line naming the change **you** chose -- it
becomes the branch name -- then say what you found when you checked, the
smallest version worth doing, and the strongest argument against doing it now.

## What the scan found

{scan}
"""


def _plan_prompt(choice: str) -> str:
    return f"""\
Plan the smallest change that carries the value of what was chosen below. You are
not making it: a writer works from your plan and a reviewer holds it to it.

{_READER_TREE}

Say the goal, the files it touches, the edits in order, and how it is checked.

## What was chosen

{choice}
"""


def _write_prompt(plan: str, branch: str, base: str, max_commits: int) -> str:
    return f"""\
Make the change below in this worktree, on branch `{branch}` cut from
`{base[:12]}`. Commit your work there; do not push.

At most {max_commits} commit(s): one is usual, and a second or third is for work
that lands on its own -- a behaviour change, the refactor it needed, the sibling
it exposes. Each commit is reviewed and submitted separately, so each must stand
alone and only depend on the ones below it. Each carries its own message in the
repository's style.

## The plan

{plan}
"""


def _refine_prompt(verdict: list[CommitVerdict]) -> str:
    return f"""\
A review of your branch blocks it. Each section below names the commit it is
about: fix that commit, on the same branch, inside the same plan (amend it, or
`git commit --fixup=<sha>` and `--autosquash` to keep the stack linear). Do not
rewrite a commit the review did not block.

{_text(verdict)}

If one of these is wrong -- the reviewer misreads the code, the fix would break
something the goal requires, or it asks for something this session cannot do
(running `scripts/autoreview.py` needs the marker cleared and a vault this run
does not have) -- do not make a change you believe is wrong. Answer it in a
`Review notes:` paragraph at the end of the commit message, one item per finding,
with what you checked: that paragraph is what the next reviewer reads.
"""


def _settle(
    run: Run, summary: Summary, options: ImproveOptions, result: Result
) -> None:
    """Turn the run's answer into the record.

    A branch is **never** deleted. A refused change is still work -- a commit that
    a person can read, cherry-pick, or fix -- and throwing it away to keep a claim
    namespace tidy is the wrong trade: the claim is a convenience, the commit is
    the product. The branch stays where it is, the record says whether it was
    delivered, and deleting it is a person's decision.
    """
    if isinstance(result, Delivered):
        summary.commits = result.commits
        return
    if isinstance(result, Nothing):
        summary.exhausted = True
        summary.stop = result.reason
        run.log.event(f"  nothing delivered: {result.reason}")
        return
    assert isinstance(result, Rejected)  # the three cases above are all of them
    summary.stop = result.reason
    if summary.branch:
        run.log.event(f"  the branch is kept for inspection: {summary.branch}")


def resolve_base(repo: Path, base: str | None) -> str:
    """The revision the branch is cut from: the merge base with upstream."""
    if base:
        return git.git("rev-parse", "--verify", f"{base}^{{commit}}", cwd=repo).strip()
    for upstream in ("origin/main", "main"):
        try:
            return git.git("merge-base", "HEAD", upstream, cwd=repo).strip()
        except git.GitError:
            continue
    return git.head_sha(cwd=repo)
