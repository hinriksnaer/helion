"""The review workflow: a change in, a verdict out.

    tell the reviewers what to review -> two reviews -> one summary -> gate

Nothing here runs git. The agents have a shell and the repository, so the prompt
says *what* is being reviewed ("the last commit", "`origin/main...HEAD`", "the
uncommitted changes") and they read it themselves. That is the same trust the
harness gives `Coder`: a competent reviewer with a clear brief beats a host-side
diff pipeline that decides what it may see.

The two reviewers answer prose, because the next model reads it. The summary is
the one typed answer, because the run acts on it: it says whether the change can
land, and that is the exit code.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated

from pydantic import Field

from ...core.config import WorkflowSettings
from ...core.lifecycle import EXIT_BLOCKED
from ...core.lifecycle import EXIT_INCOMPLETE
from ...core.lifecycle import EXIT_OK
from ...core.lifecycle import Run
from ...core.lifecycle import RunOptions
from ...core.lifecycle import RunSummary
from ...core.report import Description
from ...core.report import Report
from ...core.step import Answer
from ...core.step import Outcome
from ...core.step import Step
from ...core.step import Task
from ...core.step import Worker
from ...core.turn import versions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...core.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent


def _checklist() -> str:
    """The checklist ``scripts/autoreview.py`` also uses, loaded by path."""
    path = _REPO_ROOT / "scripts" / "review_checklist.py"
    spec = importlib.util.spec_from_file_location("_agentic_review_checklist", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load the review checklist from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.REVIEW_CHECKLIST)


def _policy(name: str) -> str:
    return (_HERE / f"{name}.md").read_text().replace("{{CHECKLIST}}", _checklist())


#: The reviewers' policy, as data: it is in the definition digest.
REVIEW_INSTRUCTIONS = _policy("instructions")
#: The summariser's policy.
SUMMARY_INSTRUCTIONS = _policy("summarize")


class Verdict(Answer):
    """What the summariser decided, and what it says to a person."""

    blocking: Annotated[
        bool,
        Field(
            description=(
                "True only if the change must not land as it is: at least one review "
                "found a real defect. Nits, style and 'could be nicer' are not "
                "blocking."
            )
        ),
    ] = False
    text: Annotated[
        str,
        Field(
            description=(
                "What a person needs: blockers first (what, where, why), then nits, "
                "then anything a reviewer could not check. Keep it short."
            )
        ),
    ] = ""

    def summary_line(self) -> str:
        if self.error is not None:
            return f"no verdict: {self.error}"
        first = self.text.strip().splitlines()[0] if self.text.strip() else ""
        return ("blocked: " if self.blocking else "clean: ") + first[:100]


@dataclass(frozen=True, kw_only=True)
class ReviewOptions(RunOptions):
    """One invocation's knobs, from the command line."""

    #: Revisions as given on the command line, handed to the reviewers verbatim.
    revisions: list[str] = field(default_factory=list)
    head: bool = False
    reviewers: list[str] = field(default_factory=list)
    #: Free text describing what to review, for what a revision cannot name.
    target: str = ""


class ReviewSettings(WorkflowSettings):
    """``[workflows.review]``: ``agents`` are the reviewers; ``summarizer`` decides."""

    summarizer: str | None = Field(
        None,
        description=(
            "Role that reads the reviewers' answers and decides. Unset means the "
            "config's default_role."
        ),
    )


#: What one reviewer answered.
AgentOutcome = Outcome[str]


@dataclass(kw_only=True)
class Summary(RunSummary):
    """The review, as data: the reviewers' answers, the verdict, the gate."""

    reviews: list[AgentOutcome] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    summarizer: str = ""
    verdict: Verdict | None = None
    what: str = ""

    @property
    def failures(self) -> int:
        return sum(1 for r in self.reviews if r.failed) + (
            1 if self.verdict is not None and self.verdict.error is not None else 0
        )

    def passed(self) -> bool:
        return (
            self.error is None
            and self.verdict is not None
            and self.verdict.error is None
            and not self.verdict.blocking
            and bool(self.reviews)
            and not any(r.failed for r in self.reviews)
        )

    def result_code(self) -> int:
        """0 clean, 2 a blocker, 3 not fully reviewed (4: could not start)."""
        if not self.reviews or self.failures:
            return EXIT_INCOMPLETE
        return EXIT_BLOCKED if self.verdict and self.verdict.blocking else EXIT_OK


def reviewer_step(*, skills_dirs: tuple[str, ...]) -> Step[str]:
    """A reviewer: the harness agent, the policy, and no answer shape."""
    return Step(
        kind="reviewer",
        instructions=REVIEW_INSTRUCTIONS,
        output_type=str,
        skills_dirs=skills_dirs,
    )


def summarizer_step(*, skills_dirs: tuple[str, ...]) -> Step[Verdict]:
    """The summariser: reads the reviewers, answers `Verdict`."""
    return Step(
        kind="review-summary",
        instructions=SUMMARY_INSTRUCTIONS,
        output_type=Verdict,
        skills_dirs=skills_dirs,
    )


def settings(config: Config) -> ReviewSettings:
    return config.workflow("review", ReviewSettings)


def reviewers(
    run: Run,
    config: Config,
    *,
    roles: Sequence[str] = (),
    snapshot_of: Path | None = None,
) -> list[Worker[str]]:
    """The configured reviewers, each reading a snapshot of ``snapshot_of``."""
    names = config.agent_roles(settings(config), list(roles))
    step = reviewer_step(skills_dirs=tuple(config.skills_dirs))
    return run.build_workers(
        [(name, config.resolve(name), step) for name in names],
        snapshot_of=snapshot_of,
    )


def summarizer(
    run: Run,
    config: Config,
    *,
    role: str | None = None,
    snapshot_of: Path | None = None,
) -> Worker[Verdict]:
    """The single worker that decides."""
    name = role or settings(config).summarizer or config.default_role
    step = summarizer_step(skills_dirs=tuple(config.skills_dirs))
    (worker,) = run.build_workers(
        [(name, config.resolve(name), step)], snapshot_of=snapshot_of
    )
    return worker


def what_to_review(options: ReviewOptions) -> str:
    """The change, in words an agent can act on. No git on this side."""
    if options.revisions:
        return (
            "the change "
            + " ".join(f"`{revision}`" for revision in options.revisions)
            + " (diff it from its merge base, e.g. `git diff origin/main...HEAD`)"
        )
    if options.head:
        return "the last commit (`git show HEAD`), including its message"
    return (
        "the uncommitted changes in this checkout, staged, unstaged and untracked "
        "(`git status`, `git diff HEAD`)"
    )


def review_prompt(what: str, target: str = "") -> str:
    return (
        f"""\
Review {what}.

You are in the tree the change ends at. Read the change yourself -- the shell
and the git tools are there -- and check the surrounding code when a diff alone
does not settle it. `git diff`, `git show`, `git log` and `git status` are the
usual starting points.
"""
        + (
            f"""
What the operator wants looked at, in their words:

{target.strip()}
"""
            if target.strip()
            else ""
        )
        + """
Answer in prose: the blockers first (what, where, why, with the evidence you
saw), then the nits. Say plainly if you found nothing blocking.
"""
    )


def summarize_prompt(what: str, reviews: Sequence[AgentOutcome]) -> str:
    listed = "\n\n".join(
        f"## {review.name} ({review.model})"
        + (" -- FAILED\n" if review.failed else "\n")
        + (review.error or review.result.strip() or "(said nothing)")
        for review in reviews
    )
    return f"""\
Two reviewers looked at {what}. Decide whether it can land, and write what a
person needs to know.

A reviewer that failed looked at nothing; that is not a pass, and not a blocker
either -- say it.

## The reviews

{listed}

Answer with `blocking` (a real defect that must be fixed first) and `summary`
(blockers first, then nits, then what was not checked).
"""


async def review_change(
    run: Run,
    config: Config,
    *,
    reviewers_: Sequence[Worker[str]],
    summarizer_: Worker[Verdict],
    what: str,
    target: str = "",
    snapshot_of: Path | None = None,
) -> tuple[list[AgentOutcome], Outcome[Verdict]]:
    """Two (or more) reviews of one change, then one summary of them."""
    task = Task(
        name=what,
        prompt=review_prompt(what, target),
        inputs={"what": what, "target": target},
    )
    reviews = await run.fan_out(reviewers_, task)
    # `turn`, not `step`: a summariser that fails is a review that did not
    # finish (exit 3), not a run that stopped -- the reviewers still answered.
    verdict = await run.turn(
        summarizer_,
        Task(
            name="summarize",
            prompt=summarize_prompt(what, reviews),
            inputs={
                "what": what,
                "reviews": "|".join(r.artifact or "" for r in reviews),
            },
        ),
    )
    return reviews, verdict


async def review_branch(
    run: Run,
    config: Config,
    *,
    repo: Path,
    base: str,
    name: str,
    reviewers_: Sequence[Worker[str]],
    summarizer_: Worker[Verdict],
) -> tuple[list[AgentOutcome], Outcome[Verdict]]:
    """How another workflow has its branch judged: ``base..HEAD`` in ``repo``."""
    what = f"the change `{base[:12]}...HEAD` in this worktree"
    task = Task(
        name=name,
        prompt=review_prompt(what),
        inputs={"what": what, "base": base},
        revision="HEAD",
    )
    reviews = await run.fan_out(reviewers_, task)
    verdict = await run.turn(
        summarizer_,
        Task(
            name="summarize",
            prompt=summarize_prompt(what, reviews),
            inputs={
                "what": what,
                "reviews": "|".join(r.artifact or "" for r in reviews),
            },
            revision="HEAD",
        ),
    )
    return reviews, verdict


async def run_review(config: Config, options: ReviewOptions, repo: Path) -> Summary:
    """Two reviews and a summary of one change, then the gate."""
    review_settings = settings(config)
    what = what_to_review(options)
    summary = Summary(what=what)
    named = options.revisions or (["--head"] if options.head else ["(working tree)"])
    command = " ".join(["autoreview", *named])
    with Run.open(
        config,
        repo,
        summary,
        command=command,
        settings=review_settings,
        options=options,
        report=REPORT,
    ) as run:
        # Resolve roles before a model is asked: a config typo is "could not
        # start", not "not reviewed".
        summary.agents = config.agent_roles(review_settings, options.reviewers)
        for name in summary.agents:
            config.resolve(name)
        summarizer_role = review_settings.summarizer or config.default_role
        summary.summarizer = summarizer_role
        config.resolve(summarizer_role)
        workers = reviewers(run, config, roles=summary.agents)
        decider = summarizer(run, config, role=summarizer_role)
        run.plan(len(workers) + 1)
        summary.reviews, verdict = await review_change(
            run,
            config,
            reviewers_=workers,
            summarizer_=decider,
            what=what,
            target=options.target,
        )
        summary.verdict = verdict.result if not verdict.failed else None
    return summary


# -- the record --------------------------------------------------------------


def _describe(summary: Summary) -> Description:
    """The review as data, and as a person reads it."""
    return Description(
        fields={
            "what": summary.what,
            "agents": [
                {"name": name, "model": summary.models.get(name, "")}
                for name in summary.agents
            ],
            "summarizer": summary.summarizer,
            "blocking": bool(summary.verdict and summary.verdict.blocking),
            "failures": summary.failures,
            "passed": summary.passed(),
            "reviews": [r.as_record() | {"answer": r.result} for r in summary.reviews],
            "verdict": (
                summary.verdict.model_dump(mode="json") if summary.verdict else None
            ),
        },
        sections=_sections(summary),
        verdict=(
            f"{'blocked' if summary.verdict and summary.verdict.blocking else 'clean'}"
            f" -- {summary.failures} failure(s), exit {summary.exit_code()}"
        ),
        text=_text(summary),
        # A stopped run still shows the reviews it got through.
        partial=_text(summary) if summary.reviews else "",
    )


#: How a review is written down; see `core.report.Report`.
REPORT: Report[Summary] = Report(describe=_describe)


def _sections(summary: Summary) -> list[str]:
    lines = [f"what: {summary.what}", "", "## Reviews", ""]
    for review in summary.reviews:
        lines += [f"### {review.name} ({review.model})", ""]
        lines += [
            "FAILED: " + (review.error or "")
            if review.failed
            else (review.result.strip() or "(said nothing)")
        ]
        lines.append("")
    if summary.verdict is not None:
        lines += ["## Summary", "", summary.verdict.text.strip(), ""]
        if summary.verdict.error:
            lines += [f"(no verdict: {summary.verdict.error})", ""]
    return lines


def _text(summary: Summary) -> str:
    lines = [summary.what, ""]
    for review in summary.reviews:
        status = "FAILED" if review.failed else "reviewed"
        if review.cached and not review.failed:
            status += " (cached)"
        lines += [f"{review.name} ({review.model}): {status}", ""]
    if summary.verdict is not None:
        lines += [summary.verdict.text.strip(), ""]
        lines.append(f"blocking: {'yes' if summary.verdict.blocking else 'no'}")
    return "\n".join(lines).rstrip()


def dry_run(config: Config, options: ReviewOptions, repo: Path) -> str:
    """Describe what a run would ask, without calling a model."""
    review_settings = settings(config)
    refs = [
        (name, config.resolve(name))
        for name in config.agent_roles(review_settings, options.reviewers)
    ]
    lines = [f"repository:  {repo}", f"reviewing:   {what_to_review(options)}"]
    for name, ref in refs:
        provider = config.provider_of(ref)
        lines += [
            f"reviewer:    {name} -> {ref.label}",
            f"             endpoint: {provider.endpoint()}",
            f"             credentials: {provider.key_status()}",
            "             thinking: "
            + (
                str(ref.thinking).lower()
                if ref.thinking is not None
                else "(the provider's own default)"
            ),
        ]
    summarizer_role = review_settings.summarizer or config.default_role
    lines.append(
        f"summarizer:  {summarizer_role} -> {config.resolve(summarizer_role).label}"
    )
    definition = reviewer_step(skills_dirs=tuple(config.skills_dirs)).definition()
    lines += [
        "capabilities: " + ", ".join(definition.names),
        "definition:  " + definition.digest()[:16],
        "versions:    " + ", ".join(f"{k}={v}" for k, v in sorted(versions().items())),
    ]
    return "\n".join(lines)
