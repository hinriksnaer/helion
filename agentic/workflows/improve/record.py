"""What a run leaves behind: `run.json`, `index.md`, `--dry-run`.

A person opens `index.md` and reads what the scan found, what the chooser decided
and why, the plan, what each attempt changed and what the reviewers said. A script
reads `run.json` and takes the branch name. Nothing here decides anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from ...core.report import Description
from ...core.report import Report
from ...core.turn import versions
from .state import ImproveSettings
from .state import independent
from .state import roles

if TYPE_CHECKING:
    from pathlib import Path

    from ...core.config import Config
    from .state import ImproveOptions
    from .state import Round
    from .state import Summary


def _fields(summary: Summary) -> dict[str, Any]:
    return {
        "branch": summary.branch,
        "base": summary.base,
        "commits": summary.commits,
        "delivered": summary.delivered,
        "exhausted": summary.exhausted,
        "stop": summary.stop,
        "blockers": sum(1 for round_ in summary.rounds if round_.blocking),
        "scan": summary.scan,
        "choice": summary.choice,
        "plan": summary.plan,
        "rounds": [_round(round_) for round_ in summary.rounds],
    }


def _round(round_: Round) -> dict[str, Any]:
    return {
        "label": round_.label,
        "changed": round_.changed,
        "blocking": round_.blocking,
        "verdict": round_.verdict,
    }


def _sections(summary: Summary) -> list[str]:
    lines: list[str] = []
    for title, text in (
        ("What the scan found", summary.scan),
        ("The choice", summary.choice),
        ("The plan", summary.plan),
    ):
        if text.strip():
            lines += [f"## {title}", "", text.strip(), ""]
    for round_ in summary.rounds:
        lines += [f"## {round_.label}", "", round_.changed.strip() or "(no answer)", ""]
        if round_.verdict.strip():
            lines += [f"review: {'blocking' if round_.blocking else 'clean'}", ""]
            lines += [round_.verdict.strip(), ""]
    return lines


def _verdict(summary: Summary) -> str:
    if summary.delivered:
        return (
            f"delivered -- branch `{summary.branch}` with {len(summary.commits)} "
            f"commit(s), {summary.blockers} blocker(s), exit {summary.exit_code()}"
        )
    if summary.exhausted:
        return f"nothing to do -- {summary.stop}, exit {summary.exit_code()}"
    reason = summary.stop or summary.error or "the run stopped"
    return f"no branch -- {reason}, exit {summary.exit_code()}"


def _text(summary: Summary) -> str:
    lines: list[str] = []
    if summary.choice.strip():
        lines.append(f"chose: {summary.choice.strip().splitlines()[0][:100]}")
    for round_ in summary.rounds:
        first = round_.changed.strip().splitlines()[0][:100] if round_.changed else ""
        lines.append(f"{round_.label}: {first or '(no answer)'}")
        if round_.verdict.strip():
            status = "blocking" if round_.blocking else "clean"
            first = round_.verdict.strip().splitlines()[0][:100]
            lines.append(f"  review: {status} -- {first}")
    if summary.delivered:
        lines.extend(("", f"branch `{summary.branch}`:"))
        lines += [f"  {commit}" for commit in summary.commits]
        lines.extend(
            ("", f"review it with: git log --oneline {summary.base}..{summary.branch}")
        )
    elif summary.stop:
        lines.extend(("", summary.stop))
    return "\n".join(lines)


#: How an improve run is written down; see `core.report.Report`.
REPORT: Report[Summary] = Report(
    describe=lambda summary: Description(
        fields=_fields(summary),
        sections=_sections(summary),
        verdict=_verdict(summary),
        text=_text(summary),
        # A stopped run still shows the candidate and the attempt it got through.
        partial=_text(summary) if summary.rounds else "",
    )
)


def dry_run(config: Config, options: ImproveOptions, repo: Path) -> str:
    """Who runs what, with what, and the writer's tool set. No credentials needed."""
    from . import pipeline as pipeline_mod

    settings = config.workflow("improve", ImproveSettings)
    lines = [f"repository:  {repo}"]
    problem = independent(config, settings)
    if problem:
        lines.append(f"WARNING:     {problem}")
    for name, role in roles(config, settings).items():
        ref = config.resolve(role)
        provider = config.provider_of(ref)
        lines += [
            f"{name:<8} {role} -> {ref.label}",
            f"         endpoint: {provider.endpoint()}",
            f"         credentials: {provider.key_status()}",
            "         thinking: "
            + (
                str(ref.thinking).lower()
                if ref.thinking is not None
                else "(the provider's own default)"
            ),
        ]
    lines.append(
        f"reviewers:   {', '.join(settings.reviewers) or '(the review table)'}"
    )
    lines.append(f"max rounds:  {settings.max_rounds} (the reviewers end the loop)")
    lines.append("")
    step = pipeline_mod._step(
        "write",
        skills_dirs=tuple(config.skills_dirs),
        reusable=False,
    )
    definition = step.definition()
    lines += [
        "writer capabilities: " + ", ".join(definition.names),
        "writer definition:   " + definition.digest()[:16],
        "versions:            "
        + ", ".join(f"{k}={v}" for k, v in sorted(versions().items())),
    ]
    return "\n".join(lines)
