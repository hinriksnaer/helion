"""What an improve run carries: options in, the record and the gate out."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import re
from typing import TYPE_CHECKING
from typing import Any

from pydantic import Field

from ...core.config import WorkflowSettings
from ...core.lifecycle import EXIT_INCOMPLETE
from ...core.lifecycle import EXIT_OK
from ...core.lifecycle import RunOptions
from ...core.lifecycle import RunSummary

if TYPE_CHECKING:
    from ...core.config import Config
    from ...core.config import ModelRef

#: The branches this workflow owns. A branch is the result and the claim at once.
PREFIX = "agentic/improve/"

#: The roles, in the order they run. `scan` and `choose` are separate on purpose:
#: the model that proposes must not be the model that decides.
ROLES = ("scan", "choose", "plan", "writer")


def slug(text: str) -> str:
    """A branch-safe slug of what the chooser named: its first line."""
    first = (text.strip().splitlines() or [""])[0]
    cleaned = re.sub(r"[^a-z0-9]+", "-", first.lower()).strip("-")
    return (cleaned or "change")[:48].strip("-")


class Result:
    """What a run produced: one of the three below."""

    def summary(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class Delivered(Result):
    branch: str
    commits: list[str]
    nits: int = 0

    def summary(self) -> str:
        return f"{self.branch} ({len(self.commits)} commit(s), {self.nits} nit(s))"


@dataclass(frozen=True)
class Nothing(Result):
    """There was nothing worth doing -- a success, not a failure."""

    reason: str

    def summary(self) -> str:
        return self.reason


@dataclass(frozen=True)
class Rejected(Result):
    """A change was made and refused. The branch is dropped unless kept."""

    reason: str
    findings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return self.reason


class ImproveSettings(WorkflowSettings):
    """``[workflows.improve]``: who does each step, and how long a loop may run."""

    scan: str | None = Field(None, description="Role that proposes candidates.")
    choose: str | None = Field(None, description="Role that investigates and chooses.")
    plan: str | None = Field(None, description="Role that scopes the change.")
    writer: str | None = Field(None, description="Role that makes the change.")
    reviewers: list[str] = Field(
        default_factory=list,
        description="Roles passed to the review stage; empty means [workflows.review].",
    )
    independent_choice: bool = Field(
        True,
        description=(
            "Refuse to start when `scan` and `choose` are one model: the second "
            "step exists to disagree with the first."
        ),
    )
    max_rounds: int = Field(
        5,
        description=(
            "Safety valve, not a policy: the reviewers end the loop by finding no "
            "blockers, and a round that adds no commit ends it too."
        ),
    )


@dataclass(frozen=True, kw_only=True)
class ImproveOptions(RunOptions):
    """One invocation's knobs, from the command line."""

    #: Keep a branch whose review refused it, for inspection.
    keep_failed: bool = False
    #: Cut the branch from this revision instead of the merge base with upstream.
    base: str | None = None


@dataclass
class Workers:
    """The four steps, and the reviewers built at the first review."""

    scan: Any
    choose: Any
    plan: Any
    write: Any
    reviewers: list[Any] = field(default_factory=list)
    summarizer: Any = None


@dataclass
class Round:
    """One pass at the change: what the writer said, and what review decided."""

    label: str
    changed: str = ""
    blocking: bool = False
    verdict: str = ""


@dataclass(kw_only=True)
class Summary(RunSummary):
    """The whole improve run, as data. Rendering and the exit code derive from it."""

    scan: str = ""
    choice: str = ""
    plan: str = ""
    rounds: list[Round] = field(default_factory=list)
    branch: str = ""
    base: str = ""
    commits: list[str] = field(default_factory=list)
    #: Why the run delivered nothing. Empty when it delivered a branch.
    stop: str = ""
    #: Nothing worth doing was found -- a success, not a failure.
    exhausted: bool = False

    @property
    def delivered(self) -> bool:
        return bool(self.branch) and bool(self.commits)

    @property
    def blockers(self) -> int:
        return sum(1 for round_ in self.rounds if round_.blocking)

    def passed(self) -> bool:
        return self.delivered and self.error is None

    def result_code(self) -> int:
        """0 delivered or nothing worth doing, 3 anything else, 4 never started."""
        if self.delivered or self.exhausted:
            return EXIT_OK
        return EXIT_INCOMPLETE


def role(settings: ImproveSettings, name: str, config: Config) -> str:
    return getattr(settings, name) or config.default_role


def roles(config: Config, settings: ImproveSettings) -> dict[str, str]:
    """Every step's role, in one place."""
    return {name: role(settings, name, config) for name in ROLES}


def independent(config: Config, settings: ImproveSettings) -> str | None:
    """Why the scan and choose roles are the same model, or ``None``."""
    if not settings.independent_choice:
        return None
    scan = config.resolve(role(settings, "scan", config))
    choose = config.resolve(role(settings, "choose", config))
    if _identity(scan) == _identity(choose):
        return (
            f"`scan` and `choose` both resolve to {scan.label}; the choosing step "
            "must be a different model, or set independent_choice = false"
        )
    return None


def _identity(ref: ModelRef) -> tuple[str, str]:
    return (ref.provider, f"{ref.model}|{ref.settings_label}")
