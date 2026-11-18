"""What every workflow run does, whatever the workflow is about.

:class:`Run` owns the run directory, the workers and their workspaces, every
turn's evidence, the decision to serve a recorded answer instead of running a
turn, and the check that decides whether an answer may be recorded at all.

No agent works in the caller's checkout (:mod:`agentic.core.worktree`):

* a **reader** gets a private tree of the tree it is about -- a checkout of the
  task's revision, or a snapshot of the working tree -- checked after every turn.
  An answer is recorded as soon as its turn ends with the tree unchanged; a turn
  that changed it gets no verdict, and the workspace is rebuilt.

The caller's checkout is still hashed before and after, as a backstop: a change
there means an agent reached outside its workspace (or someone edited the
checkout during the run), and the run says so instead of reporting a result.

:meth:`Run.fan_out` is how turns are run: independent workers over one task,
concurrently (each in its own workspace), each answer kept separate (a review,
an assessment).

Every pipeline is a coroutine, and its entry point is the one place that calls
``asyncio.run``: stages are awaited, never run on an event loop of their own.

What a turn *means* -- which workers run, how answers combine, what the gate
says -- stays in the workflow. A workflow's reusable stage takes a ``Run`` and
returns outcomes, so another workflow can run it inside its own run.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import cast

from . import artifact as artifact_store
from . import git
from . import trace as trace_mod
from . import worktree as worktree_mod
from .config import build_model
from .report import RunLog
from .report import output_paths
from .step import Outcome
from .step import Task
from .step import Worker
from .step import answer_summary
from .turn import run_turn
from .turn import versions

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence
    from pathlib import Path

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.workspaces import LocalWorkspaceBackend

    from .artifact import Artifact
    from .config import Config
    from .config import ModelRef
    from .config import WorkflowSettings
    from .report import Report
    from .stack import Composed
    from .step import Step
    from .turn import TurnResult

A = TypeVar("A")
S = TypeVar("S", bound="RunSummary")

#: Exit statuses every workflow shares. A workflow uses the subset that means
#: something for it (an assessment has no blockers, so it never exits 2).
EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_INCOMPLETE = 3
EXIT_CANNOT_START = 4


class StepFailed(RuntimeError, Generic[A]):
    """A step answered with ``error`` set: the run cannot go on.

    A pipeline is ordinary `async` code, so the way to stop it is an exception,
    not a sentinel return that every caller has to check. The pipeline catches
    this once, records why, and returns its own "no branch" result.
    """

    def __init__(self, name: str, outcome: Outcome[A]) -> None:
        super().__init__(f"{name}: {outcome.error}")
        self.name = name
        self.outcome = outcome


class RunStopped(RuntimeError, Generic[S]):
    """A run that began and then failed. ``summary`` is what was recorded.

    A ``ValueError``, ``GitError`` or ``OSError`` before the first turn is a
    usage error and keeps its own type (the entry point maps it to exit 4).
    After one, the run stopped partway and its ``run.json`` says so; the
    exception carries the summary so the entry point can print the report and
    use that exit code.
    """

    def __init__(self, summary: S, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.summary = summary
        self.cause = cause


def cause(exc: BaseException) -> str:
    """One line naming why a run stopped, for the record and the terminal.

    A cancelled turn raises `CancelledError` with no message, so the type alone
    used to be the whole story: "CancelledError: " says neither that the run was
    interrupted nor by which side.
    """
    if isinstance(exc, asyncio.CancelledError | KeyboardInterrupt):
        return "interrupted (cancelled before the turn finished)"
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


@dataclass(frozen=True, kw_only=True)
class RunOptions:
    """What every run takes from its command line. Workflows' options subclass it.

    Attributes:
        no_cache: Run every turn, even where a recorded answer would serve.
        reuse_stale: Serve a record made by another definition or environment.
        timeout: Wall-clock budget per turn; ``None`` for the settings'.
        trace: Draw the live span tree; ``False`` leaves it to
            ``HELION_AGENTIC_TRACE``.
    """

    no_cache: bool = False
    reuse_stale: bool = False
    timeout: float | None = None
    trace: bool = False


@dataclass(kw_only=True)
class RunSummary:
    """What every run records, whatever its workflow. Workflows subclass it."""

    log_dir: Path | None = None
    versions: dict[str, str] = field(default_factory=dict)
    #: Digest of what a run reads that its tasks do not show: AGENTS.md, the
    #: skill library, the library versions.
    environment: str = ""
    models: dict[str, str] = field(default_factory=dict)
    #: One digest per worker: its instructions, composition, tools and limits.
    definitions: dict[str, str] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    error: str | None = None
    #: The run never got going: a bad revision, an unknown role, an agent that
    #: could not be built. Not a finding about what the run was pointed at.
    start_failed: bool = False
    #: Whether live tracing was on. Part of the record because a traced run's
    #: findings are the same findings, but a reader who wants the span tree has
    #: to be told whether there was one.
    traced: bool = False

    def exit_code(self) -> int:
        """The process status: the run's failure if it had one, else the result.

        A run that stopped is "could not start" when it never got going and
        "incomplete" otherwise -- never a pass, and never a blocker, since what
        was not looked at cannot be judged. A run that finished exits with
        :meth:`result_code`.
        """
        if self.error is not None:
            return EXIT_CANNOT_START if self.start_failed else EXIT_INCOMPLETE
        return self.result_code()

    def result_code(self) -> int:
        """What a finished run exits with. Each workflow says."""
        raise NotImplementedError

    def run_record(self) -> dict[str, Any]:
        """The run-level fields every workflow's ``run.json`` carries."""
        return {
            "error": self.error,
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "models": self.models,
            "capabilities": self.capabilities,
            "versions": self.versions,
            "environment_sha256": self.environment,
            "definitions": self.definitions,
            "traced": self.traced,
            "exit_code": self.exit_code(),
        }


def environment(repo: Path, skills_dirs: Sequence[str], found: dict[str, str]) -> str:
    """The digest of what a run reads that the task's inputs do not show."""
    return artifact_store.environment_digest(repo, list(skills_dirs), found)


@dataclass
class Run:
    """One invocation of a workflow: its log, its workers and its records."""

    config: Config
    repo: Path
    log: RunLog
    summary: RunSummary
    timeout: float
    usage_limits: Any
    no_cache: bool = False
    reuse_stale: bool = False
    #: The readers' workspaces, removed when the run ends.
    workspaces: ExitStack = field(default_factory=ExitStack)
    #: Every worker built in this run, from every stage; a worker's index is its
    #: position here.
    workers: list[Worker[Any]] = field(default_factory=list)
    #: Each distinct task's number, assigned on first use (``Task.identity``).
    #: Numbers keep evidence files apart, so the run hands them out rather than
    #: each stage counting the tasks that ran before it.
    task_numbers: dict[str, int] = field(default_factory=dict)
    #: Each task's workspace, made on first use (``workspace_for``). A reader's
    #: is disposable; a writer's holds the change and outlives the run as a
    #: branch.
    spaces: dict[tuple[int, int], worktree_mod.Workspace] = field(default_factory=dict)
    #: The local workspaces handed to turns, one per tree (the harness's
    #: ``workspace=`` argument): the command environment is per tree.
    sessions: dict[Path, LocalWorkspaceBackend] = field(default_factory=dict)
    #: Branches this run made and refused, deleted once their worktrees close.
    _dropped_branches: set[str] = field(default_factory=set)
    #: Whether any turn (or served record) has happened yet. A failure before the
    #: first one is "could not start"; after it, the run stopped partway.
    began: bool = False

    @classmethod
    @contextmanager
    def open(
        cls,
        config: Config,
        repo: Path,
        summary: S,
        *,
        command: str,
        settings: WorkflowSettings,
        options: RunOptions,
        report: Report[S],
    ) -> Generator[Run]:
        """Start a run; on the way out, check the checkout and write the record.

        The log directory exists before anything can fail: a bad role or a
        revision that does not exist is exactly the run somebody wants to read.
        ``report`` writes ``run.json`` and ``index.md`` on every exit path. A
        ``ValueError``, ``GitError`` or ``OSError`` is a usage error
        (``start_failed``) before the first turn and a stopped run after it,
        never a finding.

        ``options.trace`` draws the live span tree (otherwise
        ``HELION_AGENTIC_TRACE`` decides). It is installed before any agent
        exists (instrumentation is read when a run starts), and it never enters
        a definition digest -- tracing shows a run, it does not change what an
        agent can see, so a recorded verdict stays valid.
        """
        from pydantic_ai.usage import UsageLimits

        log = RunLog.create(repo, command=command)
        summary.log_dir = log.directory
        summary.versions = versions()
        # Everything from here to the end writes the record, including a failure
        # in tracing or in the digest taken before the run: a run that could not
        # read the checkout still leaves a run.json saying so.
        try:
            try:
                summary.traced = trace_mod.configure(
                    enabled=options.trace or None, title=command
                )
            except Exception as exc:
                # Typically OTEL_EXPORTER_OTLP_ENDPOINT is set without the
                # exporter installed. Record it and stop as a start failure,
                # instead of a traceback with a run.json that says error: null.
                summary.error = cause(exc)
                summary.start_failed = True
                raise ValueError(f"tracing could not start: {exc}") from exc
            if summary.traced:
                log.event("tracing on: " + trace_mod.summary())
            log.event(f"repository {repo}")
            log.event(
                "versions " + ", ".join(f"{k}={v}" for k, v in summary.versions.items())
            )
            tokens = settings.max_total_tokens or config.max_total_tokens
            run = cls(
                config=config,
                repo=repo,
                log=log,
                summary=summary,
                timeout=options.timeout
                or settings.timeout_seconds
                or config.timeout_seconds,
                # The request count is not a meaningful ceiling (one small review
                # used 25); the wall clock and the token budget are.
                usage_limits=UsageLimits(request_limit=None, total_tokens_limit=tokens),
                no_cache=options.no_cache,
                reuse_stale=options.reuse_stale,
            )
            try:
                try:
                    with run.workspaces:
                        yield run
                finally:
                    # After the workspaces close: git will not delete a
                    # checked-out branch, and a refused change must not leave
                    # one behind for the next run to treat as a claim.
                    run._settle_branches()
            except BaseException as exc:
                summary.error = cause(exc)
                # A usage error is "could not start" only before the first turn:
                # after one, a bad revision or a role built late is a run that
                # stopped, not one that never got going.
                # `OSError` belongs here: a file deleted mid-run (a snapshot
                # copying it, a log path that vanished, a full disk) is a run
                # that stopped, not one that never started -- and `cli` maps an
                # `OSError` to "could not start" when it sees one.
                usage = isinstance(exc, ValueError | git.GitError | OSError)
                summary.start_failed = usage and not run.began
                if usage and run.began:
                    raise RunStopped(summary, exc) from exc
                raise
        finally:
            report.finish(summary, log)
            # Flush and stop after the record is written: an OTLP exporter is a
            # batch one, and a run that ends without a flush exports nothing.
            trace_mod.stop()

    def plan(self, total: int) -> None:
        """Tell the live view how many turns this run will run in total."""
        trace_mod.plan(total)

    # -- workers -------------------------------------------------------------

    def build_workers(
        self,
        specs: Sequence[tuple[str, ModelRef, Step[Any]]],
        *,
        snapshot_of: Path | None = None,
    ) -> list[Worker[Any]]:
        """Compose and build one agent per ``(name, model, step)``, each in a workspace.

        Every worker reads its own tree: a checkout of the task's revision when
        the task names one, or a private snapshot of ``snapshot_of`` (the
        caller's checkout unless given). Each is checked after each of its turns.

        A harness raises its own type for a missing credential or an unknown
        model; it is re-raised as a ``ValueError`` naming the worker, so the
        report points at a config line and the run exits as "could not start".
        """
        if not self.summary.environment:
            self.summary.environment = environment(
                self.repo, self.config.skills_dirs, self.summary.versions
            )
            self.log.event(f"environment {self.summary.environment[:16]}")
        source = snapshot_of or self.repo
        workers: list[Worker[Any]] = []
        for name, ref, step in specs:
            # A composition does not depend on which tree the run is pointed at --
            # the workspace is a run-time argument -- so it is made once and its
            # digest is the one every task's turn runs under.
            definition = step.definition()
            digest = definition.digest()
            provider = self.config.provider_of(ref).fingerprint()
            try:
                model = build_model(self.config, ref)
                # Build the agent from *this* definition object: the harness
                # records the tool set it offered on it, and the record reads it
                # from here. A second `step.definition()` would observe nothing.
                from . import stack

                agent = stack.build_agent(
                    model, definition, step.output_type, ref.thinking
                )
            except Exception as exc:
                raise ValueError(
                    f"could not build {name!r} ({ref.label}): {exc}"
                ) from None
            worker = Worker(
                index=len(self.workers),
                name=name,
                ref=ref,
                step=step,
                model=model,
                definition=definition,
                digest=digest,
                provider=provider,
                settings=ref.settings_label,
                source=source,
                agent=agent,
            )
            self.workers.append(worker)
            workers.append(worker)
            # Two stages may use the same role (two reviewers on one model): the
            # second is filed under its step too, not over the first.
            entry = name
            if self.summary.definitions.get(name, digest) != digest:
                entry = f"{name}@{step.kind}"
            self.summary.models[entry] = (
                ref.label
                if not ref.settings_label
                else f"{ref.label} ({ref.settings_label})"
            )
            self.summary.definitions[entry] = digest
            if not self.summary.capabilities:
                self.summary.capabilities = list(definition.names)
            self.log.event(
                f"agent {name}: {ref.label}, definition {digest[:12]}, reading {source}"
            )
        if not workers:
            raise ValueError("no agents to run")
        return workers

    def task_number(self, task: Task) -> int:
        """This task's number in the run: the same task always gets the same one."""
        return self.task_numbers.setdefault(task.identity(), len(self.task_numbers))

    def _workspace_of(self, root: Path) -> LocalWorkspaceBackend:
        """The harness workspace for ``root``, with this tree's command environment."""
        known = self.sessions.get(root)
        if known is None:
            from . import stack

            known = self.sessions[root] = stack.backend(root)
        return known

    # -- turns ---------------------------------------------------------------

    def reused(self, worker: Worker[A], task: Task) -> Outcome[A] | None:
        """The recorded answer to this question, when it may be served.

        Not when caching is off or the step is not reusable; not when there is no
        record under this contract version; not when the record is a failure (a
        transient error must not become a permanent answer); and not when it was
        produced by another definition or environment (*stale*) unless the caller
        accepted that with ``reuse_stale``.
        """
        if self.no_cache or not worker.step.reusable:
            return None
        artifact = artifact_store.load(
            self.repo, task.key(worker), schema_version=worker.step.schema_version
        )
        if artifact is None:
            self.log.event(f"  {worker.name}: no reusable record")
            return None
        try:
            result = worker.step.parse(artifact.outcome)
        except ValueError:  # pydantic's ValidationError is one: a cache miss
            self.log.event(f"  {worker.name}: unreadable record, running again")
            return None
        stale = artifact.provenance_stale(
            environment=self.summary.environment, definition=worker.digest
        )
        if stale and not self.reuse_stale:
            self.log.event(f"  {worker.name}: no reusable record")
            return None
        self.log.event(
            f"  {worker.name}: cached"
            + (" (stale)" if stale else "")
            + f" ({answer_summary(result)})"
        )
        return Outcome(
            name=worker.name,
            model=worker.ref.label,
            result=result,
            cached=True,
            stale=stale,
            elapsed_seconds=artifact.elapsed_seconds,
            artifact=artifact.key,
        )

    def workspace_for(self, worker: Worker[Any], task: Task) -> worktree_mod.Workspace:
        """The tree this (worker, task) works in, made on first use.

        A reader gets a private tree of what the task is *about*: a checkout of
        the task's revision (the tree its diff ends at), or a copy of the working
        tree when the task names no revision. A task with a ``branch`` gets a
        writable worktree on that branch instead -- one per task, shared by every
        turn about it, so a writer refines the change it already made. A cached
        answer is served before this is called, so a run that is entirely cached
        builds no workspace at all.
        """
        key = (worker.index, self.task_number(task))
        known = self.spaces.get(key)
        if known is not None:
            return known
        source = worker.source or self.repo
        if task.branch:
            space = self.workspaces.enter_context(
                worktree_mod.writer_workspace(
                    source, branch=task.branch, base=task.base
                )
            )
            self.log.event(
                f"  {worker.name}: writing {task.branch} (from {task.base or 'HEAD'})"
            )
        else:
            space = self.workspaces.enter_context(
                worktree_mod.reader_workspace(
                    source, revision=task.revision, ignore=output_paths(source)
                )
            )
            self.log.event(
                f"  {worker.name}: "
                + (
                    f"reading {task.revision}"
                    if task.revision
                    else f"reading a copy of {source}"
                )
            )
        self.spaces[key] = space
        return space

    async def step(
        self,
        worker: Worker[A],
        *,
        name: str,
        prompt: str | None = None,
        inputs: dict[str, str] | None = None,
        task: Task | None = None,
        continue_from: Outcome[A] | None = None,
        save: bool = True,
        label: str | None = None,
        revision: str | None = None,
        branch: str | None = None,
        base: str | None = None,
    ) -> Outcome[A]:
        """One delegation: build the task, run or serve the turn, stop on failure.

        This is the whole of the plumbing a pipeline should not repeat. The task
        is the cache key, and the *prompt is the question*: a prompt that embeds
        the previous answers is already keyed on them, so ``inputs`` only names
        what the prompt does not show (a tree digest, a round number).

        ``continue_from`` is how a later turn continues an earlier one: it reuses
        that turn's task (so the same tree and the same record key) and its
        message history, with ``prompt`` as the follow-up. It is not saved again;
        the caller saves the answer it accepts.

        Raises :class:`StepFailed` when the answer carries an ``error``: an
        unanswered step is a stopped run, never an empty success.
        """
        message_history: list[ModelMessage] | None = None
        if continue_from is not None:
            task = continue_from.task
            message_history = continue_from.history
            save = False
        if task is None:
            if prompt is None:
                raise ValueError(f"step {name!r} needs a prompt or a task")
            task = Task(
                name=name,
                prompt=prompt,
                inputs=dict(inputs or {}),
                revision=revision,
                branch=branch,
                base=base,
            )
        outcome = await self.turn(
            worker,
            task,
            prompt=prompt,
            message_history=message_history,
            save=save,
            label=label,
        )
        if outcome.failed:
            raise StepFailed(worker.name, outcome)
        return outcome

    def drop_branch(self, branch: str) -> None:
        """Delete a branch the run made, once its worktree is gone.

        A refused change must not leave a branch behind that the next run's
        deterministic pick treats as a claim. Removal happens after the run's
        workspaces close, because git will not delete a checked-out branch.
        """
        self._dropped_branches.add(branch)

    def _settle_branches(self) -> None:
        for branch in sorted(self._dropped_branches):
            try:
                git.git("branch", "-D", branch, cwd=self.repo)
            except git.GitError as exc:  # nothing to delete, or someone kept it
                self.log.event(f"could not delete branch {branch}: {exc}")
            else:
                self.log.event(f"dropped branch {branch}")

    async def turn(
        self,
        worker: Worker[A],
        task: Task,
        *,
        reuse: bool = True,
        save: bool = True,
        prompt: str | None = None,
        message_history: list[ModelMessage] | None = None,
        label: str | None = None,
    ) -> Outcome[A]:
        """Serve this question from a record, or run the turn and record it.

        ``prompt`` and ``message_history`` continue an earlier turn (feedback);
        the record is still filed under ``task``'s question. ``label`` names the
        evidence files when one worker takes several turns on one task. With
        ``save=False`` the answer is not recorded here; the caller records it
        once it accepts the answer.
        """
        if reuse and message_history is None:
            cached = self.reused(worker, task)
            if cached is not None:
                self.began = True
                # A served answer still answers *this* task, so a later turn can
                # continue it; it built no workspace, so it has no tree.
                cached.task = task
                return cached
        agent, workspace, definition = (
            worker.agent,
            self.workspace_for(worker, task),
            worker.definition,
        )
        # Named before the turn so every tool call is appended as it happens.
        evidence = label or worker.name
        steps_path = self.log.steps_path(self.task_number(task), worker.index, evidence)
        self.began = True
        turn = await run_turn(
            agent,
            prompt if prompt is not None else task.prompt,
            timeout=self.timeout,
            usage_limits=self.usage_limits,
            message_history=message_history,
            workspace=self._workspace_of(workspace.path),
            deps=worktree_mod.Tree(workspace.path),
            steps_path=steps_path,
            # The evidence file may be named for the attempt, but the live view
            # attributes a line by the worker that ran it: a `-retry1` label
            # matched no slot and the retry's calls vanished from the view.
            label=worker.name,
            span_attributes={
                "agentic.worker": worker.name,
                "agentic.model": worker.ref.label,
                "agentic.task": task.name,
                "agentic.task_index": str(self.task_number(task)),
            },
        )
        outcome = self._record(
            worker, task, turn, label=label or worker.name, definition=definition
        )
        # What a later step needs to continue or judge this turn: the task (the
        # same worktree and record key) and the tree it worked in.
        outcome.task = task
        outcome.tree = workspace.path
        if not workspace.writable and await asyncio.to_thread(workspace.changed):
            # The evidence above keeps what the agent said; the outcome does not,
            # because it was reached in a tree nobody can reproduce.
            self.log.event(f"  {outcome.name}: changed its workspace; no verdict")
            outcome.result = cast("A", "")
            outcome.error = (
                "the agent changed its workspace during the turn, so its answer "
                "cannot be attributed to the tree it was given"
            )
            outcome.pending = None
            await asyncio.to_thread(workspace.rebuild)
        elif save:
            self.save(worker, task, outcome)
        return outcome

    async def fan_out(
        self, workers: Sequence[Worker[A]], task: Task
    ) -> list[Outcome[A]]:
        """Every worker on one task: records where they answer, turns otherwise.

        Each worker reads its own tree, so a write by one -- a `pre-commit` run
        that reformats a file, a scratch worktree -- cannot reach another and
        spoils only its own answer. The result is in worker order, so two runs of
        the same task produce records that diff cleanly.
        """
        done: dict[int, Outcome[A]] = {}
        pending: list[Worker[A]] = []
        for worker in workers:
            cached = self.reused(worker, task)
            if cached is None:
                pending.append(worker)
            else:
                # The run did work: a record was served, so a later config error
                # is a run that stopped (3), not one that never started (4).
                self.began = True
                done[worker.index] = cached
        ran = await asyncio.gather(
            *(self.turn(worker, task, reuse=False) for worker in pending)
        )
        done.update(
            (worker.index, outcome)
            for worker, outcome in zip(pending, ran, strict=True)
        )
        return [done[worker.index] for worker in workers]

    def save(self, worker: Worker[A], task: Task, outcome: Outcome[A]) -> None:
        """Record ``outcome`` as the answer to ``task``, unless it is a failure."""
        if outcome.cached or outcome.failed or not outcome.pending:
            return
        artifact_store.save(self.repo, outcome.pending)
        outcome.pending = None

    def _record(
        self,
        worker: Worker[A],
        task: Task,
        turn: TurnResult,
        *,
        label: str,
        definition: Composed,
    ) -> Outcome[A]:
        """Write one turn's evidence and describe it, with its record ready."""
        result: A = turn.result
        number = self.task_number(task)
        timing = turn.steps.summary()
        # A failed turn is a property of the outcome, not of the answer: a prose
        # answer has no `error` field, and does not need one.
        error = None if turn.event == "ok" else turn.event
        summary = answer_summary(result)
        self.log.event(
            f"  {label}: {turn.event} in {turn.elapsed:.0f}s ({summary}); "
            f"model {timing['model_seconds']:.0f}s, tools "
            f"{timing['tool_seconds']:.0f}s in {len(turn.steps.steps)} step(s)"
        )
        if turn.usage:
            self.log.event(
                f"  {label}: " + ", ".join(f"{k}={v}" for k, v in turn.usage.items())
            )
        self.log.agent_log(
            number,
            worker.index,
            label,
            f"{worker.ref.label}\n{summary}\n"
            f"elapsed: {turn.elapsed:.1f}s\ntiming: {timing}\n"
            f"usage: {turn.usage}\nsteps: {len(turn.steps.steps)}\n"
            f"error: {error}\n",
        )
        trace_mod.turn_done(label, summary)
        key = task.key(worker)
        pending: Artifact | None = None
        if error is None:
            pending = artifact_store.Artifact(
                key=key,
                agent=worker.step.kind,
                model=worker.ref.label,
                schema_version=worker.step.schema_version,
                task=task.name,
                inputs=task.inputs,
                environment=self.summary.environment,
                definition=worker.digest,
                elapsed_seconds=turn.elapsed,
                outcome=worker.step.as_record(result),
            )
        return Outcome(
            name=worker.name,
            model=worker.ref.label,
            result=result,
            cached=False,
            elapsed_seconds=turn.elapsed,
            usage=turn.usage,
            timing=timing,
            outcome_path=self.log.write_outcome(
                number, worker.index, label, worker.step.as_record(result)
            ),
            messages_path=self.log.write_messages(
                number, worker.index, label, turn.raw
            ),
            # Written live by the recorder; a fake turn (tests, a scripted model)
            # still gets its file here.
            steps_path=(
                turn.steps.path
                or self.log.write_steps(number, worker.index, label, turn.steps)
            ),
            artifact=key,
            history=turn.history,
            error=error,
            pending=pending,
        )
