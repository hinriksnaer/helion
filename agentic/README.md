# `agentic/` — multi-step agent workflows

An **experimental** orchestration layer over `pydantic-ai` and
`pydantic-ai-harness`: point it at whatever model you have access to, and run a
workflow over a change, a tree, or a piece of work the repository has not done.

A **skill** (`.agents/skills/`) tells a model a procedure; the **CLI harness**
(`scripts/autoreview.py` driving `claude`/`codex`) gives it one isolated run and
one independent opinion. `agentic/` is the same agent with the steps written
down — so a sequence can be gated, recorded and measured.

> **Status: experiment.** It is judged on evidence, never on design: what it does
> against the tools it would replace, and what its orchestration adds over the
> same agent in one session.

Nothing here is imported by `helion/`, the test suite or `scripts/autoreview.py`,
so a change to this directory can never fail CI for a kernel.

## What belongs in code, and what belongs in markdown

**Code holds only what a model cannot know:** the tree it works in, the command
environment (`PYTHONPATH` is the workspace), the wall clock it has, and our own
resource hygiene (the jobs our shells leave behind). Everything about *how* to do the work — what to look at, how much
to verify, when to stop, whether to reach for a script — is a policy file a model
reads.

Therefore:

- a step is **one policy `.md`** plus the harness's own `Coder`. No composition of
  ours, no per-step capability, no tool we wrote.
- a behaviour that goes wrong is fixed in the policy first, and measured before
  anything else moves. The largest accuracy gain so far came from rewriting one
  104-line policy file; no runtime change has matched it.
- code is added for a fact a prompt cannot carry, and deleted when a run shows it
  is never read. The record cache went that way: every run paid for it and every
  run said `no reusable record`. The 1,197-line span renderer is the next
  candidate, not the next feature.
- **the library's documented shape wins.** A design here that duplicates the
  harness — a hand-assembled capability stack, a bespoke retry handler, a policy
  attached beside `Coder` instead of inside it — is a bug. When something has
  grown, replace it with the capability the harness ships.

Three rules follow from that and from the multi-agent guide:

- **one baseline agent**: every step runs the harness's `Coder` and adds only its
  policy text (`Coder(instructions=...)`). A release that improves the agent
  improves every workflow, and `pipeline − baseline` stays meaningful;
- **steps, not pipelines**, are the unit of reuse; a typed answer
  (`core.step.Answer`) is for what the run acts on, prose for what the next model
  reads; whatever can be computed without a model is (the order, the pick, the
  claim), and the check that decides a step passed never is;
- **level 3, not a graph**: a workflow is plain `async def` with programmatic
  hand-off until its control flow stops being a line.

## Install and configure

It needs Python 3.11 (pydantic-ai's floor) and the `agentic` extra, deliberately
not part of `dev`: nothing under `helion/` or `test/` imports it, and the harness
pulls vendor SDKs no CI job needs.

```bash
pip install -e '.[agentic]'
cp agentic/config.example.toml agentic/config.local.toml   # gitignored
$EDITOR agentic/config.local.toml
```

If `python3` on your `PATH` is not that environment, run the scripts through it
explicitly (`.venv/bin/python agentic/scripts/…`).

**The invariant:** a shared definition names a **capability role** and a
**contract**; the local config names a **provider and a model**, and how hard it
thinks. Nothing provider-specific is ever committed, so one definition serves an
internal gateway, an Anthropic key or a local model without any becoming a
special case.

```toml
thinking = "high"

[roles]
deepseek = "enmaas/rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"
glm = { model = "enmaas/rits/zai-org/glm-5-3", thinking = "medium" }
```

`thinking` is pydantic-ai's unified setting (`minimal` … `max`), the same meaning
whichever provider a role points at.

## Use

```bash
agentic/scripts/autoreview.py --dry-run      # what would run, with what: no credentials
agentic/scripts/autoreview.py --head         # the last commit
agentic/scripts/autoreview.py origin/main..HEAD
agentic/scripts/autoreview.py --target "the new parser in src/"
agentic/scripts/autoreview.py --reviewer glm

agentic/scripts/improve.py --dry-run         # who runs what, and the shape
agentic/scripts/improve.py                   # scan, pick, plan, write, review
agentic/scripts/improve.py --base origin/main   # cut the branch from a revision
```

`--dry-run` prints the target, the models, the endpoints, the composition and the
definition digest. `--trace` draws a run as a tree while it happens; anything that
is not a terminal gets append-only lines instead, from pydantic-ai's own
OpenTelemetry spans (no service, no client of ours).

**Untracked files are part of the change**: a working-tree review is `git diff
HEAD` plus a new-file diff for everything `git ls-files --others` reports.

## Exit status

`core.lifecycle.EXIT_*`, so every workflow's gate means the same thing.

| code | review | improve |
| --- | --- | --- |
| 0 | reviewed, nothing blocking (nits do not block) | a branch was delivered, or nothing was worth doing |
| 2 | a blocker was found | — (this workflow's answer is a branch, not a verdict) |
| 3 | not fully reviewed: a reviewer failed, the run stopped, nothing to review | no branch: a step did not answer, the candidate was claimed, review still blocks, or the run stopped |
| 4 | could not start: config error, unknown role, missing extra | same |

`scripts/autoreview.py` returns 0 however many findings it printed, so automation
written for the incumbent must map the codes before pointing at this one.

## What a run leaves behind

One directory per run under `agentic/logs/` (gitignored):

| file | question |
| --- | --- |
| `index.md` | what happened, for a person: target, agents, verdict, where the cost went |
| `run.json` | the same as data |
| `run.log` | a timestamped timeline: what was resolved, which worker started and finished |
| `<n>-<m>-<name>.outcome.json` | the answer the pipeline gated on |
| `<n>-<m>-<name>.messages.json` | the whole conversation: every call, result, thinking block |
| `<n>-<m>-<name>.steps.jsonl` | one record per tool call: name, arguments (both ends of a long one), duration, result size and the start of its result |

Both are appended as the turn runs, so a run that dies keeps what it had, and an
interrupted turn still writes a transcript — rebuilt from the calls and results
the recorder saw when the run left no history behind.

That record exists to answer one question: **where did the agent struggle.** A
result's opening text is what separates `command not found` from a finding (a
length cannot), and a delegate's whole answer is its result. Policy tuning is
read off `steps.jsonl`, not guessed.

## Workflows in the tree

| workflow | answer | shape | entry point |
| --- | --- | --- | --- |
| `review/` | a verdict on a change | two reviewers on one task, one summariser | `scripts/autoreview.py` |
| `improve/` | one reviewed branch | one call per step, one loop the reviewers end | `scripts/improve.py` |

Two reviewers run by default, each in its own checkout, reported separately and
never merged into one opaque verdict: independent answers are the substance of a
review, and one model agreeing with itself is not. `improve` uses a *different*
model to choose than to propose, and gates its branch through `review/`'s
`review_branch()` inside its own run — that function is the whole contract
between two workflows.

## Adding one

A directory with one policy per step and one pipeline module (`workflow.py` in
`review/`, `pipeline.py` in `improve/`): an `async def` that opens a `Run`, asks
the agents and returns a summary whose `result_code()` is the gate. Copy the
nearest and change the policy. Reach into `core/` only when the pipeline needs it:

- a step whose answer the run acts on is a pydantic model based on
  `core.step.Answer`; a step whose answer the next model reads is `str`;
- `[workflows.<name>]` settings subclass `core.config.WorkflowSettings`, options
  `RunOptions`, and the summary `RunSummary` says what its exit codes mean;
- an entry point under `scripts/` adds `_entry.add_run_arguments(parser)` and
  calls `core.cli.main(args, report=, start=)`;
- another workflow calls your function; it does not import your internals.

## Safety: a shell is not a sandbox

No turn works in your checkout — a task naming a revision gets a `git worktree`
at it, a working-tree task a private snapshot, a writer a worktree it owns — and a
reader's tree is digested, so a turn that changed it gets no verdict. But
pydantic-ai's `LocalWorkspace` "isolates nothing: tools reach anywhere on the host
this process can". `HOME` is forwarded, so `~/.ssh` and `gh` credentials are
reachable, and so is the network. Treat a unit with a shell as running with your
account, not as codex's `-s workspace-write`. Commands run detached from the turn
by design; the workspace teardown stops the ones it finds.

## Layout

```text
agentic/
  core/            the runtime: nothing here knows about a specific workflow
    step.py          Step, Worker, Task, Outcome: what a workflow declares
    lifecycle.py     Run: the run directory, workspaces, turns, fan_out
    stack.py         the baseline agent: Definition, the digest, the command env
    config.py        role -> provider/model + thinking, [workflows.<name>]
    turn.py          one agent turn: answer, cost, timing, transcript; never raises
    report.py        run.json and index.md
    steps.py         what one turn did, tool call by tool call
    trace.py         the live span tree (--trace), and the append-only fallback
    git.py           running git, and hashing a tree
    worktree.py      a disposable worktree: snapshot, revision, or writer's branch
    cli.py           an entry point after parsing: config, dry run, exit
  workflows/review/    a change in, a verdict out (workflow.py, instructions.md)
  workflows/improve/   a backlog in, one reviewed branch out (pipeline.py,
                       state.py, record.py, instructions/ per step)
  scripts/             entry points: autoreview.py, improve.py, _entry.py
  config.example.toml
```

## How it is measured

Never against its own design. Two baselines:

- **the status quo** — the skill plus the CLI, or the script: is this worth
  replacing at all?
- **the same agent in one session**, same objective, same budget: is the
  orchestration worth anything? A workflow that cannot beat this has no claim,
  however it does against the first.

Verdicts are compared as blocker *sets as data* against `scripts/autoreview.py` on
the same revisions with the model fixed. Assessments and changes have no incumbent
yet: a person reading the same tree, and merged verified changes per N runs. A
kind of answer is dropped when its own baseline wins.

## Known gaps

- **The claim is a branch, and git has no expiry**: a crashed or refused run keeps
  its candidate claimed until someone deletes the branch.
- **An agent claims no GPU for itself.** `CUDA_VISIBLE_DEVICES` is forwarded from
  whoever started the run (the helion devshell sets it), so an agent inherits the
  caller's selection; a run that needs a *different* device should claim through
  `scripts/gpu_board.py` first.
- **A delegate's conversation is not in the parent transcript**: its calls are in
  the delegating turn's `steps.jsonl` (tagged `sub` and by delegate), its messages
  only reachable through `delegate_task`'s return.
- **Temperature and seed are neither settable nor recorded**; the thinking level
  is, and is part of the record.
- **`--per-commit` does not exist**: a range is one review.
