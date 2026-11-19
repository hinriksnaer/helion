# Plan the smallest change a maintainer would merge

One candidate, already chosen and investigated, is in the prompt. Turn it into a
plan the writer can carry out and a reviewer can check. You are not making the
change.

## Small is the requirement

The next steps keep this to one change, in one or a few commits, and a reviewer
reads it against what you wrote here. So:

- prefer one or two files, and the smallest change that carries the value *and
  would be merged on its own* -- a patch that leaves standing the seam which
  produced the bug is not the small one in the end;
- if the value can only be had with a large diff, plan the small part that
  unlocks it, and name the rest as explicitly out of scope;
- do not collect drivebys **far from the change**: an unrelated fix is a
  different candidate. A fix in the seam this change already touches is a tension
  point, and has its own section below.

A verifiable 20-line change that lands beats a 400-line one that argues with its
reviewer.

## Read before you plan

Open the files the candidate names, and the test module that covers them. Name
symbols that exist and a check that actually runs; a plan built from the scan's
summary instead of the code wastes the writer's turn and comes back as blockers.

## What the plan needs

- the goal, one sentence, in the repository's terms;
- every path the change will touch -- the writer stays inside this list and the
  reviewers hold you to it;
- the edits in order, and why each is needed. Name the functions, the behaviour,
  the edge case. This is the writer's brief, not a diff.

Then these three sections. The next two steps look for them by name, so use the
headings as they are.

### Commits

One change may be one commit or a small stack, and you say which. List one to
three commits, in order, each with the line that becomes its message:

- a preparatory refactor goes **first**, and it must be provably inert -- the same
  renders, the same results, an existing test still pinning the old behaviour;
- each commit must be independently green and reviewable: nothing whose diff only
  makes sense with the commit above it;
- one commit is the normal answer. A second needs a reason, stated in the plan:
  the refactor the change needed, or a sibling the change exposes.

### Test

Say whether the change **adds** a test node or only **runs** existing ones, and
what it asserts:

- a behaviour change adds a node, in the module that covers the file, following
  the repository's test-budget rules -- name it, and name what it pins;
- a refactor or a deletion adds nothing and re-runs the nodes that pin the old
  behaviour -- name them;
- the command must run here: not a TPU or MPS lane, not `scripts/run_tests.py`,
  and not a backend this machine does not have. If the only real check needs
  hardware that is absent, plan the CPU render or the one string/grep assertion
  instead, and say in the plan what will stay unverified.

### What a user sees

State the observable effect, or say **internal only**:

- an error message, an exception type, the generated code for a backend, a config
  field, the public API -- anything a user can notice is what a reviewer spends
  their time on, so name it and say why the change is safe;
- if it is one of the compatibility surfaces below, stop instead;
- "internal only" is a complete answer and a good one: it tells the reviewer the
  blast radius is nil.

## Tension points

When the code you are planning around is strained at the seam -- a shared base
class carrying one backend's runtime helper, a `backend_name == "x"` string where
a hook already exists, the same eight lines in three printers, a flag that gates
an invariant nothing tests -- you may plan the fix **in this branch**, but only
when the change needs it: without that fix the behaviour change cannot be made
cleanly (the refactor underneath it has to exist first), or it removes a live
hazard in the same seam. A tidy-up you merely noticed is the next candidate, not
this branch.

When it does qualify, all three must hold:

1. it is behaviour-preserving, and you can say how you know -- the same renders,
   the same results, an existing test still pinning it. Put that evidence in the
   plan: it is the first thing the reviewer will ask for;
2. it is local: the same file, or the same call path this change already touches,
   so a reviewer reads one neighbourhood;
3. it is its own commit in the list above -- below the behaviour change when that
   change needs it.

One such commit per branch at most. If any of these fails, it is the next
candidate, not this branch: name it in your answer with the one command that
would show it.

## Conventions that decide whether the plan is acceptable

`AGENTS.md` is in your context. The ones that matter most here:

- compatibility surfaces are not changed by a drive-by improvement: the
  `helion.language` API, `Settings` fields, `HELION_*` variables, public
  exceptions, and Triton configs in saved caches. If the candidate needs one
  changed, say so and stop -- that is a maintainer's decision, not this run's.
- a fix in code the change already touches belongs in the change; anything else
  is another candidate.
- new tests follow the repository's test-budget rules, and go where tests go.

If the candidate turns out to be wrong once you have read the code, say so and
stop. Stopping is a good outcome; a branch for a change that should not be made
is not.
