# Plan the smallest change worth reviewing

One candidate, already chosen and investigated, is in the prompt. Turn it into a
plan the writer can carry out and a reviewer can check. You are not making the
change.

## Small is the requirement

The next steps will keep this change to one reviewable commit, and a reviewer
reads it against what you wrote here. So:

- prefer one or two files and the smallest edit that gets the value;
- if the value can only be had with a large diff, plan the small part that
  unlocks it, and name the rest as explicitly out of scope;
- do not collect drivebys. An unrelated fix is a different candidate.

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
  the edge case. This is the writer's brief, not a diff;
- how to know it worked: the test node ids or `-k` expressions, the command, and
  the property that must stay true afterwards.

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
