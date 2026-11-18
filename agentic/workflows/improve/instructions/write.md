# Make the change, on this branch

You are in a worktree on a branch of your own, cut from the revision in the
prompt. The plan is a starting point written by a competent colleague who read
the code; its goal and checks are the contract.

## Work the plan

Stay inside its files. If the plan is wrong -- the symbol is not where it says,
the approach breaks something, a cheaper edit gets the same value -- do the
smallest correct thing that still meets the goal, and say exactly what you
changed and why in your answer. Do not silently widen the scope, and do not
stop: a plan's steps are advice, its goal and checks are not.

If the change genuinely cannot be made, say what blocked it and stop.
A branch that pretends to be a fix is worse than no branch.

When a review round's finding is **wrong** -- the reviewer misreads the code, or
the fix would break something the goal requires -- do not make a change you
believe is wrong. Answer it in a `Review notes:` paragraph at the end of the
commit message, one item per finding, with what you checked. That paragraph is
what the next reviewer reads; your turn answer is not part of the review.

## Check your work

Run the checks the plan names, and anything cheap that catches a mistake first:
an import, a compile, the one test node. You have a shell.

This machine may be shared, and a GPU belongs to whoever claimed it; nothing here
claims one for you. Prefer a CPU check, or the smallest tensor that shows the
behaviour, and do not claim a GPU with an ad-hoc command. `CUDA_VISIBLE_DEVICES`
is whatever the run was started with: empty means a run that claims no GPU, which
is the safe default, and a non-empty value is a device this run may use.

Report what you ran and its result in your answer. A test you did not run is not a
test that passed, and a reviewer will look. A check the plan names that this
machine cannot run (a TPU or MPS lane, an absent backend) is not a check: run
what is possible on the CPU and say plainly what stayed unverified.

## Commit

The branch is what the run delivers, so commit your work with `git commit` in
your shell -- nothing else commits for you -- in the repository's style (see
`AGENTS.md`):

- a bracketed area and an imperative summary line, at most about 80 characters;
- why, then what changed, then a `Testing:` paragraph naming the lanes and their
  pass/fail counts.

**One commit is the normal answer; a small stack is the right answer when the
work has parts.** Each commit is reviewed and submitted on its own, so:

- one change per commit: a behaviour change, the refactor it needed, the sibling
  bug it exposes are three commits, not one padded one;
- every commit must stand alone and depend only on the commits below it -- no
  commit whose diff only makes sense with the one above it;
- each gets its own message in the style above, and its own `Testing:` line; a
  commit that borrows the next one's tests is not reviewable;
- the cap is in the prompt. Do not pad to reach it, and do not merge two things
  that deserve separate review to stay under it.

The refactor goes **first** when the behaviour change needs it, and it must be
provably inert (same renders, same results) -- that is what makes it a commit a
reviewer can approve in a minute instead of an argument.

Never push, never rebase, and never reset below the revision the branch was cut
from: that is somebody else's work.
