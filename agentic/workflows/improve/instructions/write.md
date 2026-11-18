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

## Check your work

Run the checks the plan names, and anything cheap that catches a mistake first:
an import, a compile, the one test node. You have a shell.

This machine may be shared, and a GPU belongs to whoever claimed it; nothing here
claims one for you. Prefer a CPU check, or the smallest tensor that shows the
behaviour, and do not claim a GPU with an ad-hoc command. `CUDA_VISIBLE_DEVICES`
is whatever the run was started with: empty means a run that claims no GPU, which
is the safe default, and a non-empty value is a device this run may use.

Report what you ran and its result in your answer. A test you did not run is not a
test that passed, and a reviewer will look.

## Commit

The branch is what the run delivers, so commit your work with `git commit` in
your shell -- nothing else commits for you -- in the repository's style (see
`AGENTS.md`):

- a bracketed area and an imperative summary line, at most about 80 characters;
- why, then what changed, then a `Testing:` paragraph naming the lanes and their
  pass/fail counts;
- one commit. A second commit is a second pull request, so it needs a reason.

Never push, never rebase, and never reset below the revision the branch was cut
from: the reviewers read `base..HEAD`, and a commit beneath the base is work the
run would deliver without having decided to.
