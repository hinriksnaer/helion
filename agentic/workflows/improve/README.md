# The improve workflow

A backlog in, one reviewed branch out. Four prose steps, then a review loop the
reviewers end.

```
scan ─> choose ─> plan ─> write ─> review <──> refine
  A        B        B       C       reviewers      C
                                    + summary
```

Two modules: `pipeline.py` (the steps, their prompts, the run, the gate) and
`record.py` (the report and `--dry-run`). `state.py` holds the settings, the
rounds and the exit code; `instructions/` is what each step is told.

## The design rule

**The orchestrator sequences; the models decide.** A step is one `await
run.step(...)`: the run builds the task, serves a recorded answer or takes the
turn, and stops the run when a step answers nothing. The orchestrator owns only
what a person would otherwise have to remember:

- which role runs which step;
- that `scan` and `choose` are different models (the run refuses to start when
  they agree — one model grading its own proposals is not a second opinion);
- the branch name, a slug of the chooser's first line;
- `max_rounds`, a ceiling for a writer that cannot satisfy a reviewer. The
  *policy* is the reviewers': they end the loop by finding nothing blocking.

This is the library's level 3, programmatic hand-off. A graph would add nothing:
the control flow is a line with one retry loop.

## What each step is

| step | answer | note |
|---|---|---|
| scan | prose | a numbered list of candidates: name, the case for it, the files |
| choose | prose | investigates, picks one, and its first line names the branch |
| plan | prose | the smallest change: goal, files, edits, how it is checked |
| write | prose | makes it in a worktree, on the branch, and commits with its shell |
| review | prose reviews + a typed verdict | `review.review_branch(...)`: two reviewers and a summariser |
| refine | prose | the same conversation, handed the verdict |

Every step is the harness's `Coder`, whole, plus its policy. There are no extra
tools: the writer commits with `git`, the reviewers read a revision with `git`.
The answers are prose because the next model reads them; the verdict is typed
because the run acts on it.

## What it leaves

A branch, `agentic/improve/<slug of the chosen line>`, cut from the merge base
with upstream, with the writer's commits. Nothing is pushed. A branch the review
blocks is deleted unless `--keep-failed`.

The branch is also the claim: a candidate whose branch exists is not started
again, and the scan is told which branches are in flight. Git records who is
working on what, so run N does not redo runs 1..N-1.

Exit status: `0` a branch was delivered or there was nothing worth doing; `3` no
branch (a step did not answer, the review still blocked after `max_rounds`, or
the writer stopped changing anything); `4` could not start. No `2`: the answer is
a branch, not a verdict.

## Knobs

`[workflows.improve]`, validated on use: `scan`, `choose`, `plan`, `writer` (roles;
unset means `default_role`), `reviewers` (roles for the review; empty means
`[workflows.review] agents`), `independent_choice`, and `max_rounds`.

## What it does not do

It does not push, open a pull request, or merge; a person reads the branch. It
does not change a compatibility surface to make a candidate work — the policies
say so, and such a candidate is a note for a maintainer.
