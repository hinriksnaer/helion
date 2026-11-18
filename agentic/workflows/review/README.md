# The review workflow

A change in, a verdict out. Two reviewers read the change, and a third model
reads their reviews and decides.

```
told what to review ─> reviewer A ─┐
                    └> reviewer B ─┴─> summariser ─> exit code
```

Three files: `workflow.py` (all of it), `instructions.md` (what a reviewer is
told), `summarize.md` (what the summariser is told).

## Nothing on this side reads git

The prompt says *what* is being reviewed -- "the last commit (`git show HEAD`)",
"the change `origin/main...HEAD`", "the uncommitted changes" -- and the reviewer
reads it itself. It has a shell and the tree; it can diff from the merge base,
follow a rename, or check the caller two files away. A host-side diff pipeline
would have to guess what matters and would show it a page of text where the tree
is available.

The consequence to know about: **a revision that does not exist is the reviewer's
report, not a start failure**. This tool exits 0/2/3; it does not exit 4 for a
typo in a revision.

## The reviewers

Each is the harness's `Coder`, whole: file tools, shell, delegation, compaction,
the retry budget, `RepoContext` (`AGENTS.md`), prompt caching. The only thing
this workflow adds is the policy. That is deliberate -- a reviewer that can run
the failing test, read the installed library and follow a call chain is worth
more than one restricted to what a diff shows.

Each reviewer answers **prose**: blockers first, then nits, with the evidence it
saw. It is not asked to fill in a schema, because the next model reads it, and a
schema would decide in advance which observations matter.

## The summariser

A third model reads both reviews and answers the one typed question the run acts
on: `blocking` (a real defect that must be fixed first) and `summary` (what a
person needs: blockers, nits, and what was not checked). `summarize.md` says how
to weigh a disagreement, and that a reviewer which **failed** looked at nothing --
which is neither a pass nor a blocker.

This is where the post-processing that used to live in code went: merging two
reviews, sequencing blockers first, deciding whether a nit blocks. A model does
it better, and the decision is recorded as data rather than produced by a
formatter.

## The gate

| code | meaning |
|---|---|
| 0 | both reviews answered and the summary does not block |
| 2 | the summary blocks the change |
| 3 | not fully reviewed: a reviewer failed, the summariser failed, or the run stopped |
| 4 | could not start: a config error, an unknown role, a missing extra |

A failed reviewer is exit 3 and never 0: a change nobody looked at is not a clean
change. Nits never block.

## What it does not do

It does not touch the tree it reviews: every turn works in a private tree of its
own, and one that leaves the tree changed gets no verdict. It does not push,
land, or decide for you: it reports, and the exit code is the gate.

## Knobs

`[workflows.review]`: `agents` are the reviewer roles, in order; `summarizer` is
the role that decides (default: `default_role`). `--reviewer` overrides the
reviewers for one run. `--target` is free text handed to the reviewers verbatim,
for a change a revision cannot name.

## The record

`run.json` holds the reviewers' answers, the verdict, the models and the digests.
`index.md` holds the same for a reader: each review, then the summary. The whole
conversation and each tool call are in the per-turn files, which is what makes a
verdict investigable after the fact.
