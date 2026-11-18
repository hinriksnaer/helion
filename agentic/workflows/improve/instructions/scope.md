# What a scope decides

One candidate, already picked and assessed, is in the prompt. Your job is the
plan: the smallest change that gets the value, with the checks that show it.

You are not making the change. The writer gets your plan and works from it, so
everything it needs to know that is not obvious from the code has to be here.

## The budget is real

The prompt names the file and line budget for this run. A plan over it is sent
back to you once, and then the run stops without making a change: an over-budget
plan is the workflow's way of saying "not this one".

So **split the change rather than describing a bigger one**. If the candidate's
value can only be had with a 700-line diff, the answer is the 80-line part that
makes the improvement possible, and the rest goes in `out_of_scope`. Say what is
left for later; that is a scoping decision, not a gap.

## What the plan needs

- **goal** -- one sentence, in the repository's terms.
- **steps** -- the edits in order, each small enough to check by reading.
- **files** -- every path this change touches. The run checks the writer's diff
  against this list; a file outside it fails the check.
- **tests** -- the behaviour that proves it works, named as a test node id or a
  `-k` expression. The run derives the actual lane from the changed files, so
  name what must stay true, not a command that may rot.
- **acceptance** -- concrete enough to be wrong: "`Settings` no longer
  re-validates; `test_settings.py::test_config_roundtrip` still passes".
- **max_lines** -- added + removed, honestly.
- **out_of_scope** -- the tempting adjacent work this change must not do.

## Read before you plan

Look at the code the candidate names, and at the test file that covers it. A
plan that names a symbol that does not exist, or a test module that does not
cover the path, costs the writer a turn and the reviewer a blocker. If the
candidate looks wrong once you read it, say so in `error` -- the run records it
and stops, which is better than writing a plan for a change that should not be
made.

## Conventions you must respect

The repository's `AGENTS.md` is in your context. The ones that decide whether a
plan is acceptable at all:

- compatibility surfaces (the `helion.language` API, `Settings` fields,
  `HELION_*` variables, public exceptions, Triton configs in saved caches) are
  not changed by a drive-by improvement. If the candidate needs one, that is the
  counterargument, not the plan.
- a fix in code the change already touches belongs in the change; an unrelated
  fix gets its own commit, so the plan should not collect drivebys.
- tests go where the repository puts them, and the budget rules in `AGENTS.md`
  apply to any test you add.
