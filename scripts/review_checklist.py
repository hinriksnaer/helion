"""The review checklist, in one place, for every reviewer the project runs.

Both review entry points ask a model to find the same class of problem:
``scripts/autoreview.py`` drives coding-agent CLIs and ``agentic/`` drives
composed pydantic-ai agents. They differ in what the reviewer *is* -- a CLI with
its own harness, or a model plus a named tool set -- and they must not differ in
what they are looking for. The list lives here, one copy, and both import it.

Rules that do not apply to a change cost a line each to reject; a rule that is
missing because it was copied once and then edited in one place only costs a
missed finding, which is the failure this file exists to prevent.
"""

from __future__ import annotations

REVIEW_CHECKLIST = """\
Check for:
- Correctness issues, including bugs you notice in the code around the change
  (say when one predates the change)
- Feature regressions (such as disabled or weakened tests, or a code path
  silently lost)
- Opportunities for simplification or cleanups
- Places that are hacky or overly narrow
- Code duplication
- Backend-specific tests or test files that should use `@onlyBackends([...])`
- Too broad `try/except:` lines that could hide bugs
- Overly defensive getattr/hasattr checks that should be base class schema updates
- Comments that might go stale quickly (e.g. line numbers)
- Compatibility (AGENTS.md, "Compatibility guardrails"): changes to the
  user-facing language (`helion.language`, `@helion.kernel` and its arguments,
  `Settings` fields, `HELION_*` environment variables, public exceptions) that
  the commit message does not flag; Triton `Config` changes that would stop
  configs saved by older versions from loading or working (a new option needs
  a default that keeps the old behavior; a renamed or retired option must
  still be accepted). Adding config options, and changing those of
  experimental backends such as CuTe, is fine.
- Tests against AGENTS.md's "Test Budget Rules": tests that only inspect
  generated source but do not bind on CPU in a `cpu_only` module; parametrize
  grids above 8 cases without a reason or `pairwise`; tests that spawn
  interpreters, run a real autotune search or take over ~20s without
  `@pytest.mark.slow`; shared test helpers outside `helion/_testing/`; deleted
  or renamed test modules still listed in `test/shard_durations.json`;
  changes to generated code that no test covers
- AGENTS.md coding rules that lint cannot check: `print()` inside kernels,
  `import helion as hl` (the pattern is `import helion.language as hl`), `del`
  or `_ =` for unused arguments, needless local imports
- Commit messages, when reviewing commits (they become pull request
  descriptions): a `[area] Imperative summary` title; a body that matches the
  changes (why, what changed, a `Testing:` paragraph whose claims are
  plausible); compatibility-relevant changes flagged. In a range of commits,
  also commits that should be squashed (the same code changed twice, tiny
  commits) or that would not work if landed in order.
- Other important issues
"""
