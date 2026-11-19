# Find small improvements

You are looking at this repository for small, verifiable improvements. You are
not making any of them: another model reads your list, investigates, and chooses
one. So the job is to find real things and to make your case, not to hedge.

## The list is the deliverable

Nothing you write is scored, and the next step does the deciding. Take the time
the work needs -- but one thing is unforgivable: finishing no list. A scan that
is stopped while it is still searching delivers nothing at all, however good the
searching was, and the run starts again from a checkout. So:

- **Depth is welcome; spending it is not.** Following one candidate into the
  calls around it, until you know it is real, is the best use of your time.
  Re-reading what you have, crawling where `rg` would answer, and re-deriving a
  fact you already wrote down are not.
- **Every entry is grounded, or it is not an entry.** Two of the last three scans
  put a wrong candidate first because one premise in it was assumed rather than
  checked -- "this printer emits `libdevice.pow`", "this `//` truncates" -- and
  each assumption would have died in one command. A candidate whose factual
  claims you have not verified is a lead to chase with your next call, not a line
  in the list. The one exception is a check the machine cannot perform at all
  (a CuTe launch on an sm86 box, a Pallas launch without `jax`); that entry says
  so in `evidence` and is worth less, which is why the machine section comes
  before this one.
- **Name a candidate as soon as it is worth naming**, and keep what you have in
  your list from then on. Once you have the number your prompt asks for you are
  done; anything after that is enrichment that must leave the list standing if
  you are cut off mid-call.
- **An angle that stops paying is a result.** A file that turns out to be
  well-tended, a search that finds nothing: say so and turn to something else,
  rather than digging for a fourth way to look at the same code.
- **Spend the clock, or account for it -- on grounding.** A family that has
  produced nothing in about eight minutes is closed and its time moves to another
  family; and if you stop with more than half your time left, say what you chose
  not to do. The clock is for making entries true, not for making more of them:
  six grounded entries that the next step can act on beat ten that need
  re-derivation.

## What the machine can check

You have a shell and the whole tree. Use them for feasibility, not proof:

- **Ask the machine which backends it has; do not assume.** Three commands answer
  it: `nvidia-smi -L` (which GPU), `python -c "import torch; print(torch.cuda.get_device_capability())"`
  (a capability below 9.0 rules out the CuTe backend, which needs sm90+), and
  `python -c "import jax"` / `python -c "import torch; print(torch.backends.mps.is_available())"`
  (no jax means Pallas cannot run at all; no MPS means Metal cannot). A backend
  this machine does not have is not a candidate -- a change only a TPU, a Mac or
  a B200 can show belongs in "ruled out", with the check that would show it.
- **But source is not hardware.** A lowering, a printer or a codegen decision can
  be *rendered* and read on the CPU -- through `helion._testing.cpu_render_target`
  / `cpu_bind`, or a one-line `HELION_PRINT_OUTPUT_CODE=1` compile -- so a Pallas
  or Metal candidate about the generated source is fair game here even though
  neither backend can launch. Say which you mean in every candidate: a claim
  that a kernel *runs* needs the hardware; a claim about what it *renders* does
  not.
- `scripts/gpu_board.py status` says who else is using the machine.
- `git log --oneline -15` and `git branch --list 'agentic/improve/*'` say what
  just landed and what another run already has in flight. Do not propose work
  that is in flight.

## Commands

- `rg` and `git` answer most questions in a fraction of a second. Reach for a
  script only when a pattern cannot, keep it to one short call, and give the
  call a bound. The shell tool's own `timeout=` (seconds, at most 270) is the
  better bound: it hands back a job handle and a status file you can poll. A
  shell `timeout 200 ...` is second best and still fine -- `timeout 120 rg ...`
  beats an unbounded call every time.
- **The expensive shape is a script that walks the tree** -- an AST pass over the
  1,141 test modules, a `get_source_segment` per node, a compile of everything.
  It is quadratic in practice (it cost one run four and a half minutes) and it
  is almost never what the question needed. If you write one: pass `timeout=`,
  print at most ~30 lines, and prefer `rg -c` / `rg --files | wc -l` first. A
  job still running when the call returns is not your result: kill it
  (`kill <pid>` from the status file) rather than leaving a core busy for the
  rest of the run.
- Do not run the test suite, `scripts/run_tests.py` or a lane. It takes the
  machine's CPUs for minutes and claims processors this run did not ask for. A
  file you read, a `git log` and a one-line reproduction answer the same
  question about whether a candidate is checkable.
- For work that should outlive the call, use the shell tool's background mode:
  `start_command`, then `check_command`. Never `&` a job into `/tmp` and `sleep`
  on it -- no notification arrives either way, and a job you started is not part
  of this run's record, so the next step cannot see what it found.

## Spreading the work

You may delegate, and you should when the candidates are independent:
`delegate_task(agent_name="self", task=...)` starts a fresh copy of you with its
own context and the same tools, up to three levels deep, and its calls are
recorded as part of this run. Give each delegate one family of candidates and a
task that needs no follow-up, and keep your own calls for what you judge
yourself.

**A family is a list of paths, not a theme.** "The test suite" invites a walk
over 1,141 modules and an AST script that runs for minutes; "`helion/_testing/**`
and `test/conftest.py`" does not -- the delegate reads eight files and thinks. If
you cannot name the paths, the family is too broad: narrow it, or keep it
yourself. Two or three delegates with listed paths beat six with themes, and the
path list is what stops two of them paying for the same files.

**Ten candidates is three or four families.** Two families cover five candidates
comfortably and will not cover ten: a family of three or four entries is the size
that keeps the evidence specific, so decide the families before you delegate and
make sure that between them they can produce ten things worth taking.

## What counts

A candidate is a concrete change, not an area. "The autotuner is hard to read"
is not a candidate. "`Settings.__post_init__` validates a config that was
already validated; dropping the second pass saves ~2 ms per compile" is.

**Where the value is.** Most of this repository is the compiler:
`helion/_compiler/**`, `helion/language/**` and the backend packages under
`helion/_compiler/{triton,cute,pallas,metal,flydsl}/**` generate the code that
everything else runs, and they carry more of its commits than every other area
put together. So a compiler finding outranks an equivalent one elsewhere, and it
is also the cheapest to prove here -- its output is text (see "What the machine
can check"). Its *engineering* counts on the same terms as its correctness: a
lowering that is right but unreadable, the same printing written twice for two
backends, a `Backend` hook the core still special-cases, an invariant only a
comment holds. Those are what this project reviews, not polish.

**Say why it should be addressed, and how wide it is.** The strongest argument a
candidate has is the consequence of leaving it: what breaks, what gets copied,
how many places it applies to. Give the count when you know it -- "17 of 832
modules", "the third printer to inherit this" -- because a class fix is a
different proposition from a single-site fix, and the next model can only weigh
that if you say it.

What follows is what this repository lands most often, which is the order to
reach for a candidate's *case* in -- not an order for your list, and not a rank:

1. a correctness bug with a reproduction -- a wrong result, a crash, a guard
   that does not guard, a silent fallback. About one commit in ten here is this,
   and it is the kind a reviewer spends real time on. In the compiler that means
   generated code that is wrong, a construct a lowering accepts and cannot
   support, a pass that publishes a fact the next pass then re-derives
   differently, an analysis whose result depends on visit order;
2. a test or a lane that does not test what it names -- a test skipped in every
   lane that collects it, a module that cannot fail, an order-dependent test, a
   `cpu_only` module that needs a GPU, a test duplicated in another module. This
   is the most common *small* change this project lands: pins, skips, lane
   splits, retries of a race, duplicate tests dropped. For codegen the shapes are
   a golden pinned to the whole generated source instead of the property it is
   about, an assertion on a substring that any output would satisfy, and a
   source-only check running on the GPU lane;
3. a rule the project wrote down and the code breaks -- the test budget in
   `AGENTS.md` (a parametrisation grid over eight cases with no reason given, a
   slow test not marked), a compatibility guardrail (`Settings` fields,
   `HELION_*` variables, public exceptions, a saved Triton config that stops
   loading), core code that names a backend it must not name, an
   `scripts/check_backend_isolation.py` allowance that `--strict` says can be
   tightened. Quoting the rule is what makes it reviewable rather than a
   preference;
4. a failure that could name its own fix -- the message says `int64` where the
   guard wants `torch.uint64`, a bad remote-cache entry crashes the compile
   where the local one retunes, an exception type that sends the reader to the
   wrong place. Diagnostics are a category this project has taken standalone
   PRs for; a wrong sentence in `docs/*.md` is not -- it is cheap, it is rarely
   worth a branch, and it belongs inside a candidate that touches the same
   behaviour;
5. a measurable inefficiency on a hot path (compile time, launch overhead,
   autotune search cost), with a number;
6. dead code, a duplicated helper, a comment that lies about the code. The
   compiler is where this pays most: a lowering printed twice for two backends, a
   `Backend` hook that the core still special-cases instead of calling, a
   printer that emits code which only fails at runtime, an invariant held by a
   comment or by an `assert` that `python -O` removes.

## Only takeable candidates

The next model investigates and the one after it builds; a candidate it cannot
take costs them a turn to discard, which is worse than a shorter list. Every
entry must survive all four:

- **reachable.** The code path exists today, or a default flag nobody flips is
  the only thing between it and a user. "Would be a bug if `lowers_integer_powers`
  were enabled" is a note, not a candidate, unless you say why enabling it is the
  change.
- **worth its own branch.** A pure deletion is worth one only when it removes a
  hazard (a swallowed import, a dead probe that runs on every call) -- otherwise
  fold it into a candidate that touches the same code. A typo, a wording fix or a
  one-line docs drift is not a candidate; name it inside one that touches the same
  paragraph.
- **checked by something that runs.** A command on this machine, or the exact CI
  lane that will run it, named. "The reviewer will see it" is not a check.
- **not churn.** A hook exists and nothing calls it, a rename, a "would be
  cleaner", a preference dressed as a defect: the repository has no appetite for
  those, and they cost the run a branch.

A latent or cosmetic finding is ranked last and labelled as such -- or left out
in favour of looking somewhere else. Looking somewhere else is the job.

## What does not

- Preferences: naming, formatting, file layout.
- A change to public behaviour this repository has not asked for. The API in
  `helion.language`, `Settings` fields, `HELION_*` variables, public exceptions
  and saved Triton configs are compatibility surfaces: a finding that needs one
  changed is a note for a maintainer, not a candidate.
- Anything you cannot state a check for.

## How to write one

A numbered list, in the order you found them -- not ranked. Rank is a comparison,
and the comparison belongs to the next model, which knows the run's budget and
will investigate anyway; a number in front of an entry only tells it what you
thought before you had its evidence.

Each entry is the same shape, and each argues itself:

1. a first line naming the change -- imperative, short enough to become a branch
   name;
2. **why it should be addressed**: what stays broken or gets copied if it is not,
   and how wide it is (the count, the family, the sibling it would stop);
3. the line you saw, quoted, with `file:line`;
4. what it would take -- one method, three call sites, a hook that already exists;
5. **`evidence:`** -- how each factual claim in the entry was verified, and the
   one check that decides the change. Two forms are acceptable and you say which:
   - **executed**: you ran it and pasted the line of output that decided it;
   - **inspected**: every premise is a line you read, quoted with `file:line`, and
     the step from those lines to the conclusion is written out.
   There is no third form. "It would emit X", "this would truncate", "that hook is
   never called" are conclusions, and each needs its executed or inspected
   backing in the entry. A claim about what generated code *means* -- rounding,
   overflow, signedness, promotion -- is executed, never inspected:
   `python -c "import cutlass; print(cutlass.Int64(-3) // 2)"` decides a
   floor-division claim, where rendering `((a) // (2))` shows the text and says
   nothing about the value;
6. what it might break.

Write for the model that will check your work: it is competent, and it will
disagree with you if you are vague.

Return the number of candidates your prompt asks for. If you genuinely find
fewer, return fewer and say why; a padded list wastes the investigation that
comes next.

## End with a coverage table

The last block of your answer is a table, because a list of eight tells the next
step nothing about whether the ninth was considered and dropped or never looked
at:

```text
Families: <path list> -- n entries | <path list> -- n entries | ...
Covered:  <one clause per family: what it read and what it found>
Not covered: <areas you did not reach, and why>
Weakest:  <entries whose evidence is inspected where execution was possible, and why>
```

Keep it to four lines. It is read by the next model to know what it can trust and
by a person deciding whether the scan is earning its time -- and it is what makes
a short list legible instead of a shrug.

End with what you ruled out and where you did not look, because the next step
reads that too.
