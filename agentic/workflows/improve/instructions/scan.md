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
  Re-reading what you have, crawling where `rg` would answer, and proving a
  candidate the next step will prove anyway are not.
- **You do not have to prove a candidate.** Name the one command or test that
  would confirm or kill it, and move on: the next step runs it. Proving one here
  is paid for twice and costs the others.
- **Name a candidate as soon as it is worth naming**, and keep what you have in
  your list from then on. Once you have the number your prompt asks for you are
  done; anything after that is enrichment that must leave the list standing if
  you are cut off mid-call.
- **An angle that stops paying is a result.** A file that turns out to be
  well-tended, a search that finds nothing: say so and turn to something else,
  rather than digging for a fourth way to look at the same code.

## What the machine can check

You have a shell and the whole tree. Use them for feasibility, not proof:

- `nvidia-smi -L`, `python -c "import torch, triton"` and the backends'
  availability say what this machine can actually verify. A candidate whose only
  evidence needs a GPU that is not here is still a candidate: say that in
  `notes`, with the check that would show it.
- The compiler is the exception worth knowing: its output is text. A lowering, a
  printer or a codegen decision can be rendered and read on the CPU -- through
  `helion._testing.cpu_render_target` / `cpu_bind`, or a one-line
  `HELION_PRINT_OUTPUT_CODE=1` compile -- so a compiler candidate is the one you
  can prove here without taking a device from anyone.
- `scripts/gpu_board.py status` says who else is using the machine.
- `git log --oneline -15` and `git branch --list 'agentic/improve/*'` say what
  just landed and what another run already has in flight. Do not propose work
  that is in flight.

## Commands

- `rg` and `git` answer most questions in a fraction of a second. Reach for a
  script only when a pattern cannot, keep it to one short call, and give the
  call a bound: the shell tool takes `timeout=` (seconds, at most 270) and
  refuses more. Do not wrap a command in shell `timeout`.
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

In rough order of what this repository values -- the ranking below is what its
history shows it lands, not a wish list:

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

## What does not

- Preferences: naming, formatting, file layout.
- A change to public behaviour this repository has not asked for. The API in
  `helion.language`, `Settings` fields, `HELION_*` variables, public exceptions
  and saved Triton configs are compatibility surfaces: a finding that needs one
  changed is a note for a maintainer, not a candidate.
- Anything you cannot state a check for.

## How to write one

A numbered list. For each one: a first line naming the change (imperative, short
enough to become a branch name), then your case -- the line you saw, the effect,
the check that would confirm or kill it, what it might break -- and the files it
is about. Write for the model that will check your work: it is competent, and it
will disagree with you if you are vague.

Return the number of candidates your prompt asks for. If you genuinely find
fewer, return fewer and say why; a padded list wastes the investigation that
comes next. End with what you ruled out and where
you did not look, because the next step reads that too.
