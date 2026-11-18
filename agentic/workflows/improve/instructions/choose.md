# Investigate the candidates, then choose one

Another model proposed the candidates in the prompt. You decide which one this
run works on. Nobody is going to average your scores or second-guess a ranking:
the choice is yours, and your reasoning is what the next step works from.

## Investigate first

You have a shell and the tree. For the candidates that look plausible:

- read the code they are about, and confirm the claim is true;
- run the thing that shows the effect -- a test, a small reproduction, a timing,
  a grep for the other callers;
- check that the change would not need a public behaviour change to work.

Two or three candidates checked properly beat ten skimmed. You do not have to
assess all of them; you have to choose well and say why.

## Spreading the investigation

You may delegate, and with candidates that check independently you should:
`delegate_task(agent_name="self", task=...)` starts a fresh copy of you with its
own context, and its calls are recorded as part of this run. This permission is
narrower than the scan's -- it covers checking candidates, not making the choice.

- **One candidate per delegate.** Name the one candidate in the brief, the file
  or line it is about, and what you want back: a verdict with the command that
  proves it, and what would make the change unsafe. A brief that carries two
  candidates gets you two skimmed ones.
- **Do not hand two delegates the same candidate.** The scan already spreads its
  families; a delegate that re-derives another's work costs five to ten minutes
  for an answer you already have.
- **Keep what you decide yourself.** Delegate the mechanical part -- does this
  claim hold, does this test really not run, does the reproduction fail without the
  fix -- and judge the answers yourself. A delegate that gets it wrong is
  invisible unless you read what it checked.
- **Worth it from three candidates up.** A delegate re-reads what it needs and
  takes five to ten minutes, the same as checking one yourself, so delegation
  pays when there are candidates left over after your own reading, not instead
  of it.

Say in your answer which candidates you verified yourself and which a delegate
brought back, so the plan and the reviewers know how much of a claim is second
hand. End it with a short **already proven** list: the commands you or a delegate
actually ran and what they showed. The planner and the reviewers read your answer,
and that list is what stops them paying for the same reproduction twice.

Do not delegate a check the scan already wrote out for you -- run it yourself,
it is usually one command. Delegate the verification you *cannot* do quickly:
several candidates at once, a module you would have to read in full, a render
that needs a fixture you have not learned yet.

## Then choose

Pick the one that is most worth doing **now**, given:

- how real it is -- you reproduced it, or you read the code that makes it true;
- how much a reviewer can check afterwards;
- how small the change can be. A change nobody can review is not an improvement,
  and the next step will keep the plan small;
- whether **this** machine can show it. A kernel run needs a backend that is
  installed and a device that supports it: no `jax` means no Pallas run, no MPS
  means no Metal run, a compute capability under 9.0 means no CuTe run. A CPU
  *render* of the generated source does count when the claim is about the source
  (see the scan's `notes`), but a candidate whose only check needs a TPU, a Mac
  or a B200 is the last one to pick -- and if you pick it anyway, say why.

This machine may be shared, and it is not yours to take a device from: a GPU
belongs to whoever claimed it, and nothing here claims one for you. Check what
is there (`nvidia-smi -L`, `scripts/gpu_board.py status`) and prefer a CPU check;
if you do run something on a device, keep it to the one command that shows the
behaviour, and never leave a job running behind you.

Small and certain beats large and speculative. A correctness bug with a
reproduction beats a nicer API.

## What to say

Start with **one imperative line** naming the change you chose: it becomes the
branch name, so make it specific and short. Then the useful part, which the next
two steps read:

- what you found when you checked, including what surprised you;
- the smallest version of the change that carries the value, if it is smaller
  than what the scan proposed;
- the strongest argument against doing it now -- say it even when you choose it,
  so the planner and the reviewers are not surprised later.

If every candidate turns out to be wrong, say so plainly and stop. That is a good
outcome, and it is better than choosing something you do not believe in.
