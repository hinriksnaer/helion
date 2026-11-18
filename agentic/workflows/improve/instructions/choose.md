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

## Then choose

Pick the one that is most worth doing **now**, given:

- how real it is -- you reproduced it, or you read the code that makes it true;
- how much a reviewer can check afterwards;
- how small the change can be. A change nobody can review is not an improvement,
  and the next step will keep the plan small;
- whether the hardware here can show it. Prefer what can be verified on this
  machine.

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
