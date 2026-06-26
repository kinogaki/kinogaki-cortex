# The Ten Commandments of Fragile Ideas — a research rule for kinogaki-cortex

A rule for how we judge ideas. The search space is **very high-dimensional**: a weak first result (e.g. "no
better than a bigram") is the *normal* state of a real idea, not a verdict. We must often go **10–20 development
steps** down a weak idea before a win appears. So we protect fragile ideas instead of killing them at the gate.

> Honest note: there is **no canonical "10 commandments" by Jony Ive** — the famous numbered list is Dieter
> Rams' *Ten Principles of Good Design*, which Ive reveres. These ten are **synthesized** from Ive's genuine,
> documented philosophy on the fragility of ideas (verbatim quotes below) + our high-dimensional-search insight.

## The quotes this is built on (Jony Ive, verbatim)

- "**Ideas, by definition, are always fragile. If they were resolved, they wouldn't be ideas.**"
- "While ideas ultimately can be so powerful, they begin as **fragile, barely formed thoughts, so easily missed,
  so easily compromised, so easily just squished.**"
- "I feel that ideas are very fragile, so **you have to be tender when they are in development.**"
- "You have to make an **extraordinary effort not to focus on the problems**, which are implicated with any new
  idea. You have to focus on the actual idea, which is **partial, tentative, and unproven.**"
- "If you don't **actively suspend your disbelief** — if you don't believe there is a solution to the problems —
  of course you will lose faith in your idea."
- "**Criticism and focusing on the problems can be so damaging**, particularly in the absence of a constructive idea."
- "**Opinions are not ideas.** Opinions are not as important as ideas. Opinions are just opinions."

## The Ten Commandments

1. **An idea is fragile by definition.** A first result no better than a bigram is the normal state of a real
   idea — *if it were already resolved, it wouldn't be an idea.* Don't read weakness as a verdict.
2. **Be tender in development.** Never hold a one-step-old idea to a ship-quality bar. Judge nascent ideas by
   **trajectory, not level**.
3. **Focus on the idea, not the problems.** Every new idea arrives wrapped in problems — that's expected.
   Actively **suspend disbelief**: assume a solution exists and hunt for it.
4. **Go 10–20 steps before judging** (the high-dimensional rule). Set an explicit **development budget** (≥10
   real variations) before any kill decision. The space is vast; wins compound late.
5. **Criticism without a constructive alternative is destructive.** "Not better than bigram" only counts as a
   kill when paired with a *better* idea or an *exhausted budget* — never on its own.
6. **Opinions are not ideas.** "It probably won't work" is an opinion; only an experiment is evidence. Never kill
   on priors.
7. **Measure the right axis.** A fragile idea usually first wins on a dimension you weren't headlining —
   calibration, robustness, generalization, rare-context, inspectability. *Before killing, check the other dials.*
   (Exp S: ties top-1 but 3× perplexity. Exp R: loses clean, wins under noise. Exp T: loses overall, wins on the
   backoff slice. All three would have been wrongly killed on the headline metric.)
8. **Keep a graveyard, not a trash can.** Record every shelved idea with *why* and *the step it died at*, so it
   can be resurrected when a complementary piece arrives.
9. **Combine fragile ideas.** Weak ideas often win only in combination — offset-attention + evidence + ignition,
   not any one alone. Don't judge an idea in isolation from the stack it's meant to live in.
10. **Protect the time and space to develop.** Fragile ideas die of premature optimization pressure. Budget
    exploration explicitly; **nurture before you prune.**

## Graveyard / resurrection list (fragile ideas parked, NOT killed)

- **Raytracing / proximity as predictor** (Exp P): lost to bigram on next-word. *Parked, not dead* — resurrect as
  a **rare-context backoff modulator** (its untested winning axis, per commandment 7), and as inspection/similarity.
- **Recency/topic cache** (Exp H/K): small, hurt phrase-coherence. *Parked* — likely wins once scoped to a
  discovered topic segment (boundaries) rather than a fixed window.
- **Evidence-drop as a boundary signal** (Exp R): lost to branching entropy. *Parked* — may win fused with entropy,
  or at the word/phrase level where the drop is sharper than at chars.
- **Ray extrapolation** (Exp P): 1.79% vs 21%. *Mostly dead* at the word level, but the trajectory idea may have
  life at the **topic** level (smoother) — budget a couple more steps before final burial.
