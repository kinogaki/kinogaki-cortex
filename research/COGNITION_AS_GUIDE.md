# Global rule — Human cognition is the guiding model

The fourth standing rule of kinogaki-cortex, with **online-only**, **bounded memory**, and **fragile ideas**.
It explains the other three: they are the shape they are because *that is the shape of a mind*.

> **Human intelligence — flaws and all — is our model and our intuition. It is the only thing that achieves
> general intelligence under the constraints we've adopted (online, single-pass, finite memory). So we treat it
> as both the existence proof and the design oracle, and we keep pulling cognitive science into the framework.**

## Why

Every other rule is a fact about minds. Online learning (we learn as we live). Bounded memory (we forget, we
generalize, we write things down). Fragile ideas (a hunch needs nurturing before it pays). Human cognition is
where those constraints *coexist with* general intelligence — so when we're unsure how a piece should work, the
right question is not "what's optimal?" but **"how does a person do this?"** The answer is usually buildable on
counts, because the brain isn't doing gradient descent either.

## The flaws are features

We follow human cognition *including* its documented quirks, because under real constraints those quirks are
adaptive priors, not defects:

- **availability** ≈ recency-weighted counts (leaky accumulators) — we already lean on this.
- **representativeness** ≈ similarity/prototype matching — our leader-clustering.
- **anchoring / order effects** ≈ sticky context, hysteresis — our event slots.
- **good-enough / satisficing** ≈ stop at the first confident level — effort-gated depth.
- **capacity limits & forgetting** ≈ the bounded-memory rule itself, which *forces* generalization.

So a bias that hurts on a contrived puzzle is often the right bet on the real distribution. We keep them.

## The big gap this names: System 1 vs System 2

Everything we have built is **System 1**: fast, parallel, associative, intuitive — counts voting, instantly. It
is exactly the system that recognizes a word, predicts the next letter, feels a topic shift. What we have *not*
built is **System 2**: slow, serial, deliberate, working-memory-bound — the system that follows a chain of
steps, holds and manipulates a few items, checks its own work, and reasons about something not directly present.

Our open frontiers map onto this precisely:
- compositional **reasoning** (AD: the parallelogram is in System-1 counts, but combining relations in steps is
  System 2);
- **discourse coherence** (AC: we *detect* the event-shift with System-1 surprise, but *maintaining and
  reasoning over* a situation model is System 2);
- the **count-native sharpening combiner** is, plausibly, what a System-2 step does to System-1 representations.

**A count-native System 2 — a small, capacity-limited, serial workspace that manipulates System-1 outputs under
metacognitive control — is now a primary research direction.** Mechanisms to mine and build: working memory
(Baddeley/Cowan, 7±2), the central executive / cognitive control, deliberate serial inference, metacognitive
confidence as the trigger to "think harder" (we have calibration — Exp AB — as the seed), and the global
workspace as the bottleneck that serializes parallel System-1 votes into one deliberate thought.

## How to apply

- Prefer mechanisms with a cognitive analogue; name the analogue when you build one.
- Keep a standing cog-sci mining habit (this rule is why the cognition fan-outs happen).
- Don't "fix" a human-like bias before checking whether it's an adaptive prior under our constraints.
- Treat **System 2** as the project's next architectural layer, not an afterthought.
