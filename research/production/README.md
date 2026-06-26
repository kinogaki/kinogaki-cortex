# The Production Library

*The distilled, cited, build-queued knowledge base for the move from **mimicking**
language to **producing** it — and the loops that force the crossing. Written on counts,
honest about the debates, every mechanism tied to a standing rule and a real experiment.*

This library is the second half of the bridge the [acquisition library](../acquisition/README.md)
began. Acquisition asked how a reader's counts come to exist; production asks what turns a
reader into a **speaker who means it** — a system that emits a token *for its effect*, not for
its fidelity to the input. Everything here is a **count-native organ or eval** that plugs into
the AT reactive harness (`observe()` / `act()` / `Turn.signal`) and the [generation turn](../acquisition/MECHANISMS.md)
the acquisition round already built (AU chunk lexicon → BD/BL coverage-competition producer).
No gradients, no k-means, no SVD, no backprop; single streaming pass; bounded memory; human
cognition (flaws and all) as the guide.

## The thesis

A reader does not automatically become a speaker. Reading is mimicry — predict the next token
to match the stream. Production becomes **communication** only when an emitted token is selected
for its **effect on a listener**, not for fidelity to the input. The project has already crossed
the single hardest threshold this predicts: **a contingent reply teaches more than the same
tokens overheard cold** (Exp BE, +0.45 bpc over the scrambled-timing yoke). Contingency is the
engine. What is missing is the **steering**: *which* utterance to repeat (the reward attaching to
a producible unit, not to a whole warm turn), and *for whom* (a second count table approximating
the listener — recipient design as the **divergence** between a speaker table and a listener
table, not a literal "theory of mind"). The honest frame is layered: the reactive loop (AT) plus
contingency (BE) is the motor; the new organs are a steering wheel — a function split that says
whether a form was an echo, a name, or a request that got something; an intrinsic vocal-play
drive so the loop produces *before* any listener exists; a metacognitive controller that reads
the project's existing confidence scalar three ways (emit / deliberate / **ask**); and an
audience model scored on the listener's surprisal, never the speaker's. **Production is
comprehension read the hard way — and communication is production read for its consequence.**

## The crossing, in one loop

1. **Endogenous play** — an always-on learning-progress drive (`act()` probes where its own
   counts are still sharpening) so the cortex produces with no listener at all (>90% of infant
   protophones are to no one). This breaks AT's cold-start without inventing a reward.
2. **Contingency** — a reply that answers you (Exp BE) up-weights what you said. *Already won.*
3. **Function** — the same form, tagged by what *preceded and followed* it: echo (just heard),
   name (referent present), request (got an answer). Reward attaches to the function, not the
   surface.
4. **Audience** — a second count table over the listener's own tokens. Informativity is the
   **divergence** of speaker-surprisal from listener-surprisal against a *static* listener prior
   (Exp AM's law) — recipient design, scored on the listener, judged on referential success and
   redundancy suppression, **never** on held-out bpc.
5. **Metacognition** — the existing confidence scalar (Exp AG conflict + Exp AB f·c) read as a
   three-way control: emit when sure, **deliberate** when the gap is in the form, **ask** when the
   gap is in the goal.

## The files

- **[SCIENCE.md](SCIENCE.md)** — the cog-sci ground truth, organized by the eight production
  angles (mimicry→generative, communicative pressure, metacognition, world/situation model,
  theory of mind, reward & motivation, active-inference control, inner speech). Per angle: the
  summary, the strongest cited findings (with evidence strength and URLs), and 2023–2026 work.
- **[MECHANISMS.md](MECHANISMS.md)** — every surviving count-native mechanism, grouped by the loop
  it serves (**production / world-model / metacognition / social-ToM / motivation**). Each carries
  the rule(s) it honors, the experiment id(s) it refines/extends, an honest novelty mark, and a
  full experiment sketch (corpus / metric / baseline / kill-condition). New experiments are named
  **BN onward**, continuing the A…BJ lineage; cross-linked to the acquisition
  [BUILD_QUEUE](../acquisition/BUILD_QUEUE.md).

## How this library was built

Eight production angles — each grounded onto the real experiment lineage (A→BL) with its
mechanisms — were checked against two adversarial lenses: **rule-compliance + reinvention** (does
it obey online-only / bounded / fragile / cognition-as-guide, and is it honestly new or
AM/G4/BE/AL relabelled?) and **evidence honesty + category errors** (is the cited science
overstated? does it reify a mental state as a count table?). The recurring proposal across nearly
every angle was *"a second count table approximating the listener."* That table is rule-legal —
additive counts, leaky, bounded, no SVD — and it is **not** a reinvention of Exp AM **iff** it is
scored on referential/communicative **success**, an axis the offline spine never had, because
AM's static-prior verdict only pre-kills *bpc* claims. So the decisive cut kept the **first** clean
instance of each genuinely new organ, revised the ones that pin a distinct angle on the same idea,
and cut the duplicates and the one true category error (per-entity **belief** tables routed by a
verb frame — that reifies false-belief tracking as topic-conditioned backoff). The standing rules
live two directories up: [FRAGILE_IDEAS](../FRAGILE_IDEAS.md),
[MEMORY_CONSTRAINT](../MEMORY_CONSTRAINT.md), [COGNITION_AS_GUIDE](../COGNITION_AS_GUIDE.md); the
experiment lineage is in [PROVENANCE](../PROVENANCE.md).

## The one headline

**The audience model is the first top-down state with a license to change the 99% slice.** Every
persistent-state mechanism the project tried as a *predictor* — the situation model, the topic
prior, the narrative-event chain (Exp AM) — landed on the dead 0.9% backoff slice; a static prior
from the same count mass matched it. The listener table escapes that grave **only** because it is
not a predictor: it is a generation **bias** scored by whether a modeled listener recovered the
intended referent. Build it as the inverse-read of the grounding axis (Exp AV's word→referent
counts, read backwards as referent-given-utterance), rerank the BD producer's candidates by it
minus a cost, and judge it on referential success against a *yoked* (scrambled) listener. See
mechanism **A1 (L0-listener + S1 speaker)** and queue item **BN**.
