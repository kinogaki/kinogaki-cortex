# CURRICULUM — the developmental spine

The order in which a child's language comes online — and the order in which the cortex's
organs ([MECHANISMS.md](MECHANISMS.md)) become *usable*. This is the spine for staged
probing in the harness.

> **The standing rule (Exp AK).** This is **NOT a training schedule.** A count learner
> cannot get stuck — an early noisy high-order count is *outvoted*, not *frozen*, so
> sequencing the input by difficulty is a **no-op** (AK reproduced this; BabyLM confirmed it,
> β≈−3.6). "Starting small" was a property of the gradient optimizer, not of learning. So
> this curriculum is a **diagnostic ordering**: a description of *which mechanism's
> preconditions (enough types, a ripe frame, a stable slot, a live interlocutor) are met
> when* — and which **gate** a stage opens. The only thing that legitimately gates is a
> ripeness/confidence/entropy threshold the stream crosses, never a hand-timed lesson plan.

The one structural exception (embedding-depth self-gating for recursion) is **thin** and
gated as a fragile idea — see the morphology-syntax rung.

---

## Rung 0 — BABBLE → FIRST-WORDS (segmentation, the floor)

**What the cortex learns:** a sub-word/word-unit inventory from an unsegmented stream — the
8-month-old's task. Carve the stream into word-like tokens before anything binds to meaning.

**Gate to open it:** enough char-level counts accumulated to make branching-entropy estimates
non-degenerate.

**Mechanisms here:**
- **M1** chunk lexicon with sub-unit interference — discovers the unit inventory (proto-words)
  and mints the chunks that become `act()`'s vocabulary. *The babble→first-words organ.*
- **M3** reliability-gated boundary detectors — the adaptive segmenter (scoped to the
  head-final drift).
- **M15** attentional salience gate + pause/punctuation-as-boundary prior — gates segmentation
  itself; upstream of everything.
- Exp **A** `branch_chunk` is the already-built floor everything sits on. Keep the entropy
  **signal** and the boundary **commit** threshold as two dissociable knobs (Benjamin 2023).

**Honest category note:** a char/grapheme stream has no syllabic CV structure, so do not
invoke "canonical babbling" unless running on a syllabified/phonemic stream; otherwise frame
M1/M3 plainly as chunk discovery. The chunk-coverage **ratio** is a useful *precursor metric*
(developmental-risk readout), **not** a stage gate (AK: gating first-words on it is likely
ceremony).

---

## Rung 1 — FIRST-WORDS (form acquires a referent and a graduation threshold)

**What the cortex learns:** which chunks become **words** — a chunk earns word-status when it
(a) acquires a **referent** (meaning) and (b) **graduates** (crosses a count-maturity
threshold). This is the meaning gate that turns the segmenter's units into a lexicon and gives
generation something to be *about*.

**Gates:** a stock of segmented chunks (rung 0); for grounding, a **scene-bearing
environment** emitting co-present referent-ids (M2's substrate); for graduation, **AB**
confidence over **diverse** contexts (M14).

**Mechanisms here:**
- **M2** cross-situational word→referent learning (dual-variant) — the first-words meaning gate.
- **M14** recency + context-diversity accumulator — graduation; moves a token from first-words
  into word-combos (a word must graduate before it can lead an AF construction).
- **M17** two-threshold comprehension/production gate — the comprehension read (`t_low`) runs
  from the very start, so **comprehension precedes production at every rung by construction**.
- **M11** mutual exclusivity (fan-divided, cue-gated) — gated by and useful only **after** a
  stock of known names exists to compete.
- **M13** two-rate fast-map slot with spacing-sensitive consolidation — runs throughout, matters
  once the lexicon outgrows the slot budget.
- **M22** exemplar-chaining for overextension — fires here: overextension is the error of a
  child with words but no rule, **before** productive grammar.
- **M7** function-word anchor voter + **M8** seeded label propagation — need only a tiny lexicon
  (the top-frequency band exists from the first thousand tokens; 8 seeds suffice), so noun/verb
  categories emerge before any rich syntax.
- **M12** shape-bias meta-counter — **developmentally gated**: fires only after ~50 consolidated
  nouns (the emergence curve is the test), at the late-first-words / word-combos boundary.

---

## Rung 2 — WORD-COMBOS (item-based → abstract; the two-word floor)

**What the cortex learns:** to combine graduated words into open-slot constructions; verb-islands
fuse into productive schemas; categories sharpen.

**Gates:** an AF **frame** must be ripe (≥40 tokens, AF's ripeness gate); a join clears only when
**both** slots and their joining bigram/frame are over production-threshold (the two-word MLU
floor); for the merge, enough **distinct fillers** cross a slot (Baayen P / type-count).

**Mechanisms here:**
- **M5** streaming-association slot strength (ΔP/PPMI) — the substrate; association gates
  over-generation where raw frequency would not.
- **M6** two-sided frequent frames + cross-anchor merge — verb-islands → schemas (the
  concrete→abstract trajectory; measurable only on CDS/exposure-order).
- **M9** dual order-free / order-sensitive routing matures here (noun/verb dissociation).
- **M16** variation-set minimal-pair miner — minimal-pair diffing is how a child carves
  recombinable slots; feeds AF and the phrase-boundary level (M).
- **M18** threshold-gated decoding begins to combine (telegraphic "cat here"; MLU floor).

---

## Rung 3 — MORPHOLOGY-SYNTAX (inflection, agreement, the productive system)

**What the cortex learns:** morphology (inflection, the +ed default, the U-shape), agreement,
non-adjacent dependencies, and the productive emission of grammar.

**Gates:** a tense-cue exists only once a stem reliably combines with a tense-marking context
(M6 frame ripe); the irregular must be **behaviorally mastered** before the U appears ("went"
before "goed", **AH** StabilityMonitor); enough **regular** exposure for the +ed default's
branching-entropy to clear the productivity threshold; for production, **AH** stability promotes
a construction to a generation-eligible SlotObject.

**Mechanisms here:**
- **M19** dual-route inflection head with f·c blocking — the words-and-rules knob; rare
  item-specific micro-U.
- **M20** R-W recovery loop — recovery without feedback (the first organ that learns from a
  **reply**, bridging toward discourse).
- **M21** mastery-mines-the-rule redescription — the micro-U as a side effect of compression,
  applied **per-item** (never a macro-U).
- **M18** threshold-gated decoding admits Brown's morphemes in frequency×frame-reliability order
  (reported as correlation, not explanation).
- **G1** coverage-competition production is where the constructicon **speaks** — gated by AH's
  stability/mastery trigger.
- **G7** UID re-ranker shapes optional-element omission (presupposes syntax with optional
  elements).
- **Structure-graded recursion exposure** (the **one** defensible curriculum, Lai & Poletiek;
  **thin**, gated as fragile) sits at the far end: order by **embedding depth**, self-gated on the
  agent's own entropy (admit depth d+1 only after depth-d transition entropy stabilizes). It is
  expected to possibly **confirm AK by losing** — a clean negative is the publishable outcome.
  See BUILD_QUEUE **BJ**.

---

## Rung 4 — DISCOURSE → Rung 5 — PRAGMATICS (the reactive loop, the top)

**What the cortex learns:** to align with a partner, retreat from over-generalization on
correction, prefer grounded forms, resolve reference in context — the developmental moment a
child stops over-generalizing because a caregiver models the conventional form.

**Gates:** the **AT InterlocutorEnv** must be live (a responder returning competing forms); a
**generation organ** must exist at all (G1); everything below (a stable lexicon, ripe frames,
productive constructions) — you can only ground what you can already emit and resolve.

**Mechanisms here:**
- **G2** contingency-gated learning rate (+ **G5** two-state cadence, bound to it) — a reply that
  answered the model teaches more than passive reading; the proactive speaker.
- **G3** repair as a paired count edit — reformulation corrects the conventional form.
- **G4** per-conversation common-ground overlay — ratified shared state, grounded-form preference,
  reference resolution (a generation bias, **not** the long-span predictor AM killed).
- **G8** dual-counter structural priming / alignment — interactive alignment; couples comprehension
  to production with a partner.
- **M4 retreat loop** (mechanism M4's reactive twin in the construction angle) and **E3**
  communicative-success reweighting — the say-it/get-corrected/retreat contract.

**Out of this library's altitude:** the ~0.9% global-coherence frontier (event/situation model)
is **AC/AM/AG/AL** territory — the slice that won't yield to stacked state. Distributional
acquisition earns its keep **below** it; these mechanisms feed it operands but do not reach
discourse-level coherence themselves.

---

## The bottom-up gating chain (one line)

`segment (M1/A) → graduate + ground (M14/M2) → comprehension>production gate (M17) → combine
into frames (M5/M6/M16) → inflect + speak (M19/G1) → align with a partner (G2/G3/G4/G8)` —
each rung gated by a **ripeness/confidence/entropy threshold the stream crosses**, never by a
schedule. Consolidation/sleep (M23–M26) runs **across** all rungs, deciding which specifics
survive the budget (governed by **AE/AI/AR**).

See [BUILD_QUEUE.md](BUILD_QUEUE.md) for the order to actually build these in (which is *not*
the developmental order — it is ordered by what unblocks the generation organ and the live
reactive loop fastest).
