# The Language-Acquisition Library

*The distilled, cited, build-queued knowledge base for the move from **modelling** text
to **acquiring** and **generating** it — written on counts, honest about the debates,
every mechanism tied to a standing rule and an existing experiment.*

This library is the bridge between the offline spine (A→AS, which only ever *read* and
*scored* a corpus) and the reactive harness (AT, which let the cortex *speak* and a world
*reply*). Everything here is a **count-native organ or eval** that plugs into AT's
`observe()` / `act()` / `Turn.signal` seams. No gradients, no k-means, no SVD, no backprop;
single streaming pass; bounded memory; human cognition (flaws and all) as the guide.

## The thesis

Children acquire language as a **bank of parallel accumulators over a Zipfian stream**, not
as an optimizer that locks weights. Boundaries fall out of branching-entropy spikes
(Exp A); units become a **chunk lexicon** whose sub-units *decay* as the whole commits;
words become *grounded* when a co-present scene binds a token to a referent (a second
co-occurrence axis the spine never had); categories emerge from frequent frames and
function-word anchors; constructions abstract from item-based islands by **association**
(ΔP/PPMI), not raw frequency; and **production is comprehension read the hard way** — a
many-to-one selection over the *same* counts, gated on a winner-take-all margin so the
cortex "understands but won't yet say." The whole arc is **gated, not scheduled** (Exp AK:
a count learner can't get stuck, so a difficulty curriculum is a no-op; only the *final*
memory and which cues have *ripened* matter). The single thing that turns acquisition into
generation is the reactive loop: a token that *closed a turn* — or that a reply *corrected*
— is worth more counts than the same token read passively. Acquisition and generation are
**one mechanism read in two directions**: the cue that predicts the next word during reading
is the guardrail that constrains the next word during speaking.

## The files

- **[SCIENCE.md](SCIENCE.md)** — the cog-sci ground truth, organized by the 12 research
  angles. Per angle: the summary, the strongest cited findings (with evidence strength and
  URLs), and the latest 2023–2026 work. Read this for *what is true* and *how sure we are*.
- **[MECHANISMS.md](MECHANISMS.md)** — every surviving count-native mechanism, grouped
  acquisition vs generation. Each carries the rule(s) it honors, the experiment id(s) it
  refines/extends, an honest novelty mark, and a full experiment sketch
  (corpus / metric / baseline / kill-condition).
- **[CURRICULUM.md](CURRICULUM.md)** — the developmental ordering
  (babble → first-words → word-combos → morphology-syntax → discourse → pragmatics): what
  the cortex learns at each rung, what *gates* progression, which mechanisms operate there.
  This is the spine for staged probing in the harness (a description of ripeness order, **not**
  a training schedule — AK forbids that).
- **[BUILD_QUEUE.md](BUILD_QUEUE.md)** — the prioritized, numbered queue continuing the
  A…AT naming from **AU** onward. Ordered by what unblocks the generation organ and the live
  reactive loop fastest. Each entry is one sentence + its kill-condition, cross-linked to
  MECHANISMS.md.
- **[READING.md](READING.md)** — annotated bibliography across all twelve angles, one line
  each on why it matters and what to take, with URLs.

## How this library was built

Twelve research angles were each grounded onto the existing experiment lineage (A→AT),
their proposed mechanisms checked against three adversarial lenses — **rule-compliance**
(does it really obey online-only / bounded / fragile-ideas / cognition-as-guide?),
**reinvention** (is it honestly new, or AB/AF/AJ/AO relabelled?), and **evidence honesty**
(is the cited science overstated? does it commit a category error?). Mechanisms that
survived are in MECHANISMS.md with their fixes applied; the cuts and merges are recorded
there too. The standing rules live one directory up:
[FRAGILE_IDEAS](../FRAGILE_IDEAS.md), [MEMORY_CONSTRAINT](../MEMORY_CONSTRAINT.md),
[COGNITION_AS_GUIDE](../COGNITION_AS_GUIDE.md); the experiment lineage is in
[PROVENANCE](../PROVENANCE.md) and the [lab notebook](../../experiments/LAB_NOTEBOOK.md).

## The one headline

**The chunking-vs-transitional-probability verdict is the sharpest unbuilt lever we own.**
Our current `CortexAgent` counts fixed-order n-grams — which is *exactly* the pure-TP table
the literature (Isbilen 2023; PARSER) says is **wrong**. The missing organ is a chunk
lexicon that **commits to whole units and lets their sub-unit transitions decay**. Build
that, and its discovered chunks become `act()`'s emission vocabulary — generation stops
being per-char gibberish. See mechanism **M1** and queue item **AU**.
