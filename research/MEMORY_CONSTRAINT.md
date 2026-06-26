# Global rule — Bounded memory

The third standing rule of kinogaki-cortex, with **online-only** (no gradients, single pass) and **fragile
ideas** (judge on the right axis, nurture 10–20 steps). It is the one that makes the project honest.

> **The model operates under a MEMORY BUDGET. It may not keep unlimited counts. Like a mind in the real world,
> it must cope with finite memory — through generalization, sleep, and the environment.**

## Why this changes everything

Until now most experiments assumed an *unbounded* store — `np.unique` over the whole stream, every context kept
forever. The capstone finding, **"what survives scale,"** was measured in exactly that regime: with infinite
memory, more data subsumes any mechanism that merely re-predicts what raw counts already hold, so the top-down
topic prior, the noise→concept shift, sleep's bpc gain, and the retention edge all *vanished* at scale.

**A memory budget reverses that.** When you cannot keep every count, you must throw most of them away — and the
only way to stay good after throwing them away is to have **generalized** first. So the mechanisms that looked
redundant at infinite scale are precisely the ones a bounded model needs:

- raw high-order counts are the first thing you can't afford → you must back off to **concepts / constructions**;
- you can't store every instance → **sleep/consolidation** (distill specific→generic, merge, prune) is how you
  pay down the table;
- you can't keep everything that's stale → **importance-based eviction** (STI/LTI, ART vigilance) decides what
  survives (Exp AE: this is the whole game under a cap);
- and what won't fit internally goes **outside** — to a store you re-read.

Bounded memory is what *forces* generalization. Generalization is the point. The constraint and the goal are the
same thing.

## The three ways to cope (build & measure these)

1. **Generalization** — compress many instances into reusable abstractions (online concept clusters, open-slot
   constructions, prototypes). Success = dropping instances without losing prediction. (Exp C, U, AF.)
2. **Sleep cycles** — offline consolidation that prunes lossy counts, distills specific→generic (lossless where
   the specific ≈ its backoff), merges duplicates. One pass helps; over-refinement degrades (Exp AA). Under a
   budget this is not optional housekeeping — it's how the table stays within the cap.
3. **Environment as memory** — offload durable knowledge to an EXTERNAL store the model writes and re-reads (a
   retrieval index, a note/reminder, a `.prism` document). The internal store stays small and fast; the world
   holds the long tail. (We write things down and use technology for exactly this reason.) Untested — a frontier.

## How to apply (methodology change)

- **Impose an explicit memory budget** in every experiment (cap on stored contexts / concepts / table bytes).
  No unbounded `np.unique`-and-keep.
- **Report quality-per-bit**, not just quality — and how gracefully each model degrades as the cap tightens.
- **Eviction policy is a first-class variable** (recency vs STI/LTI vs ART-vigilance), not an afterthought.
- **Re-ask "what survives scale" under a budget** — the dichotomy is expected to flip: the unbounded survivors
  (boundaries, similarity-for-unseen, calibration, constructions) plus the *bounded* survivors (consolidation,
  retention, generalization) should now all matter, because the budget is what gives them a job.

Open question worth a real experiment: **the environment-as-memory loop** — a bounded internal cortex that
writes its rare/important knowledge to an external store and retrieves it on demand, vs. a model that tries to
hold everything internally, under the same total budget.
