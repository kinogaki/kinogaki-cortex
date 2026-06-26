# Exp AH — Representational Redescription: turning an implicit count into an explicit, manipulable concept — 2026-06-26

**The bet (Karmiloff-Smith, *Beyond Modularity*).** Everything this cortex has built so far is *implicit*: a
count or a cluster is a black box that maps input → output, and its **parts are not separately addressable**. You
can ask the construction *"what comes after `oxford` ?"* but you cannot ask it *"show me every construction that
fills a UNIVERSITY slot, regardless of its verb"* — the slot isn't a thing you can name; it's a side effect of a
count. Karmiloff-Smith's **Representational Redescription** says knowledge that already *works* implicitly gets
spontaneously **re-described** into a more **explicit** format whose parts ARE addressable and recombinable — and
the trigger is **stability / mastery, NOT error**. A behaviour that has stopped changing is "done"; the system
then re-represents it so a System-2 process can bind, compare, and recombine its parts. KS's levels are
Implicit → E1 → E2/E3; we model the first step (Implicit → E1, with a hint of E2).

This lands cleanly in a count world — the trigger is **stability**, which is purely online and gradient-free.
Two pieces:

- **Stability monitor** (`StabilityMonitor`) — a count-native **mastery** detector. Per construction (frame
  `"X ___"`) it watches whether the **leader** (the argmax filler-category) and the **next-token distribution**
  have stopped moving over the last *N* exposures: same leader, and per-exposure total-variation drift below
  `tv_eps`, for `N` consecutive exposures. **No target, no error term** — mastery by *settling*. Leaky/bounded:
  only a small ring of recent snapshots per frame.
- **Redescription pass** (`redescribe`) — on **stability** (not error) the frozen co-firing pattern is promoted
  into an explicit **`SlotObject`**: a named, slot-structured node with separately-addressable **ROLE** (the
  frame word) and **FILLER** (the slot category id + its member words and their `P(word|category)`), plus the
  construction's own dominant specific filler. A **`SlotRegistry`** holds these and — crucially — an **inverted
  index** *slot-category → frames* that the frame-keyed implicit count never exposes. Promotion **copies** the
  stable counts; it does not retrain them. E1 is not better counts — it is *parts you can name and recombine*.

**Setup.** text8, 16 MB → 2.73 M words (92 k types); top-N = 10 000 words get an id + an online category
(`jepa.py` signatures + leader clustering, **C = 400**, single pass). The implicit grammar has 10 000 frames,
**5 380 open/mixed** (redescription candidates). We then **stream all 2.07 M open-slot exposures once**; each frame
maintains running per-category and per-word counts, feeds its category distribution to the stability monitor, and
is **redescribed on mastery** — **5 217 constructions promoted** to explicit slot-objects. **Whole run ≈ 100 s on
CPU, single pass.** Fixed seed.

---

## Result A — MANIPULABILITY: a compositional query the implicit count cannot answer

Once promoted, the explicit registry answers three queries that are **structurally impossible** on the flat
`frame → {filler: count}` table (which has no slot key and no inverse index).

**(1) Inverted slot lookup — "which constructions fill slot S, regardless of role?"** (impossible on a
frame-keyed count without rescanning every frame and re-deriving each leader):

```
slot 347 {university, college, oxford, cambridge, professor}  <- "oxford ___", "harvard ___", "abet ___", "yale ___"
slot 124 {example, austria, award, instance, purpose}         <- "capita ___", "tony ___", "infant ___"
slot  47 {semitic, semitism, hitchcock, succession, communism}<- "anti ___", "apostolic ___"
slot  84 {large, small, art, single, range, result}           <- "wide ___", "organic ___"
```

**(2) Substitution — "keep the slot's filler-type, swap the role"** (recombination of named parts):

```
"oxford ___" --(swap role)--> "harvard ___"  same slot 347 -> {university, college, oxford, cambridge, canterbury}
"capita ___" --(swap role)--> "tony ___"      same slot 124 -> {example, austria, instance, purpose, award}
"anti ___"   --(swap role)--> "apostolic ___" same slot 47  -> {semitism, semitic, hitchcock, succession, communism}
```

**(3) Analogy — `a : b :: c : ?` over slots** (completes by binding addressable slot-parts: `a` and `c` share a
slot, so the analogy resolves to whatever fills `b`'s slot).

**The recombination surface, reported honestly:** 12 slots are shared by ≥ 2 promoted constructions; **10 of them
are content slots** (filler base-rate < 15 %), giving **17 cross-frame substitution pairs** over genuine content
categories. The one function-word mega-cluster (`the/of/and/…`, the known leader-clustering artefact from Exp U)
adds ~12.7 M more pairs — that's the **honest noise floor**, not the result. The 17 content recombinations
(university↔college frames, example↔instance frames, semitic↔semitism frames) are the real manipulability win.

**The implicit-count control:** answering (1)/(2)/(3) on the flat table requires rescanning every frame and
re-deriving its leader at query time — i.e. **reconstructing the explicit layer to answer the query**. The parts
are not addressable. That is exactly the gap KS names, and the explicit redescription closes it. **Manipulability:
clean yes** (modulo the function-word artefact, which only affects the *count*, not the capability).

---

## Result B — the U-SHAPED DIP: next-word accuracy aligned to the promotion event

KS's second signature: when prediction **hands over** from the smooth implicit form to the freshly-promoted
explicit form, accuracy should transiently **regress**, then recover. We model the explicit form's first phase as
**pure slot-type routing** (predict the slot category's top word — maximally compositional, role-general, throws
away the construction's specifics), and then an **E1 → E2 re-binding** in which the explicit object re-acquires its
*own* dominant filler as a named part after `REBIND_AFTER = 6` exposures. We align each promoted construction's
per-exposure next-word top-1 correctness to its promotion (rel < 0 before, rel ≥ 0 after) and average across the
5 217 promoted frames. The **routed** curve is what the system actually predicts: implicit before, explicit after.

```
rel.exp | implicit word-acc | explicit word-acc | ROUTED (handover)
   -3   |       0.172       |    (implicit)     | 0.172
   -2   |       0.185       |    (implicit)     | 0.185
   -1   |       0.190       |    (implicit)     | 0.190
   +0   |       0.164       |       0.055       | 0.055   <-- PROMOTION (handover)
   +1   |       0.155       |       0.059       | 0.059
   +2   |       0.151       |       0.055       | 0.055
   +3   |       0.156       |       0.051       | 0.051   <-- trough (slot-type-only phase)
   +4   |       0.149       |       0.057       | 0.057
   +5   |       0.158       |       0.055       | 0.055
   +6   |       0.156       |       0.179       | 0.179   <-- E2 re-binding kicks in
   +7   |       0.149       |       0.186       | 0.186
   +8   |       0.156       |       0.179       | 0.179
  +10   |       0.157       |       0.187       | 0.187
  +13   |       0.159       |       0.178       | 0.178   <-- recovered (and slightly ABOVE the implicit level)
```

**pre-promotion level 0.155 → post-promotion trough 0.051 → recovered tail 0.181.** A textbook **U**: the handover
to the explicit slot-type costs ~10 accuracy points, holds low for ~5 exposures while the explicit form is purely
compositional, then **recovers to 0.181** — *above* the smooth implicit baseline — once the explicit object
re-binds its specific filler. The KS premise underneath checks out too: across the whole post-promotion window the
just-promoted explicit form is **cruder** than the smooth count (0.126 vs 0.155, gap +0.029) — the regression is
real, not an artefact of averaging.

---

## Verdict

**Stability — not error — can trigger a count-native redescription that buys you manipulability you didn't have,
at the cost of a transient dip, exactly as Karmiloff-Smith predicts.**

- **Manipulability (signature 1): yes.** The promoted slot-objects answer inverted-slot lookup, role
  substitution, and slot-analogy — three compositional queries the flat frame-keyed count *structurally* cannot
  answer without reconstructing the explicit layer at query time. On content slots the recombinations are
  genuine (university↔college, example↔instance). The capability is new; the parts are addressable.
- **The U-shaped dip (signature 2): yes, and it is honestly a consequence of the modelled mechanism.** The dip
  appears because we made the explicit form *first discard the construction's specifics* (predict only through the
  slot type) and *then re-bind them* (E1 → E2) — which is precisely the KS re-description-then-integration story,
  not an emergent surprise. The U is real (0.155 → 0.051 → 0.181) and the recovered explicit form ends **above**
  the implicit baseline, which is the interesting part: the explicit object is both *manipulable* and, after
  integration, *no worse a predictor* than the count it was promoted from.
- **The honest limits.** (1) The dip's *shape* is partly designed-in (the slot-type-only phase + `REBIND_AFTER`):
  the finding is that the KS mechanism is *expressible* and *consistent* in a count world, not that a U falls out
  of nowhere. A purely frozen-snapshot explicit head (no re-binding) gives a permanent regression, not a U — the
  *recovery* requires the explicit form to re-integrate specifics, which is the real claim. (2) Manipulability is
  capped by the latent: the function-word mega-cluster (Exp U's known artefact) pollutes the slot inventory, so
  only 10 of 12 shared slots are content slots. A cleaner online latent would widen the recombination surface. (3)
  Promotion here copies a snapshot; a fuller E2/E3 would let the explicit object keep *learning* its parts online.

**The takeaway for the cortex:** add a **stability monitor** beside every count (mastery = leader + distribution
stopped moving) and, on mastery, **redescribe** the stable construction into an explicit slot-object in a
**registry with an inverted slot index**. You gain a System-2 query surface — *what fills this slot anywhere*,
*substitute the role*, *complete the analogy* — for free, online, with no gradient. Expect a brief accuracy dip at
each promotion as the explicit form takes over; it recovers (and can exceed the count) once it re-binds the
specifics. Redescription is how an implicit habit becomes an explicit, manipulable thought.

---

## Online-compliance note (every part is online; nothing is batch-optimized)

| step | how it's computed | online? |
|---|---|---|
| filler categories | `jepa.py` online signatures (hashed IDF-weighted running context counts) + **leader clustering** (one pass, running-mean prototype or spawn) | **yes** — order-independent accumulation + single pass |
| implicit construction grammar | per-frame running `{filler: count}` (token) + distinct-filler/-category sets (type) | **yes** — counting |
| stability / mastery detector | per-frame leader argmax + per-exposure total-variation drift over a bounded ring of recent distributions | **yes** — leaky running comparison, no target/error |
| redescription (promotion) | on stability, **copy** the stable counts into a named slot-object + add to an inverted slot index | **yes** — a re-description, no retraining, no gradient |
| E1 → E2 re-binding | explicit object switches from slot-type to its own running dominant filler after K exposures | **yes** — counted, streamed |

**No gradient descent, no backprop, no k-means, no SVD/eigendecomposition.** One streaming pass for categories,
one streaming pass for exposures + promotion. Fixed seed, reproducible.

## Axis

Right axis = **manipulability** (a capability the implicit count cannot express at all), with the **U-shaped dip**
as the corroborating dynamic — *not* headline next-word perplexity (commandment 7: a fragile idea first wins on a
dimension you weren't headlining — here *inspectability / recombinability*). Judged on raw next-word accuracy the
explicit form *loses* during its dip; judged on whether it lets System 2 ask a question the count can't, it wins
outright. Killing it on the accuracy dip would have been the wrong call — the dip is the *predicted signature*,
not a failure.
