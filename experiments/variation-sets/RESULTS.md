# Exp BB — Variation-set minimal-pair miner (M16) — RESULTS

**Mechanism.** A bounded ring buffer of recent utterances. Each new utterance is token-aligned against the
previous one by **longest-common-subsequence** (the anchored diff, no gradient). When their overlap ≥ a
threshold, the agreeing tokens are a **frame** and each disagreeing run is a **(slot, filler) substitution** — a
minimal pair the input itself isolates. Two products: (1) extra (frame, filler) counts folded into AF's
construction tables (`constructions.py`); (2) the agree→disagree transition recorded as a phrase **boundary**
(a new source for `boundaries.py`). Reusable in `lib/varsets.py`.

**Corpus (SUBSTITUTION — stated honestly).** The spec asks for CHILDES (Brown/Manchester) natural variation
sets. CHILDES is **not in `data/`**. We substitute a **synthetic child-directed-ish variation-set generator** (7
frames over 3 filler categories, 53-word vocabulary; each variation set repeats one frame with different fillers
from its slot category — the clean kill-test the literature uses) plus a **text8 negative control** (natural text
has almost no adjacent near-repeats → the miner should stay quiet there). A `noise` dial scrambles a fraction of
slots into distractor utterances, burying the variation-set signal so the miner has something to recover.

**Axes (the only ones judged, per Haga "helps syntax, not world knowledge").**
1. **Compositional generalization** — hold out (anchor, filler) pairs the model never reads adjacent; score them
   through AF's open-slot **category** head, AF-alone vs AF + the miner's extra counts.
2. **Phrase-boundary F1** — diff cuts vs branching-entropy boundaries vs their union, against the true open-slot
   edges (±1 token tolerance).

Single streaming pass, fixed seed (`SEED=0`). `python exp_bb_varsets/run.py`, wall ≈ 7s.

---

## Result 1 — PHRASE BOUNDARIES: a clean, decisive win (the headline)

Gold boundaries = the open-slot edges (where the repeated frame stops and the swapped filler begins — exactly
what the diff is built to find). Cut rate matched across detectors.

| detector | precision | recall | **F1** |
|---|---:|---:|---:|
| branching entropy (`boundaries.py`, baseline) | 1.000 | 0.580 | 0.734 |
| **diff (the miner)** | 1.000 | 0.780 | **0.877** |
| **combined (union)** | 1.000 | 0.907 | **0.951** |

The variation-set diff finds the open-slot boundary **+0.143 F1** over branching entropy alone, and the **union
reaches 0.951** — the two signals are complementary (the diff catches slot edges the entropy detector misses
when the pre-slot word is itself predictable). This holds across **every** config in the FRAGILE grid
(F1_diff = 0.877, F1_comb ≈ 0.950 throughout). This is the new boundary source M16 promised for M.

## Result 2 — COMPOSITIONAL GENERALIZATION: a wash (Haga's "not the syntax slice it wins on")

AF's open-slot category head already generalizes the held-out pairs **without** the miner (held-out perplexity
~13–15 vs the n-gram floor of 180–16000 — AF wins the unseen-combination slice by 1–3 orders of magnitude, the
Exp AF result reproduced). Because filler→category membership is supplied by AF's categories, the head **always
fires** (coverage 1.00) and the miner's extra counts have nothing to fix; they mostly **over-sharpen** the
category distribution toward the variation-set fillers, costing a few % of perplexity.

| noise | ppl AF-alone | ppl AF+miner | Δ% | coverage AF→AF+miner |
|---:|---:|---:|---:|---:|
| 0.0 (clean) | 14.75 | 14.78 | −0.2% | 1.00 → 1.00 |
| 0.3 | 13.26 | 13.51 | −2.0% | 1.00 → 1.00 |
| 0.5 | 13.18 | 13.41 | −1.8% | 1.00 → 1.00 |
| **0.7 (heavy)** | 13.47 | **13.24** | **+1.8%** | 1.00 → 1.00 |

The miner only crosses into a (small, +1.5–1.8%) **win** at very heavy noise, where its selective up-weighting
of the *true* frame→filler pairs partially cancels the count-dilution from distractor utterances. Everywhere
else it is neutral-to-slightly-negative. Honest read: **the count-reweighting product of M16 buys little on
compositional generalization once AF's category head is in place** — the variation set does not teach the open
slot anything the open slot did not already abstract. (Coverage never moves because the hand-given categories
keep the head firing; a noisier *learned* category latent would be the place to retest whether the miner rescues
abstention.)

## Result 3 — text8 negative control: the miner is specific, not trigger-happy

40,000 text8 pseudo-utterances (8-word chunks of natural Wikipedia text) → only **130** registered as variation
sets (overlap ≥ 0.60), **236** substitutions. Natural text has almost no adjacent near-repeats, so the miner
stays quiet and does not corrupt counts — confirming it fires on genuine variation-set structure, not noise.

---

## FRAGILE budget

24 variations: `noise ∈ {0.0, 0.3, 0.5, 0.7} × overlap ∈ {0.50, 0.60, 0.75} × bonus ∈ {2.0, 4.0}`. The boundary
win is flat across all 24. The comp-gen result is consistently a wash except the heavy-noise corner. We did
**not** kill on the first weak comp-gen result — we ran the budget, found the one regime where the miner helps
comp-gen, and reported it honestly as marginal.

## KILL-CONDITION

*Kill if it does not improve compositional-generalization OR boundary F1 on **any** slice — Haga's syntax-only
help is a PASS; kill only if it loses on syntax too.*

**KILL DID NOT FIRE.** The miner wins **decisively on the segmentation axis** (boundary F1 0.877 vs 0.734;
combined 0.951) across every config, and clears comp-gen on the heavy-noise slice (+1.8%). Segmentation *is* a
syntax slice (the open-slot edge), so this is a PASS by the kill rule.

## Verdict — **PARTIAL win**

- **Boundaries: a real, robust win.** Adjacent-utterance diffing is a strong, complementary phrase-boundary
  source — F1 0.877 alone, 0.951 unioned with branching entropy. This is the deliverable to keep: feed the diff
  cuts into M as a second boundary signal. (Honestly synthetic; the *shape* — variation sets expose slot edges
  cleanly — should survive to CHILDES, but the magnitude is a toy.)
- **Compositional generalization: a wash.** Once AF's open-slot category head exists, the miner's extra
  (frame, filler) counts add essentially nothing (and slightly over-sharpen) — it only helps in a heavy-noise
  corner. This matches Haga's prediction (variation sets help syntax-as-segmentation, not as much elsewhere) and
  the Exp AF finding that the category head already carries compositional generalization.
- **Take to CHILDES / a learned-category retest:** the open question the toy can't answer is whether the miner
  **rescues abstention** when categories are *noisy/learned* (here coverage was pinned at 1.00 by hand-given
  categories). That is the slice where its count-reweighting could earn its keep on comp-gen.

---

## Online / bounded / no-backprop compliance

| step | how | compliant? |
|---|---|---|
| utterance alignment | token-level LCS dynamic program (counts matches) | online, no gradient |
| variation-set detection | single overlap ratio vs a fixed threshold | online |
| frame/filler harvest | additive counts into AF tables (identical to incrementing each pair `bonus` extra times) | online counting |
| boundary harvest | record diff cut positions; F1 read off | online |
| memory | a fixed **ring buffer** of N utterances + the bounded AF tables + a (frame,filler)→bonus dict keyed on existing frames | **bounded** |

**No gradient descent, no backprop, no k-means / SVD / eigen / word2vec.** One streaming diff per utterance
against a bounded ring. Fixed seed, reproducible.
