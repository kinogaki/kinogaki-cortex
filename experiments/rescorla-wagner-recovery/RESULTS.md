# Exp BC — Rescorla-Wagner recovery loop — RESULTS

**Verdict: WIN** (clean, on the axis the idea can win). The kill condition did **not** fire.

Run: `exp_a_boundary/.venv/bin/python exp_bc_rescorla/run.py` (fixed `SEED=0`, single streaming pass).

## What was tested

A stem is a **cue**; its inflected forms are **competing outcomes**. Two learners stream the *same*
two-phase event sequence once:

- **RWCompetition** (M20): predict the form, then on observing apply Rescorla-Wagner over all of the
  cue's outcomes — `V[o] += α·(λ·1[o==heard] − ΣV)`. The heard form rises; every *expected-but-absent*
  form is decremented by `α·ΣV`. No label, no reward.
- **CountOnly** (load-bearing baseline): hear a form, increment its count. The over-applied form's
  absolute count never falls.

**Corpus.** A synthetic, frequency-matched child-directed-ish plural stream built in `run.py`
(15 regular noun stems taking `+s`, plus the irregular `mouse`). Phase 1 (6000 events): only the
over-applied `mouses` is heard for the target. Phase 2 (9000 events): the correct `mice` arrives at
fraction `irr_ratio`, the rest still `mouses` (lingering wild over-application), **with no
correction signal**. The M20 spec asks for CHILDES order for the realistic version; **CHILDES is not
in `data/`, so it is SUBSTITUTED** by this hand-built synthetic stream (and a text8 frequency sanity
import). Sweep: `irr_ratio ∈ {0.3, 0.5, 0.7, 0.9}` × `alpha ∈ {0.05, 0.15, 0.30}` = **12 variations**
(FRAGILE budget satisfied). Recovery = a **sustained** (3-consecutive-probe) crossover where
`p(mice) > p(mouses)`.

## The table (sweep)

| irr_ratio | alpha | RW xover | CO xover | RW slope | CO slope | RW Δp(mouses) | CO Δp | winner |
|----------:|------:|---------:|---------:|---------:|---------:|-------:|------:|:------|
| 0.3 | 0.05 | never | never | 0.000 | 0.004 | 0.000 | 0.183 | tie |
| 0.3 | 0.15 | never | never | 0.000 | 0.004 | 0.000 | 0.183 | tie |
| 0.3 | 0.30 | never | never | 0.000 | 0.004 | 0.000 | 0.183 | tie |
| 0.5 | 0.05 | **2600** | never | 0.016 | 0.007 | 0.709 | 0.305 | **RW** |
| 0.5 | 0.15 | **2600** | never | 0.014 | 0.007 | 0.619 | 0.305 | **RW** |
| 0.5 | 0.30 | **2600** | never | 0.013 | 0.007 | 0.585 | 0.305 | **RW** |
| 0.7 | 0.05 | **400** | never | 0.019 | 0.009 | 0.875 | 0.409 | **RW** |
| 0.7 | 0.15 | **200** | never | 0.018 | 0.009 | 0.823 | 0.409 | **RW** |
| 0.7 | 0.30 | **200** | never | 0.019 | 0.009 | 0.869 | 0.409 | **RW** |
| 0.9 | 0.05 | **400** | 7600 | 0.022 | 0.012 | 1.000 | 0.542 | **RW** |
| 0.9 | 0.15 | **200** | 7600 | 0.022 | 0.012 | 1.000 | 0.542 | **RW** |
| 0.9 | 0.30 | **200** | 7600 | 0.022 | 0.012 | 1.000 | 0.542 | **RW** |

**Tally over 12 variations: RW recovers-first 9, CountOnly-first 0, tie 3.**
Crossed at all — RW 9/12, CountOnly 3/12. Mean `(CO_xover − RW_xover)` where both cross = **7333
events** in R-W's favor. The 3 ties are exactly the `irr_ratio=0.3` cases (the corrective form stays
the *minority* of target tokens) — there genuinely isn't enough signal, and **neither** model should
recover; correctly, neither does. R-W never recovers *spuriously* and CountOnly never recovers *fast*.

## The decisive contrast (weak signal, `irr_ratio=0.5`)

With `mice` and `mouses` near-balanced, increment-only `p(mouses)` is still **0.70** by the end of
9000 corrective events (it crawls from 1.00 toward 0.50 by frequency alone and never crosses). R-W
drives `p(mouses)` down to a clear minority — sustained-crossing the irregular by event ~2600 — even
though `mouses` is still being heard ~half the time. That gap *is* "recovery without correction": the
cue's budget is competitively reallocated to `mice`, so `mouses` bleeds out from being expected and
not seen, not from being out-frequencied.

## Rules compliance

- **Online single pass:** predict-then-update *is* the definition; one streaming pass, no epoch.
- **No gradient / k-means / SVD / backprop:** R-W is the canonical *non-gradient* associative rule.
- **Bounded memory:** the decrement frees budget; outcomes below `floor` are pruned.

## Honest caveats (sub-negatives, reported not hidden)

1. **R-W oscillates trial-to-trial.** With a single combined cue, each event yanks the whole
   distribution by `α·(λ−ΣV)`, so the raw per-event `p(mouses)` jitters hard (visible in the
   decisive-contrast table). The *trend* is unambiguously downward and the **sustained** (3-probe)
   crossover marker is robust to it — but the instantaneous readout is noisy and a real learner would
   want a leaky-averaged read-out. This is faithful R-W behavior, not a bug.
2. **Ramscar check is flat (a clean local negative).** Exposing the model to more regular `+s`
   plurals before the corrective phase did **not** slow the irregular's recovery (xover stayed at
   400 across `reg_warmup ∈ {0, 4000, 12000}`). Reason: the mechanism uses **per-stem elemental
   cues**, so regular exposure matures *other* stems' cues and never touches the `mouse` cue. Ramscar's
   count-maturity sign needs a **shared `+s` outcome cue** that the regulars and the irregular both
   compete over — a configural/shared-cue extension this elemental version deliberately did not build.
   Honest: M20's recovery claim holds; its Ramscar-sign sub-claim is **not** reproduced here.
3. **Memory footprint tied (17 vs 17).** With only two competing forms per target cue, the bled-out
   `mouses` rarely sits below `floor` *at probe time*, so the prune didn't bite — the bounded-memory
   advantage is structural (decrement *can* free budget) but did not show numerically at this scale.

## Kill condition

**M20 / BUILD_QUEUE BC kill:** "increment-only passive reading recovers just as fast (recovery is
just frequency) → keep the simpler increment-only loop." **It did NOT fire.** Increment-only never
recovers-first in any of the 12 variations, crosses at all in only 3 (and only very late, at the
strongest signal), and trails R-W by ~7300 events where both cross. Recovery is **not** just
frequency: R-W recovers from prediction-error exposure alone, with no label and no reward.
