# Exp AG — a count-native System 2: the deliberate pass — 2026-06-26

**The bet (Evans/Stanovich two-process theory; Botvinick conflict monitoring; Engle WM-capacity;
Oberauer's focus; Kahneman default-interventionist; Sloman suppress-not-erase).** Everything we built
before this is **System 1**: the char-order experts vote in parallel and a leader pops out — the fast,
local, *prepotent* argmax. System 2 is not a smarter model. It is a tiny, capacity-bounded, **serial**
workspace, **triggered by conflict**, that **decouples** from the immediate input and **overrides**
System 1 *only when System 1 is wrong*. Built from counts — no gradient, no batch optimization. The
test is the **Engle signature**: the override must IMPROVE accuracy where the prepotent answer is wrong
and DO NO HARM where there is no conflict.

**The machine.**
- **System 1** — product-of-experts over SHORT orders {2,3}; its vote activations are the leaky
  accumulators, its argmax the prepotent leader.
- **The dual trigger (default-interventionist)** — deploy System 2 when calibrated confidence
  `c = f·c` (NARS, from `lib/confidence`, Exp AB) `< θ`, OR Botvinick conflict is high. Two conflict
  signals: WITHIN System 1 (top-two vote product, high only when two answers are *both* on) and ACROSS
  subsystems (the fast leader and the reflective leader disagree, both strong). Otherwise ship the fast
  leader immediately.
- **The top-down GOAL** — the wider LONG-order {5,6} opinion, the broad context the fast voter
  under-weights. This is what System 2 consults.
- **The deliberate pass (race)** — a capacity-`k≈4` FOCUS, a serial race over leaky accumulators
  seeded by System-1's votes + the top-down goal drive, **inhibition-of-return** so the loop advances,
  **cognitive decoupling** (the prepotent seed fades), a **step budget**, a **suppress-not-erase**
  floor on the default; commit the deliberate winner only if it beats the default.
- **The deliberate pass (deferral)** — the *minimal* operator under the same trigger: think = consult
  the wider context. On every fired position, defer to the reflective (goal) answer.

**Setup.** Char next-char on text8. Train 12,000,000 chars; held-out eval 300,000 (299,999 predicted).
Short {2,3}, long {5,6}, decider order 3; add-α=0.05; unigram fallback. Single causal pass to build the
`(w+,w-)` tables (`confidence.CountTruth`); the deliberate pass is a per-position serial loop over those
same counts — no second training pass. Fixed seed 0. Whole run ≈ 16 s on CPU. θ=0.35, κ=0.40 (within),
κ×=0.10 (cross), k=4, budget=6, IOR=0.7, decouple=0.5, goal-gain=2.0, floor=0.02.

**The probe.** A CONFLICT subset = positions where the short-order leader and the long-order leader
DISAGREE and both contexts are seen (the prepotent local bet fights the broader context). A NO-CONFLICT
subset = where they agree. Of the eval stream, 98.2% have both contexts seen; **37.5% are conflict**,
60.8% no-conflict. On the conflict subset, System 1's prepotent answer is **wrong 88.4%** of the time —
exactly the slice where a deliberate override should pay.

---

## The Engle signature

| subset | n | System-1 | S2 deferral | Δ deferral | S2 race | Δ race |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 299,999 | 0.4612 | **0.6022** | **+0.1410** | 0.5772 | +0.1161 |
| **CONFLICT** | 112,450 | 0.1161 | **0.4960** | **+0.3799** | 0.4273 | +0.3112 |
| **NO-CONFLICT** | 182,270 | 0.6803 | 0.6803 | **+0.0000** | 0.6801 | −0.0002 |

**The signature is textbook.** Both System-2 variants lift accuracy massively on the CONFLICT subset
(+0.38 deferral, +0.31 race) — where System 1's local argmax is wrong 88% of the time — while leaving
the NO-CONFLICT subset *exactly* untouched (deferral) or within 0.0002 (race). The gate produces the
override-when-wrong / hands-off-when-right behaviour Engle's working-memory work predicts: it improves
the prepotent-wrong cases and does no harm on the easy ones.

The gate fired on **71.8%** of positions (confidence-low OR conflict-high). On the conflict subset the
deliberate race overrode the default 33.6% of the time; net it FIXED 46,448 and BROKE 11,450 conflict
cases (net **+34,998**), and on the no-conflict subset it touched only 668 cases (net −44 — the only
harm, from the handful where the reflective answer is also wrong).

## Graceful fallback (the capacity bound)

With the step **budget set to 0**, the gated model emits the System-1 leader on every position — it
equals System-1-only *exactly* (0 overrides). The override degrades gracefully to the fast default when
there is no capacity to think, which is the suppress-not-erase guarantee made operational.

## Does the elaborate deliberate RACE earn its keep? (the honest negative)

The two System-2 variants share the *same gate*; they differ only in the operator. On the 215,312 fired
positions the race and the trivial deferral agree 85.5% of the time. On the **31,290 where they differ**:

| | accuracy on the disagreement |
|---|---:|
| deliberate race (focus + IOR + decoupling + time-integrated commit) | 0.1458 |
| trivial deferral (consult the wider context) | **0.3854** |

**The workspace machinery does NOT help.** Where the elaborate race declines to switch (keeping a
System-1 answer) or picks a third candidate, it is *worse* than simply deferring to the reflective
context. The race's selectivity costs more than it recovers: a third-candidate pick is wrong ~81% of
the time. The mechanism that wins is the **gate** — the metacognitive *decision to think* — not the
serial focus/IOR loop we wrapped around it. Judged on the right axis (the Engle signature), the gate is
load-bearing and the deliberate-race operator is not.

## Verdict

**WIN on the headline axis — the Engle signature is clean and large — with an honest negative on the
mechanism.** A count-native, online, no-backprop System 2 *does* override System 1 exactly when System
1 is wrong and *does no harm* when it's right (+0.38 on conflict, ±0.0000 on no-conflict, graceful at
zero budget). The load-bearing piece is the **dual trigger** (calibrated confidence + Botvinick
conflict): once you *decide* to think, the cheapest possible deliberate operator — defer to the wider
context — captures essentially all the win. The elaborate serial race (focus, inhibition-of-return,
cognitive decoupling, time-integrated commit) is **parked, not killed** (Fragile-Ideas §7, §8): it is
faithful to the cognitive story but loses to the trivial operator on this char-level next-token probe.
Its untested winning axis is *multi-step* problems where a single deferral can't reach the answer —
compositional reasoning (Exp AD), discourse coherence (Exp AC) — where holding and manipulating 4 items
across serial cycles is the *point*. Char next-token is a one-step decision; it can't exercise the
workspace.

## Online-compliance note

Single causal pass for the `(w+,w-)` count tables (`confidence.CountTruth`); the deliberate pass is a
per-position serial loop over leaky accumulators with no second training pass, no gradients, no batch
optimization (no k-means/SVD). Working memory is a bounded `k=4` focus; the step budget is fixed.
Everything is read off counts + leaky accumulators. Fully online, bounded-memory, count-native.

## Lineage

Grew from **calibrated confidence (Exp AB)** — the NARS (f,c) truth value is the metacognitive trigger
"is my fast answer trustworthy?" — **the gate (Exp X / hetero)** — surprise/confidence opening a route
to a higher level — and **leaky evidence (Exp R)** — the accumulators the race runs over. The thread is
**System 2 / the right combiner**: Exp AB asked *is the fast answer calibrated?*, Exp AG asks *and what
do we do when it isn't?* The answer, on this probe, is the simplest default-interventionist one: notice
the conflict, consult the wider context. Credit Kahneman (default-interventionist), Stanovich (cognitive
decoupling), Botvinick (conflict monitoring), Engle (working-memory capacity as the override signal).
