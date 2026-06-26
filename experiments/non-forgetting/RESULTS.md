# Experiment AE — online non-forgetting under REAL domain shift — 2026-06-25

**The right axis: BACKWARD RETENTION.** Exp B tried to demonstrate the transformer-differentiator — *no
catastrophic forgetting* — on four English registers and found there was nothing to forget (English-at-the-char
is one distribution; more of it only helps). The honest fix it called for: test under a GENUINE register shift.
Here we stream three truly different registers — **Darwin** (Victorian science) → **Shakespeare** (Early-Modern
verse) → **KJV Bible** (archaic scripture) — in ONE pass, NO replay, and after each phase measure bits-per-char
on a held-out slice of EVERY register. Forgetting = how much an earlier register's bpc *degrades* after later
ones.

**Why a memory cap.** A plain count model is additive — it never overwrites, so it can't forget. The interesting
question only exists under **bounded memory**: cap the per-order context table so a new register's flood of
n-grams FORCES eviction. Then the whole experiment turns on the eviction/retention policy. Both models below use
the identical add-α highest-order-seen backoff and count all orders the same way (so peak quality is comparable);
they differ ONLY in **what they keep when the table overflows**.

- **FLAT** — single-timescale recency (the baseline that forgets). One leaky use-score per context; evict the
  least-recently-used. A new register's contexts are all "fresh and used", so they wash out the earlier register.
- **DUAL** — the brain-inspired stack: **ECAN STI/LTI** (fast salience + slow importance) · **CLS** fast/slow
  stores · **ART vigilance** (resonate-or-spawn) · **LIDA broadcast** (one winner per step). Evict the **lowest
  LTI** (protect rare-but-important); each step only the **most-specific context that RECOGNIZED the input**
  (ART resonance, match ≥ ρ) is broadcast and gets a strong LTI write — novelty gets only a weak write and must
  earn retention over time.

Char-level, orders 1–5, 120k chars/register, cap 3,000/order, ρ=0.15, fixed seed 0, single pass.

## Result — the retention matrix (bpc, lower=better)

`M[after row][eval col]` — bpc on each register *after* training through the row's register.

**FLAT (recency eviction)**

| after \ eval | darwin | shakespeare | bible |
|---|---:|---:|---:|
| darwin       | **2.522** | 3.435 | 3.254 |
| shakespeare  | 2.790 | **2.856** | 3.048 |
| bible        | 2.863 | 2.968 | **2.188** |

**DUAL (STI/LTI + broadcast + ART)**

| after \ eval | darwin | shakespeare | bible |
|---|---:|---:|---:|
| darwin       | **2.174** | 3.306 | 3.145 |
| shakespeare  | 2.123 | **2.526** | 2.695 |
| bible        | 2.172 | 2.550 | **1.965** |

**Backward forgetting (Δ bpc = final − right-after-training-it; + = forgot):**

| register | FLAT peak | FLAT final | FLAT Δ | DUAL peak | DUAL final | DUAL Δ |
|---|---:|---:|---:|---:|---:|---:|
| darwin      | 2.522 | 2.863 | **+0.341** | 2.174 | 2.172 | **−0.002** |
| shakespeare | 2.856 | 2.968 | **+0.113** | 2.526 | 2.550 | **+0.024** |
| **total**   |       |       | **+0.454** |       |       | **+0.021** |

- **DUAL forgets ~21× less** (total backward Δ +0.021 vs +0.454). Darwin — the first register, most exposed to
  later floods — is essentially **flat** under DUAL (−0.002) while FLAT loses a third of a bit on it.
- And it does so at **better, not worse, peak**: mean diagonal bpc DUAL 2.222 vs FLAT 2.522. Retention is *not*
  bought with quality — protecting the contexts that actually predicted well also sharpens the current register.
- Mechanism check: after the full stream, DUAL keeps **1,274 vs 1,059** of Darwin's order-4 contexts (+20%) at
  the same table size — the LTI/ART policy evicts *the right things*.

## Findings

1. **POSITIVE: the two-timescale + broadcast + vigilance stack measurably defeats forgetting under bounded
   memory.** This is the demo Exp B couldn't run in English-only — under a *real* register shift with a memory
   cap, a flat recency cache forgets and the brain-inspired policy does not. The differentiator is real once the
   test has something to forget.
2. **The load-bearing piece was ART resonance, not the leaks alone.** A first DUAL that wrote LTI to the
   *highest-STI* context (low orders win) actually retained *worse* than FLAT (peak 3.05, Δ +0.92) — broadcasting
   to the generic short contexts starves the rare specific ones that carry a register's identity. Switching the
   broadcast winner to *the most-specific context that recognized the input* (ART best-match, match ≥ ρ)
   concentrated LTI on exactly the predictive high-order n-grams and flipped the result. Commandment 4/7: the
   idea looked dead at step 1; the win came two steps in, on a dial (which context is reinforced) we weren't
   headlining.
3. **NUANCE / honest scope:** the effect is a *bounded-memory* phenomenon. With an uncapped table both models
   are non-forgetting (additive counts) and identical — so this result is a statement about graceful degradation
   under a memory budget, not about counts-vs-gradients in the abstract. It is, however, exactly the regime any
   real lifelong learner lives in.

## Online note

Strictly single streaming pass, no replay, no gradient descent, no batch optimization. STI/LTI are per-entry
leaky recurrences keyed on a global step clock (lazy decay on touch); ART vigilance is an online nearest-match
(resonate-or-spawn) decision; LIDA broadcast is a per-step winner; eviction is reservoir-sampled lowest-LTI
(O(1) amortized, no global sort). Nothing iterates to convergence; nothing backprops. Fixed seed 0.

## Axis

The headline axis is **backward retention under domain shift at bounded memory** — not raw bpc. On that axis the
brain-inspired policy wins decisively (21× less forgetting) while also tying/beating on peak. Honest caveat: with
unbounded memory there is nothing to measure.
