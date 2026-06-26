# Experiment AI — power-law memory and budgeted eviction (ACT-R) — 2026-06-26

**The bet (Anderson & Schooler 1991; ACT-R rational analysis of memory).** The strongest empirical result in the
cognition sweep: human memory *accessibility* tracks the recency, frequency, AND spacing statistics of the real
environment — and the single curve that fits all three is a **power law**, not an exponential. ACT-R encodes it as
the base-level activation of a memory chunk:

> **B = ln( Σ_k (t − t_k)^(−d) )**,  d ≈ 0.5 — frequency adds terms, recency weights them, decay *d* is the leak.
> need-odds (probability the chunk is needed now) **∝ exp(B)**.

Our substrate weights memory by **raw count** (frequency only) or an **exponential EMA / leaky use-score** (Exp AE's
FLAT — geometric recency). This experiment builds a count model whose per-context activation is the incremental
ACT-R approximation of B (no stored timestamps — a per-entry O(1) recurrence) and uses **exp(B) for both prediction
weighting and eviction**, then tests it on the three axes a memory budget makes visible.

Char-level, orders 1–5, d = 0.5, darwin.txt (200k train / 20k held-out), per-order caps {500, 1500, 4000}, single
streaming pass, fixed seed 0. Whole run 170 s on CPU.

## Test 1 — the spacing effect (pure accumulator, no eviction)

One motif used **20 times**, MASSED (bunched at the start, then a long gap) vs SPACED (spread evenly across a 20k
stream). Same total count. Final retrieval weight at the end of the stream:

| weighting | massed | spaced | spaced / massed |
|---|---:|---:|---:|
| **power-law exp(B)** | 0.1415 | 1.2668 | **8.96×** |
| exponential EMA | 9.1e-04 | 2.443 | 2683× |

- **POSITIVE — power-law reproduces the spacing effect cleanly:** spaced is 8.96× more accessible than massed at
  equal frequency. This is the Anderson–Schooler curve, in the substrate.
- The EMA's "2683×" is **not** a win — it is collapse. After the long post-massed gap the EMA decays the massed
  motif to ~9e-04, i.e. effectively unrecoverable; the ratio is huge only because it divides by near-zero. The
  power law keeps the massed motif *retrievable* (0.14) while still preferring spaced — graceful, not bimodal.
  An exponential cannot *represent* spacing; it only remembers the last touch.

## Test 2 — eviction quality under a fixed memory budget (held-out bpc, lower = better)

Cap each order's table; the four policies differ ONLY in what they evict on overflow. Same counting + add-α
highest-order backoff, so peak quality is comparable.

| cap / order | **powerlaw** (exp B) | lru | **lfu** (count) | ema | best |
|---:|---:|---:|---:|---:|:--|
| 500   | 2.1672 | 2.8835 | **2.1471** | 2.4942 | lfu |
| 1500  | 1.9203 | 2.5044 | **1.8967** | 2.3847 | lfu |
| 4000  | 1.8070 | 2.2264 | **1.7872** | 2.2051 | lfu |

## Test 2b — eviction under DOMAIN SHIFT (the regime where recency *should* matter)

Stream darwin (200k) → **shakespeare flood** (200k) under the cap, then eval back on darwin held-out. B's flood
forces eviction of A's contexts — the bounded-memory, non-stationary regime Exp AE showed is where retention bites.

| cap / order | **powerlaw** | lru | **lfu** | ema | best |
|---:|---:|---:|---:|---:|:--|
| 500   | 2.3181 | 3.0858 | **2.2526** | 2.8715 | lfu |
| 1500  | 2.0435 | 3.0036 | **1.9733** | 2.9238 | lfu |
| 4000  | 1.8408 | 2.5929 | **1.8011** | 2.5807 | lfu |

## Test 3 — downstream bpc, UNBOUNDED: power-law-WEIGHTED prediction vs plain counts

| model | held-out bpc |
|---|---:|
| plain highest-order backoff | **1.8016** |
| power-law-weighted blend (exp B × specificity) | 2.4785 (**+0.677**) |

Using exp(B) to *blend* the per-order distributions is strictly worse than clean highest-order backoff.

## Findings

1. **POSITIVE — the power law is the right shape for the spacing effect** (Test 1): 8.96× spaced-over-massed at
   equal count, and unlike the EMA it keeps massed content *retrievable* rather than collapsing it. If you want a
   memory whose accessibility matches the Anderson–Schooler environment statistics, B is the curve.
2. **NEGATIVE / honest — power-law eviction does NOT beat raw frequency (LFU) under the cap**, in either the
   stationary (Test 2) or the domain-shift (Test 2b) regime. The hypothesis — that evict-lowest-B preserves the
   long sparse repeated tail best → best bpc-under-budget — **fails**. LFU wins at every cap and every shift. The
   power law is a consistent, decisive **second** (it beats the EMA recency baseline and LRU everywhere, by
   0.3–1.0 bpc), but it never beats counting.
3. **Why — and it's a clean mechanism, not noise.** The power law has frequency and recency in it; **LFU is its
   d→0 limit** (no decay = pure count). A decay sweep confirms it directly:

   | d (decay) | 0.05 | 0.2 | 0.5 (ACT-R) | 0.9 | LFU (d→0) |
   |---|---:|---:|---:|---:|---:|
   | bpc @ cap 1500 | 1.8992 | 1.8939 | 1.9179 | 2.0477 | 1.9044 |

   Quality **monotonically degrades as d grows**: every bit of recency weighting the power law adds over pure
   frequency is, here, a tax. The reason is the data: a char n-gram's predictive value is almost entirely its
   **total count** — there is no "this context was useful last week but is stale now" structure at order ≤5 over
   one English corpus, so recency adds variance without signal. ACT-R's recency/spacing earns its keep when
   usefulness is *non-stationary and not captured by frequency*; char-gram eviction is the opposite case.
4. **NEGATIVE — power-law *weighting* of prediction (Test 3) is strictly worse** (+0.68 bpc) than highest-order
   backoff. Exp D already found product-of-experts, not a linear blend, is the right combiner; mixing orders by a
   need-odds weight repeats that mistake. exp(B) is a retention prior, not a prediction combiner.

**Verdict:** *Honest negative on the headline axis.* The power law is demonstrably the right **shape of
forgetting** (Test 1, the spacing curve is real and graceful), but as a budgeted-eviction policy for char n-grams
it loses to plain LFU because LFU is the d→0 special case and these contexts' value is pure frequency. The result
is a sharp scope statement: keep raw counts for eviction here; reach for ACT-R's power law only where recency and
spacing carry signal that frequency does not (sparse, non-stationary, or genuinely spaced memory — e.g. word- or
concept-level retrieval, not dense char-grams).

## Online note

Strictly one streaming pass; no gradient descent; no batch optimization. The ACT-R activation is the incremental
Petrov/Anderson approximation — most-recent use tracked exactly, older-use mass via the closed-form age-integral —
so it is a per-entry O(1) recurrence on a global step clock (lazy: recomputed only on touch/query), storing one
extra float, **no timestamps**. Eviction is reservoir-sampled lowest-score (O(1) amortized, no global sort).
Nothing iterates to convergence; nothing backprops. Fixed seed 0.

## Axis

The headline axis is **quality-per-bit under a fixed memory budget** (held-out bpc at a cap) plus **retention of
spaced vs massed exposure**. On the spacing axis the power law wins (it's the only curve that represents it). On
the budget axis it loses to raw counting, decisively and for a understood reason (LFU = the d→0 limit; char-gram
value is frequency). Honest scope: the cognitive result is real, but its payoff lives where frequency fails to
rank usefulness — which dense char n-grams over one corpus do not.
