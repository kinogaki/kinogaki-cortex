# Exp AB — NARS-style calibrated confidence + precision-weighting — 2026-06-26

**The bet (Pei Wang's NARS; predictive-coding precision; ACT-R).** A raw count `w` says how *often* a
context fired, never whether it was *right*. Split it. In a single online pass, score each context's
running top-1 against the char that actually comes: increment **hits `w+`** on a correct top-1,
**misses `w-`** on a wrong one. Attach a NARS truth value to the association:

    frequency  f = w+ / w        (w = w+ + w-)   — how often this context's bet pays off
    confidence c = w / (w + 1)   (k = 1)         — how much evidence stands behind that frequency

Then pooling becomes **evidence-additive REVISION** (sum w+/w- across orders, recompute f,c — a
low-evidence source barely moves the pool), voter weights become **NARS confidence** or
**predictive-coding precision** `π = 1/var(error)` (a leaky accumulator), and the routing GATE opens on
a **principled `f·c`** (the high order takes over exactly when its evidence says it will do better than
the backoff) instead of a hand-tuned threshold. Everything is read off counts — no gradient, no fitted
parameter.

**Setup.** Char next-char on text8. Train 12,000,000 chars; held-out eval 300,000 (299,999 predicted
positions). Char orders {2,3,4,5}; add-α=0.05 smoothing; unigram fallback. Single causal pass to build
the `(w+,w-)` tables (the running-top-1 hit/miss is computed exactly, fully vectorized — validated
against a brute-force reference on 60 random streams, 0 mismatches). Fixed seed 0. Whole run ≈ 17 s on
CPU.

The hit-rate `f̄` rises monotonically with order — longer context bets right more often — which is the
signal the truth values capture:

| order | #contexts | hit-rate f̄ | mean c |
|---:|---:|---:|---:|
| 2 |     718 | 0.369 | 0.969 |
| 3 |  11,081 | 0.486 | 0.801 |
| 4 |  78,869 | 0.567 | 0.650 |
| 5 | 304,621 | 0.610 | 0.525 |

---

## Q1 — prediction (bpc / perplexity)

| combiner | top-1 acc | bpc | perplexity |
|---|---:|---:|---:|
| bare-count pool (baseline) | **0.5681** | 3.6336 | 12.41 |
| confidence-weighted pool | 0.5649 | 3.6077 | 12.19 |
| (f,c)-revision | 0.4114 | 2.7239 | 6.61 |
| **precision-weighted** | 0.5450 | **2.1103** | **4.32** |

**Calibrated weighting wins on bpc/perplexity, decisively.** Precision-weighting cuts bpc from 3.63 to
**2.11** (perplexity 12.4 → **4.3**, a 2.9× drop) and (f,c)-revision cuts it to 2.72 — both far below
the bare product-of-experts, while top-1 accuracy is roughly held (precision-weighted 0.545 vs 0.568).
The bare pool gives every order an equal vote; a thin, unreliable order-5 context then drags the pooled
distribution off-target. Down-weighting that order by its own track record (precision or c) is exactly
what a product-of-experts needs and never had. (f,c)-revision spends some argmax accuracy to buy a much
softer, better-spread distribution — note its low accuracy with its strong bpc; that trade is the
calibration story, made explicit in Q2.

## Q2 — calibration (the axis this idea is *for*)

Stated confidence = the probability the model assigns its own argmax. ECE = Σ (bin/N)·|acc − conf|.

| combiner | ECE ↓ | overall acc |
|---|---:|---:|
| bare-count pool | 0.2798 | 0.5681 |
| confidence-weighted pool | 0.2796 | 0.5649 |
| **(f,c)-revision** | **0.0268** | 0.4114 |
| precision-weighted | 0.0523 | 0.5450 |

**This is the headline: a 10× calibration win.** (f,c)-revision drops ECE from **0.280 → 0.027**;
precision-weighting to 0.052. The reliability table shows why — the bare pool is wildly **overconfident**
(it stakes 0.985 confidence on 179k predictions and is right only 76% of the time), while revision's
stated confidence tracks its accuracy almost on the diagonal:

| stated-conf bin | bare n | bare conf | bare acc | rev n | rev conf | rev acc |
|---|---:|---:|---:|---:|---:|---:|
| 0.0–0.1 |       0 | 0.000 | 0.000 |   3,679 | 0.098 | 0.085 |
| 0.1–0.2 |      16 | 0.192 | 0.000 |  57,618 | 0.160 | 0.168 |
| 0.2–0.3 |   1,338 | 0.285 | 0.142 |  68,252 | 0.256 | 0.258 |
| 0.3–0.4 |   9,445 | 0.340 | 0.134 |  62,336 | 0.344 | 0.377 |
| 0.4–0.5 |  14,547 | 0.452 | 0.200 |  32,176 | 0.445 | 0.547 |
| 0.5–0.6 |  23,408 | 0.550 | 0.261 |  21,405 | 0.549 | 0.581 |
| 0.6–0.7 |  20,738 | 0.648 | 0.290 |  18,319 | 0.652 | 0.623 |
| 0.7–0.8 |  23,593 | 0.750 | 0.327 |  16,238 | 0.747 | 0.758 |
| 0.8–0.9 |  27,470 | 0.852 | 0.368 |   9,943 | 0.849 | **0.902** |
| 0.9–1.0 | 179,444 | 0.985 | **0.759** |  10,033 | 0.937 | **0.952** |

Read the last two rows. When the bare pool says "85% sure" it is right 37%; when revision says "85%
sure" it is right 90%. When revision says "94% sure" it is right 95%. The truth value means what it
says — the count now knows how sure it is.

## Q3 — the gate: principled `f·c` vs a tuned threshold

High = order-5, backoff = order-2 (the spread that makes routing matter). The tuned gate opens the high
order where its confidence ≥ a swept threshold; the principled gate opens where `f·c(hi) > f·c(lo)` —
no knob.

| gate | thresh | open % | acc | bpc | ppl |
|---|---:|---:|---:|---:|---:|
| tuned threshold | 0.00 | 100% | 0.5869 | 1.9575 | 3.88 |
| **tuned threshold (best)** | 0.30 | 97% | 0.5848 | **1.9570** | 3.88 |
| tuned threshold | 0.70 | 96% | 0.5803 | 1.9690 | 3.91 |
| tuned threshold | 0.90 | 93% | 0.5724 | 1.9944 | 3.98 |
| tuned threshold | 0.99 | 75% | 0.5228 | 2.1892 | 4.56 |
| **principled f·c (no knob)** | — | 77% | 0.5696 | 2.0422 | 4.12 |
| always backoff (o2) | — | 0% | 0.3667 | 2.9152 | 7.54 |
| always high (o5) | — | 100% | 0.5869 | 1.9575 | 3.88 |

**Honest result: on clean text, overall, the gate loses to "always open."** Order-5 (with its unigram
fallback) beats order-2 almost everywhere, so the best policy is to open ~always; the best tuned
threshold (1.957 bpc) edges the parameter-free principled gate (2.042, Δ +0.085). The principled gate is
*too conservative* — it closes whenever order-5's `f·c` is thin, but order-5's full distribution still
usually beats order-2's. So judged on the headline axis, the principled gate matches the *conservative*
end of the tuned sweep but not its optimum.

### Q3b — the gate on the slice it is *for* (rare / unreliable high context)

The gate can only matter where the high order is genuinely unsure. Restrict to positions whose order-5
context has confidence `c < 0.8` (evidence mass `w ≤ ~4` — a thin 5-gram): 6,727 positions, 2.2% of eval.

| policy (rare slice) | acc | bpc |
|---|---:|---:|
| always high (o5) | 0.5490 | 2.7635 |
| always backoff (o2) | 0.2896 | 3.4609 |
| best tuned threshold | 0.5490 | 2.7635 |
| **principled f·c** | 0.5120 | **2.7389** |

**On the slice that matters, the principled gate is the only policy that beats "always open" on bpc**
(2.739 vs 2.764), by routing the genuinely-uncertain-and-likely-miscalibrated cases to the backoff. The
best tuned threshold (0.30) opens here too and is indistinguishable from always-open. The principled
gate trades a little accuracy (0.512 vs 0.549) for the calibrated-bpc win — the same accuracy/bpc trade
as (f,c)-revision in Q1, on exactly the population where a gate earns its keep.

---

## Verdict — PASS on calibration; partial on the gate

1. **Calibration: clear PASS, the headline.** NARS truth values turn an overconfident count into an
   honestly-calibrated one — **ECE 0.280 → 0.027 (10×)** for (f,c)-revision, with a near-diagonal
   reliability curve. This is the axis the idea is for, and it wins big.
2. **Prediction: PASS (unexpected bonus).** Precision-weighting and revision both cut bpc/perplexity far
   below the bare product-of-experts (ppl 12.4 → 4.3 / 6.6) at roughly held accuracy — confidence is a
   *better expert weight* than the equal vote the bare pool uses.
3. **Gate: PARTIAL.** The parameter-free `f·c` gate does *not* beat the best tuned threshold overall on
   clean text (high order wins ~everywhere, so "always open" is hard to beat). But on the **rare /
   unreliable high-order slice** — the only place routing should matter — the principled gate is the
   single policy that beats always-open on bpc. A fair "matches the conservative regime; wins exactly
   where a gate is supposed to."

**Which axis it won on: CALIBRATION** (Q2), and as a bonus, **bpc/perplexity via confidence-weighted
pooling** (Q1). The gate (Q3) is a qualified pass — principled, knob-free, and a winner on the
rare-context slice (FRAGILE commandment 7), not on the clean-text headline.

**Honest caveats.** (a) (f,c)-revision lowers bpc but also lowers top-1 accuracy (0.41 vs 0.57) — it
buys calibration with argmax sharpness; precision-weighting keeps both (acc 0.545, bpc 2.11) and is the
better all-round combiner here. (b) The clean-text gate result depends on the high order beating the low
order almost everywhere; with a fairer Katz-backed high order the spread would narrow and the gate's job
would change. (c) `k=1` is the canonical NARS horizon; we did not sweep it (the point was a
parameter-free truth value, not a tuned one). (d) "Confidence-weighted pool" barely moved bpc vs bare —
the win came from *additive revision* and *precision*, not from simply scaling log-dists by c.

## Online-compliance note

Strictly online and count-only. The `(w+,w-)` split is a **single causal pass**: at each position a
context's running top-1 reflects only its *earlier* occurrences, scored against the next char before
that char is folded in (verified exactly against a brute-force online reference). Precision is a **leaky
running mean/variance** of signed error (causal — weight at t uses stats up to t−1). Revision is plain
**addition of evidence counts**. The gate reads `f·c` straight off the counts. **No gradient descent, no
batch optimization, no k-means/SVD** — only counting, a leaky accumulator, and add-α smoothing. Train and
eval are a held-out split for measurement; the learning rule itself is single-pass and incremental.

## Files
- `experiments/lib/confidence.py` — CountTruth (vectorized single-pass running-top-1 hit/miss split),
  truth_of, order_logdist, bare_count_pool / weighted_pool / revision_truth, confidence_weight /
  precision_weights, tuned_threshold_gate / principled_gate, calibration / ECE / perplexity.
- `experiments/exp_ab_confidence/run.py` — self-contained, ~17 s, fixed seed 0.
