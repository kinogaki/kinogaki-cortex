# Experiment R — evidence accumulation with decay — 2026-06-25

**The bet (Thousand Brains).** Don't recompute the next-char belief fresh every step. Keep a LEAKY
log-evidence accumulator `E_t = gamma·E_{t-1} + Σ_k log p_k(·)` over the same char-order experts, so
one noisy step can't tank the prediction, and read the accumulator's DROP as a free boundary signal.

**Setup.** Char next-char prediction on text8. Train 10,000,000 chars; eval a 200,000-char slice
(199,999 predicted positions). Experts = char context-orders {2,3,4,5}, true Katz-style backoff to
order-1/unigram, add-α=0.05 smoothing. Baseline = FRESH product-of-experts (sum of the per-order
log-probs) recomputed each position. Evidence model = the SAME experts via the leaky accumulator,
scaled by the effective window mass `(1−gamma)` so `softmax(E)` stays calibrated (an *average*
log-evidence, not a growing sum). Fully vectorized; whole run is ~5 s on CPU. Reproducible (fixed seeds).

---

## Result 1 — robust voting under context corruption

A fraction of the CONTEXT chars are randomized; targets are the clean true next-chars. gamma = 0.8.

| context noise | fresh acc | fresh bpc | evid acc | evid bpc | acc Δ fresh | acc Δ evid |
|---|---:|---:|---:|---:|---:|---:|
| 0%  | **0.570** | **3.670** | 0.275 | 5.609 | 0.000 | 0.000 |
| 5%  | 0.505 | 5.493 | 0.253 | 6.061 | 0.065 | 0.021 |
| 10% | 0.446 | 7.209 | 0.234 | **6.497** | 0.125 | 0.041 |
| 20% | 0.349 | 10.199 | 0.204 | **7.275** | 0.221 | 0.071 |
| 30% | 0.275 | 12.762 | 0.180 | **7.996** | 0.295 | 0.094 |

**The crossover is real.** On clean text the fresh pool is far better (sharp, exact: 3.67 bpc, 57%
acc). But it is brittle: every corrupted context char is a full-strength bad step, so its bpc explodes
(+9.1 bpc from clean→30% noise) and accuracy collapses (−0.295). The accumulator degrades gracefully:
bpc rises only +2.4 over the same range, accuracy drops only −0.094. **At ≥10% noise the evidence
model wins on bpc** (6.50 vs 7.21 @10%, 7.28 vs 10.20 @20%, 8.00 vs 12.76 @30%) — exactly the
"one noisy step can't tank it" prediction.

### Result 1b — the leak/robustness tradeoff (gamma sweep, fresh has no gamma)

Fresh pool reference: clean bpc 3.670 / acc 0.570 · 20%-noise bpc 10.164 / acc 0.350.

| gamma | clean bpc | clean acc | 20%-noise bpc | 20%-noise acc |
|---:|---:|---:|---:|---:|
| 0.5 | 4.216 | 0.451 | 7.789 | 0.296 |
| 0.6 | 4.591 | 0.400 | 7.583 | 0.268 |
| 0.7 | 5.051 | 0.344 | 7.415 | 0.239 |
| 0.8 | 5.609 | 0.275 | 7.261 | 0.204 |
| 0.9 | 6.318 | 0.192 | 7.108 | 0.165 |

Clean bias/variance tradeoff: more leak (higher gamma) ⇒ blurrier on clean text but ever more robust
under noise. **Even gamma=0.5 beats the fresh pool at 20% noise (7.79 vs 10.16 bpc)**, and the noisy
bpc keeps falling monotonically as gamma rises while the clean bpc keeps rising. The decay is buying
exactly what it should: noise immunity, paid for in clean-case sharpness.

---

## Result 2 — boundary signal (vs true word boundaries)

Confidence `conf_t = gamma·conf_{t-1} + logP(observed char)`; boundary = a sharp DROP. Compared
head-to-head with the forward branching-entropy RISE (Exp A's winner). Truth = new-word-start
positions (the char right after a space — spaces themselves are highly predictable, so the
prediction-difficulty lands on the first char of the next word). 33,650 boundaries, density 0.168.
Top-K thresholded at the true rate; greedy ±tol matching.

| signal | F1 (±1) | F1 (±2) |
|---|---:|---:|
| random | 0.420 | 0.571 |
| instantaneous surprisal | 0.459 | 0.573 |
| **entropy RISE (Exp A winner)** | **0.755** | **0.783** |
| confidence DROP (evidence) | 0.507 | 0.627 |

**Negative result, and it's instructive.** The confidence-DROP signal beats random (0.507 vs 0.420 @±1)
but loses decisively to entropy-RISE (0.755). The reason is the same decay that helps Result 1: the
leaky accumulator *smooths* the surprisal stream, and boundary detection wants a *sharp* per-position
spike, not a decayed one. Accumulation and boundary-sharpness pull in opposite directions. Entropy-RISE
stays the right boundary mechanism (consistent with Exp A); the evidence accumulator is a
*prediction-robustness* tool, not a boundary detector.

---

## Verdict — QUALIFIED PASS

1. **Robust voting: PASS.** Leaky log-evidence accumulation degrades far more gracefully under context
   noise than a fresh product-of-experts. Clear bpc crossover at ≥10% corruption; accuracy drop cut to
   ~⅓ of the baseline's at 30% noise; monotone leak/robustness tradeoff across gamma. The Thousand-Brains
   "evidence over time, not a fresh vote each step" idea earns its keep when the input is unreliable.
2. **Boundary-from-drop: FAIL (vs the incumbent).** Confidence-drop carries weak boundary signal
   (F1 0.51 @±1) but is dominated by branching-entropy RISE (0.76). The accumulator's smoothing, which
   *helps* prediction, *hurts* boundary sharpness — the two objectives are in tension.

**Honest caveats.** (a) On clean text the accumulator is strictly worse at next-char prediction — it
predicts a blur of the last ~1/(1−gamma) chars, so its clean-case accuracy (0.27 @gamma=0.8) is well
below the fresh pool (0.57). The win is *only* in the noisy regime. (b) "Noise" here is uniform random
char substitution in the context; structured noise (typos, OCR confusions) might behave differently.
(c) The accumulator is the simplest leaky form; a noise-gated version (down-weight a step whose experts
disagree) would likely keep more clean-case accuracy while staying robust — the natural next experiment.

## Files
- `experiments/lib/evidence.py` — ExpertBank (vectorized per-order next-char log-dists with backoff),
  fresh_pool_logp, evidence_logp (leaky accumulator), running_confidence / forward_entropy /
  drop_signal / rise_signal, f1_at_rate, corrupt_context.
- `experiments/exp_r_evidence/run.py` — self-contained, ~5 s, fixed seeds.
