# Exp AC — discourse coherence via a Bayesian-surprise EVENT MODEL — 2026-06-26

**The bet (Zacks event-segmentation; Kumar 2023 Bayesian-surprise; Franklin SEM sticky-CRP; Zwaan situation
models).** A reader carries a persistent **event slot** — a leaky profile over the currently-active
phrase/topic clusters — that acts as a soft top-down prior on the next token. At every step we measure the
**Bayesian surprise**

    S_t = KL( P_t || P_{t-1} )

over the top-k next-word distribution, where `P_{t-1}` is the belief *before* the word and `P_t` the belief
*after* conditioning on it (the one-step posterior update). A leaky running mean/var turns `S` into a
z-score; `z > θ` fires an **event boundary**: the live slot is ARCHIVED into a non-forgetting long-term store
and a new slot is SELECTED via a **sticky-CRP bank** (prefer the current slot by a stickiness pseudocount;
open a new slot only when no existing slot beats a new-slot pseudocount). The committed slot's cluster
profile is mapped to a word prior and mixed into the predictor **only on the backoff slice** (the Exp T
lesson: top-down belongs where local prediction has run out).

**Two right axes.** (1) **Boundary detection** — does KL/Bayesian-surprise locate enwik9 article (`<page>`)
boundaries better than per-token **surprisal** and than **branching-entropy**? This is the Kumar-2023 claim;
test it. (2) **Discourse prediction** — does the persistent slot prior lower perplexity beyond local context,
especially on words far from local n-gram support? Honest if it doesn't.

**The online rule (hard).** Single streaming pass; **no gradient descent, no k-means, no SVD**. Concept
clusters = online signatures (hashed, IDF-discounted running context counts) + **online leader clustering**
(`lib/jepa`). The slot bank is leader clustering with a stickiness prior. The surprise normalizer is a leaky
EMA mean/var. The per-token predictor is a word trigram→bigram→unigram **count** model. Every learned object
is an order-independent accumulation or a single-pass leader assignment.

**Setup.** enwik9, **36 MB → 5,177,506 words** (vocab 80 k + UNK), **4,598 `<page>` boundaries** (≈ every
1,126 words = topic-boundary ground truth). Top-12,000 words clustered online into **C = 400** concept
clusters (all 12 k clustered). KL over **top-k = 12**; surprise z-normalizer half-life 6,000 words, θ = 2.0,
refractory 150. Fixed seed (0). **Whole run 449 s on CPU, single pass.**

---

## Axis 1 — boundary detection vs the `<page>` truth

All three signals are thresholded at the **same rate** (= #gold = 4,598 cuts), so precision = recall = F1; a
predicted cut counts as a hit if it lands within `±tol` words of a real article boundary (greedy one-to-one).

| tol (words) | **KL (Bayes-surprise)** | surprisal | branching-entropy |
|---:|---:|---:|---:|
| 10 | **0.091** | 0.003 | 0.000 |
| 25 | **0.154** | 0.027 | 0.001 |
| 50 | **0.156** | 0.093 | 0.012 |

**KL wins decisively, on every tolerance.** At `tol = 25` words: **KL F1 0.154 vs surprisal 0.027 (5.7×) vs
branching-entropy 0.001 (≈120×).** The gap *widened* with scale (at a 3 MB pilot KL was 0.099; at 36 MB it is
0.154, while the others stayed near the floor). At the loosest tolerance (`±50`) surprisal partly catches up
(0.093) — raw per-token spikes are *somewhere* near a boundary but poorly localized — yet KL still leads.
Branching-entropy, the Exp A/M phrase signal, is essentially useless for *topic* boundaries: it fires on
syntactic fan-out, which is uniform across articles.

The event model's own fired boundaries (θ = 2.0) are very **high-precision, very low-recall**: it fired only
**20** cuts in 5.18 M words; ~10 % of them are real article boundaries (precision 0.10 at ±25) but they cover
<0.5 % of the 4,598 gold cuts. The z-gate is calibrated to fire only on the *largest* belief jolts — it finds
a handful of unmistakable topic shifts and ignores the rest. As a ranked *signal* (the table above) KL is
strong; as a hard *segmenter* at this θ it is a precision instrument, not a recall one.

> **Why KL beats surprisal.** Surprisal is `−log P(actual word)` — it spikes on *any* rare or hard word
> (every long content word inside an article lights it up), so its boundary signal drowns in within-topic
> noise. KL(P_t‖P_{t-1}) measures how much the *whole next-word belief moved* — a rare word that was already
> expected barely moves the distribution, while the first word of a new article reshapes it. **Bayesian
> surprise isolates the belief-update, not the token-cost.** That is exactly the Kumar-2023 claim, and on
> real multi-article text it reproduces.

---

## Axis 2 — does the persistent event-slot prior help prediction?

The slot's cluster profile is turned into a word prior, `P(w) ∝ slotprofile[clu(w)] · unigram(w)`, and
blended into the predictor **only on the backoff slice** — the words where the local trigram *and* bigram
context are both unseen (local prediction has run out). `no-slot` is the identical model with the prior off
(an exact apples-to-apples baseline).

| prior weight | overall bpw — no-slot | overall — with-slot | Δ | backoff slice (0.9 % of words) no-slot → with-slot | Δ on slice |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 10.7828 | 10.7816 | +0.0012 | 11.7228 → **11.5802** | **+0.143** |
| 0.6 | 10.7828 | 10.7822 | +0.0006 | 11.7228 → 11.6485 | +0.074 |

**The slot prior helps, concentrated exactly where predicted — and the win is small because the slice is
small.** On the **backoff slice the persistent event slot saves 0.143 bits/word** (w = 0.3): when the local
word context is exhausted, the active topic carries real information about which word comes next. But at
36 MB the trigram/bigram tables cover so much that the backoff slice is only **0.9 % of words** (it was 6 %
at 3 MB), so the diluted overall gain is a near-invisible +0.0012 bpw. A lighter blend (w = 0.3) beats a
heavier one (w = 0.6): the topic prior is a gentle nudge, not a replacement for the unigram — over-trusting
it (0.6) erases half the gain.

This is the **Exp T result, reproduced through a completely different mechanism.** Exp T committed a global
topic *G* by ignition and found it helps only on the word-level backoff slice; here a Bayesian-surprise event
slot — discovered online from belief jolts rather than from a recency histogram — lands in the *same* place
with the *same* shape: **top-down priors pay off only where local context fails, and only at an altitude
whose tokens are topic-bearing.** Two independent roads to one law.

---

## Verdict

**Bayesian surprise is a genuinely good topic-boundary detector; the event slot it drives is a real but
narrow predictive prior.**

- **Axis 1 — the Kumar-2023 claim holds, clearly.** KL(P_t‖P_{t-1}) beats per-token surprisal **5.7×** and
  branching-entropy **~120×** at locating real enwik9 article boundaries (F1 0.154 vs 0.027 vs 0.001 at
  ±25 words), and the lead grows with data. The mechanism is exactly the one the literature names: surprise
  measured as *belief-update* (KL) isolates topic shifts, while surprise measured as *token-cost* (surprisal)
  is swamped by ordinary rare words. This is the headline win.
- **Axis 2 — the slot prior helps only where local prediction fails.** +0.143 bits/word on the backoff slice,
  but that slice is 0.9 % of a 36 MB run, so the overall gain is +0.0012 bpw — honest, real, tiny. The right
  reading is *altitude*, not failure: the prior is correctly inactive on the 99 % of words the local n-gram
  already nails, and correctly informative on the 1 % it can't. A gentle blend (w = 0.3) beats an aggressive
  one — the topic is a prior, not a predictor.
- **The honest negative.** As a *hard segmenter* the z-gate at θ = 2.0 is precision-only (20 cuts fired,
  recall < 0.5 %). The strong boundary result is KL-as-ranked-signal, not KL-as-fired-events; matching the
  event-rate to the article rate would need a much lower θ (and would trade away the precision). The slot
  bank's archive/sticky-CRP machinery ran correctly online but its *predictive* payoff is the small
  backoff-slice gain above — we did not find a large coherence win from the persistent slot beyond it.

**Takeaway for the cortex.** Carry **Bayesian surprise (KL of the one-step belief update), not surprisal**, as
the topic-boundary signal — it is the right axis and it reproduces a published result on real text. Keep the
event slot as a **soft, low-weight prior fired only on the backoff slice** (the Exp T altitude law,
re-confirmed). Do not expect the slot to sharpen the 99 % of tokens local context already owns.

---

## Online-compliance note (every part is online; nothing is batch-optimized)

| step | how it's computed | online? |
|---|---|---|
| concept clusters | online signatures (hashed IDF-weighted running context counts) + **leader clustering** (single pass, nearest running-mean prototype or spawn) — `lib/jepa` | **yes** — accumulation + one-pass assignment |
| per-step Bayesian surprise | KL over the top-k next-word counts before/after the token | **yes** — counting |
| surprise z-score | **leaky EMA** running mean/variance, threshold + refractory | **yes** — leaky accumulator |
| event-slot profiles | per-slot **leaky** cluster-count profile | **yes** — leaky accumulator |
| slot selection | sticky-CRP = leader clustering with a stickiness pseudocount + non-forgetting archive | **yes** — single nearest-prototype-or-spawn decision per boundary |
| next-word predictor | word trigram→bigram→unigram **counts** | **yes** — counting |

**No gradient descent, no backprop, no k-means, no SVD/eigendecomposition.** The vectorized count builders
(`np.unique`/`np.bincount`) are batched implementations of order-independent accumulation — identical to a
token-at-a-time online update. `topk_dist` is memoized by context-dict identity (each frozen context sorted
once) purely for speed; it changes no result. Reproducible at fixed seed 0.
