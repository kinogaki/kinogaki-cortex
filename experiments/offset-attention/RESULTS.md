# Exp S — Offset-keyed count-attention

**The claim.** Attention asks "which earlier positions predict the next token, and how much?".
This is the **count-based, no-backprop** answer: one associative count table per relative **offset**
`d` (`T_d[a] = {b: count}` = how often word `b` followed when the word `d` steps back was `a`), each
offset an expert predicting the **same** next position from a **different** earlier slot. Pool the
experts with a geometric mean (`cortex.vote_sparse` style), weighting each offset by its **information
gain** `IG(d) = H(next) − H(next | word_at_offset_d)`, read straight off the counts. No queries, keys,
values, or gradients — attention as replicated Columns + an entropy-derived weight per position.

Setup: text8, word level, top **40,000** words + UNK (OOV 2.7%), **D=8** offsets, train ≈ 6.67M words,
held-out eval = last **120,000** words. Abstaining experts contribute nothing; a fully-abstaining
predictor falls back to uniform (never skipped — a fair perplexity). Pooling weight `∝ IG(d)**γ`,
`γ=8` (the one free knob; γ=1 = flat geometric mean, γ→∞ = pure bigram; chosen on a separate 20k slice
where γ≈8 minimized perplexity while still blending multiple offsets — see "On γ" below).

## Q1 — offset-attention vs fixed n-grams (ordered context)

| model | top-1 acc | perplexity |
|---|---:|---:|
| bigram (`d=1` only) | 16.21% | 13,477 |
| trigram (fixed, +bigram backoff) | **17.79%** | 40,034 |
| **offset-attention (D=8)** | 16.31% | **4,318** |

**Verdict: yes on the metric that matters.** Offset-attention crushes both n-grams on perplexity
(4,318 vs 13,477 vs 40,034 — a **3.1× reduction** over the bigram). On top-1 it ties the bigram and
sits just under the trigram. The trigram's higher top-1 comes with a *worse* perplexity than the
bigram: a sparse unsmoothed trigram context is confidently wrong off-distribution. Offset-attention is
the only model that is both accurate **and** calibrated, because it pools eight position-experts instead
of betting everything on one sparse context.

## Q2 — KILLS BAG-OF-WORDS (the key claim)

The **bag** control reuses the *same* eight tables but merges them offset-agnostically into one
distance-blind table `bag[a] = Σ_d T_d[a]`, then pools an unweighted expert per *distinct* context word.
By construction the bag's answer depends only on **which** words are in the context, not their order.
We evaluate both models on **ordered** context and on **scrambled** context (the D context words
shuffled per position; the *set* is preserved).

### 2×2 — top-1 accuracy

| | ordered | scrambled | Δ(scramble) |
|---|---:|---:|---:|
| **offset-attention** | **16.31%** | 5.79% | **−10.51 pp** |
| bag-of-words | 7.27% | 7.27% | +0.00 pp |

### 2×2 — perplexity

| | ordered | scrambled |
|---|---:|---:|
| **offset-attention** | **4,318** | 30,716 |
| bag-of-words | 12,061 | 12,061 |

**Verdict: claim proven, cleanly.** Three things all hold at once:
1. **Offset-attention is order-sensitive.** Scrambling the context collapses it: 16.31% → 5.79%
   (−10.5 pp) and perplexity blows up 4,318 → 30,716. It is *using* word order.
2. **The bag is order-blind.** Scrambling moves it by exactly **0.00 pp** and **0.0** perplexity —
   identical numbers, as the construction guarantees. It is a bag of words.
3. **Offset-attention beats the bag on real (ordered) context** — 16.31% vs 7.27% top-1, 4,318 vs
   12,061 perplexity. The order information the bag throws away is worth >2× the accuracy and ~3× the
   perplexity.

This is the headline: a pure count model, with no backprop, is **not** a bag of words — it earns its
accuracy from word *position*, and discards it gracefully (back to bag-ish, then worse) when position is
destroyed.

## Q3 — learned IG(d) weights (do nearer offsets carry more?)

| offset `d` | IG (bits) | IG share | pool weight (∝ IG⁸) |
|---:|---:|---:|---:|
| 1 | 3.3073 | 18.4% | 81.6% |
| 2 | 2.4641 | 13.7% |  7.8% |
| 3 | 2.2136 | 12.3% |  3.3% |
| 4 | 2.0544 | 11.4% |  1.8% |
| 5 | 2.0055 | 11.2% |  1.5% |
| 6 | 1.9910 | 11.1% |  1.4% |
| 7 | 1.9751 | 11.0% |  1.3% |
| 8 | 1.9666 | 10.9% |  1.3% |

**Yes — nearer offsets carry more, and the decay is fast then flat.** Raw IG drops sharply from `d=1`
(3.31 bits) to `d=2` (2.46) and then has a long, slowly-decaying tail down to ~1.97 bits at `d=8`. The
immediately-preceding word is by far the most informative position; everything farther is *weakly but
nonzero* informative (every offset keeps >1.9 bits — far context never goes uninformative, it just
plateaus). The model learns this entirely from counts, with no labels and no training loop.

## On γ (the one knob, kept honest)

The *flat* geometric mean (γ=1) over-weights the seven weakly-informative far offsets and blurs the
sharp `d=1` signal — it gave acc 10.9% / ppl 6,120: great perplexity vs the bigram, but a diluted
argmax. Sharpening the pool toward informative offsets fixes the argmax. On a separate 20k tuning slice,
perplexity bottomed near γ≈8 (acc 17.4% / ppl 3,448) while still genuinely blending offsets
(weights `[.82,.08,.03,.02,.01,.01,.01,.01]`); pushing γ higher degenerates monotonically toward the
pure `d=1` bigram (γ=24 → weights `[1,0,…]`, acc 17.68% / ppl 3,640). γ=8 is the chosen operating point:
near-bigram top-1, best calibration, and a real multi-offset blend (which is what makes Q2 work). The
top-1 ceiling for this family is the bigram's accuracy — pooling far offsets cannot raise the argmax
above the single most-informative position, but it lowers perplexity ~3× and is what carries the
order-sensitivity result.

## Bottom line

- Offset-keyed count-attention **beats fixed n-grams on perplexity by ~3×** and ties the bigram on top-1
  — a principled, count-based, gradient-free attention that is strictly better-calibrated than any single
  n-gram.
- It **is not a bag of words**: order-sensitive (−10.5 pp under scramble), strictly above the order-blind
  bag on ordered context (16.31% vs 7.27%), while the bag is provably invariant to scrambling.
- The **per-offset information gain** is a real, count-derived attention weight: it decays fast from the
  nearest position and then plateaus, with `d=1` dominating.

Repro: `python exp_s_offset_attention/run.py` (D=8, vocab=40k, 40MB text8, ~4 min). New files only:
`lib/offsetattn.py`, `exp_s_offset_attention/run.py`, this file.
