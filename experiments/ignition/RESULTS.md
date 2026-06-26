# Exp T — Top-down prior / ignition broadcast

**Question.** A higher level commits to ONE global topic state **G** (global-workspace / Thousand-Brains
ignition); the lower predictor conditions its counts on it — `P(next | local-ctx, G)`, backing off to
`P(next | local-ctx)` when `(ctx, G)` is unseen. **Ignition/hysteresis:** G only switches when one topic
cluster decisively dominates the recent window (margin threshold), else it is held. Does conditioning on the
committed topic reduce surprisal on topically-varied text (enwik9, real article boundaries)?

**Setup.** enwik9, 30 MB train / ~0.82 MB held-out test (107 article boundaries in test). G built online:
content words → K=128 topic clusters (PPMI co-occurrence + spherical k-means) → recency-weighted topic
histogram (half-life 40 words) with ignition margin 0.18; broadcast to every position. G switched 250× on
test (~every 550 words). Models share the *exact* count machinery so WITH-G vs WITHOUT-G is apples-to-apples
(`use_g=False` recovers the plain FastChar / backoff baseline bit-for-bit).

## Results

### Char level (order-6 backoff; G folded into orders 6,5,4,3)

| metric | without-G | with-G | Δ (lower=better → +Δ = G helps) |
|---|---|---|---|
| **overall bpc** | **1.8645** | 2.0821 | **−0.2176** (G HURTS) |
| shuffled-G control | — | 2.2499 | (−0.3854 vs no-G) |
| post-boundary (first 400 chars) | 1.1853 | 1.4043 | −0.2190 |
| post-boundary (first 50 chars) | 0.9560 | 1.0987 | −0.1427 |

### Word level (trigram→bigram→unigram; unigram fallback optionally G-conditioned)

| metric | without-G | with-G | Δ |
|---|---|---|---|
| **overall bits/word** | 12.9877 | **12.9688** | **+0.0189** (G helps) |
| **backoff slice** (5.5% of words, local ctx exhausted) | 12.6156 | **12.2746** | **+0.3410** (G clearly helps) |

## Verdict

**Top-down G helps where, and only where, local context is exhausted — and the altitude decides whether that
ever happens.**

1. **Char level: G does not help; it hurts (−0.22 bpc).** Next-*character* prediction is already saturated by
   the local 3–6 char context — topic is fully subsumed. Folding G into the key makes each `(G, ctx)` cell
   ~128× sparser, so add-α smoothing dominates and the conditioned counts predict *worse* than the pooled
   table. The shuffled-G control is worse still (2.25 vs 2.08), proving the machinery is sound — real G is
   far better than random G — but **no G beats any G** at this altitude. The post-boundary surprisal spike is
   *not* reduced either: the spike comes from low-order/unseen char contexts that the un-fragmented plain
   table smooths better, and topic governs word *choice*, not character *spelling*.

2. **Word level: G helps, concentrated exactly where predicted.** Overall the gain is tiny (+0.019 bits/word)
   because 94.5% of words are nailed by the local bigram/trigram, where G is redundant. But on the **5.5%
   backoff slice — words where the local word-context is unseen — the committed topic saves 0.34 bits/word.**
   That diluted 0.34 is the whole overall gain. This is precisely the global-workspace claim: when local
   prediction fails, the broadcast global state carries real information about which word comes next.

3. **Why the asymmetry.** Topic lives in the *vocabulary distribution*, not in spelling. Conditioning on G
   pays off only at an altitude whose tokens are topic-bearing (words/concepts) AND only in the regime where
   local context has run out. Char-level n-grams never reach that regime — local spelling context is almost
   always sufficient — so G can only fragment, never inform.

**Implication for the cortex.** The top-down prior is real but it belongs **higher and as a soft blend, not a
hard re-key at the char level.** The productive design is: let the word/concept Level commit G via ignition,
and inject it as a *prior on the word distribution* used only when the lower n-gram backs off (or as a small
mixed-in term) — never as an extra key dimension on the dense char counts, which only sparsifies them. Bigger
wins would come from longer-range word context and a topic-conditioned word prior with shrinkage toward the
global unigram (so sparse `(G, w)` cells degrade gracefully), not from pushing topic down onto characters.
