# Experiment C — does a grown concept hierarchy earn its keep? — 2026-06-25

**Setup.** English (Austen), char-level, vocab=27 (a-z + space). Train 589k chars / test 104k. Pure
**associative** substrate (order-8 backoff char n-gram + frequency-weighted lexicon trie), no backprop —
the substrate Exp B pointed to. Question: does conditioning prediction on a learned **word concept layer**
lower bits-per-char vs the flat char model? Lexicon written to a `.prism` doc (the inspectable concept store).

## Result — the ladder (bits-per-char, lower = better)

| model | bpc | vs flat |
|---|---:|---:|
| flat char 8-gram (baseline) | 2.124 | — |
| **+ word concepts, TRUE words (L1)** | **1.653** | **+22.2%** |
| + word concepts, DISCOVERED words (unsupervised L1) | 1.773 | **+16.6%** |
| + word→word context, first-char (L2) | 1.653 | +22.2% (no gain) |
| + context-aware lexical marginalization (L2′) | 1.692 | +20.3% (worse) |
| + word cache / recency, clean additive (L1 + cache) | **1.650** | **+22.3%** (best) |

## Findings — what worked, what didn't

1. **WORKED (the real result): word concepts earn their keep — a robust +22% bpc reduction**, and **+16.6%
   fully unsupervised** (words discovered by branching entropy, no labels). Pure association, online,
   non-forgetting, **and inspectable** — the discovered concept store is real English words
   (`the, a, to, and, he, her, was, she, not, …`) written into a `.prism` document. This validates the core
   north-star claim: *grown higher-level concepts make the model measurably better, transparently.*
2. **DIDN'T compound (honest negative): word→word context (L2) added nothing.** Reason found: the char 8-gram
   already reaches ~8 chars *into the previous word*, so an explicit word-bigram is redundant. Context-aware
   marginalization (L2′) was *worse* — capping candidates to top-400 words discards the long tail the full
   trie nails.
3. **Recency/activation adds a sliver.** A clean additive decaying word **cache** (recently-seen words boost
   their first char) beats L1 by +0.1% (1.6504). The mechanism works — *and it's exactly our north-star
   recency rule* — but the effect is tiny at this scale because it only touches word-first-chars.
4. **Unsupervised discovery is noisy** (over-segments: `e, t, d, s, n` appear as standalone "words"),
   which is the whole +16.6% → +22.2% gap. Better segmentation → closer to the ceiling.

## What this means

- The **first level of the hierarchy is real and valuable** (concepts → +22%, unsupervised +16.6%,
  inspectable, online, associative). That's a genuine, north-star-validating result.
- **Compounding to a much lower bpc is NOT yet demonstrated** — higher levels (word-context, recency) only
  shave first-chars because the char n-gram subsumes local context. Real compounding needs a level that
  affects *more than first-chars* and reaches *beyond the char window*: **phrases / sentence-topic structure**,
  or a weaker/shorter char baseline that doesn't already encode word context.
- At **1.65 bpc**, the fully-online/associative/inspectable model is competitive-ish (char-LSTM ≈ 1.4, PPM ≈ 1.9)
  — promising for something that learns online, never forgets, and you can read.

## Next (toward a bigger result)

- **Better unsupervised segmentation** (close the +16.6%→+22% gap): fix over-segmentation (min-length, both-
  direction branching with a learned threshold, iterative re-segmentation using the lexicon).
- **A real second level**: phrases (frequent word sequences) or a topic/cache over a longer horizon that
  affects whole-word prediction, not just first chars — the place compounding can actually happen.
- **The voting field**: multiple views (different orders/directions) voting — does consensus beat the best
  single view (Exp D)?
