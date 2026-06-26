# Exp H — concept hierarchy (C/E) scored on the generalization suite — 2026-06-25

**Setup.** text8, train 10MB / test 2MB, vocab 71k. Reusable metrics (`../lib/metrics.py`): bpc + overfit gap,
prediction horizon, real-word % (text-vs-gibberish), and phrase-coherence (% of generated word-bigrams seen in
training). Models: char backoff (K=6); + word lexicon (Exp C); word-level unigram·bigram·cache (Exp E).

## Scorecard — what each level actually buys

| level | test bpc | overfit gap | real-word % (t1.0) | phrase-coh % |
|---|---:|---:|---:|---:|
| char-only | 2.004 | +0.528 | 77.2 | 56.9 |
| + word concepts (C) | **1.897** | **+0.315** | **88.8** | 45.6 |
| + phrases / word-level (E) | — | — | **100.0** | **82.4** |
| *(real held-out text)* | — | — | 95.0 | 61.4 |

## Findings (each level improves the property it models — and nothing more)

1. **Word concepts (C) = generalization + word-validity.** Overfit gap nearly halves (0.528 → 0.315): the
   lexicon is a *regularizer* — train-bpc rises (1.48→1.58) while test-bpc falls (2.00→1.90), the textbook
   less-overfit/better-generalize signature. Real-word generation 77%→89%. **But it does NOT improve phrase
   coherence** (45.6%, even below char-only) — it doesn't model word→word transitions, and its richer vocabulary
   lands in rarer bigrams.
2. **Phrase level (E) = phrase coherence.** Word-bigram coherence jumps to 82.4% (samples read like English:
   *"very successful campaign against … military aircraft … edition of the"*), 100% real words. Models
   transitions → produces locally-coherent phrases.
3. **Prediction horizon is local-char-dominated** — concepts don't move it (1.7→1.6 greedy run). Expected.
4. **Global/discourse coherence is still absent in all three** — every model produces locally-plausible
   *word-salad* (E loops on topical words; C/char ramble). The metric ceiling itself is informative: even real
   text has only 61% "seen" bigrams on 10MB — novel-combination generalization is its own challenge.

## What this establishes

- **The measurement framework works** and cleanly attributes value per level: words→word-validity+generalization,
  phrases→phrase-coherence. Every future change is now scored on {bpc, overfit gap, horizon, real-word%,
  phrase-coh%}.
- **The north-star claim ("grown concepts improve the model") is validated with real metrics, not just bpc** —
  concepts measurably reduce overfitting and gibberish; phrases measurably add coherence.
- **The frontier is explicit: GLOBAL coherence** (discourse/topic/meaning), which no current level models. That
  is the next level up — and the place the architecture's *reasoning/movement* ambitions must eventually deliver.
  We now have the dials to know if they do.
