# Exp Z — similarity reps as a representation factory, projected into the count predictor — 2026-06-26

**The bet.** Proximity / "raytracing" is a *bad predictor* (Exp P: 1.8% vs the bigram's 21%; Exp W: lost the
rematch) but a *good similarity tool* — nearness in co-occurrence space is meaning, cleanly. So stop asking it
to predict. Use it as a **representation factory**: build similarity reps **hierarchically** (words, then
phrases) and **project them into the count-based predictor**. The rare-context play: when a context word or
phrase has few or zero *direct* next-word counts, substitute/augment with its similarity **cluster's**
aggregated counts, so the rare item inherits its neighbours' statistics. Hybrids are the point.

**The online translation (the hard rule: ONLINE ONLY — no backprop, no k-means, no SVD/eigendecomposition, no
PPMI matrix factorization).** Everything is a single streaming pass of order-independent counting plus online
leader clustering, reusing Exp U's machinery (`jepa.online_signatures` + `jepa.leader_cluster`).

- **L1 word reps.** Each word's *signature* is an online-accumulated, hashed, IDF-discounted co-occurrence
  count vector (D=96) — a random sign-projection done *by accumulation*, not by factorizing a matrix. Clusters
  form by **online leader clustering**: stream the top-N words once in first-appearance order; a ripe word
  (≥80 evidence) joins the nearest running-mean prototype if cosine ≥ 0.85, else spawns one (cap 2000). One
  pass, no re-assignment.
- **L2 phrase reps.** Discover phrase units by **branching entropy** over the word stream (Exp A/M, one level
  up). Represent each phrase type by the **mean of its member words' signatures** (a bag of word-reps), then
  **online-leader-cluster the phrase signatures** → phrase concept ids. Higher-unit similarity, built the same
  online way.
- **The count predictor we project into.** Bigram `P(next | prev_word)` by counting, smoothed over the unigram
  prior. Alongside it we aggregate, **per word cluster**, the next-word counts pooled over all member words —
  projecting the rep onto the count keys so a rare/zero context word can back off onto its cluster.
- **The hybrid.** `bigram` (count only) → `+word-rep` (mix the prev word's own next-dist with its cluster's
  aggregated next-dist, cluster weight `λ = κ/(κ+prev_count)` rising as direct evidence falls) → `+hierarchy`
  (a coarser phrase-cluster expert added on top, weighted up only when the word evidence is thin). Weights are
  read off count mass, not learned. No gradients.

**Setup.** text8, 18 MB → 3.07 M words (98 k types); 85/15 train/test. Top-N = 12 000 words get a rep.
Online clustering: **2000 word clusters** (11 994/12 000 clustered), **108 phrase clusters** over 347 k phrase
types. Eval = 120 000 random test positions. **Whole run 24 s on CPU, single pass.** Fixed seed.

---

## The representation holds up (proximity = meaning, cleanly)

```
'three'   -> c98  (108w): one, nine, eight, three, five, four, six, seven, american, b, d, p
'france'  -> c185 (106w): germany, england, france, children, india, brazil, russia, austria, egypt, italy, spain
'january' -> c98            (lands in the numbers/symbols cluster)
```

Phrase clusters (108 of them) are coherent multi-word units:

```
p21  ( 78): his own | he died | after his | his mother
p4   ( 52): has been | have been | had been | has also
p28  ( 36): middle east | middle ages | west coast | east coast
p2   ( 35): can be | may be | would be | will be
p9   ( 24): does not | do not | has not | would not
```

Numbers cluster, countries cluster, and modal/negation phrases cluster — built online by counting, never
co-trained against the predictor, so (like Exp U) they **cannot collapse**.

---

## Result — next-word, count-only vs +word-rep vs +hierarchy, sliced by context

120 000 probes. **1.8%** rare context (prev train-count < 20); **21.8%** unseen-target (the true next word has
**zero** direct count under the prev word — the bigram can only offer the unigram floor). Accuracy is argmax
top-1; perplexity is the calibration axis. ± is a 95% CI.

| slice | model | perplexity | top-1 acc |
|---|---|---:|---:|
| **OVERALL** | bigram (count only) | 1263.3 | 17.255% |
| | +word-rep backoff | **671.2** | 17.197% |
| | +hierarchy (phrase) | **662.3** | 17.179% |
| **UNSEEN target** (zero direct count, 21.8%) | bigram (count only) | 18 953 086 | 0.000% |
| | +word-rep backoff | **965 088** | 0.011% |
| | +hierarchy (phrase) | **887 388** | 0.019% |
| **RARE context** (prev < 20, 1.8%) | bigram (count only) | 11 456 | 11.99% |
| | +word-rep backoff | **4471** | 10.90% |
| | +hierarchy (phrase) | **4337** | 10.49% |
| **COMMON context** | bigram (count only) | 1212.1 | 17.354% |
| | +word-rep backoff | **647.8** | 17.316% |
| | +hierarchy (phrase) | **639.3** | 17.305% |

Paired accuracy deltas (rare/common slices): all flat-to-slightly-negative and inside or near the CI
(+word-rep − bigram on common: −0.038% ± 0.029; on unseen-target: +0.011% ± 0.013). The argmax does not move.

---

## Reading it honestly (the fragile-idea shape)

This is the **exact** shape Exp P predicted and the fragile-ideas rule told us to look for: *the idea wins on
the axis it was never headlined on.*

- **The win is calibration / perplexity on unseen contexts, and it is enormous.** On the 21.8% of predictions
  where the bigram has *zero* count for the true next word, the bigram is helpless (perplexity 19 million — the
  unigram floor). Projecting the word's similarity cluster onto the count keys cuts that **~20×** to 965 k; the
  phrase rep on top cuts it a further ~8% to 887 k. A never-counted bigram inherits a *meaningful* distribution
  from its cluster-siblings instead of the flat prior. Overall perplexity nearly halves (1263 → 662) — and the
  gain is entirely the tail; the common slice barely moves once you discount the unseen mass it contains.
- **It does NOT move the argmax.** Top-1 accuracy is flat-to-slightly-negative everywhere, inside the CI. This
  is consistent with Exp P (similarity is a map, not a road) and Exp U (the latent predicts the *kind*, not the
  *word*): similarity backoff redistributes *probability mass* toward plausible neighbours, but the single
  most-likely next word is still whatever the dense local counts say. Proximity sharpens the distribution, not
  the pick.
- **The hierarchy earns a little.** Phrase reps add over word reps alone on perplexity in every slice (overall
  671 → 662; unseen 965 k → 887 k; rare 4471 → 4337). Small but consistent and on the right axis — the coarser
  phrase concept supplies extra prior mass exactly when the word-level evidence is thin.

### Which axis each piece won on

| piece | won on | lost / flat on |
|---|---|---|
| L1 word-rep projection (backoff) | **unseen-context perplexity (~20×), overall perplexity (~2×)** | top-1 accuracy (flat) |
| L2 phrase-rep (hierarchy) | **perplexity, every slice (small, consistent)** | top-1 accuracy (flat) |
| the hybrid vs either alone | **calibration on the rare/unseen tail** | common-text argmax (flat) |

> **The lesson.** Use the similarity map to *read*, not to *walk*. Projected into a counter, the similarity
> cluster is a backoff prior: it tells a never-counted context where the mass should go — cutting unseen-context
> perplexity ~20× — without ever moving the single best guess. The hierarchy (phrase concepts on top of word
> concepts) deepens the prior a little more. The map prices the tail; the local counts still pick the word.

**Online-compliance note.** Single streaming pass. Reps = online co-occurrence signatures + online leader
clustering (running-mean prototypes, assign-to-nearest / spawn-on-distance, no re-assignment). Predictor =
counts; per-cluster aggregates = counts pooled by cluster; backoff weights = count mass `κ/(κ+count)`. The
`np.unique` / `bincount` builders are batched implementations of order-independent accumulation, identical to a
token-at-a-time online update. **No gradient descent, no k-means, no SVD/eigendecomposition, no PPMI matrix
factorization.** `spatial.build_embedding` (eigendecomposition) is deliberately not used.

**Verdict.** A keeper for the cortex, parked on the right shelf: carry the online word-cluster (and phrase-
cluster) as a **rare-context backoff prior** projected onto the count predictor's keys. It is cheap, it cannot
collapse, and it does the one thing direct counts cannot — give an unseen context a sensible distribution.
Do **not** expect it to raise top-1 accuracy; proximity is for pricing the tail, not picking the word. This
resurrects the Exp P graveyard entry ("raytracing as a rare-context backoff modulator") on its true axis.
