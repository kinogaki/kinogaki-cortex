# Exp AM — discourse coherence as PREDICTION: a persistent SITUATION MODEL over long spans — 2026-06-26

**Lineage.** Event-model (AC) showed Bayesian-surprise (KL of the one-step belief update) *detects* topic
boundaries beautifully and the gap grows with scale — but the single persistent event-slot it drove helped
*prediction* only on the ~1% backoff slice (the words where the local n-gram had already run out). Ignition
(T) found the same altitude law for a global topic *G*. Concept clusters (U) gave us the online latent. The
open question on the **global-coherence** thread: can a persistent **situation model** lower perplexity
*generally* — beyond the backoff slice — and keep generated text on-topic over long spans?

**The bet.** AC's slot was one undifferentiated bag of active clusters. AM adds the two things the discourse
literature says a situation model actually needs, both built **online** (single streaming pass; no gradients,
no k-means, no SVD):

- **(a) Narrative-schema event chains (Chambers-Jurafsky 2008).** C-J learn "narrative event chains" by
  counting verb pairs that share a coreferring protagonist, then PMI-scoring them so the current event
  predicts the *expected next* event. With no parser and no coreference, we approximate at the cortex's
  altitude: an **event = the concept-CLUSTER of a content word**; "coreference" = **recency of a shared
  entity-ish cluster** (two events count as a chain link when the same *who*-slot is hot for both). We count
  ordered cluster-pair co-occurrences in an 8-event window, PMI-score them online, and read off
  `P(next event-cluster | current)` — a top-down prediction of the next event, spread over its member words.
- **(b) Multi-dimensional typed situation slots (Zwaan's event-indexing model).** Not one slot but a few
  **typed** persistent leaky accumulators — **who** (entity-ish clusters, fast half-life 120), **where**
  (location-ish clusters that follow locative anchors, 200), **topic** (everything, slow 2000) — each emitting
  its own cluster prior. A token's cluster loads each slot weighted by that slot's *typing* of the cluster
  (entity-ness / location-ness / 1.0). Typings are cheap online stand-ins for NER: who-ness = IDF-weighted
  mass (protagonists are rare, content-bearing); where-ness = fraction of occurrences following
  `in/at/on/near/…`.

The combined situation state is a soft **top-down prior over words**, blended into the predictor **at every
step** (not gated to backoff) — that is the whole point: AC restricted it to backoff; AM asks whether a
richer, typed, schema-driven situation earns its keep *generally*.

**Setup.** enwik9, **36 MB → 5,177,506 words** (vocab 80 k + UNK). Top-12,000 words → **C = 400** online
concept clusters (jepa signatures + leader clustering, all 12 k clustered). 352 clusters typed who-ish
(>0.5), 9 where-ish. Word trigram→bigram→unigram **count** predictor. Fixed seed 0, single pass, **400 s**.

---

## The honesty control that decides everything

A sparse trigram table renormalized with add-α smoothing is *badly calibrated on its tail*: blending it
against **any** broad word prior repairs the smoothing and lowers bits, **whether or not that prior tracks the
situation.** So the naive "with-situation vs no-situation" delta is not the test — it conflates *smoothing
repair* with *situation tracking*. The decisive baseline is a **static control prior**: the corpus unigram (a
fixed, situation-agnostic broad prior) blended in *identically*. The situation only earns its keep if the
**live** prior beats this **static** one. Every number below reports `vs-STATIC`.

---

## KEY TEST — does the situation help BEYOND the backoff slice?

Bits/word, mixed everywhere (w = blend weight) or backoff-only. The **non-backoff** column is the
beyond-backoff test. `vs-no` = with-sit − no-sit (the *naive*, misleading delta); `vs-STATIC` = static − with-sit
(the *honest* delta — positive = the live situation beats a fixed prior).

| config | slice | no-sit | static | with-sit | **vs-no** (naive) | **vs-STATIC** (honest) |
|---|---|---:|---:|---:|---:|---:|
| everywhere w=0.15 | overall | 10.783 | 10.167 | 10.235 | **+0.548** | **−0.068** |
| | **non-backoff (99.1%)** | 10.775 | 10.153 | 10.223 | — | **−0.070** |
| | backoff (0.9%) | 11.723 | 11.723 | 11.646 | — | **+0.077** |
| everywhere w=0.05 | non-backoff | 10.775 | 10.395 | 10.446 | — | **−0.051** |
| | backoff | 11.723 | 11.723 | 11.667 | — | **+0.056** |
| backoff-only w=0.30 | overall | 10.783 | 10.783 | 10.782 | +0.001 | +0.001 |
| | backoff (0.9%) | 11.723 | 11.724 | 11.664 | — | **+0.060** |

**The naive everywhere-win is a mirage.** Blending the situation prior in at every step *looks* like a huge
**+0.548 bpw** gain over no-situation. But a **static corpus unigram** blended in identically gets a *larger*
gain (10.167 vs 10.235) — the live situation is **−0.068 bpw WORSE than a fixed prior** on the non-backoff
99.1% of words. The "beyond-backoff" win is **entirely smoothing repair, none of it situation tracking.** A
lighter blend (w=0.05) is less wrong but still loses to static.

**On the backoff slice the situation is genuinely informative — and only there.** Restricted to the 0.9% of
words where local context is exhausted, the *live* situation beats the *static* unigram by **+0.060–0.077
bpw**: when the n-gram has nothing, the active who/where/topic state really does carry which word comes next.
But that slice is 0.9% of a 36 MB run, so the honest overall effect (backoff-only config) is **+0.0005 bpw** —
real, tiny, exactly AC's shape.

---

## Ablation — which dimension carries the (apparent) signal?

Non-backoff Δ vs no-sit (everywhere, w=0.15) — note these are the *naive* deltas, all of which are mostly
smoothing repair:

| component | non-backoff Δ | backoff Δ | reading |
|---|---:|---:|---|
| full | +0.552 | +0.077 | — |
| **topic-only** | **+0.682** | +0.040 | the slow background topic = the best smoothing-repair prior |
| who/where-only | +0.652 | +0.095 | who/where carries the most *backoff* signal |
| schema-only | +0.041 | **−0.027** | the C-J event chain barely moves prediction, and **hurts** on backoff |

Two honest readings. **(1)** The *topic* dimension produces the biggest non-backoff number — because the
slowest, broadest slot is the closest thing to the static unigram, i.e. it is the best *smoothing repairer*,
not the best *situation tracker*. **(2)** The **narrative-schema event chain — the headline new mechanism —
essentially does not predict the next word** (+0.04 non-backoff, **−0.03 on backoff**). The C-J PMI chain over
*content-word clusters* is too coarse and too noisy a stand-in for a parsed, coreference-linked verb chain;
without real predicate-argument structure the "expected next event" is barely better than chance and on the
sparse slice it actively misleads.

---

## Generation — topic-consistency over long spans

Topic-consistency of 4000-word samples = mean within-window pairwise cosine of generated content-cluster
embeddings (higher = stays on a few related topics).

| sampler | topic-consistency | Δ |
|---|---:|---|
| no-prior | 0.646 | — |
| **static-prior** (control) | **0.763** | +0.117 vs no-prior |
| with-situation | 0.647 | **−0.116 vs static** |

**Another honest negative.** Generation steered by the live situation is **no more coherent than unsteered**
(+0.001) and **markedly *less* coherent than steered by a fixed unigram** (−0.116). The static prior wins this
metric trivially — it pulls every step toward high-frequency function words, which sit close together in
cluster space and inflate "consistency" without saying anything. The situation prior, by *tracking* the text,
keeps moving its mass and so scores lower. The metric rewards *staying still*, and a fixed prior stays
stillest; tracking the situation is the opposite of what maximizes it.

---

## Verdict

**A richer, typed, schema-driven situation model does NOT predict over long spans any better than AC's single
slot did. The altitude law holds, the new mechanisms add nothing, and the apparent wins are artifacts.**

1. **Beyond-backoff: honest negative.** Mixed everywhere, the situation prior is **worse than a fixed corpus
   unigram** (−0.07 bpw on the 99.1% non-backoff slice). The eye-catching "+0.55 bpw beyond backoff" is
   **smoothing repair** — any broad prior earns it — not situation tracking. The control is the whole story.
2. **Backoff-only: real but tiny, exactly AC.** Where local context is exhausted (0.9% of words) the live
   situation beats a fixed prior by **+0.06 bpw**. The active state genuinely carries information *only* where
   the n-gram has none. Overall honest gain +0.0005 bpw. **AC's law reproduced through richer machinery:
   top-down priors pay off only where local prediction fails.**
3. **The narrative-schema chain failed.** Chambers-Jurafsky event chains over *content-word clusters* barely
   predict (+0.04) and *hurt* on the slice that matters (−0.03). Without a parser and coreference, "event" =
   "content-cluster" is too lossy a proxy; the PMI chain is noise. This is the clearest negative: the marquee
   mechanism did not transfer to the count substrate.
4. **Generation: situation steering does not improve coherence**, and a static prior "wins" the metric only by
   standing still — a caution that topic-consistency-by-self-similarity rewards degeneracy.

**Takeaway for the cortex.** Carry AC's lesson unchanged: a persistent top-down state is a **soft, low-weight,
backoff-only prior** — and *measure it against a static prior*, never against no-prior, or you will mistake
smoothing repair for understanding. Typed who/where/topic slots and a C-J event chain add machinery but no
predictive power at this altitude. **The situation model, as a general next-word predictor, stays a
backoff-only effect** — the global-coherence frontier is not crossed by stacking more persistent structure on
a count predictor whose 99% is already owned by local context.

---

## Online-compliance note

| step | how | online? |
|---|---|---|
| concept clusters | jepa online signatures (hashed IDF context counts) + leader clustering, single pass | **yes** |
| typed-slot typings | IDF-weighted mass (who); locative-follow fraction (where) — running counts | **yes** |
| narrative-schema chain | ordered cluster-pair co-occurrence counts in an 8-event window; PMI = closed-form transform | **yes** — counting |
| shared-protagonist weight | leaky who-histogram peakedness, one streaming pass | **yes** — leaky accumulator |
| typed situation slots | per-type leaky cluster histograms, own half-lives | **yes** — leaky accumulators |
| next-word predictor | word trigram→bigram→unigram counts | **yes** — counting |

No gradient descent, no backprop, no k-means, no SVD. The vectorized count builders are batched
order-independent accumulation (identical to a token-at-a-time update). The cluster→word prior is refreshed
every 256 words (the AC refresh trick — the slowly-drifting prior is unchanged in substance, ~8× faster).
Reproducible at fixed seed 0; the verdict is identical at the 3 MB pilot (where the backoff slice is 6.1% and
the same vs-STATIC signs hold).
