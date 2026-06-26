# Exp U — JEPA-style prediction in REPRESENTATION space, ONLINE — 2026-06-25

**The bet (LeCun's JEPA).** Don't reconstruct the raw input; *mask* part of it and predict the masked part's
abstract **representation** from the visible context; add **sparsity** so the representation doesn't collapse.
The promise is that rep-space prediction (a) generalizes better — especially on rare/unseen inputs where the
exact token is too sparse to count but its *class* is dense — and (b) is more *robust* to corrupted context.

**The online translation (the hard rule: ONLINE ONLY — no backprop, no k-means, no SVD/eigen, no PPMI
factorization).** Everything below is a single streaming pass of order-independent counting plus one online
leader-clustering pass.

- **Representation (the latent) = a word's concept CLUSTER, built ONLINE.** Each word's *signature* is an
  online-accumulated, hashed, IDF-discounted context-count vector (D=64): every neighbour within ±5 is mapped
  by a *fixed random sign-hash* into one of 64 buckets and added with weight `sign/log(2+count)` — a random
  projection done *by accumulation*, not by factorizing a matrix. Clusters are then formed by **online leader
  clustering**: stream the top-N words once in first-appearance order; a "ripe" word (≥40 evidence) joins the
  nearest running-mean prototype if cosine ≥ 0.75, else **spawns** a new prototype (cap 400). One pass, no
  re-assignment, no iteration-to-convergence.
- **Masked prediction.** Hide a word; build a **bidirectional ±5 offset-keyed** context (each visible
  neighbour is an expert `(signed-offset, word) → {target}`; experts pooled log-linearly over the union of
  their supports). Two heads share the *same* counted evidence: **token** (predict the masked WORD = input
  space) and **cluster** (predict the masked CLUSTER = rep space / JEPA).
- **Inter-layer sparsity.** Predict the masked cluster from a **sparse top-k code over context clusters** —
  only the k most-frequent context clusters carry signal. Sweep k ∈ {1, 3, 10, 30, dense}.

**Setup.** text8, 12 MB → 2.05 M words (79 k types); 85/15 train/test split. Top-N = 12 000 words get a
cluster (the rest are OOV and dropped as both target and context). Online clustering produced **C = 400**
clusters, all 12 000 top-words clustered, sizes min 1 / median 10 / max 2179. Eval = 60 000 masked test
positions whose target has a cluster. **Whole run 296 s on CPU, single pass.** Reproducible (fixed seeds).

---

## What the online representation looks like (grounded in counts → non-collapsing)

A few of the 400 clusters, members shown by descending train-count:

```
cluster 225 (51w): languages, atlantic, programming, pacific, argentina, os, amateur, shah, dynamic, routes, unix, chile
cluster 254 (56w): states, kingdom, amber, archaeology, bureau, goals, belt, chronicles, uniform, petroleum, geology
cluster  76 (48w): astronauts, olympic, di, micro, richmond, effectiveness, dramatically, hosts, pittsburgh, yields
cluster 376 (59w): aircraft, architecture, behavior, broad, norway, processing, input, excellent, ethical, storage
```
(Plus a clean numbers cluster `{one, zero, nine, two, … seven}` and a function-word cluster `{the, of, and, in,
a, to, is, …}` that dominates the target distribution — see the baseline caveat below.) The clusters are
*real but noisy*: locally coherent (programming/unix/os; archaeology/geology/bureau) yet polluted by the
hash-projection's collisions. They are computed once by counting and never co-trained against the predictor,
so — unlike a gradient JEPA encoder — **they cannot collapse to a trivial solution; we get rep-space targets
with zero collapse-prevention machinery (no stop-grad, no VICReg, no EMA teacher).**

---

## Result 1 — masked prediction: input space vs representation space

60 000 probes; **11.0 % rare** (train-count < 50), 89.0 % frequent.

| head | ppl | acc (overall) | acc (rare) | acc (freq) |
|---|---:|---:|---:|---:|
| **token** (input space) | 6.5e8 † | 6.70 % | **0.00 %** | 7.53 % |
| **cluster** (rep space / JEPA) | 2268.9 † | **65.37 %** | **11.20 %** | 72.04 % |
| cluster **majority-baseline** | — | 63.48 % | — | — |
| cluster **chance** (1/C) | — | 0.25 % | — | — |
| rep→token (decide via predicted cluster) | — | 6.30 % | 0.00 % | 7.08 % |

† **The perplexities are not comparable and not the headline.** The log-linear (geometric-mean) pool is a
product-of-experts: when ±5 neighbours disagree it drives the *true* class's probability toward zero, so a
single contrarian expert blows up ppl. With a 12 000-way token vocab this is catastrophic (6.5e8); with a
400-way cluster vocab it is merely bad (2269). Accuracy (arg-max, robust to this) is the trustworthy metric.

**Reading it honestly:**

1. **On its own metric, rep-space "wins" — but most of that win is the majority cluster.** 65.37 % cluster
   accuracy is only **+1.9 points over a baseline that always guesses the single biggest cluster (63.48 %)**.
   The function-word mega-cluster is the target so often that predicting "function word" is most of the score.
   The genuine, above-baseline signal from context is small.
2. **The one place rep-space clearly helps is RARE words — exactly JEPA's claim.** On rare targets, token
   accuracy is **0.00 %** (the exact rare word was never counted in this `(offset, ctxword)` pair in train),
   while cluster accuracy is **11.2 %** (44× chance). The token is too sparse to predict; its *class* is dense
   enough to. This is the rep-space-generalizes-on-the-tail effect, real but modest.
3. **Routing token decisions THROUGH the latent does not help.** "rep→token" (pick the top cluster, then the
   best token inside it) gets **6.30 %**, *below* direct token prediction (6.70 %). The cluster prior is too
   coarse/noisy to sharpen the within-cluster token choice; the latent helps you predict the *class*, not the
   *word*.

---

## Result 2 — robustness to corrupted context (JEPA claims rep-space degrades gracefully)

A fraction of visible context words are replaced by random in-vocab words; the true clean target is scored.

| context noise | tok acc | clu acc | tok ppl | clu ppl |
|---:|---:|---:|---:|---:|
| 0 %  | 6.70 % | 65.37 % | 6.5e8 | 2268.9 |
| 10 % | 6.63 % | 64.75 % | 6.3e8 | 2730.0 |
| 20 % | 6.50 % | 64.28 % | 6.0e8 | 3314.1 |

At 20 % noise token accuracy retains **97 %** of clean, cluster accuracy retains **98 %**. **The JEPA
robustness claim is only weakly supported here — both heads are about equally (and very) robust.** The reason
is structural, not representational: the bidirectional ±5 window holds ~10 context experts and they are pooled,
so corrupting 1–2 of them is outvoted regardless of whether the target is a token or a cluster. Robustness
comes from the *ensemble of context experts*, not from the rep-space target. (Contrast Exp R, where a leaky
*temporal* accumulator gave a real robustness crossover — the robustness lever is accumulation/voting, not
the choice of input-vs-rep target.)

---

## Result 3 — inter-layer sparsity: predict the masked cluster from a top-k cluster code

| k | train acc | test acc | overfit gap (train−test) | test ppl |
|---:|---:|---:|---:|---:|
| **1** | 63.97 % | **66.26 %** | −2.29 % | **6.94** |
| 3 | 60.13 % | 63.62 % | −3.50 % | 46.56 |
| 10 | 59.95 % | 63.53 % | −3.58 % | 103.69 |
| 30 | 59.95 % | 63.53 % | −3.58 % | 103.69 (= dense; window has ≤10 clusters) |
| dense | 59.95 % | 63.53 % | −3.58 % | 103.69 |

**Sparsity helps, and the optimum is the *most* sparse: k = 1.** A single-active-cluster code beats the dense
code by **+2.7 accuracy points (66.3 % vs 63.5 %)** and, dramatically, on perplexity (**6.9 vs 103.7, ~15×
sharper**). More active context clusters *dilute* the prediction: the log-linear pool of many cluster-experts
becomes diffuse, so the calibrated probability of the truth collapses (ppl explodes) and the arg-max drifts.
The "overfit gap" is negative everywhere (test slightly above train — the two probe samples differ; there is
no classic over-fitting at this data scale), so the story is **bias/sharpness, not variance**: a sparse code is
a sharper, less-interfering predictor. k ≥ 10 is identical to dense because a ±5 window rarely contains more
than ~10 distinct clusters.

---

## Verdict

**Rep-space prediction is real and grounded, but its advantage over input-space is narrow and concentrated;
sparsity is the clearer win.**

- **Does rep-space beat input-space?** *Yes on its own axis, but mostly trivially.* Cluster accuracy (65 %)
  towers over token accuracy (7 %), but ~63 of those 65 points are "guess the function-word cluster." The
  **honest, non-trivial win is on RARE words** (cluster 11 % vs token 0 %) — JEPA's tail-generalization claim
  holds, modestly. Predicting the *class* succeeds where the exact token is uncountable; **converting that
  class prediction back into a token does not help** (rep→token 6.3 % < direct 6.7 %).
- **Is rep-space more robust?** *Barely.* 98 % vs 97 % retention at 20 % context noise — both heads ride the
  same context-expert ensemble, so the robustness comes from voting, not from the target space. JEPA's
  robustness claim does not reproduce strongly in the count world.
- **Does sparsity help generalization, and at what k?** *Clearly yes, at k = 1.* Maximum sparsity is best:
  +2.7 acc and ~15× sharper ppl vs dense. The mechanism is anti-dilution, not anti-overfitting.
- **The collapse argument is the strongest part.** Because the latent is *counted, not co-trained*, it cannot
  collapse — we ran rep-space prediction with **none** of JEPA's collapse-prevention machinery and the clusters
  stayed meaningful (numbers cluster, function-word cluster, topic clusters). That is the count world's real
  structural advantage over gradient JEPA: **collapse-free rep-space targets, for free, online.** What it does
  *not* buy here is a large predictive or robustness edge over plain token counting.

**Why the modest edge (analysis).** The hash-projection signatures are noisy, so the 400 clusters are coarse;
and the target distribution is so dominated by function words that the cluster task is easy-but-shallow. A
sharper online latent (a larger/cleaner signature space, or clustering only *content* words) would likely widen
the rare-word gap — that is the natural follow-up. As built, the takeaway for the cortex is: **carry an online,
counted, collapse-free concept-cluster as a cheap auxiliary representation (it nails the rare-word *class* the
token counts miss), feed the predictor a SPARSE k=1 cluster code, but keep the exact-token prediction in input
space — routing tokens through the latent loses.**

---

## Online-compliance note (every part is online; nothing is batch-optimized)

| step | how it's computed | online? |
|---|---|---|
| word signatures | per-word running count vector over D=64 hashed, IDF-weighted context features; IDF itself a running count | **yes** — order-independent accumulation (the canonical online learner) |
| concept clusters | **leader clustering**: single pass in stream order, nearest running-mean prototype by cosine *or spawn*; no re-assignment | **yes** — one pass, no iteration-to-convergence |
| masked-prediction counts | bidirectional offset-keyed `(offset,word)→{token,cluster}` counts | **yes** — counting |
| sparse-cluster counts | `active-cluster → {target-cluster}` counts over top-k codes | **yes** — counting |

**No gradient descent, no backprop, no k-means, no SVD/eigendecomposition, no PPMI matrix factorization.** The
vectorized builders (`np.unique` / `np.bincount` / scatter-add) are batched implementations of order-independent
accumulation — the resulting counts are identical to a token-at-a-time online update. The random-projection
signature is applied *by accumulation* (a fixed sign-hash), never by factorizing a matrix. The earlier draft of
this experiment used PPMI+eigen embeddings and k-means and was **discarded** for violating the online rule;
this version replaces both with the online signature + leader-clustering pipeline above.
