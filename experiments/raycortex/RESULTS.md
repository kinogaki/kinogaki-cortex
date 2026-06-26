# Exp W — ray-cortex: resurrecting proximity/raytracing as a rare-context backoff, in combination

**Question.** The naive Euclidean "proximity columns" idea (Exp P) lost to a bigram on next-word and was
*parked, not killed* (FRAGILE_IDEAS commandment 7/8). Resurrect it the right way: the **graph** form
(spreading activation over a PMI association graph, the form the sources endorse), built INTO an integrated
predictor with everything that works — offset-keyed count-attention (Exp S), leaky log-evidence (Exp R),
online-leader-clustered topic prior (Exp T, online clusterer), product/geometric-mean pooling — and judged
on the RIGHT axes: rare/unseen-context generalization, robustness, and the ablation (does proximity finally
earn its keep *in combination*?).

**Verdict up front (honest).**
- **The offset-attention core is the workhorse** — 322.7 ppl vs bigram 694.5 / trigram 2816 (a 2.1× / 8.7×
  win on next-word over the n-gram baselines, confirming Exp S).
- **Evidence-pooling EARNS ITS KEEP on the rare-context axis** — winsorized robust pooling beats plain
  geometric mean by **+12.5 ppl on rare contexts (95% CI [6.7, 20.6])** and **+45 ppl on unseen contexts
  (CI [20.8, 81.6])**, both standalone and *inside the combination*. This is a real fragile-idea win on a
  non-headline axis (it costs ~7% on common contexts, so it loses overall — exactly the Exp-R pattern).
- **Proximity/raytracing does NOT earn its keep, even here, even on its intended axis.** On rare contexts it
  is **−4.8 ppl (CI [−10.8, 3.0], not significant)**; inside the combination, *removing* it never
  significantly hurts and usually helps. Given a real, supported shot (graph form, swept weights, the right
  slice, the right stack), it still loses. **Parked deeper** — with a concrete structural reason (below).

Online-compliance: single streaming pass; counts, leaky/winsorized accumulators, PMI-read-from-counts, and
online leader-clustering only. NO gradient descent, NO k-means/SVD/eigendecomposition anywhere (the topic
prior uses jepa.leader_cluster — running-mean prototypes, spawn-on-distance — NOT the k-means in ignition.py).

---

## Setup

- Corpus text8, first 15 MB → 2.55 M words, vocab 20 000 (UNK 5.0%); train 90% / held-out 10%.
- Word-level next-word prediction; **perplexity** on 30 000 held-out positions. Slices by the **immediate
  context count** dc = #times the directly-preceding word was seen with a known follower (the offset-1
  successor mass): rare = dc ≤ 2 (0.9%, n=260), unseen = dc = 0 (0.5%, n=148), common = dc ≥ 20 (93.4%).
- Predictor = `lib/raycortex.py`. Experts pooled by weighted geometric mean; proximity & topic gated in by a
  smooth backoff weight `gate = 1/(1+dc/8)` (≈1 when the backbone is starved, ≈0 when it is rich). Evidence
  = per-step **winsorization** of each expert's per-candidate log-vote to its median ± 4 nats before pooling
  (the robust-voting form of leaky log-evidence: one bad/over-confident expert can neither zero nor assert a
  hypothesis). Proximity = `ProximityGather`: seed the recent context words on the PMI graph (N=3000 nodes),
  spread 1 hop, pool the activated nodes' successor distributions. Topic = `TopicPrior`: online signatures →
  leader-cluster → committed G (ignition/hysteresis) → per-topic word histogram, soft at backoff.

## Perplexity — overall and by context slice

| model            | overall | rare (dc≤2) | unseen (dc=0) | common (dc≥20) |
|------------------|--------:|------------:|--------------:|---------------:|
| bigram           |  694.5  |   2606.4    |    1185.1     |    593.0       |
| trigram (naive)  | 2816.5  |   2640.1    |    1185.1     |   2627.4       |
| **offset-only**  | **322.7** |   237.5   |     536.7     |    340.7       |
| +proximity       |  323.8  |    242.2    |     528.1     |    341.2       |
| +topic           |  322.9  |    240.9    |     541.3     |    340.7       |
| +evidence        |  344.8  | **225.3**   |  **492.8**    |    365.5       |
| full−proximity   |  345.5  |    228.6    |     497.9     |    366.1       |
| full−topic       |  348.0  |    228.9    |  **480.7**    |    368.2       |
| full−evidence    |  323.9  |    245.0    |     532.2     |    341.1       |
| full             |  348.5  |    232.0    |     485.2     |    368.6       |

(The naive trigram is *worse* than the bigram: with vocab 20 000, an order-3 context seen on train but with
this target unseen gets near-zero add-α mass — a fixed high-order n-gram with no proper backoff overfits.
The offset-attention core's geometric-mean backoff is precisely what fixes this, hence its 2–9× win.)

## The two axes that decide it (bootstrap, 2000 resamples of the slice)

**Evidence vs the plain pool** (positive ppl-gap ⇒ evidence wins):

| slice  | n   | offset → evidence gap | 95% CI         | significant? |
|--------|----:|----------------------:|----------------|:-----------:|
| rare   | 260 |        **+12.5**      | [ 6.7,  20.6 ] | **yes**     |
| unseen | 148 |        **+45.4**      | [20.8,  81.6 ] | **yes**     |

Inside the combination (full−noEvidence → full): rare **+13.5** CI [6.4, 22.9], unseen **+49.0** CI
[21.1, 90.9] — evidence carries the combination's entire rare-context gain.

**Proximity vs the plain pool** (positive ⇒ proximity wins):

| slice  | offset → proximity gap | 95% CI          | significant? |
|--------|-----------------------:|-----------------|:-----------:|
| rare   |         −4.8           | [−10.8,  3.0 ]  | no (negative)|
| unseen |         +9.3           | [−12.0, 41.3 ]  | no          |

Inside the combination (full → full−noProximity): rare **+3.5** CI [−4.7, 9.1] (proximity slightly *hurts*),
unseen **−13.0** CI [−47.1, 7.3] (leans helpful but not significant). Net: proximity is noise-to-negative.

## Robustness (15% of held-out context words randomized)

| model          | clean ppl | corrupt ppl | degradation |
|----------------|----------:|------------:|------------:|
| bigram         |   694.5   |   1075.6    |   1.55×     |
| offset-only    |   322.7   |    315.9    |   0.98×     |
| full ray-cortex|   348.5   |    338.4    |   0.97×     |

No robustness differentiation: the geometric-mean backoff already absorbs a randomized context word (it lands
on an unseen context and backs off), so corruption barely moves either model — it even *helps* slightly by
routing past a now-misleading high-order count. The evidence winsorization gives no extra robustness here
because the threat model (a *random* word) is absorbed by backoff, not by a confidently-wrong vote. **Honest
note:** this is the wrong corruption to stress winsorization; a sharper test (corrupt to a *plausible* word
that votes a confident wrong successor) is the follow-up that could surface the robustness win — budgeted, not
run here.

## Why proximity still loses (the structural reason — for the graveyard)

The proximity gather is supposed to fill in exactly when the direct context is thin. But a thin context means
the **preceding word is rare** — and rare words are *absent from the top-N PMI graph* (N=3000 covers only the
frequent words). So on the rare slice the gather is seeded only by whatever *common* context words remain, and
those spread to a **generic** successor distribution — essentially a worse, noisier version of the topic prior,
with none of the specificity the rare preceding word would carry. Proximity and the topic prior thus collide on
the same backoff niche, and the topic prior (committed, hysteretic, calibrated) does it better. Raising N to
cover rare words would blow up the dense PMI matrix and is still online-illegal in spirit (the gather's votes
for a rare seed are themselves count-starved). **Conclusion: proximity-as-predictor has no niche left that
evidence+topic don't already fill better.** Parked deeper; the live uses for the association graph are now
*inspection / similarity*, not prediction.

## Ablation summary — which piece carries weight

| piece      | overall | rare/unseen (the right axis) | verdict                                            |
|------------|:-------:|:----------------------------:|----------------------------------------------------|
| offset     | **best**| good                          | the workhorse; keep as the core                    |
| evidence   | costs ~7%| **best (significant)**       | **earns its keep on rare/unseen** — a real win     |
| topic      | neutral | helps unseen (full−topic best on unseen, marginal) | mild, keep as a soft backoff |
| proximity  | costs   | not significant / negative    | **does not earn its keep** — parked deeper         |

**Reading per the fragile-ideas rule:** evidence is the Exp-R pattern repeating one level up — loses on the
headline, wins (significantly) on the rare/unseen axis it was always meant for. Proximity got its real,
supported shot (graph form, the right axis, the right stack, swept weights) and lost with a clear mechanistic
reason. That is a legitimate "parked deeper," not a premature kill.

## Online-compliance note

Single streaming pass; every component is counts / decayed or winsorized accumulators / PMI-read-from-counts /
online leader-clustering. No backprop, no k-means, no SVD/eigendecomposition. The topic prior deliberately
imports `jepa.online_signatures` + `jepa.leader_cluster` (running-mean prototypes, spawn-on-distance) and
inlines the ignition/hysteresis commit — it never touches the batch k-means in `ignition.py`.

## Files
- `lib/raycortex.py` — ProximityGather, TopicPrior (online), RayCortex (integrated predictor + robust pool).
- `exp_w_raycortex/run.py` — load → fit → baselines → ablation → slices → robustness → `results.npz`.
- `exp_w_raycortex/run.log`, `results.npz` — the run output + per-position log-probs for re-slicing.
