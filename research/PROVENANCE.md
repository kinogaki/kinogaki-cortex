# Provenance — how each experiment led to the next

The honest lineage of kinogaki-cortex: most experiments exist because an earlier one *failed in a way that
suggested them*, or *succeeded and raised the next question*. Edges are "A → B because …". For the blog: each
post should link **grew from** (its parents) and **led to** (its children).

## The spine (A → V)

- **A — branching-entropy boundaries** (words recoverable, F1 0.775; Bayesian surprise failed at char level)
  → **M** (same signal one level up → phrases), **V** (change/trajectory boundaries), and the whole *surprise*
  through-line. Bayesian-surprise-for-themes deferred to the topic level.
- **B — catastrophic forgetting** (negative: English-only has nothing to forget) → the load-bearing accident:
  *the count model beat the gradient nets* → **the substrate is associative counts, not SGD** → grounds the
  later ONLINE-ONLY rule and everything downstream.
- **C — word concepts help char prediction** (+22%) → **E** (does it compound higher?).
- **D — voting / product-of-experts** (negative on char-bpc: it saturates) → two lessons: *product-of-experts is
  the right combiner, not linear voting*, and *bits-per-char is the wrong axis for higher levels* → directly
  seeds **I**'s pooling and the later FRAGILE-IDEAS "judge on the right axis" rule.
- **E — word-level compounding** (perplexity 476→247) → confirms the hierarchy idea → **I**.
- **F — scaling, data × capacity** (not saturated; optimal cortex grows with data) → "scale pays" → **J**, **N**.
- **G — metric suite** → **H**.
- **H — concepts on the metrics** (each level buys its own axis; *global coherence is the frontier*) → names the
  target that motivates **T** (ignition) and the whole attention/boundary push.
- **I — the uniform Column** (one part, wired bigger) → **J** (is it a good, fast base?).
- **J — vectorized backend** (same model, 9×+, scale law now reachable) → unlocks **K**, **N**, **O**.
- **K — 3/4/5 levels at scale** (more data helps a lot; *4th local level flat; topic cache a constant, not a
  scaling, win*) → THE FORK: more fixed local levels won't get us there; we need something *beyond local
  context* → motivates **L** (attention), **M** (boundaries), and the source mining.
- **N — gigabyte scaling** + **O — GPU/Metal** (≈50×) → the fast, well-fed substrate the rest rides on.
- **P — raytracing/proximity** (negative: proximity ≠ a next-word predictor) → confirmed the sources' warning;
  *parked, not killed* → resurrect as a rare-context backoff modulator.
- **Source mining** (10 readers over all 167 TBP transcripts + papers/brain/gofai/grounding/forums →
  `IDEAS_FROM_SOURCES_V2.md`) → the consensus build queue → **R, S, T, U**.
- **R — evidence accumulation** (mining idea #1) → robust voting under noise (won on that axis); evidence-as-
  boundary *parked* (lost to A's entropy-rise — resurrect fused, or at word level).
- **S — offset count-attention** (mining idea #2) → the principled, count-based, online form of the attention
  **L** was reaching for (3× perplexity win; provably not bag-of-words).
- **T — top-down prior / ignition** (mining idea #3; ← H's frontier) → helps exactly on the word-backoff slice.
- **U — JEPA-style** (user request + the sparsity/representation ideas) → online leader-clustering, rep-space
  prediction + sparsity sweep.
- **V — trajectory/change memory** (← the *Trajectory Memory* talk y3zBQoueYDg; ← A's boundary signal) → change
  models transfer across content; trajectories are directional; affordances prime onsets.
- **W — ray-cortex / fair rematch** (← **P** proximity, parked; ← **R** evidence; ← **T** ignition; ← **S** offset-
  attention) → the fair rematch P earned: proximity in its graph form, inside the best stack, on the rare-context
  slice, swept weights. Evidence *earns its keep* on rare/unseen (significant, +12.5 ppl rare); proximity still has
  no prediction niche (rare gap −4.8, not significant; rare preceding word absent from the PMI graph) → *parked
  deeper*, live use = inspection/similarity, not prediction.
- **X — heterogeneous specialized stack** (← **I/J** one-part-repeated, the opposite axis; ← **T** ignition, the slow
  topic level; ← **S** offset-attention, the word level; ← source mining) → specialization-by-level *loses* on bpc
  (uniform 1.985 vs full 2.369); each piece wins on its own axis (word calibration, 8000 phrase chunks, 143 online
  topics); the one clean win is the **gate** — dynamic confidence routing ≫ static pool (~0.9 bpc). When parts are
  unequal, the arbiter is load-bearing.
- **Y — noise forces concept-reliance** (← **R** leaky-evidence char pooling; ← **X** the gated four-level stack) →
  perception-time noise on the INPUT only; the stack degrades ~2.7× slower than a flat bigram, and the gate routes
  prediction mass from letters to concepts (86%→95% as p 0→0.3) with no noise signal given. Two parked negatives:
  noisy *training* hurt clean rare-context accuracy (count tables don't overfit like nets → fix is consistency-
  counting); second-level word→topic takeover is trending but hasn't crossed over by q≈0.3.

## How the two rules were born

- **ONLINE-ONLY** ← **B** (counts beat gradients; the count substrate is inherently online) — made an explicit
  MUST after T/U/P used batch k-means/eigen as a stand-in.
- **FRAGILE IDEAS** ← **K** ("didn't pay off" was a 2 MB artifact) + **P** (killed at the bigram gate) + **D**
  (wrong axis) + **S/R/T/V** (all would've been killed on the headline metric, all won on the right axis).

## The cross-cutting threads (link posts by these too)

- **Surprise / robustness**: A → M → V → R → Y (one signal: boundaries, attention, learning, and leaning on the
  idea when the surface fails).
- **The right combiner**: D → I → S → X (product/geometric-mean pooling, calibrated, weighted; X's gate is the
  arbiter when the parts are unequal).
- **Scale**: F → J → N → O (data wasn't the problem; capacity + speed were).
- **Global coherence (open)**: H → K → T → X (and where attention/boundaries must eventually deliver).
- **Fragile ideas, earned the honest way**: P → W (the meaning-map got its fair rematch and was parked deeper with
  a reason, not killed on a headline).
