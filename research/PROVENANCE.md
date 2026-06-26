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

## How the two rules were born

- **ONLINE-ONLY** ← **B** (counts beat gradients; the count substrate is inherently online) — made an explicit
  MUST after T/U/P used batch k-means/eigen as a stand-in.
- **FRAGILE IDEAS** ← **K** ("didn't pay off" was a 2 MB artifact) + **P** (killed at the bigram gate) + **D**
  (wrong axis) + **S/R/T/V** (all would've been killed on the headline metric, all won on the right axis).

## The cross-cutting threads (link posts by these too)

- **Surprise**: A → M → V → R (one signal: boundaries + attention + learning).
- **The right combiner**: D → I → S (product/geometric-mean pooling, calibrated, weighted).
- **Scale**: F → J → N → O (data wasn't the problem; capacity + speed were).
- **Global coherence (open)**: H → K → T (and where attention/boundaries must eventually deliver).
