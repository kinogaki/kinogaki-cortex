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

- **Z — similarity hybrid** (← **P** raytracing, re-aimed onto the rare-context slice it was parked for; ← **U**
  JEPA online concept clusters as the reps; ← **S** offset-attention as the counter it pours into) → use the map to
  *read*, not to *walk*: a similarity cluster projected into the counter is a rare-context backoff prior — cuts
  unseen-context perplexity ~20× without ever changing the top guess. The map prices the tail; the counts pick the
  word → motivates **AD** (is the analogy itself in the counts?), and confirms leader-cluster pooling as the
  right shape *for backoff*.
- **AA — sleep consolidation** (← **U** counted latent cannot collapse; ← **R** bounded accumulation; ← **I/J**
  one-part-repeated, whose count tables are the memory slept over; idea ← Letta *Towards Agents that Learn*) → one
  offline pass of pruning + lossless distilling shrinks the memory 37% and *improves* rare-context bpc; a second,
  harder pass grinds specifics into generic mush (Letta's "generic and lossy", reproduced in counts). Dream once.
- **AB — calibrated confidence** (← **R** evidence/leaky accumulator → the precision weight; ← **X** the gate,
  whose tuned threshold this replaces with a knob-free `f·c`; idea ← Pei Wang / NARS) → splitting a count into
  hits/misses gives a NARS truth value `(f,c)` that calibrates for free (ECE 0.280→0.027, 10×) and, as an expert
  weight, cuts perplexity 12.4→4.3. The knob-free gate loses to a tuned threshold on clean text but is the only
  policy that beats always-open on the rare/unreliable slice → *parked there*. Confirms **the right combiner**
  thread: the truth value is the weight a product-of-experts always wanted.
- **AC — event model / discourse coherence** (← **T** ignition, whose altitude law this re-confirms by a different
  road; ← **M** boundaries, the phrase signal it beats at *topic* boundaries; ← **U** JEPA concept clusters as the
  event slots; ideas ← Zacks event segmentation, Kumar 2023 Bayesian surprise) → Bayesian surprise `KL(Pt‖Pt-1)`
  beats per-token surprisal 5.7× and branching-entropy ~120× at finding real enwik9 article boundaries (F1 0.154
  @±25; lead *grows* with scale 0.099→0.154 at 3→36 MB). Honest negative: precision-only as a hard segmenter; the
  win is KL-as-ranked-signal. The slot prior helps only on the ~1% backoff slice (+0.143 bpw) — the **Exp T law,
  re-confirmed**.
- **AD — analogy in counts** (← **Z** similarity hybrid, whose leader-clustering is the right shape for backoff and
  *wrong* for relations; ← **S** offset-attention, relations live in pair patterns; ideas ← NARS, the
  parallelogram-in-counts result PMC11493305) → analogy IS already in raw counts: 3CosAdd on raw PPMI profiles
  solves `a:b::c:?` at 56/94% (~4× baseline), no SVD/word2vec. TWO negatives: (i) online leader-clustering can't
  substitute for SVD — it *blurs* the relation axes (paris & tokyo co-cluster), flat-to-worse at every strength;
  (ii) NARS transitive induction spreads mass too broadly to beat a direct counter. The recurring **right combiner**
  gap: the representation is in the counts, but we lack a count-native combiner that sharpens without blurring.
- **AE — non-forgetting under real domain shift** (← **B** catastrophic forgetting, the test that had nothing to
  forget, now given teeth; ← **X** the gate, kin to the LIDA broadcast winner; ← **U** the specific-context clusters
  ART learns to protect; ideas ← ECAN STI/LTI, CLS, ART vigilance, LIDA broadcast) → stream Darwin→Shakespeare→Bible,
  one pass, no replay, bounded memory: the brain-inspired DUAL model forgets ~21× less (total backward +0.021 vs the
  recency cache's +0.454; Darwin flat −0.002 vs +0.341) AND has the better peak (2.22 vs 2.52). Load-bearing piece =
  ART resonance (reinforce the most-*specific* recognizing context) — looked worse-than-baseline at first, won two
  steps in (FRAGILE 4/7). Scope: a bounded-memory phenomenon, the only regime a lifelong learner lives in.
- **AF — usage-based constructions** (← **C** word concepts, the first counted concept that earned its keep; ← **M**
  boundaries, the frames an open slot abstracts over; ← **U** JEPA online categories the slot routes through; ideas
  ← Bybee, Goldberg, Tomasello usage-based grammar) → count TWO things (token + type frequency) and a flat n-gram
  becomes compositional: on held-out unseen (frame,filler) pairs the open-slot construction beats the n-gram 4.3×
  on perplexity (5405 vs 23461), higher prob on 80%; froze idioms (*such as / based on / part of*) and abstracted a
  NUMBER+UNIT slot; statistical preemption cut over-generation −39.5%. Honest: a backoff for the UNSEEN, not a
  replacement (specific count is sharper when seen).

## How the two rules were born

- **ONLINE-ONLY** ← **B** (counts beat gradients; the count substrate is inherently online) — made an explicit
  MUST after T/U/P used batch k-means/eigen as a stand-in.
- **FRAGILE IDEAS** ← **K** ("didn't pay off" was a 2 MB artifact) + **P** (killed at the bigram gate) + **D**
  (wrong axis) + **S/R/T/V** (all would've been killed on the headline metric, all won on the right axis).

## The cross-cutting threads (link posts by these too)

- **Surprise / robustness**: A → M → V → R → Y (one signal: boundaries, attention, learning, and leaning on the
  idea when the surface fails).
- **The right combiner**: D → I → S → X → AB → AD → AF (product/geometric-mean pooling, then calibrated/weighted by
  the NARS truth value AB, then the count-native combiner AD/Z still lack — one that sharpens without blurring;
  AF's open-slot head pours frame-preference into the category lexicon).
- **Scale**: F → J → N → O (data wasn't the problem; capacity + speed were).
- **Global coherence (open)**: H → K → T → X → AC (and where attention/boundaries must eventually deliver; AC's
  Bayesian surprise is the topic-boundary signal, AC's event slot the soft top-down prior).
- **Online learning / non-forgetting**: B → AA → AE (counts beat gradients and never forget; sleep tidies the
  memory once; the brain-inspired eviction policy keeps non-forgetting alive under bounded memory and real shift).
- **Representations / the map**: P → W → Z → AD → AF (the meaning-map prices the tail but doesn't pick the word; the
  analogy is in the counts; constructions make a counted concept productive).
- **Fragile ideas, earned the honest way**: P → W (the meaning-map got its fair rematch and was parked deeper with
  a reason, not killed on a headline); AB (gate parked on the rare slice), AD (induced links parked), AE (ART
  resonance won two steps in) all judged on the axis they can win.
