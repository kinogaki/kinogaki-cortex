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

## The System-2 round (AG → AK)

- **AG — the deliberate pass / a count-native System 2** (← **AB** calibrated confidence, the metacognitive
  trigger "is my fast answer trustworthy?"; ← **X** the gate, surprise/confidence opening a route to a higher
  level; ← **R** leaky evidence, the accumulators the deliberate race runs over; ideas ← Kahneman default-
  interventionist, Stanovich cognitive decoupling, Botvinick conflict monitoring, Engle WM-capacity) →
  everything before this was System 1; a dual gate (calibrated `f·c` + Botvinick conflict) deploys a deliberate
  pass that OVERRIDES System 1 only when it is wrong. Engle signature clean: +0.38 acc on conflict cases (S1
  wrong 88%), 0.000 harm on no-conflict, bit-for-bit fallback at zero budget. Honest negative: the elaborate
  serial workspace (focus + IOR + decoupling) loses to a trivial "defer to wider context" (0.15 vs 0.39 where
  they differ) — the GATE is load-bearing, the workspace is *parked* for the multi-step task (AC/AD).
- **AH — representational redescription** (← **C** word concepts, the first counted concept that earned its
  keep; ← **U** JEPA online categories the slots route through; ← **AF** the implicit constructions this
  promotes; idea ← Karmiloff-Smith *Beyond Modularity*) → stability (not error) promotes a mastered
  construction into an explicit, slot-addressable concept answering queries the flat count can't (inverted slot
  lookup, role substitution, slot analogy). KS U-shaped dip confirmed (0.155 → trough 0.051 → recovered 0.181
  above baseline). Honest: the recovery is a consequence of modeled E1→E2 re-binding; the win is that the KS
  mechanism is expressible/self-consistent on counts, manipulability the real prize. Architecturally this
  provides the explicit OPERANDS the AG workspace deliberates over.
- **AI — power-law memory / budgeted eviction** (← **AE** non-forgetting, which made the budget load-bearing and
  the leaky use-score the thing to beat; ← **AA** sleep, the other place a bounded memory decides what to keep;
  ← the scaling studies, the budget-re-elevates rule; idea ← Anderson & Schooler, ACT-R rational analysis) →
  the power law (B = ln Σ tₖ⁻ᵈ) is the right SHAPE (spacing: spaced 8.96× more accessible; EMA can't represent
  it) but raw-count LFU wins eviction for dense char-grams at every cap (LFU = the d→0 limit; decay sweep
  degrades monotonically), and power-law-weighted prediction loses (+0.68 bpc). Right shape, wrong place: keep
  LFU for char-grams, reserve the power law for sparse word/concept-level retrieval.
- **AJ — take-the-best / less-is-more** (← **S** offset-attention, whose per-offset experts are the cues
  take-the-best ranks; ← **AB** calibrated confidence, whose hit/miss truth value IS the ecological validity
  that orders the cues; ideas ← Gigerenzer & Goldstein take-the-best, Simon satisficing, the bias–variance
  theorem) → validity-ordered, noncompensatory, early-stopping inference beats full geometric-mean integration
  on every axis (acc 15.00% vs 9.71%, ppl 1918 vs 7160, 4.56 vs 8 cues/step); less-is-more (α>β) confirmed on
  sparse contexts. Honest negative: a base-rate prior γ>0 on the single-pass clusterer lowers stability (γ=0
  stays). REVISES the standing combiner: validity-ordered take-the-best, not full pooling — the sharpening rule
  the *right-combiner* thread kept asking for (sharpen by ignoring, not blurring).
- **AK — memory-budget-as-curriculum** (← the scaling studies / **F**, the question of how capacity grows with
  data; ← **AE** the leaky-accumulator budget this schedules; ideas ← Elman "starting small", Vygotsky ZPD) →
  growing the budget on a schedule does NOT beat full-from-start (FULL 2.744 vs GROW 2.751, robust; fixed-small
  loses 30%). "Starting small" was a property of the gradient OPTIMIZER (which freezes early guesses), not of
  learning — a count learner can't get stuck, so no curriculum is needed, only enough final memory. ZPD overlay
  hurts (−5.8%). The bounded-memory rule stands; scheduling it is a no-op.

## The budget-binding-reasoning round (AL → AS)

- **AL — the multi-step workspace** (← **AG** the parked serial workspace, here given the multi-step task it was
  built for; ← **AH** the redescribed slot-concepts it chains; ← **AD** the count-relation it applies twice) → the
  workspace REACHES a 2-hop target (acc 1.00) where System 1 and one-step deferral score 0.00 (trapped on the
  intermediate) — reachability is real and new. Honest negative #2: on a single deterministic chain it ties a blind
  "apply-twice"; the focus/IOR machinery is for SELECTING among competing chains, which this probe lacks. Parked
  again, sharper reason; next axis named — competing candidate chains.
- **AM — the situation model** (← **AC** the event model + 1%-backoff law; ← **T** ignition, the same line for a
  global topic; ← **U** JEPA categories that stand in for events; ideas ← Chambers & Jurafsky narrative event
  chains, Zwaan event-indexing) → a persistent who/where/topic situation model does NOT predict over long spans.
  The +0.55 bpw "everywhere" win is pure smoothing repair — a STATIC frozen unigram beats the live situation by
  −0.07 bpw on the 99% non-backoff slice; the situation helps only the same 0.9% backoff slice (third mechanism to
  land there). Coherence will not yield to stacked state. Methodological law: measure a top-down prior against a
  STATIC prior, never against no prior.
- **AN — VSA / resonator decode** (← **AD** the analogy-in-counts reasoning probe; ← **Z** the sim-hybrid map; ←
  **AH** which supplies the slots this reads; ideas ← Plate HRR, Kanerva HDC, Frady/Kent/Olshausen/Sommer resonator
  networks) → role-filler decode WITH known roles = 100% (robust to 8 bound pairs, even D=512): compositional
  reading solved when structure is supplied. The blind RESONATOR (factor a product with unknown roles) FAILS at
  affordable dimension (≈0% all-F over a 4000 codebook at D ≤ 2048 — the capacity wall). Payoff: VSA-decode works
  GIVEN structure (AH supplies, AL manipulates) — don't factor blindly. Analogy probe not recovered (follow-up).
- **AO — cue-based retrieval** (← **S** offset-attention, whose *position* key this generalises to a *feature* key
  weighted by *fan*; ideas ← Lewis & Vasishth cue-based retrieval, Jaeger/Engelmann/Vasishth interference,
  Anderson fan effect) → content-addressable cue retrieval binds long-distance subject–verb agreement to a
  correct-number antecedent 99.96% vs offset-attention's 65.26% (a wall past its modal offset), and reproduces the
  human agreement-attraction interference (0.00% → 2.11% only with a recent opposite-number distractor; the fan
  divides activation). New primitive: long-distance binding by content cue + fan.
- **AP — permutation-bound n-grams + FlyHash** (← **M** phrase boundaries + the phrase-sparsity problem; ← **AF**
  the constructions whose sparsity this addresses; ← the VSA mining of AN; ideas ← Kanerva HDC, Dasgupta/Stevens/
  Navlakha FlyHash) → similar phrases pool counts (beats floored literal on 67% of unseen-phrase probes) AND
  preserve order (×2.29 under scramble vs the bag's ×1.00), but FlyHash crosstalk loses the tail (aggregate ppl
  1206 vs literal 831; poor exact memory 423 vs 2.2). Fix: use as a backoff layer under the literal table.
- **AQ — environment-as-memory** (← the bounded-memory rule; ← **AE** evict-the-tail and ← **AA** consolidate-the-
  head, the two prior coping routes this completes with externalize; idea ← Ericsson & Kintsch long-term working
  memory) → at EQUAL memory budget a bounded-internal + external store does NOT beat one bigger internal table
  (evidence fragmentation: the confident-internal path answers from the internal fragment alone); it wins only in
  the cost-asymmetric regime (cheap/big external → −0.23 bpc). Externalizing is a COST ARBITRAGE, not a better use
  of the same bytes.
- **AR — power-law memory at the word level** (← **AI** the char-gram negative + its explicit prediction; ← **AE**
  backward-retention-under-register-shift; idea ← Anderson & Schooler environmental power law) → the parked
  resurrection from AI: power-law (ACT-R) eviction BEATS LFU at the word level under non-stationarity + a tight
  budget (cap 10k −0.008, 30k −0.006 bpw; LFU re-wins once loose) — the sign FLIPPED from AI's char-gram result,
  exactly as AI predicted. Wins by serving the present, not protecting the past (LFU forgets the stale register
  *less* but predicts the live registers worse).
- **AS — what survives a budget** (← **what-survives-scale**, the unbounded capstone whose open twist this tests;
  ← **AA** consolidation and ← **AF/C** concepts and ← **T** ignition, the three "vanished" mechanisms re-run under
  a cap; ← the bounded-memory rule) → the bounded-memory rule's prediction CONFIRMED: mechanisms that vanish at
  unbounded scale RETURN under a budget. Consolidation flips −0.006 (unbounded) → +0.144 → +0.307 bpc as the cap
  tightens (a CURVE; vanishes when the budget stops binding). Lossless generalization (consolidation) wins broadly;
  lossy (concepts) only as tight as the budget forces; the topic prior stays neutral. Generalization is how a
  bounded model approximates the unbounded one.

## How the two rules were born

- **ONLINE-ONLY** ← **B** (counts beat gradients; the count substrate is inherently online) — made an explicit
  MUST after T/U/P used batch k-means/eigen as a stand-in.
- **FRAGILE IDEAS** ← **K** ("didn't pay off" was a 2 MB artifact) + **P** (killed at the bigram gate) + **D**
  (wrong axis) + **S/R/T/V** (all would've been killed on the headline metric, all won on the right axis).

## The cross-cutting threads (link posts by these too)

- **Surprise / robustness**: A → M → V → R → Y (one signal: boundaries, attention, learning, and leaning on the
  idea when the surface fails).
- **The right combiner**: D → I → S → X → AB → AD → AF → AJ → AO (product/geometric-mean pooling, then
  calibrated/weighted by the NARS truth value AB; AF's open-slot head pours frame-preference into the category
  lexicon; AJ answers the sharpening question — validity-ordered take-the-best beats full pooling by *ignoring* the
  weak cues; AO generalises S's *position* key to a *feature* key weighted by *fan* — reaching back by the right cue
  is a strictly better long-distance combiner than reaching back by a fixed offset).
- **Long-distance binding (new)**: S → AO (offset-attention's fixed position key reaches a fixed window and decays
  with distance; AO replaces it with content-addressable cue retrieval — feature bundle + fan — binding a verb to
  its correct-number subject across a clause 99.96% vs 65%, and reproducing the human agreement-attraction
  interference because activation divides by the fan).
- **System 2 / the reasoning route (new)**: AB → X → AG, with AH supplying the operands, AN reading them, AL moving
  them (AB asks *is the fast answer calibrated?*; X is the gate routing higher; AG is the metacognitive gate that
  overrides System 1 only when wrong; AH redescribes implicit counts into explicit, manipulable concepts; AN's
  VSA-decode reads role-filler structure back out of a sum *given* the slots — 100%, but the blind resonator fails,
  so don't factor blindly; AL's serial workspace manipulates those bundles and reaches a multi-hop target one-step
  deferral can't). The route is **AH → AN → AL** (supply structure → decode → workspace). Open: AL's
  focus/inhibition is for *selecting among competing chains* — the named next probe is competing-chains selection.
- **Scale**: F → J → N → O (data wasn't the problem; capacity + speed were).
- **Global coherence (open)**: H → K → T → X → AC → AM (and where attention/boundaries must eventually deliver; AC's
  Bayesian surprise is the topic-boundary signal, AC's event slot the soft top-down prior; AM is the third
  persistent-state mechanism to land on the same 0.9% backoff slice — coherence will NOT yield to stacked state;
  whatever crosses it must change the 99%, not rescue the 1%. Methodological law from AM: measure a top-down prior
  against a *static* prior, never against no prior).
- **Online learning / non-forgetting / memory budget**: B → AA → AE → AI → AK → AR → AQ → AS (counts beat gradients
  and never forget; sleep tidies the memory once; the brain-inspired eviction policy keeps non-forgetting alive
  under bounded memory and real shift; AI fixes the eviction *shape* — LFU for char-grams, the power law for sparse
  levels; AK shows the budget needs no curriculum; AR confirms AI's prediction — power-law eviction *wins* at the
  word level under non-stationarity, the sign flipped; AQ shows environment-as-memory is a *cost arbitrage*, not a
  free lunch at equal budget; AS confirms the bounded-memory rule outright — under a budget the vanished mechanisms
  return, consolidation flipping −0.006 → +0.307 bpc as the cap tightens, a measured curve).
- **Representations / the map / VSA**: P → W → Z → AD → AF → AH → AN → AP (the meaning-map prices the tail but
  doesn't pick the word; the analogy is in the counts; constructions make a counted concept productive; AH
  redescribes a mastered construction into an explicit slot-addressable concept; AN reads those role-filler
  structures back out of a sum given the roles; AP's permutation-bound FlyHash addresses pool similar phrases and
  keep order, but the addressing blurs the tail — use it as a backoff layer under the literal table).
- **Fragile ideas, earned the honest way**: P → W (the meaning-map got its fair rematch and was parked deeper with
  a reason, not killed on a headline); AB (gate parked on the rare slice), AD (induced links parked), AE (ART
  resonance won two steps in), AL (the workspace parked again, with a sharper reason — it's for selection, not a
  single chain), AP (FlyHash parked as a backoff layer, not killed) all judged on the axis they can win.
