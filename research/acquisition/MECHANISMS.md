# MECHANISMS — the surviving count-native organs

Every mechanism that survived the three adversarial lenses (rule-compliance, reinvention,
evidence-honesty), with the **revise** fixes folded in and the **cuts/merges** recorded at
the bottom. Grouped **acquisition** vs **generation** (several inform both — marked).

Each entry carries:
- **Rules honored** — online-only / bounded-memory / fragile-ideas / cognition-as-guide.
- **Refines/extends** — the experiment id(s) it builds on (see [PROVENANCE](../PROVENANCE.md)).
- **Novelty** — honest: NEW, refines-X, or reinvents-X-on-a-new-axis.
- **Experiment** — corpus / metric / baseline / kill-condition.

The cog-sci behind each is in [SCIENCE.md](SCIENCE.md); the build order is in
[BUILD_QUEUE.md](BUILD_QUEUE.md); the developmental rung is in [CURRICULUM.md](CURRICULUM.md).

> **Reconciliation note.** Three proposals were **cut** (folded elsewhere) and two
> production proposals **merged** into one organ. The count of cuts is recorded at the end.
> Where the evidence-honesty lens demanded it, an overstated claim has been softened *in
> place* (e.g. PMI "approximates" ME rather than gives it "for free").

---

# ACQUISITION

## M1 — Chunk lexicon with sub-unit interference (the PARSER/Isbilen organ)
**The headline mechanism.** Add a `ChunkColumn`: a count table over **variable-length**
candidate units (not fixed-order n-grams). Each `observe()` greedily **covers** the buffer
with the highest-weight chunks (longest-confident-first = AJ take-the-best). When two adjacent
covering chunks recur together, **mint their concatenation** (leader-clustering spawn-on-novelty).
The decisive twist: as a longer chunk's weight grows, **leak weight from its constituent
sub-chunks** (a leaky subtraction, Exp R shape) — the model commits to whole units and the
sub-unit transition decays (Isbilen splice result). Bounded by LFU eviction (Exp AI). The
committed chunks become `act()`'s emission vocabulary.

- **Rules:** online (additive + leaky subtraction + leader spawn), bounded (LFU + sub-unit
  decay frees memory), no-backprop, cognition-as-guide (chunk-over-transition).
- **Refines/extends:** refines **AF** (adds the sub-unit decay AF lacks); reinvents
  `cortex.branch_chunk`'s output as a *stored lexicon*. Decay rule is **NEW** to the project.
- **Novelty:** NEW (the sub-unit-interference decay).
- **Experiment.** *Corpus:* (1) Saffran 1996 frequency-matched syllable stream (clean
  kill-test); (2) space-stripped Pride & Prejudice (Exp A corpus); (3) harness CorpusEnv.
  *Metric:* the **splice test** (train language-1, splice a learned ABC into BCA, measure
  recognition of the preserved B–C transition); plus boundary F1 vs A's 0.775; plus chunk-agent
  held-out bpc; plus type/token of emitted chunks. *Baseline:* pure forward-TP-dip segmenter
  (Saffran null) AND A's entropy-rise AND fixed-order n-gram agent. *Kill:* sub-unit B–C does
  NOT decay below pure-TP **and** chunk-agent generation/bpc does not beat the n-gram agent —
  give it the FRAGILE budget (≥10 variations of decay rate + cover policy); first weak result
  is **expected**. → queue **AU**.

## M2 — Cross-situational word→referent learning (dual-variant, in a scene-bearing env)
Make the harness meaning seam real. Give the environment's `step` a **scene** (a small set of
co-present referent-ids the agent did **not** get from the token stream — `Turn.signal` carries
them). Ship **both** variants the literature is unresolved between:
- **(A) dense associative** — a word×referent co-occurrence count matrix; score a mapping by
  PMI-like `c(w,o)/(c(w)c(o))`. This gives a competition pressure that **approximates** mutual
  exclusivity (it does *not* inherit the Bayesian ME guarantee — measure it, don't assume it).
- **(B) bounded single-slot propose-but-verify** (Trueswell) — ONE referent guess per word +
  a confidence counter; if the guess is present, increment; if absent, decrement and on zero
  re-propose from currently-present **unbound** objects. The memory-budget-honoring variant.

The learned map gives `act()` a reason to say a word: the referent it wants — the first
grounding of generation. **(Absorbs the former "referent-binding Column / SceneEnv" proposal —
its `SceneEnv` + entity-id store + Pursuit single-winner slot + McMurray S-curve check are
variant B here.)**

- **Rules:** online (counts / leaky single-slot), bounded (B natively; A capped + LFU — the
  contrast *is* the equal-bytes experiment), no-backprop, cognition-as-guide (propose-but-verify).
- **Refines/extends:** extends **AT** (the scene-bearing env); reinvents **AB**+**AJ** on a new
  (word×scene-entity) axis; ME-by-competition reuses **AO** fan.
- **Novelty:** NEW (the binding layer / second co-occurrence axis the spine never had).
- **Experiment.** *Corpus:* Yu & Smith scenes (ambiguous word+object sets) at controlled
  referential uncertainty 1..N through the scene env; later a Haiku responder that names a
  referent. *Metric:* mapping accuracy after N scenes; ME rate; the propose-but-verify signature
  (**at-chance after a disconfirmed trial**); McMurray vocabulary S-curve. *Baseline:* dense A
  vs bounded B at **equal memory budget** (AQ-style equal-bytes); random mapping; full word×referent
  table as the rejected strawman. *Kill:* neither variant reaches above-chance from co-occurrence
  alone, OR B fails the at-chance-after-disconfirm signature (then it's a cheap heuristic, not the
  human mechanism) — keep both on the FRAGILE budget; pick by which improves harness grounding,
  not raw accuracy. → queue **AV**.

## M3 — Reliability-gated boundary detectors → the head-final drift (scoped down)
Run three boundary detectors online: forward-TP-dip, backward-TP-dip (Pelucchi), and A's
branching-entropy rise. Each carries an **AB** hit/miss truth value (f,c) tallying how often its
boundary coincided with an eventually-**stable** chunk (from M1). Combine via **AJ** take-the-best.
**Scoped (reinvention fix):** the combiner is AJ+AB reused and Exp D showed flat voting saturates
— so the *deliverable* is **one** falsifiable claim, the developmental **forward→backward-TP
drift on a head-final corpus** (Japanese/Korean romanized). Gate a nonadjacent (a_X_b, Exp S)
detector ON only when middle-slot entropy is high and adjacent TP low (Gómez variability).

- **Rules:** online (histograms + reliability tallies), bounded (O(#detectors)), no-backprop,
  fragile (judged on the drift axis it can win, not English F1 which is A/AF territory).
- **Refines/extends:** refines **A**, **S**; combiner is **AJ**+**AB** (honestly reused).
- **Novelty:** the per-detector boundary-reliability tally + the head-final drift it produces.
- **Experiment.** *Corpus:* Saffran frequency-matched stream + a head-final sample. *Metric:*
  does the reliability gate shift weight toward backward-TP on head-final text (the drift)? —
  with frequency-vs-TP discrimination as a secondary check, **not** a kill axis. *Baseline:*
  each single detector + flat linear vote (Exp D, which saturated). *Kill:* the gate does NOT
  show the forward→backward drift on head-final text after the FRAGILE budget. *(Evidence-honesty
  caveat: the ~13mo drift is thin/contested — a clean negative is acceptable.)* → queue **AZ**.

## M4 — Identity / repetition channel (the replicable residue of "rule learning")
Marcus 1999 algebraic ABA/ABB rule-learning **failed a 4-lab ManyBabies replication** → do NOT
build a variable-binding engine. The robust residue is identity/repetition. Add a cheap count
feature "is `token_i == token_j` at offset k" (`c(repeat@1)`, `c(repeat@2)`) as one expert
channel into M1's vote — reduplication sensitivity without the symbolic claim.

- **Rules:** online (one counter/offset), bounded (trivial), no-backprop, fragile
  (graveyard-friendly, allowed to lose at the gate).
- **Refines/extends:** stands alone; explicitly NOT the parked variable-binding engine
  (**AN** blind resonator failed; **AH** supplies structure instead).
- **Novelty:** NEW (small, deliberately scoped).
- **Experiment.** *Corpus:* Marcus ABA/ABB synthetic + naturalistic reduplication
  ("bye-bye", "no-no"). *Metric:* does the repeat-channel improve chunk boundary placement /
  next-token prediction around reduplicated spans, and generalize the pattern to novel tokens
  **without** claiming abstract rules? *Baseline:* M1 without the repeat channel. *Kill:* never
  improves any axis after the FRAGILE budget — **park in the graveyard with the step it died at,
  do not delete** (may revive with morphology: reduplication is morphological).

## M5 — Streaming-association slot strength (ΔP / PPMI) as the construction substrate
Replace **AF**'s raw token/type slot strength with a streaming **association** score per
(filler f, slot s): keep four marginals `c(f,s), c(f,·), c(·,s), N`; derive
`ΔP = P(s|f) − P(s|¬f)` and `PPMI = max(0, log(c(f,s)·N / (c(f,·)·c(·,s))))` on demand. Use
association — not raw frequency — to rank prototype fillers (the Casenhiser-Goldberg skewed-input
anchor = argmax-association), decide productivity (Baayen P = hapax/token), and set the preemption
veto. This is the one dial AF left on the table; raw co-occurrence over-generates (the 2025
"LLMs learn constructions humans don't know" warning).

- **Rules:** online (four additive marginals, closed-form on demand), bounded (association
  *prunes* — zero/negative-PPMI fillers drop, table shrinks), no-backprop.
- **Refines/extends:** refines **AF** (swaps scoring from frequency to association).
- **Novelty:** refines AF (association was never the slot substrate).
- **Experiment.** *Corpus:* text8 (AF's pipeline); held-out unseen (frame,filler) pairs;
  CHILDES/BabyLM CDS subset. *Metric:* compositional perplexity on unseen pairs (assoc-ranked vs
  raw-count); over-generation mass on weak-competitor links vs AF's commitment-ratio veto;
  centrality agreement vs a collostruction gold list. *Baseline:* AF raw token/type + commitment-
  ratio preemption at equal memory. *Kill:* association does not lower over-generation below AF's
  −39.5% AND does not beat raw-count on held-out perplexity at equal memory — after 10–20 nurture
  steps checking **both** dials — park as "raw counts suffice for English text", not killed.
  → queue **AW**.

## M6 — Two-sided frequent frames + cross-anchor merge (verb-islands → schemas)
AF's frame is a one-sided left bracket ("X ___"); build Mintz's two-sided frame
key = `(w_left, w_right)` → distribution over the **middle** word (an emergent category). Then
the verb-island→schema step: per anchor keep a random-projected slot-signature; online
leader-cluster anchors by signature cosine; when two anchors' fillers distribute alike, **merge**
their islands into one schema in **AH**'s registry. Gate merge on type-count (productivity), keep
frozen on token-count (entrenchment). Honest caveat: frames are weak in morphologically-rich /
free-word-order languages — combine with AF's one-sided frame and let entropy decide. *Quote the
91–98% target only for top-k frames on CDS; treat the German free-word-order result as a
first-class outcome, not a control to discount.*

- **Rules:** online (bracket counters + leader-clustering + random projection), bounded
  (merging N island tables → one schema **is** the memory win), no-backprop. Uses leader-clustering
  for **similarity only** (heeds **AD**: it blurs relation axes — never for relation directions).
- **Refines/extends:** refines **AF** (category bootstrap), extends **AH** (registry).
- **Novelty:** NEW (two-sided frame + cross-anchor merge).
- **Experiment.** *Corpus:* CHILDES/BabyLM CDS primary; text8 secondary; a German subset as the
  free-word-order outcome. *Metric:* emergent-category purity (top-k frames, Mintz ~91–98% on CDS);
  the abstraction trajectory (filler-type-entropy per anchor over exposures — only measurable on
  CDS/exposure-order, **not** text8); cross-anchor transfer after merge. *Baseline:* AF one-sided
  + U context-signature categories, no merge. *Kill:* two-sided does not beat one-sided on CDS
  purity AND merge yields no transfer (after 10–20 steps; check German **separately**) — park the
  two-sided frame, keep the merge if transfer alone holds.

## M7 — Function-word anchor voter (free top-k frequency bootstrap)
Rank tokens by raw leaky count; the top band (~20 tokens) **is** the closed-class set, no labels.
Per anchor, keep right/left-neighbor category-tally counts. At predict time, "follows
determiner-anchor" is one cue with counted validity `v = hits/(hits+misses)`, fed into **AJ**
take-the-best alongside the S/AF frame cue. The cheapest possible category bootstrap — a frequency
threshold plus adjacency counts. As a generation guardrail: after emitting an anchor, its
right-neighbor category sharpens the next-token distribution.

- **Rules:** online (top-k + adjacency counts), bounded (~20 anchors), no-backprop. Informs **both**.
- **Refines/extends:** substrate S/AF/AJ; the anchor cue itself is new.
- **Novelty:** NEW (we have never mined raw frequency **rank** as a category signal).
- **Experiment.** *Corpus:* text8 (English) + a German slice (the honest-negative: anchor should
  degrade). *Metric:* POS-cluster purity of anchor-voter vs AF frame-voter vs the two combined
  under AJ; next-token perplexity where the anchor fires. *Baseline:* AF frame-category voter
  alone. *Kill:* anchor adds <2% purity over AF on English AND degrades the AJ-combined result on
  either language — but per FRAGILE confirm it isn't just mis-validitied before killing.
  → queue **AX**.

## M8 — Seeded single-pass label propagation (semantic-seed bootstrap)
Hand-seed a handful of words (object→noun, action→verb) — the only "innate" injection the stack
considers (Cassani). For a novel target, vote its category as the most-frequent **same-context**
seed/known word; propagate the label into **AH**'s registry. **Fixes applied:** strictly **one
forward pass, no re-propagation** (each token labeled once at first confident encounter; never
revisited); use a sharp nearest-**context-counter**, **not** leader-cluster centroids (AD showed
those blur paris/tokyo); gate every propagation behind **AB** high-confidence and **abstain**
otherwise so a wrong early label cannot cascade. The defensible claim is the **precision/coverage
tradeoff** of seeding, not propagation accuracy per se.

- **Rules:** online (single pass, no convergence loop), bounded (seeds fixed; abstention caps
  growth), no-backprop, fragile (drift is the primary risk axis).
- **Refines/extends:** refines **AH** (adds seed-injection + propagation policy).
- **Novelty:** the seed-injection idea is new to the project.
- **Experiment.** *Corpus:* text8 + held-out low-frequency targets with gold POS (UD tags).
  *Metric:* noun/verb precision at categorize-or-abstain (Cassani ~90% noun, ~80% verb),
  **reported with coverage** (abstention rate). Report **drift** (wrong early label cascading) as
  the primary risk axis. *Baseline:* unseeded AF leader-clusters by best-cluster-to-POS alignment.
  *Kill:* cannot beat ~70% noun precision at any non-trivial coverage, OR drift drops precision
  below the unseeded baseline — check the abstention dial before killing.

## M9 — Dual order-free / order-sensitive count routing (noun-vs-verb asymmetry)
**The sharpest falsifiable claim in the set.** Maintain TWO parallel reps per word: (a) a
bag-of-context skip-gram vector (order-**ignoring**) — the noun-like signal; (b) an ordered
argument-frame signature from **S** (offset-sensitive) — the verb-like signal. At prediction,
weight the two voters by the **entropy** of each rep (the more-peaked wins). The noun/verb split
falls out of which rep is sharper — no POS labels. Directly tested by the 2025 ablations:
shuffle-order should hurt verb-routed words; replace-context should hurt noun-routed words.

- **Rules:** online (two additive tables + per-lookup entropy), bounded (same word cap, AR
  power-law eviction), no-backprop. Informs **both**.
- **Refines/extends:** refines **S**/**AF**/**U** (each supplies only one half).
- **Novelty:** the dual-rep-plus-routing and the dissociation test.
- **Experiment.** *Corpus:* CHILDES (matches the cited human result) AND text8, with
  order-shuffle and context-replace ablations. *Metric:* per-class (noun vs verb, gold UD)
  accuracy under intact / shuffled / replaced — the **signature** is the *dissociation* (shuffle
  hurts verbs more, replace hurts nouns more). *Baseline:* a single unified skip-gram voter
  (dissociation must be absent without dual reps). *Kill:* no dissociation appears under any
  entropy-weighting.

## M10 — Cross-layer agreement-gated validity update (renamed from "joint inference")
Run three count layers concurrently (boundary = A; category = M7 anchor + AF frame; verb-frame =
S) voting into a shared per-token category belief. When two layers **agree**, give each a small
Hebbian bonus count to its hit tally. **Fixes applied:** renamed from "joint inference" (category
inflation); add a guardrail against agreement-amplified error — **cap the co-increment** and
require the agreeing cues to be conditionally independent (two correlated cues must not reinforce
a shared error). The bootstrap order is **not** committed (emerges from which counts saturate,
per AK). Always measure against a **static** single-layer prior (AM's law).

- **Rules:** online (concurrent counters + capped bonus), bounded (one leaky accumulator/cue),
  no-backprop. Informs **both**. Honestly: **collapses to AJ if the co-increment is removed**.
- **Refines/extends:** modifies **AJ** validities (AJ ranks fixed validities; this updates them).
- **Novelty:** the agreement-driven validity update (narrow; see honesty note).
- **Experiment.** *Corpus:* text8 / CHILDES, three layers live in AT's `run()`. *Metric:*
  cluster purity + held-out bpc vs each layer alone vs **fixed-validity AJ** AND a **static
  single-layer prior**; time-to-saturation per layer. *Baseline:* fixed-validity AJ over the same
  three cues + the static prior. *Kill:* does not beat **fixed-validity AJ** on the **non-backoff
  (99%) slice**, OR the gain is pure smoothing (a static prior matches it on the non-backoff
  slice). If it only helps the 1% backoff slice it has landed where AC/AM/T already are — **cut**.

## M11 — Mutual exclusivity as fan-divided novelty reservation (cue-gated)
ME-by-competition as **AO**'s fan over **entities**, gated by a contrast cue:
`novelty_score(e) = (c(w,e)+α)·(1 − max_other P(e|known-word))` — strength minus what e is already
owed. A well-named entity divides its activation by **fan** (#word-owners), so a novel word routes
to the **unclaimed** entity. Good-Turing held-out mass **approximates** the ME bias (a smoothing
choice, not an identity). **Gate** the inhibition by a focus/contrast count (Brody — *contested*,
so test whether gating helps, don't assume it).

- **Rules:** online (counts + a smoothing constant), bounded (AO store + per-word top-k),
  no-backprop. Informs acquisition.
- **Refines/extends:** refines **AO** (fan over entities) + the cue-gate + novelty-mass smoothing.
- **Novelty:** new use of AO + new smoothing role.
- **Experiment.** *Corpus:* novel-name-nameless-category probes over the scene env (known-named +
  novel entity; novel word with/without a focus flag); a Gandhi & Lake ME battery. *Metric:* %
  novel-word binds to the unclaimed entity (ME rate), focus-on vs focus-off; held-out-mass
  calibration. *Baseline:* plain softmax over present entities (predicted to spread mass, the
  deep-net negative); always-on ME. *Kill:* fan + held-out mass doesn't lift ME above
  plain-normalization after the budget, OR the gate makes no difference — park the gate (may need
  a real prosody proxy).

## M12 — Shape-bias meta-counter (second-order dimension weighting)
A meta-accumulator learning **which** feature dimension predicts shared-label, then weighting
similarity by it (the learned shape bias, emerges after ~50 nouns). Per category, count per
dimension how constant its bucket stays across same-label exemplars; accumulate a global leaky
tally of which dimension predicts co-labeling; at generalization, weight similarity by these
counts. **Honest scope (evidence fix):** without a perceptual channel a text/hashed proxy has no
real "shape" — so this tests **dimension-weighting machinery, not the shape bias per se**. Gate
on a grounded env with real perceptual features; otherwise label **inspection-only**.

- **Rules:** online (count bucket agreement + leaky tally), bounded (O(dimensions), reuses Z
  clusters), no-backprop. Informs both.
- **Refines/extends:** reuses **U**/**Z** clustering; the dimension-weight meta-counter is new.
- **Novelty:** NEW (second-order accumulator).
- **Experiment.** *Corpus:* novel-noun-generalization battery over the scene env; vary prior
  vocab size (10 vs 50+). *Metric:* % shape-based extensions vs vocab size — expect near-zero
  early, **rising past ~50 nouns** (the emergence curve is the test, not absolute accuracy).
  *Baseline:* unweighted similarity; fixed innate shape weight. *Kill:* never separates shape
  above unweighted similarity after the budget, OR buckets can't isolate a shape proxy — park as
  an inspection tool.

## M13 — Two-rate fast-map slot with spacing-sensitive consolidation
Two-tier memory: confidence-dependent decay (low-c decays fast = fragile fast-map; decay slows as
confidence/visits cross thresholds = consolidated). Spacing helps **mechanistically**: decay
between exposures forces re-retrieval; a successful effortful re-retrieval credits more. The
**AA** sleep pass evaporates unconsolidated fast-maps. **Scope fix:** the two-store machinery
**reuses AE verbatim** (don't rebuild CLS); the one thing AE/AA/AI did not test is the
**massed-vs-spaced retention dissociation at the word→referent (sparse) grain** — lean on **AR**
(power-law wins there), not AI.

- **Rules:** online (state-dependent decay + confirm-credit; one offline sleep pass, AA),
  bounded (**is** the coping mechanism — evicts unconsolidated maps), no-backprop. Informs both.
- **Refines/extends:** refines **AA** + **AE** + **AR**; new = confidence-dependent decay +
  spacing/effortful-retrieval credit.
- **Novelty:** the spacing-sensitive consolidation at the word-referent grain.
- **Experiment.** *Corpus:* massed vs spaced exposure of the same novel (word,referent) pairs over
  the scene env; delayed retention probe after intervening scenes / one sleep pass. *Metric:*
  retention at delay, massed vs spaced. *Baseline:* single-rate constant-decay slot (no spacing
  effect); no-sleep. *Kill:* spaced and massed retain equally after the budget (decay does
  nothing) — revert to single-rate Pursuit (M2-B).

## M14 — Recency + context-diversity accumulator (graduation, not raw frequency)
Replace the plain count in the acquisition path with a leaky integrator that (a) rewards
**spaced** re-encounters over bursty repetition (ACT-R spacing, **AI**), and (b) carries a
context-diversity tally. A token **graduates** (becomes slot-addressable, eligible to lead an AF
construction) when its **AB** (f,c) over **diverse** contexts clears confidence. Penalize learning
of tokens in high-entropy/long utterances. The graduation timeline **is** a count-native
age-of-acquisition curve. **Fix applied:** context-diversity must be a **bounded approximate
distinct-counter** (HyperLogLog / small sketch), never an exact per-context set; evict ungraduated
rows under AI/AR.

- **Rules:** online (leaky lazy decay + diversity sketch + AB), bounded (sketch + evictable rows),
  no-backprop, fragile (judged on AoA correlation even if bpc is flat). Informs acquisition.
- **Refines/extends:** refines **AJ**/**AB** (applied to acquisition order, not inference).
- **Novelty:** refines AJ/AB (new application + a new AoA eval metric).
- **Experiment.** *Corpus:* CHILDES + Wikipedia same-size slices; compute per-word graduation
  step. *Metric:* Spearman of model graduation order vs CHILDES AoA norms; bpc as a guardrail;
  spacing-aware vs raw-frequency. *Baseline:* raw additive frequency (current observe path).
  *Kill:* spacing/diversity neither improves AoA correlation NOR holds bpc within noise — if it
  tracks children better but loses bpc, **FRAGILE-PASS**, park don't kill.

## M15 — Attentional salience gate (IDS-as-attention, not IDS-as-text)
IDS's benefit is pre-linguistic/attentional → simulate it as a multiplier on the increment in
`observe()`: up-weight (i) utterance **edges**, (ii) **repeated** tokens within the variation-set
window, (iii) high-**surprisal** onsets. Pair with a hard pause/punctuation-as-boundary prior into
**M**. The multiplier reads off counts the agent already has. **Risk guard:** high-surprisal
up-weighting amplifies noise (a typo is high-surprisal) — compose with **AA** consolidation and
**AB** confidence to prune low-confidence salient spikes (the Exp Y noisy-training trap).

- **Rules:** online (per-token multiplier), bounded (no new storage; low-salience rows decay),
  no-backprop, cognition-as-guide (infant attention orienting). Informs acquisition.
- **Refines/extends:** new role (M used branching-entropy for boundaries; this weights *learning*).
- **Novelty:** NEW.
- **Experiment.** *Corpus:* CHILDES through AT (real utterance edges), punctuation preserved.
  *Metric:* word-segmentation F1 salience-gated vs flat; per-word sample-efficiency (bpc at fixed
  budget); does it help **more** on CDS than Wikipedia (the literature says it should). *Baseline:*
  flat observe; branching-entropy segmentation without the edge/pause prior. *Kill:* improves
  neither F1 nor per-word efficiency over flat counting on CDS, OR does not differentiate CDS from
  Wikipedia (the honest negative).

## M16 — Variation-set minimal-pair miner (adjacent-utterance diffing)
A fixed-size ring buffer (N=4–8 utterances). After each utterance, token-align it against the
previous (LCS/anchored diff, no gradient). When overlap ≥60%, agreeing spans are **frames** and
each disagreeing span is a (slot, filler) substitution → up-weighted counts into AF's tables and a
new boundary signal into **M**. In the InterlocutorEnv, a variation-set responder (or Haiku
prompted for motherese) makes the world emit variation sets — the count-native realization of
"reactive teaches more than passive." Report the **syntax / world-knowledge split** honestly
(Haga: helps syntax, not world knowledge).

- **Rules:** online (single diff vs bounded ring buffer), bounded (ring buffer + AF tables under
  AI/AR), no-backprop. Informs both.
- **Refines/extends:** refines **AF** (adds adjacent-utterance alignment) + a boundary source for M.
- **Novelty:** NEW as a harness Environment property.
- **Experiment.** *Corpus:* CHILDES (Brown/Manchester) natural variation sets + synthetic
  injection into text8; L0 and L2. *Metric:* compositional generalization on unseen (frame,filler)
  pairs with/without the miner; BLiMP-style minimal pairs; phrase-boundary F1 (M) from
  agree/disagree points vs branching-entropy alone. *Baseline:* AF without diffing; branching-
  entropy boundaries without the diff. *Kill:* does not improve compositional-generalization OR
  boundary F1 on **any** slice (matching Haga's syntax-only help is a PASS on the syntax slice) —
  kill only if it loses on syntax too; judge on the syntax axis, not flat bpc. → queue **BB**.

## M17 — Two-threshold comprehension/production gate (the C>P lag organ)
Split every binding into the SAME leaky accumulator read at two operating points.
**Comprehension** (a recognition read): "understood" if ANY binding count clears a **low**
threshold. **Production** (a generation read used by `act()`): a form may be emitted only if it is
the **argmax** of the pooled vote AND its **AB** (f,c) clears a **high** bar. Concretely, gate
`act()`'s sampling: zero out candidates failing the production test, renormalize, sample. The lag
falls out and **widens** for words with many competitors (high **AO** fan) — reproducing the
"understands but won't say it yet" stage.

- **Rules:** online (two fixed scalars + AB + argmax), bounded (no new table; production gate
  *reduces* emitted vocabulary), no-backprop, cognition-as-guide (robust C>P finding). Informs both.
- **Refines/extends:** composes **AB** truth-value + **AO** fan; the dual-operating-point gate is new.
- **Novelty:** NEW.
- **Experiment.** *Corpus:* text8 or CHILDES-CDS through InterlocutorEnv; the 680-word CDI list as
  a watch-set. *Metric:* per-word comprehension-onset vs production-onset; median lag (in
  stream-time bins) and its correlation with competitor density. *Baseline:* single-threshold
  acquisition (predicts zero lag). *Kill:* the C-before-P lag is absent or does NOT widen with
  density across two seeds — check the lag at the form/grammatical level before killing (it should
  appear at every level). → queue **AY**.

## M18 — Threshold-gated decoding → telegraphic speech & Brown's morpheme order
Make `act()` emit ONLY slots whose frame-conditioned accumulators (AF's two counts) cross a
per-slot productivity threshold; low-count function/morpheme slots are dropped → telegraphic "cat
here" for free. Morphemes come online in an emergent order = input-frequency × frame-reliability
(a **hypothesis** for Brown's order — **report correlation honestly, don't claim to explain it**).
MLU is an output: a join clears only when both slots and their joining frame are over-threshold →
MLU climbs as offset-chunk counts accumulate. Overregularization rides the same surface (M19's
U-shape at the morphology grain).

- **Rules:** online (AF token/type + joining-frame counts), bounded (gating drops low-count slots;
  the default is one generalized pattern), no-backprop, fragile (each axis judged separately).
  Informs generation.
- **Refines/extends:** refines **AF** (telegraphic = threshold-gated decoding) + **AH** (the U).
- **Novelty:** MLU-as-self-metric on the harness decoder is new.
- **Experiment.** *Corpus:* CHILDES-CDS for input; emit via InterlocutorEnv; held-out irregular
  pasts (went/goed) tracked over stream time. *Metric:* running MLU(t) (a Brown-Stages-I–V climb);
  morpheme-onset order **correlated** vs Brown's 14; overregularization rate(t) (a low-amplitude U,
  ~2.5%). *Baseline:* ungated `act()` (raw temperature sampling). *Kill:* morpheme-onset order does
  not correlate with Brown's order across seeds, OR the U amplitude is implausibly large (≫2.5%) —
  re-tune (FRAGILE) before discarding; telegraphic and MLU judged on their own axes.

## M19 — Dual-route inflection head with f·c blocking (words-and-rules as one knob)
An inflection Column keyed by (stem-id, tense-cue). **Route A (memory):** a leaky-accumulator
count of the attested form, **AB**-split into (f,c). **Route B (default):** **AF** open-slot
construction over orthographic suffix counts — a productive +ed vote whose strength is the
branching-entropy/neighborhood-density of the suffix slot (**graded**, per Weissweiler/AD, **not**
a crisp boolean). Production = **AJ** take-the-best: if Route A's c exceeds the gate it
noncompensatorily **blocks** B; else the default fires. Overregularization happens exactly when a
low-frequency irregular's leaky count decays below the default — automatically **rare and
item-specific (micro-U)**. The single gate threshold slides Pinker↔Rumelhart (a measurable knob,
**not** a resolution of the debate).

- **Rules:** online (leaky counts + AB + AF suffix counts + AJ gate), bounded (small decaying
  irregular store + one suffix table, AR eviction), no-backprop, fragile (per-item-rate axis).
  Informs both.
- **Refines/extends:** fuses **AB** + **AF** + **AJ**.
- **Novelty:** the tunable dual-route gate + the rarity/micro-U law (components exist).
- **Experiment.** *Corpus:* AO-CHILDES / CDS + a synthetic irregular stream; held-out child
  error-TYPE distributions (Marcus/Maslen). *Metric:* per-item overregularization **rate**
  correlated vs child/corpus per-item rates (Weissweiler **graded**, not pass/fail); aggregate rate
  low (~2.5–10%) and roughly constant, **never a macro-U**. *Baseline:* single-route (gate=0);
  pure-memory; recency n-gram. *Kill:* no gate setting reproduces a low-constant item-specific rate,
  OR dual-route matches child error-TYPE no better than single-route → buys nothing over AF+AB.
  → queue **BA**.

## M20 — Rescorla-Wagner recovery loop (recovery without feedback)
The count-native answer to "why does *mouses* disappear without correction." In `run()`, before
observing each token the Agent **predicts** the inflected form (M19). On observing: an R-W update
— **increment** the heard form, and (the part AB lacks) **decrement** the predicted-but-absent form
by `(λ − Σ V)`. The repeatedly-predicted-but-never-confirmed "mouses" association decays purely
from being expected and not seen — cue competition / blocking. Reuses AT's `obs_surprise` to gate
the update. The **first acquisition use** of AT's reactive contract; recovery is impossible for a
passive increment-only reader.

- **Rules:** online (predict-then-update is the definition), bounded (the decrement **frees**
  budget), no-backprop (R-W is the canonical non-gradient associative rule). Informs both.
- **Refines/extends:** extends **AT**; the offline spine only ever *incremented* — the decrement
  is new.
- **Novelty:** NEW (predict-before-update + competitive decrement).
- **Experiment.** *Corpus:* InterlocutorEnv staged responder: feed regulars+over-applied stems so
  the model over-generalizes ("mouses"), then feed the correct irregular ("mice") **without any
  correction signal**; CHILDES order for the realistic version. *Metric:* does the over-applied
  count decay below the irregular's **purely** from prediction-error exposure (no label/reward)?
  P("mouses") trajectory; recovery slope; Ramscar's sign (exposure to other regulars affects it by
  count-maturity). *Baseline (load-bearing):* passive CorpusEnv with increment-only `observe()` —
  it **cannot** recover. *Kill:* increment-only passive reading recovers just as fast (recovery is
  just frequency) → keep the simpler increment-only loop. → queue **BC**.

## M21 — Mastery-mines-the-rule redescription for inflection (the micro-U, staggered)
Specialize **AH**'s stability-triggered redescription to inflection under sleep (**AA**). When a
SET of (stem→past) mappings has each been produced above a count threshold (mastery = settling,
**not** error), mine the shared suffix via branching-entropy and promote a compressed "+ed"
default into AH's registry; the act of trusting the new rule transiently raises
overregularization. **Fixes applied:** the only headline is the **macro-vs-micro contrast** — the
pass must yield **per-verb (micro)** U-dips that stay rare in aggregate and **never** a
synchronized macro-U; apply the mined rule **per-item** as each stem's leaky count decays (build
the staggering in, don't hope for it). Recovery comes from **M20** (not AH's designed
`REBIND_AFTER`).

- **Rules:** online (mastery counts) + one offline sleep pass (AA), bounded (compression **is** the
  coping move — AS: generalize before discarding), no-backprop. Informs both.
- **Refines/extends:** refines **AH** (mine across a SET; source recovery from M20).
- **Novelty:** the cross-item rule-mining + honest end-to-end U.
- **Experiment.** *Corpus:* synthetic morphology stream (tunable irregular:regular ratio) +
  CHILDES; track per-verb production across the sleep boundary. *Metric:* per-verb (micro) U,
  staying rare (~2.5–10% Marcus) in aggregate; memory freed by compression (AS curve). *Baseline:*
  AH's designed-recovery `REBIND_AFTER` (does M20 recovery match/beat it on realism?); no-sleep
  M19 (does rule-mining add item-specific U beyond decay-driven errors?). *Kill:* the pass produces
  a **synchronized macro-U** (contradicts Marcus) — rejected; also killed if M19's decay already
  gives the full U with no rule-mining.

## M22 — Exemplar-chaining lexical layer for overextension
A lexical layer for first-words reusing **U**: represent each word by the SET of referent-context
vectors (cheap random-projection sketches + any coarse perceptual code). To name a new referent,
score each word by nearest-exemplar similarity `exp(−d²/h)`, softmax over words. **Overextension**
= the model picks a known word ("dog" for all four-legged things) before the correct word's
exemplars exist (a flaws-as-features prediction). **Fixes applied:** **hard-cap exemplars per
word at K** and AA-consolidate the rest into one prototype (mandatory, not optional — kernel-over-
all-exemplars is O(memory)); judge on the **comprehension>production asymmetry** (`p(c|w)` vs
`p(w|c)`) but report 55%/top-5 recall as primary and the asymmetry as **exploratory** (inversion
direction is **not** the sole cause); guard against U's function-word mega-cluster artefact (the
kill-condition).

- **Rules:** online (exemplars + leader-clustering + kernel vote), bounded (K-cap + prototype
  consolidation), no-backprop (leader, not Lloyd). Informs both.
- **Refines/extends:** reuses **U** clustering as substrate; new as a lexical extension layer.
- **Novelty:** NEW (word-meaning extension + the C>P asymmetry).
- **Experiment.** *Corpus:* Ferreira-Xu child overextension dataset (55%/top-5 benchmark);
  multimodal if perceptual codes are available. *Metric:* top-5 recall of attested overextensions
  (~55% vs 12% baseline) — **primary**; the asymmetry (production overextends, comprehension does
  not) — exploratory. *Baseline:* frequency baseline (~12%); single-channel associative model.
  *Kill:* predictions dominated by U's function-word mega-cluster, OR cannot reproduce the
  asymmetry — park until a clean content-word latent exists.

---

# GENERATION

## G1 — Coverage-competition production: the open slot drives `act()` (merged organ)
**The generation turn.** Today `act()` samples a flat geometric-mean vote → gibberish. Replace
with construction-driven production: (1) **retrieve** candidate constructions whose role/context
matches state (AH registry, cue-keyed via **AO** feature+fan, not a fixed offset); (2) score each
by **coverage × frequency** — coverage = kNN density of the intended filler against the slot's
**bounded leader centroids** (not raw exemplars), frequency = the construction's leaky use-score;
(3) **compete** by **AJ** take-the-best (validity-ordered, noncompensatory, early-stopping — not a
flat pool); (4) fill the slot through its category lexicon and emit.

**Merged:** the former "Frame-then-content incremental decoder (Levelt pipeline)" is folded in as
the **decoder scaffold** — three leaky buffers (conceptualize/formulate/articulate), function
words ride **with** the frame (never through content selection), a branching-entropy switch (Exp
A) escalates from fast articulation to slow message/frame. Its separately-judgeable claim — **frame
survival under content error** — is kept as a kill-test. *(Evidence-honesty: do **not** claim this
adjudicates radical-exemplar vs emergent-abstraction, nor that it is "the Levelt pipeline" — present
coverage×frequency as one Goldberg-inspired heuristic; Levelt seriality is debated.)*

- **Rules:** online (dict lookup + kNN over bounded centroids + take-the-best argmax; no learning
  step), bounded (constructicon under leaky Dunn-2022 decay = AR eviction; kNN over centroids),
  no-backprop. Informs generation.
- **Refines/extends:** extends **AT**; reuses **AJ** (combiner), **AH** (registry), **AB**
  (coverage truth-value), **AF** (category fill, preemption), **AO** (retrieval key).
- **Novelty:** NEW as a production organ (AF/AH only read and scored).
- **Experiment.** *Corpus:* L0 CorpusEnv (text8) to fit the constructicon, then generate;
  comprehension side via BLiMP / a constructional minimal-pair set. *Metric:* fraction of emitted
  (frame,filler) pairs well-formed vs flat-sample; over-generation rate (should fall);
  constructional-contrast battery preference; **frame survival under an injected wrong content
  label** (80–95% category sanity); does production prefer frozen idioms verbatim and open slots
  productively? *Baseline:* the current flat geometric-mean sampler (gibberish floor) at equal
  memory; single-table no-split sampler. *Kill:* not measurably more well-formed / less
  over-generating than the flat sampler on the **constructional** battery (its right axis, not raw
  perplexity — AF's verdict: constructions win only on the unseen-combination slice) — park as
  "needs the situation model (Exp AM frontier)", do NOT kill the constructicon. → queue **BD**.

## G2 — Contingency-gated learning rate (the temporal-contingency dial)
In `observe()`, multiply each increment by a contingency gain `g = exp(−Δt/τ)` (Δt = steps since
the agent last emitted a non-empty `act()`). Keep TWO registers: `tab_hot` (gain g, for input
arriving while a self-emission is warm — a reply that answered the model) and `tab_cold`
(background, gain ~0.2). Prediction pools both, hot weighted higher. A contingent Haiku reply is
worth more counts than the identical tokens read passively (Goldstein & Schwade contingent-vs-
yoked). **Guard (joint-attention is contested):** gain stays **soft** and cold input still updates
`tab_cold` at low weight — never a hard joint-attention flag.

- **Rules:** online (per-increment scalar), bounded (hot is small/specific = AE/ART-protectable;
  LFU eviction unchanged), no-backprop, cognition-as-guide. Informs both.
- **Refines/extends:** extends **AT** (the dial it deferred); reads **AT**'s `surprise_probe`.
- **Novelty:** NEW.
- **Experiment.** *Corpus:* InterlocutorEnv with a contingent responder over CDS/dialogue.
  *Metric:* held-out bpc + a **turn-overlap** contingency metric (reported separately, not as bpc).
  *Baseline:* passive CorpusEnv (gain=1, single table) AND the critical **YOKED** ablation
  (identical reply text, scrambled timing → random g). *Kill:* contingency-ON matches YOKED-OFF on
  bpc AND turn-overlap at matched tokens — surface as the honest negative (register the yoked
  baseline **before** running; check turn-overlap/rare-context before declaring death). → queue **BE**.

## G3 — Repair as a paired count edit (reformulation = negative + positive evidence)
When an external turn restates the agent's prior span with edits (detected cheaply: **AC**
branching-entropy/KL spike at the join + **high lexical overlap** with the agent's last emission),
compute the span diff. The n-grams unique to the **agent's** version get a **MISS** increment on
**AB**'s split; the n-grams in the **reformulation** get a **HIT** increment. Eve Clark's
repair-as-both-evidences as two count edits — riding AB + AF preemption, **not** a parallel penalty
table. *(Drop the Tomasello intention-reading framing: the responder has no communicative intent;
report entrenchment and preemption as two count-gaps that MAY be collinear, not assume it.)*

- **Rules:** online (two edits per repair), bounded (edits existing fields), no-backprop. Informs both.
- **Refines/extends:** refines **AB** + **AF**; new = the interactive trigger + span-diff routing.
- **Novelty:** refines AB+AF (new only in the interactive loop closure).
- **Experiment.** *Corpus:* InterlocutorEnv with a responder issuing clarifications/reformulations
  on malformed spans (Haiku can play this). *Metric:* uptake (next emission of the same slot moves
  toward the reformulated form, edit-distance over turns) + over-generation rate. *Baseline
  (load-bearing):* **G2 contingency-gated counting WITHOUT the repair edit** — does plain contingent
  input already capture it? *Kill:* plain contingent counting gives the same uptake as the explicit
  paired edit → repair is subsumed by G2; park the diff machinery (AB increments with extra steps).

## G4 — Per-conversation common-ground overlay (grounded-form bias for generation)
A small, leaky per-dialogue table of entities/constructions that have been **both** emitted by the
agent AND acknowledged (reappeared in the partner's turn — **AO** cue retrieval). Acknowledged
items promote to a high-confidence shared store; unacknowledged decay. At `act()`, **bias** the
next emission toward grounded forms (least-collaborative-effort) and use the overlay as a
reference-resolution prior multiplied onto the global cross-situational counts. Strictly a
per-conversation overlay; resets/decays between dialogues. **AM killed the long-span predictor** —
this is licensed ONLY as a generation/reference bias, measured against a **static** prior.

- **Rules:** online (leaky per-dialogue table), bounded (discarded at conversation end =
  AQ environment-as-memory), no-backprop. Informs both.
- **Refines/extends:** refines **AM** (generation bias only, not the dead predictor).
- **Novelty:** new as a generation bias + reference prior.
- **Experiment.** *Corpus:* multi-turn InterlocutorEnv where referents recur and get acknowledged.
  *Metric:* reference-resolution accuracy under the social prior vs counts-only; grounded-form reuse
  in generation. *Baseline:* **AM's STATIC frozen prior** (never no-prior) for the predictor claim;
  counts-only resolution for binding. *Kill:* does not beat a static prior on reference resolution
  AND does not raise grounded-form reuse → collapses into AM's 0.9%-slice result, park inspection-only.

## G5 — Two-state cadence: interpersonal-foraging emission scheduler (bound to G2)
Give `act()` a leaky accumulator A that self-correlates over steps (carry-over burstiness) and gets
a downward kick to the next-emit interval when a contingent reply arrives. Threshold into two
states: **EXPLOIT** (short intervals, high learning gain — feeds G2's g) and **EXPLORE** (long
intervals, the agent probes on its own). Makes the loop **proactive** instead of input-starved.
**Fixes applied:** **bind it to G2 as one experiment** (don't run as an independent learning
mechanism); **drop the specific burstiness-β targets** as success criteria (curve-matching a
borrowed exponent proves little) — judge only on whether exploit-burst tokens yield higher learning
gain than explore tokens at matched count.

- **Rules:** online (one accumulator + threshold), bounded (O(1) state), no-backprop,
  cognition-as-guide (arousal/availability, not optimization). Informs generation.
- **Refines/extends:** refines **AT** `act()` (fixed cadence today); supplies the pacing **AG**'s
  gate decides over.
- **Novelty:** NEW (engage/explore states + intrinsic oscillator).
- **Experiment.** *Corpus:* InterlocutorEnv with varied reply rate/contingency. *Metric:* learning
  gain in exploit bursts vs explore at matched count (the cadence-realism is a secondary fragile
  axis only). *Baseline:* fixed-cadence `act()`. *Kill:* exploit-burst tokens yield no higher gain
  than explore at matched count → park (keep only if it wins on cadence-realism, and say so up front).

## G6 — Margin-gated production (read the counts the hard way)
**Production is comprehension read backwards.** `produce(cue)`: gather candidate labels via **AO**
cue-retrieval (cue = message/role bundle + left context), score each by `count(cue,label)` with
**AB** (f,c) as activation divided by **fan**; emit only if
`margin = activation(top)/activation(2nd) ≥ θ_emit` (AJ self-setting bar), else stay silent or back
off to a generic higher-margin label (dog before retriever) or defer the slot. Comprehension uses
the SAME table with NO gate (one-to-many is forgiving). Reproduces the C>P gap and an "understands
but won't say it yet" stage on the *production* side (M17 is the acquisition-threshold twin; this
is the read-direction asymmetry).

- **Rules:** online (reverse read of existing counts), bounded (AO per-cue store; production is a
  query, no new table), no-backprop, fragile (judged on production precision, not bpc). Informs both.
- **Refines/extends:** extends **AO**; reuses **AB**, **AJ** — none of which gates emission on a
  many-to-one margin.
- **Novelty:** NEW (the read-direction many-to-one margin gate).
- **Experiment.** *Corpus:* text8 word-level; (message-cue, label) pairs, held-out production and
  comprehension probes over the SAME pairs. *Metric:* comprehension accuracy (no gate) vs production
  recall (gated) vs evidence/θ; the **gap shrinking** + production precision. *Baseline:* ungated
  argmax production; offset-attention production without fan/margin. *Kill:* gated precision not
  above ungated at matched recall, OR the gap does not appear/shrink with evidence → the structural-
  gap claim fails. → queue **BF**.

## G7 — UID surprisal-band re-ranker (decode-time information smoothing)
Make surprisal a **decoding control**. Per step compute surprisal = −log(pooled count-share) per
candidate; keep a running-mean band (leaky). Re-rank top candidates to keep local surprisal near
the band: penalize **spikes** (low-count continuation → keep optional material: a determiner,
"that", a fuller form) and **troughs** (predictable → reduce/omit). Among frame-permitted
candidates pick the one whose branching-entropy (Exp A) is closest to the running mean. The same
predictor vetoes (AB) implausibly-high-surprisal candidates before commit — generator and checker
are one engine (Levy-Jaeger UID).

- **Rules:** online (one leaky scalar + argmin over computed surprisals), bounded (no new memory),
  no-backprop, fragile (judged on omission-pattern + surprisal variance, not bpc). Informs generation.
- **Refines/extends:** reuses Exp **A** entropy + **AB** veto; UID-band decoding is new.
- **Novelty:** NEW.
- **Experiment.** *Corpus:* text8; "that"-droppable and determiner-droppable probes from the
  corpus. *Metric:* omission rate vs following-clause predictability (Levy-Jaeger: drop when
  high-prob, keep when low); per-token surprisal variance (lower than unsmoothed). *Baseline:*
  argmax/temperature sampling with band control OFF. *Kill:* omission does not track predictability
  in the Levy-Jaeger direction, OR variance is not reduced.

## G8 — Dual-counter structural priming / alignment (free comprehension→production transfer)
Two counters per AF frame. **LEXICAL-BOOST** = fast-decaying leaky accumulator keyed (lemma+frame),
spikes on use, biases only the next few utterances (Bock & Griffin transient). **ABSTRACT
PERSISTENCE** = slowly/never-decaying frame count incremented on **every** produced OR comprehended
instance (implicit learning). Frame vote = base AF + persistence + boost. Both modalities increment
the same persistent counter → comprehension→production transfer is free: a structure just *read* in
the InterlocutorEnv reply becomes more likely in what the agent *says* next (interactive alignment).

- **Rules:** online (two additive/leaky counters), bounded (keyed on bounded AF frames; boost
  decays), no-backprop. Informs both.
- **Refines/extends:** refines **AF** (the dual-timescale counter is new).
- **Novelty:** NEW (priming/alignment as fast-leak + slow-persist with read/write sharing).
- **Experiment.** *Corpus:* InterlocutorEnv, responder alternating two construction variants
  (active/passive, DO/PO datives) over a text8-trained agent. *Metric:* priming effect
  `P(reproduce X | comprehended X) − P(X | comprehended Y)`; two-timescale separation; the AT
  kill-test (does priming beat passive reading per token?). *Baseline:* single persistent counter
  (no boost); no-priming generation. *Kill:* no above-chance priming, OR fast/slow decay does not
  separate, OR no comprehension→production transfer.

---

# CONSOLIDATION & SLEEP (informs both; gates durability)

## M23 — Two-rate memory with a lexical-competition delay (fast episodic slot, slow via sleep)
Two count stores. **FAST:** a bounded sparse episodic buffer, written at full weight on first sight
(single-shot recallable). **SLOW:** the existing backoff table — the **only** store consulted for
**competition**. A novel token is recallable immediately but does NOT slow its neighbours until a
sleep pass folds it in. Sleep = an **AA** pass restricted to the fast buffer, interleaving each
promoted token with an **AE** LTI-weighted replay of existing slow entries (count-native SWIL: pull
nearest neighbours via the random-projection/LSH index, replay-increment only those). Marker:
competition ~0 pre-sleep, >0 post-sleep.

- **Rules:** online fast/slow writes + slow reads; **the sleep pass is a SECOND bounded pass
  (offline, flagged like AA)** — honest, not pure single-pass. Bounded (fast buffer is sparse/
  localist; merges into the bounded slow table), no-backprop. Informs both.
- **Refines/extends:** refines **AE** (no two-stage delay / no SWIL replay) + **AA** (flat→localist).
- **Novelty:** the lexical-competition timecourse marker + SWIL neighbour-replay.
- **Experiment.** *Corpus:* text8 through CorpusEnv + ~50 injected nonce tokens near real
  neighbours (cathedruke ~ cathedral), each introduced once then re-encountered. *Metric:*
  competition score = post-introduction drop in the real neighbour's pooled next-token probability,
  immediately and after one `sleep()`; held-out bpc; fast-buffer size. *Baseline:* AT single-store
  (instant competition, biologically wrong); AA single-table sleep (global, not neighbour-local).
  *Kill:* pre-sleep competition not ~0, OR neighbour-only SWIL doesn't match full-interleave within
  noise at <10% cost, OR bpc degrades vs AA single sleep → the split buys nothing.

## M24 — Inverse-count / surprise-prioritized spindle replay (selective, frequency-stratified)
Replace AA uniform replay with a **prioritized** offline pass. Maintain an online min-heap of
(cumulative_count, token); during `sleep()`, re-increment / protect with probability ∝ 1/count (or
local branching-entropy) — concentrating the bounded offline budget on the **rare/uncertain tail**
(Schapiro 2018; spindles protect infrequent words). Two stratified subroutines (vocabulary vs
grammar): one walks the rare TOKEN tail (AR power-law protected), one walks the rare OFFSET/skip-gram
counters (Exp S) and promotes specific contexts into their shared bucket (Gómez: decay items,
increment the rule). A developmental **anneal** knob shifts replay from form-merge (early) toward
meaning/co-occurrence (late) — one knob for the child-form/adult-meaning split.

- **Rules:** online heap maintenance + offline replay (flagged), bounded (spends the scarce offline
  budget on exactly the tail AS's heavy-hitter cap would drop), no-backprop. Informs acquisition.
- **Refines/extends:** refines **AA** (uniform → prioritized) + **AS** (heavy-hitter → rare-tail).
- **Novelty:** the dual item-vs-regularity replay budget + developmental anneal.
- **Experiment.** *Corpus:* text8 60M (AS protocol) + the AE domain-shift stream. *Metric:*
  rare-context bpc and overall bpc at a fixed entry budget, uniform-replay vs inverse-count;
  backward-retention Δ of rare contexts specifically. *Baseline:* AA gentle (uniform) sleep at equal
  offline budget; AS heavy-hitter cap without selective replay. *Kill:* inverse-count does not beat
  uniform on **rare-context** bpc at equal budget, OR harms common-context more than it helps rare
  (the AA over-distillation trap). → queue **BG**.

## M25 — Schema-consistency gate: surprise-routed fast-vs-immediate-slow consolidation
One cheap online decision per novel token. Compute surprise under current counts (**AB** (f,c)).
**LOW** surprise (schema-consistent, e.g. a new unit in an attested AF slot) → **immediate
full-weight merge** into the slow table (McClelland 2020 fast integration). **HIGH** surprise →
the fast episodic buffer for slow interleaving (M23). Converts the rigid "integration-needs-a-night"
time-lock into a **consistency gate** (the replicable version). **Guardrail baked in:** **NO**
one-shot direct-to-slow path on its own (the failed-to-replicate Coutanche route, deliberately not
built) — only schema-consistency earns the fast lane.

- **Rules:** online (one surprise lookup + threshold), bounded (consistent tokens never occupy a
  fast slot), no-backprop. Informs acquisition.
- **Refines/extends:** composes **AB** + **AF** + M23 fast buffer.
- **Novelty:** NEW routing gate (the consistency-gate framing).
- **Experiment.** *Corpus:* text8 with two injected classes — schema-consistent (novel filler in a
  high-type-frequency AF slot) and schema-inconsistent (novel token in a low-entropy deterministic
  context). *Metric:* integration latency (exposures/sleeps to asymptotic slow weight) by class;
  interference (backward bpc change on contexts that previously owned the slot). *Baseline:* no gate
  (all novel → fast buffer); the forbidden one-shot-direct-to-slow (to show it causes interference).
  *Kill:* schema-consistent fast-merge causes **more** backward interference than routing through the
  fast buffer → the gate is unsafe; revert to all-through-buffer.

## M26 — Cued / curriculum-weighted replay (TMR) + conditional early-exit sleep
Two cheap controls. **(a) TMR knob:** a tag-set of tokens the writer/teacher wants reinforced
(recently-introduced Atlas entity names, or a supplied vocabulary) gets **extra** replay increments
during `sleep()` — fired ONLY in the offline pass. **(b) Conditional sleep:** gate whether to
consolidate at all by count AND local entropy — only consolidate where encoding was weak/sparse AND
there is extractable regularity; otherwise early-exit (Belia 2023 heterogeneous benefit; avoids
AA over-refinement). **Fix applied:** drop the borrowed ~9.5% cued-over-uncued figure as a target;
keep TMR as a directional prediction (cued > matched uncued) only.

- **Rules:** online tagging + online gate + offline extra-increment replay (flagged), bounded (tiny
  tag-set; conditional exit spends budget only where needed), no-backprop. Informs both.
- **Refines/extends:** refines **AA** (stop-after-one-cycle → run-only-where-needed).
- **Novelty:** NEW (TMR cueing + conditional early-exit). Strong Atlas fit.
- **Experiment.** *Corpus:* a small Atlas-flavoured corpus (sparse named entities) via CorpusEnv +
  text8 for the conditional-gate ablation. *Metric:* recall on tagged (cued) vs equally-rare untagged
  (uncued) tokens after one sleep; for conditional sleep, bpc + total offline increments vs
  unconditional AA. *Baseline:* AA unconditional uniform sleep. *Kill:* cued tokens show no recall
  advantage over matched uncued → drop TMR; conditional early-exit underperforms always-on at
  equal-or-worse compute → drop the gate.

---

# EVAL ORGANS (read-side; the acquisition phase is judged by these)

## E1 — BLiMP / minimal-pair scoring Probe + impossible-language ablation
A Probe scoring grammaticality the field's way: for a minimal pair (s+, s−), sum per-position
−log P_count(token | K-tail) via the existing vote; prefer the lower-surprisal sentence. Bundled
with the Kallini **impossible-language** ablation: train the same Column band on natural English vs
a position-shuffled scramble; report whether our counter geometry learns natural grammar more
easily (the **locality bias** baked into backoff). **Fix:** **control for scramble complexity** (or
report the confound) — the gap may be entropy-driven, not naturalness-driven.

- **Rules:** read-side (the eval does not learn), bounded (O(sentence×band); ablation under the AS
  budget cap), no-backprop. Informs both.
- **Refines/extends:** the eval **AT** deferred; over the existing vote.
- **Novelty:** NEW (the eval bar the rules implicitly demand).
- **Experiment.** *Corpus:* train on 10M and 100M-word CDS/transcribed mix (sentence-bounded
  windows); eval = BLiMP 67 sets; ablation = natural vs scrambled. *Metric:* BLiMP accuracy
  (per-phenomenon + macro); ablation = natural-minus-scramble gap. *Baseline:* bigram/5-gram
  backoff at the same budget; report distance to BabyLM transformer (~0.85) and human (~0.88)
  **honestly**. *Kill:* count band ≤ bigram on BLiMP, OR natural and scramble learned equally
  easily — do NOT kill on the gap to transformers; agreement (give **AO**) and interrogatives are
  expected weak slices, report per-phenomenon. → queue **BH**.

## E2 — Count-native age-of-acquisition tracker (running-KL stabilization Probe, POS-split only)
Per-word neural AoA = the count at which P_count(·|word) stabilizes (running KL between successive
snapshots drops below threshold). **Fix (reinvention):** M14 already produces an AoA curve — keep
this probe ONLY for what M14 doesn't test, the **POS dissociation** (function words stabilize first
under bigram counts; nouns need wider-offset **S** Columns, the Ficarra effect n-grams miss). Make
that dissociation vs the **frequency-rank null** the sole deliverable.

- **Rules:** online (running KL over log-spaced snapshots; keep only the previous snapshot per
  tracked word), bounded (AoA log external-memoried), no-backprop. Informs acquisition.
- **Refines/extends:** offset counters = **S**; stabilization = count analogue of Ficarra 2025.
- **Novelty:** NEW as a Probe (POS-split deliverable).
- **Experiment.** *Corpus:* CHILDES-derived CDS (where AoA norms exist). *Metric:* Spearman of
  count-AoA vs human AoA, **split by POS** (predicate/function vs noun); wider-offset Columns should
  recover the noun effect bigrams miss. *Baseline:* bigram-only AoA (under-predicts nouns); raw
  word-frequency rank (the honest null). *Kill:* count-AoA correlates with human AoA no better than
  frequency rank — it is just a frequency proxy.

## E3 — Communicative-success as count-reweighting (the interaction signal, gated follow-up)
After the agent acts, the responder's reply carries a **success scalar**; up-weight constructions
that preceded successful replies, down-weight failures (AF preemption-style — a multiplicative count
reweight, **not** policy gradient). **Fixes applied:** the success signal must NOT be another LLM's
quality judgment (grader-leakage: the grader's competence becomes the teacher) — use
**task-completion / referent-resolution success** in a grounded env, or explicitly bound the claim
to "success-as-low-responder-surprise" and watch for leakage. **Differentiate from G2:** keyed to a
**success** scalar (what-worked-I-say-again), not timing (Δt) — run **after** G2, with G2 as the
baseline; if a graded success bit adds nothing beyond contingent-timing reweighting, **cut**.

- **Rules:** online (per-turn reweight), bounded (construction table; no new structures),
  no-backprop, fragile (≥10 nurture steps). Informs generation.
- **Refines/extends:** reuses **AF** inhibition + **AT** signal channel.
- **Novelty:** new for the loop (AT shipped the seam, only a surprise Probe over it).
- **Experiment.** *Corpus:* InterlocutorEnv with a grounded comprehension/task-completion bit;
  matched passive CorpusEnv of equal token budget. *Metric:* per-token improvement in generation
  quality / construction appropriateness, reactive vs passive (the AT kill-test). *Baseline:* **G2
  contingency-gated** (does the success bit add anything beyond contingent timing?) + passive
  reading of equal tokens. *Kill:* reactive+reweight gives no per-token advantage over passive, OR
  no advantage over G2 timing-reweight → the success signal is inert; cut.

---

# CUTS & MERGES (the adversarial pass)

- **CUT — "The referent-binding Column (SceneEnv + Pursuit slot)"** → folded into **M2** variant B
  (same Pursuit single-winner, same word×scene-entity axis, same propose-but-verify/chance-after-
  miss signature, same full-table strawman). Its SceneEnv + entity-id store is now M2's substrate.
- **CUT — "Surprise-as-learning-gate (N400 written onto counter-update rate)"** → duplicates the
  write-rate-by-surprisal idea; the **monotone** "high-surprise writes more" is the naive version
  the **Goldilocks inverted-U** corrects. *(Note: a separate Goldilocks learning-rate gate was the
  buildable form; per the reconciliation it is itself a fragile budget-efficiency experiment — see
  **BUILD_QUEUE BI** — and the N400/cloze validation is folded in there as a **read-out
  correlation**, kept distinct from the write-gate per the category-error fix.)*
- **MERGE — "Frame-then-content incremental decoder (Levelt)"** → merged into **G1** as the decoder
  scaffold; its load-bearing claim (frame survival under content error) is kept as a G1 kill-test.

**Cut count: 3** (two hard cuts + one merge).
