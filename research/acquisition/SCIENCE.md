# SCIENCE — the cog-sci ground truth

The distilled science behind language acquisition and production, organized by the twelve
research angles. Per angle: the summary, the strongest cited findings (with an
**evidence-strength** tag and URL), and the latest 2023–2026 work. This is the ground truth
that [MECHANISMS.md](MECHANISMS.md) operationalizes on counts and that
[CURRICULUM.md](CURRICULUM.md) sequences.

Evidence tags: **robust** (multiply replicated / meta-analytic), **solid** (well-supported,
some debate), **mixed** (real effect, contested determinants), **thin** (single-lab or
recently challenged), **negative** (a non-replication or null we must respect).

---

## 1. Statistical & distributional learning

**Summary.** Infants segment a stream by tracking conditional structure (transitional
probabilities, branching entropy). But the field's sharpest recent correction is that the
unit is a **chunk**, not a transition: once a sequence is learned as a whole, its internal
transitions *weaken*. Frequency alone is not what's used — conditional structure is.

- **robust** — Saffran, Aslin & Newport 1996: 8-month-olds segment by transitional
  probability; the frequency-matched control proves it's TP, not frequency.
  https://www.science.org/doi/10.1126/science.274.5294.1926
- **solid** — Harris 1955 / successor-variety; branching entropy *rises* at boundaries
  (this is **our Exp A**, F1 0.775). https://doi.org/10.1080/00437956.1955.11659552
- **solid (the verdict)** — Isbilen et al. 2023: chunk-based segmentation, not pure TP —
  a learned ABC chunk *weakens* the B–C transition (the splice test). The decisive result
  no current experiment implements. https://doi.org/10.1111/cogs.13193
- **mixed** — Mintz 2003 frequent frames (A_x_B): two-sided frames yield 91–98%
  same-category middle words *on top-k frames in child-directed speech*; degrades off-corpus
  and in free-word-order languages. https://doi.org/10.1016/S0010-0277(02)00247-3
- **negative** — Siegelman et al.: individual statistical-learning benchmark scores have
  **poor reliability** and inconsistently predict language outcomes → forbids validating a
  capability on a single segmentation F1. https://doi.org/10.1016/j.jml.2017.01.001
- **2023** — Benjamin et al.: neonates *track* the statistic but boundary-*commitment* is a
  separable stage → keep the entropy signal and the commit threshold as two knobs.

## 2. Usage-based & construction grammar

**Summary.** Grammar emerges item-by-item (verb islands) then abstracts to open-slot
constructions. The lever the field hands us: score slots by **association** (ΔP/PMI), not
raw frequency — raw co-occurrence over-generates.

- **robust** — Tomasello, *Constructing a Language* (2003): item-based → abstract; no verb
  is an island. https://www.hup.harvard.edu/books/9780674017641
- **solid** — Stefanowitsch & Gries collostructional analysis: association beats raw
  frequency for which fillers belong in a slot. https://doi.org/10.1075/ijcl.8.2.03ste
- **solid** — Goldberg coverage/competition: a creative use is licensed by exemplar
  coverage and *blocked* by a better-fitting competitor (statistical preemption).
- **2025 (warning)** — "LLMs learn constructions humans don't know": raw co-occurrence
  stores statistically-real-but-non-human patterns; **association + preemption** is the gate
  that keeps the inventory human-like. (Dunn and colleagues, association-over-frequency.)
- **2024** — Royal Society Open Science: count models show the holophrase → item-based →
  abstract trajectory only on child-directed-speech-scale data with an exposure-order axis.

## 3. The bootstrapping problem

**Summary.** Prosodic/TP bootstrapping (our Exp A), frequent frames (our S→AF), and
syntactic bootstrapping (verb argument-frames cue meaning) — but syntactic bootstrapping is
**small** (d≈0.24) and English-favoring. The honest discipline: weight the frame cue
modestly, never as the sole signal.

- **solid** — Mintz frequent frames (as above).
- **mixed/thin** — Syntactic bootstrapping meta-effect d≈0.24 (Cao & Lewis-style
  re-analyses); CDS is "not optimal" for it. Treat as one validity-weighted voter.
- **2025** — Word-order ablations: removing order most hurts **verbs**; removing
  co-occurrence most hurts **nouns** — a clean, falsifiable dual-representation claim.
- **2024–25** — Joint-inference reframing: segmentation, categorization and frame learning
  run *concurrently* and mutually constrain — but any "coherence" win must beat a **static**
  single-layer prior (our Exp AM's law), or it's smoothing.

## 4. Word learning, reference & constraints

**Summary.** Gavagai is solved by **cross-situational accumulation** plus constraints
(mutual exclusivity as competition, the shape bias, fast-mapping). The mechanism question
the field is unresolved between: dense associative accumulation vs **propose-but-verify**
(single-winner). Both are count-native; pick by memory budget and the human signature.

- **robust** — Yu & Smith 2007 cross-situational word learning. https://doi.org/10.1111/j.1467-9280.2007.01915.x
- **solid** — Trueswell et al. 2013 propose-but-verify: learners keep **one** hypothesis;
  at-chance after a disconfirmed trial (the human signature). https://doi.org/10.1016/j.cogpsych.2012.10.001
- **solid** — Frank, Goodman & Tenenbaum 2009: a *Bayesian* model gives ME via marginal
  competition. Note: PMI **approximates** this pressure, it does not inherit the guarantee.
- **solid** — Smith et al. shape bias emerges after ~50 nouns (not innate).
- **mixed** — Brody-style: ME may be modulated by focus/contrast (contested).
- **2023–25** — CVCL (Vong et al., *Science* 2024): grounded word learning from a child's
  head-cam via simple online association + contrast. https://www.science.org/doi/10.1126/science.adi1374

## 5. Developmental sequence as curriculum

**Summary.** The vocabulary spurt is a **mathematical inevitability** of parallel
accumulation over a Zipfian difficulty distribution — *almost exactly* our Column bank.
Milestones (comprehension-before-production, telegraphic omission, MLU climb, U-shaped
overregularization) should *fall out* as sanity checks, not be engineered.

- **robust** — McMurray 2007: the spurt is parallel accumulation × right-skew, no stage
  switch. https://doi.org/10.1016/j.cognition.2007.07.015
- **robust** — Comprehension precedes production (low recognition threshold vs high
  winner-take-all production threshold).
- **mixed** — Brown's 14-morpheme order is real but its *determinants* (frequency vs
  semantic/syntactic complexity) are debated — don't claim to have explained it.
- **negative (hard rule)** — Chang 2025 / BabyLM: low surprisal ≠ "learned"; distributional
  trajectories correlate **poorly** with children's age-of-acquisition → forbids defining
  acquisition by held-out bpc.

## 6. Input, child-directed speech & plausible training

**Summary.** A vindication-by-deflation: children's edge is the **learning algorithm, not
the data**; curriculum-by-difficulty/age/length **hurts** at ≤100M words (our Exp AK
reproduced this). The buildable levers are orthogonal to curriculum: variation sets,
recency+diversity weighting, an attentional salience gate.

- **2024** — Feng et al.: at ≤100M words the algorithm is the lever, not CDS content.
- **negative** — BabyLM curriculum β≈−3.6, linguistic-difficulty bias β≈−7.3 (curriculum
  consistently hurts). Confirms Exp AK on an external substrate.
- **solid** — Onnis/Waterfall variation sets: adjacent near-duplicate utterances give
  minimal-pair slot/filler structure for free.
- **mixed** — Haga et al.: variation sets help **syntax** (BLiMP/GLUE), **not** world
  knowledge (EWoK); optimal proportion is benchmark-dependent.
- **solid** — Ferry/Thiessen: infant-directed speech's benefit is **attentional/prosodic**,
  not textual → model it as a salience multiplier, not as privileged text.
- **thin** — Lai & Poletiek: embedding-**depth** ordering helps recursion (the one curriculum
  survivor; small, not robustly replicated).

## 7. Social-pragmatic & interactive grounding

**Summary.** **Interaction** (a reply causally tied to what the model just said), not raw
exposure, is the active ingredient — but the effect is **modest** on a competent base and
will not cure global coherence (our AM/AC altitude law). Contingency is a credit-assignment
*layer*, reported on its own metric (turn-overlap), never sold as a bpc breakthrough.

- **robust** — Goldstein & Schwade: contingent vs **yoked** replies dissociate learning
  with identical text → the yoked ablation is the required control.
  https://doi.org/10.1111/j.1467-9280.2008.02117.x
- **solid** — Clark, reformulation-as-correction (both negative and positive evidence);
  least-collaborative-effort / common ground.
- **mixed** — Ambridge 2018: preemption and entrenchment are **collinear** — implement as
  **one** expectation-violation signal; do not claim to dissociate them.
- **2025** — Salhan ContingentChat (BabyLM): contingency gives modest gains on a competent
  base, hard to install on a weak one.
- **contested** — Akhtar & Gernsbacher: joint attention is **not** a hard gate (blind /
  overhearing learners acquire) → contingency gain must be **soft**, cold input still counts.

## 8. Errors, U-shaped learning & words-and-rules

**Summary.** Overregularization is **rare (~2.5–10%) and roughly constant** — a shallow
*item-by-item micro-U*, never a system-wide macro-U. Falls out of a leaky memorized-form
count competing a productive default vote. Recovery-without-feedback = Rescorla-Wagner cue
competition (implicit negative evidence), the cleanest fit to ONLINE-ONLY.

- **robust** — Marcus et al. 1992: rarity + micro-U, no macro-U.
  https://doi.org/10.2307/1166115
- **solid** — Ramscar et al.: recovery without correction; hearing "mice" when you predicted
  "mouses" is implicit negative evidence (R-W blocking).
- **2025** — Weissweiler et al.: production is **graded**, not rule-perfect ("spling"→
  "splung") → the default vote must stay continuous in suffix entropy/neighborhood density.
- **mixed** — Words-and-rules (Pinker) vs single-route (Rumelhart) is genuinely contested;
  reproducing the U does **not** settle it.
- **solid** — Ferreira & Xu, exemplar chaining for overextension (55% top-5 vs 12% baseline);
  comprehension > production asymmetry.

## 9. Prediction & predictive processing

**Summary.** Surprisal (−log p), entropy reduction (Hale 2006), branching-entropy
boundaries are **already our native idiom** (A, AC, S, AB). The genuinely new work is on
generation: prediction is a **modulator over a base count-learner, not the engine** — every
prediction-error gate ships with an ablation showing the base learner still acquires.

- **robust** — Hale 2006 entropy reduction as a difficulty signal (distinct from surprisal).
  https://doi.org/10.1207/s15516709cog0000_64
- **solid** — Huettig & Mani 2016: prediction is a *helping hand*, not necessary.
  https://doi.org/10.1080/23273798.2015.1072223
- **solid** — Reuter et al. 2019: encoding is gated **conjunctively** (predict AND redirect),
  not by surprise alone.
- **2023** — Gambi et al.: the error→durable-memory boost **fails in children** because
  episodic binding is immature → splits onto live counts vs consolidated store.
- **negative** — 2025 VOE pupillometry null + Kidd's Goldilocks is about *attention/look-away*,
  not memory write-rate → don't claim an inverted-U "learning law."

## 10. Memory, consolidation & sleep

**Summary.** The best-matched external angle: complementary learning systems (fast
hippocampal / slow neocortical), two-stage word learning (lexical competition needs sleep),
**selective replay of weak/rare items**, sleep-as-abstraction, schema-consistency gating.
Three of these we built under other names (AA, AS, AE); the new refinements are
inverse-count replay and the schema-consistency gate.

- **robust** — McClelland, McNaughton & O'Reilly 1995 CLS. https://doi.org/10.1037/0033-295X.102.3.419
- **robust** — Gaskell/Davis: novel-word lexical competition emerges **after sleep**.
- **solid** — Schapiro et al. 2018: replay is biased toward **weak/infrequent** items (the
  single most-buildable signal AA/AS did *not* implement). https://doi.org/10.1098/rstb.2016.0049
- **2020** — McClelland: rigid "needs-a-night" time-lock → **schema-consistency** gate (the
  replicable version: consistent items integrate fast).
- **negative** — Coutanche fast-mapping hippocampus-independent route **failed to replicate**
  (2023) → no one-shot direct-to-cortex shortcut.
- **2024–25** — Ball et al.: novel-word **semantic** priming is fragile/strategic and
  **dissociates** from form competition → keep form and meaning consolidation separate.

## 11. Production is not comprehension backwards

**Summary.** Production is a harder **many-to-one** selection on the same counts, run as an
incremental frame-then-content pipeline with a forward-model self-monitor. We own every
comprehension organ but only *read* with them.

- **solid** — Levelt's conceptualize/formulate/articulate (seriality is debated —
  cascading/interactive views compete; treat as one modular instantiation).
- **robust** — Comprehension > production gap (a first-words phenomenon).
- **solid** — Levy & Jaeger Uniform Information Density: "that"-omission tracks following
  predictability. https://papers.nips.cc/paper/2006/hash/c6a01432c8138d46ba39957a8250e027
- **solid** — Bock & Griffin structural priming: a **transient** lexical boost + a
  **persistent** abstract (implicit-learning) component → two timescales.
- **mixed** — Pickering & Garrod one-shared-engine / forward-model self-monitor (real debate;
  present as engineering convenience with a cognitive analogue).

## 12. Cognitive-neuro / BabyLM learnability

**Summary.** A permission slip and an honesty bar, not new mechanisms: TP segmentation = A,
surprisal/Bayesian-surprise = AC, constructions = AF, calibrated prediction-error = AB —
in biological vocabulary. The new work is **eval** (BLiMP / impossible-language / N400 / AoA
probes) and harness wiring; the data-mix lessons are configuration.

- **robust** — BabyLM Challenge: ~80–85% BLiMP achievable on 10–100M words (transformers,
  heavily epoch-engineered → don't quote 97%/0.85 as our bar).
- **solid** — Kallini et al. impossible-language ablation (natural vs scrambled) — but the
  gap may be driven by **complexity/entropy**, not naturalness (control or report the
  confound). https://arxiv.org/abs/2401.06416
- **solid** — Michaelov et al.: LM surprisal predicts N400 — a **read-out correlation**, not
  evidence the *mechanism* matches the brain (functional ≠ mechanistic; Warstadt & Bowman).
- **2025** — Ficarra: function words stabilize first under bigram counts; nouns need wider
  context — recoverable with offset Columns (our Exp S), missed by n-grams.

---

*Standing negatives that cut across angles, kept honest in MECHANISMS.md:* SL-benchmark
unreliability (require ensemble convergence, never a single F1); Marcus non-replication
(no variable-binding engine; identity/repetition only); curriculum-hurts (AK + BabyLM; gate,
don't schedule); coherence won't yield to stacked state (AC/AM 1%-backoff law; measure
top-down priors against a **static** prior, never no prior); poverty-of-stimulus boundary
(rare long-distance constructions may be unreachable at ≤100M words — an honest boundary,
not a bug).
