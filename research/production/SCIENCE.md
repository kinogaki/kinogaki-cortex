# SCIENCE — the cog-sci ground truth for production

The distilled science behind the crossing from mimicking language to producing it, organized by
the eight production angles. Per angle: the summary, the strongest cited findings (with an
**evidence-strength** tag and URL), and the latest 2023–2026 work. This is the ground truth that
[MECHANISMS.md](MECHANISMS.md) operationalizes on counts.

Evidence tags: **robust** (multiply replicated / meta-analytic), **solid** (well-supported, some
debate), **mixed** (real effect, contested determinants), **thin** (single-lab or recently
challenged), **negative** (a non-replication or null we must respect).

> A standing methodological note from Exp AM governs this whole library: any top-down or
> partner-state prior must be measured against a **static** prior built from the same count mass,
> never against no-prior — and judged on the right axis (referential/communicative success), not
> on held-out bpc, where AM already ruled persistent state dead on the 99% slice.

---

## 1. From mimicry to generative production (the developmental staircase)

**Summary.** Production is not one switch but a staged path, and every rung has a count/entropy/
contingency analogue. Vocal-imitation reward loop → the functional split (echoic / mand / tact:
the *same form* is a different operant by its antecedent and consequence, not its shape) →
item-based frame-and-slot → type-variation productivity → entrenchment/preemption brakes →
audience design. Our project has crossed the hardest single rung (contingency teaches, Exp BE);
the missing layer is the **functional** one above raw timing — the same form is currently learned
identically whether it was an echo, a name, or a request that got something.

- **robust** — Skinner, *Verbal Behavior* (1957): the operant split is **functional, not
  topographic** — mand (request, reinforced by getting the thing) vs tact (name, reinforced by
  generalized social consequence) vs echoic. The same word is three operants.
  https://en.wikipedia.org/wiki/Verbal_Behavior
- **robust** — Tomasello, item-based / verb-island → abstract frame-and-slot productivity
  (also the acquisition spine). https://www.hup.harvard.edu/books/9780674017641
- **solid** — Goldberg coverage/competition + statistical preemption as the over-generation
  brake; entrenchment is the robust half, preemption the contested half (Ambridge).
- **mixed** — Chomsky's review of *Verbal Behavior* (1959): the mand/novelty objection — novel
  utterances are not directly reinforced. Answered count-natively by **skip-gram backoff** (a
  novel form inherits reward from its sub-units).
- **2024 (warning)** — Kodner et al.: derivability ≠ explanation — 49–89% of *scrambled* target
  utterances also "derive" under a permissive grammar. Any "the producer emits well-formed novel
  utterances" claim (our BD/BL at 87.5% frame-survival) must guard against deriving scrambled
  targets too. Function routing does **not** address word-order well-formedness and must not be
  credited with it.

## 2. Communicative pressure (production for an effect on a listener)

**Summary.** Production becomes communication when a token is selected for its **effect** on a
modeled listener. The Rational Speech Acts (RSA) move is a second, separately-indexed table: a
*listener* model L0(referent | utterance) the speaker scores its own candidates against, minus a
cost. Implicature and informativity fall out of that scoring. The load-bearing part is that the
speaker reasons against a **different** agent's counts, scored on **that** agent's surprisal.

- **robust** — Frank & Goodman 2012, RSA: pragmatic production = a speaker reasoning about a
  literal listener; informativeness − cost predicts human reference. https://www.science.org/doi/10.1126/science.1218633
- **solid** — Goldstein & Schwade infant vocal learning: a **contingent** caregiver reply
  reshapes the *phonological form* of babble — reward attaches to the form, not the content (our
  Exp BE is this in counts; the "reward the form" clause is still unbuilt). https://doi.org/10.1111/j.1467-9280.2008.02117.x
- **solid** — Bates proto-imperative (GET) vs proto-declarative (SHOW): two communicative goals,
  dissociable in development — the dual-goal split that warns against ONE production table.
- **2025** — Nikolaus et al.: communicative-success signals (was the utterance understood / acted
  on) shape child production beyond raw frequency. https://doi.org/10.1016/j.cognition.2024.105977
- **2024–25 (caution)** — Liu & Steedman and "On the Same Wavelength": pragmatic speaking is
  **not** a free byproduct of next-token prediction in LLMs — recipient design must be built, not
  assumed; and a simulated (model) listener risks the agent learning to please *that* model.

## 3. Metacognition & self-monitoring in production

**Summary.** The pre-emit monitor turns mimicry into goal-directed production. The decisive
cog-sci verdict is that the cheapest account is the one already running in our codebase:
conflict-based monitoring (the activation gap between top competitors), which is exactly Exp AG's
vote-margin. Perceptual-loop and forward-model accounts read the same signal a second, more
expensive way. The genuinely new outcome the production turn needs is a **third** branch — not
just emit-or-defer but emit / deliberate / **ask** — and the routing key is whether the
uncertainty is in the *goal* or the *form*.

- **solid** — Nozari, Dell & Schwartz 2011: conflict-based monitor — error likelihood = the
  activation gap between top competitors, per layer, no comprehension loop needed (maps 1:1 onto
  Exp AG). https://doi.org/10.1016/j.cogpsych.2011.07.001
- **solid** — Levelt perceptual loop; Pickering & Garrod forward model (prediction-by-simulation):
  the same signal read more expensively. https://doi.org/10.1017/S0140525X12001495
- **solid** — Goupil & Kouider: pre-verbal infants ask for help **selectively** when they know
  they don't know — metacognition is procedural and effortful, not a free reflex. https://doi.org/10.1073/pnas.1606015113
- **solid** — Thompson, feeling-of-rightness as the System-1 cue that recruits System-2;
  Nelson & Narens FOK/TOT distinction (tip-of-the-tongue = leader cold, neighborhood hot).
- **2023–24 (debate live)** — Teghipco et al. failed to replicate the lexicality effect central
  to the perceptual-loop account; the comprehension-based vs conflict-based monitoring debate is
  **unresolved**. Build agnostically; do not claim to resolve it.

## 4. The world / situation model (comprehension's running state)

**Summary.** Comprehension builds a running situation model; production is the inverse — encode a
chunk of that model into words. But our project already ran this bet and it **lost**: Exp AM's
typed who/where/topic slots plus a Chambers-Jurafsky narrative-event chain bought nothing beyond
the 0.9% backoff slice; a static unigram from the same count mass matched it, and the live
situation was −0.07 bpw *worse* on the 99% non-backoff slice. So the productive use of the
situation model is **not** a richer predictor — it is a **salience** signal for production:
*which* dimension just changed is *what is worth saying* (Levelt's conceptualizer / Slobin's
thinking-for-speaking).

- **robust** — Zwaan & Radvansky event-indexing; Zacks event-segmentation theory: prediction-error
  spikes mark event boundaries / what is reportable. https://doi.org/10.1037/0033-2909.123.2.162
- **robust** — Levelt: the **conceptualizer** selects a preverbal message-tuple *before*
  formulation. https://mitpress.mit.edu/9780262620895/speaking/
- **solid** — Slobin thinking-for-speaking: what gets selected to say is shaped by the language's
  habitual encoding, not by raw next-token likelihood.
- **negative (ours)** — Exp AM: a persistent situation model does **not** predict over long spans;
  the "everywhere" win was pure add-α smoothing repair, matched by a static prior. The law: measure
  against a static prior, on the right axis.
- **2024–25 (contested)** — strict Levelt seriality (conceptualizer → formulator) is challenged by
  interactive/cascading production models; claim only "a message-selection stage helps on-target
  production," not "we validated the Levelt architecture."

## 5. Theory of mind & recipient design

**Summary.** RSA's load-bearing move is a second table answering the *inverse* question (given an
utterance, which referent does a modeled listener recover?). The category error to avoid: the
load-bearing part of theory of mind is **recursive belief attribution** and tracking a divergence
between an agent's representation and the world; a frequency table of what the listener has heard
captures only first-order **common ground / audience exposure** — recipient design, the behavioral
residue, *not* belief. Scope every "second table" to audience-design-by-divergence and judge it
only on redundancy suppression / reference resolution against a static prior.

- **robust** — Clark & Marshall common ground; least-collaborative-effort: speakers tailor to what
  the addressee can recover. https://doi.org/10.1017/CBO9780511620539
- **solid** — Keysar et al.: an **egocentric default** with cue-gated correction — listeners and
  speakers do *not* run full perspective-taking on every word; it is resource-bounded. https://doi.org/10.1111/1467-9280.00211
- **mixed** — de Villiers complementation hypothesis: mastering sentential complements
  ("X thinks that P") scaffolds false-belief; Hale/Schipper-Sluijter show it is **facilitative,
  not necessary**. Build belief-routing and complement-frames as decoupled, not a pipeline.
- **solid** — Pyers & Senghas (Nicaraguan Sign Language): mental-state vocabulary **precedes**
  false-belief performance (the first-cohort signature) — a count-threshold emergence story.
- **negative / caution (ours + field)** — a count table over partner surface tokens **cannot**
  represent the listener knowing something the speaker knows is false (the real false-belief test).
  Liu & Steedman 2024: pragmatic ToM is not free from next-token prediction. Claim recipient
  design, never mind-reading.

## 6. Reward, motivation & contingency for producing

**Summary.** The project proved the **social** half (Exp BE: a contingent reply teaches more than
the yoked same tokens). The missing halves are **intrinsic** (endogenous vocal play / curiosity
that produces *before any listener exists*) and **audience-conditioned** (communicative success as
a per-form reward, not a global learning-rate dial). All three are leaky accumulators, no
gradients.

- **robust** — Oller / Long: >90% of infant protophones are produced **to no one** — vocal play is
  endogenous, not response-contingent. https://doi.org/10.1111/desc.13069
- **solid** — Oudeyer & Kaplan / Forestier: **learning-progress** curiosity (interest ∝ rate of
  competence change) yields self-organizing developmental stages and an automatic curriculum
  (abandons mastered *and* impossible regions). https://doi.org/10.3389/fnbot.2007.00006
- **solid** — Goldstein & Schwade contingency (above); Franklin et al. still-face / extinction
  burst: when contingent replies stop, vocal rate/variance spikes then extinguishes. https://doi.org/10.1037/a0036244
- **2024–25** — Zhang et al.: **in-kind** caregiver replies (re-using the child's channel) sustain
  vocal bouts more than off-channel replies; interpersonal-foraging models — a contingent reply
  makes the next vocalization come *sooner* (bursty cadence).

## 7. Production as goal-directed action / active inference

**Summary.** A speaker is a controller that emits tokens to make the input it gets back match what
it wants (Friston other-directed control). The gap between Exp BE's self-referential dial ("weight
the input my output shaped") and active inference is **the target**: emit so the input comes back
as wanted. Honestly scoped, the audience model is a forward **predictor** (Pickering-Garrod
prediction-by-simulation), gated on its own reliability so a thin model degrades to silence rather
than confabulation. Turn-taking and anticipatory launch fall out of the same precision/gain
account.

- **solid** — Friston active inference / expected free energy (EFE) = pragmatic value + epistemic
  value; the two-term decomposition is **contested as principled vs post-hoc** (2023 debate) —
  present any count version as a *heuristic motivated by the EFE shape*, not as the free-energy
  functional. https://doi.org/10.1016/j.neunet.2017.09.012
- **solid** — Levinson turn-taking: ~200ms inter-turn gaps vs ~600ms production latency forces
  launching production **while the incoming turn is still unfolding** (anticipation, not reaction).
  https://doi.org/10.3389/fpsyg.2015.00731
- **solid** — Pickering & Garrod self-monitoring forward model: simulate the partner with the
  **same** machinery; self-audio attenuation (don't re-ingest your own output as the world's reply).
- **caution** — there is no variational posterior in a count model; "epistemic entropy-drop +
  pragmatic reward" is two count heuristics summed, not EFE. Report it as such.

## 8. Inner speech as a second, self-addressed channel

**Summary.** Vygotsky's other-to-self transfer: regulatory speech from a caregiver is copied
inward, condenses (predication/abbreviation) as it matures, and re-externalizes under load. As
counts: a second self-addressed token band, seeded by copying the partner's regulatory n-grams,
volume-gated on task difficulty (branching entropy), condensing by mutual information as its counts
ripen. The hard honesty check from 2024 is that inner speech is **recruited, not load-bearing**.

- **solid** — Vygotsky: private speech → inner speech; other-regulation internalized to
  self-regulation; abbreviation/predication on mastery. https://www.hup.harvard.edu/books/9780674576292
- **solid** — Berk: private speech **rises with task difficulty** (the falsifiable signature).
- **2024 (the honesty constraint)** — anendophasia (Nedergaard & Lupyan): people with little/no
  inner speech are impaired on verbal working memory and rhyme but **unimpaired** on task-switching
  and categorical perception — inner speech is a recruited tool, not a substrate for every path.
  https://doi.org/10.1177/09567976241245111

---

## Cross-angle laws this library obeys

1. **AM's law** — any partner/common-ground/situation prior is measured against a **static** prior
   from the same count mass, judged on referential/communicative success, never on bpc.
2. **The yoked control** (Exp BE) — every contingency/audience claim rides a registered-before-running
   yoked baseline (scrambled timing or scrambled listener), never PASSIVE-vs-ON (which conflates
   structure with timing).
3. **The grader-leakage fix** (E3) — reward is task/referent-resolution success or low-responder-
   surprise, **never** another LLM's quality judgment; the live partner replies in character, it
   does not grade.
4. **No bpc for the chunk/listener vote** (Exp BK) — the listener table is a production-side rerank,
   judged on uptake, never voted into the next-char prediction pool (BK showed an extra expert hurts
   bpc at every positive weight).
5. **The category-error guard** — a count table over partner surface tokens is recipient design, not
   theory of mind; never claim belief tracking.
