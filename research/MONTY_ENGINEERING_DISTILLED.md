# Monty / TBP — engineering distilled (how they actually built it) — 2026-06-25

This is the grounded engineering reference for **kinogaki-cortex**, mined from 629 findings across
167 Numenta Thousand Brains Project research-meeting transcripts (2021–2026). It synthesizes what
their builders *actually do and decided* — merging duplicates, tracking how positions evolved across
the dated meetings, and resolving (or flagging) contradictions.

**Evidence-strength caveat — read this first.** These are *auto-caption transcripts of informal,
in-flux research meetings*, not papers, not shipped specs. Treat them as **direction and reasoning,
not settled fact**. Many "decisions" reverse within months (e.g. whether morphology and behavior share
a column; whether child IDs go up the hierarchy). Where something is shipped/measured I say so; where
it's a whiteboard hypothesis I say so. Quotes are paraphrase-grade. Dates in the video titles are the
only reliable ordering signal, and I use them to call out evolution. Do not cite this document as if
Numenta has *published* any of it.

---

## 1. Headline takeaways (the 6–10 things that should shape kinogaki-cortex)

1. **There is exactly one message format — a pose + features — and one shared mutable "State" object.**
   Everything (sensor module, learning module, motor) both consumes and emits the same `State`: a
   location + orientation in a *common reference frame*, plus a feature list, plus routing metadata
   (confidence, sender id/type). A learning module **cannot tell** whether its input came from a sensor
   or another module — that uniformity is what buys compositional stacking, parallelism, and cross-modal
   voting "for free" (video: 2023/06 - The Cortical Messaging Protocol; video: 2024/12 Overview of the TBP).

2. **Models are NEVER passed between modules; only objects-at-poses, votes, and goal-states are.** This
   is the load-bearing constraint. Internal models stay private; the wire only carries *states of the
   world* (video: 2024/11 - Brainstorming on Compositional Policies - Part 3).

3. **Voting is associative consensus over IDs, not weight-multiply attention.** Each module keeps its
   *own private SDR* for an object and learns *co-occurrence associations* with other modules' codes.
   No shared dictionary, no transmitted SDRs. A sparse random subsample (~20–30 active cells) is enough
   to force the whole population to settle (video: 2024/09 - Q&A; video: 2025/02 - Review of the Cortical
   Circuit; video: 2025/10 - Using Episodic Memory Replay). Voting mainly **cuts the number of sensing
   steps**, it is not required for recognition (video: 2024/01 - Current Capabilities).

4. **Voting on *location* is the unsolved hard part; voting on *ID/orientation* is fine.** Sharing object
   ID or orientation = the *same* message to everyone (cheap). Sharing relative *location* requires a
   *different* reference-frame transform per module pair — O(n²) and biologically implausible. The
   current "voting" is admittedly **message-passing**, not the original settling dynamic (video: 2025/08 -
   Review of Reference Frame Transforms; video: 2026/03 - New Insights Around Deformations).

5. **Attention is a 3D region in egocentric/body space, broadcast globally, that GATES who votes.** Not a
   sensor command, not a retinal patch — a *volume* (that can collapse to a point). Only columns sensing
   *inside* the attended region get to vote and pass information up; others are silenced even if receiving
   input. Recognition starts with the largest object in the region and *narrows* until something is
   recognized (video: 2025/11 - Discussing Attention; video: 2025/12 - Off-Object Observations).

6. **Morphology and behavior are the SAME machinery with different input.** Morphology = static
   *features at locations*; behavior = *changes at locations*. By 2025 they committed to *separate columns*
   for the two (one reference frame each). "Behavior is just a twist in the input" — and it's the idea they
   consider genuinely novel (video: 2025/08 - Recap of Behavior Solutions; video: 2025/12 - 2026 and Beyond).

7. **Causality = behavior-to-behavior association, learned via spike-timing on a broadcast bus.** Only
   changes can cause changes; a static object causes nothing. Recognized behaviors are *broadcast widely*
   to layer-1 apical dendrites and the receiver associates whatever was active just before it fired — the
   receiver doesn't know if the broadcast carries an object ID, a behavior, a state, or an imagined goal
   (video: 2025/11 - More Discussions Around Learning Causality).

8. **Everything is sensorimotor and motor-first.** Every learning module emits a motor output, which is a
   *target location* (a 3D point + "blurriness"/confidence Gaussian), never a muscle command — subcortex
   translates location → movement (video: 2022/02 - Object Representation; video: 2025/12 - 2026 and Beyond).

9. **Evidence accumulation is a leaky-integrate predict-then-compare loop with dynamic hypotheses.** Match
   adds evidence, mismatch subtracts; a *bounded, decaying* EMA fixes an age-bias bug; the *slope* of the
   best hypothesis (not episode boundaries) triggers "burst" resampling of new hypotheses — the route to
   unsupervised continuous inference (video: 2025/09 - Brighton Final Presentations; video: 2026/03 -
   Burst Sampling).

10. **The interface contract is admittedly NOT yet a stable spec.** "CMP v1 published" is an explicit
    unmet v1.0 blocker; the informal protocol is named as the source of API instability (video: 2025/12 -
    Our Path to an Easy-to-Use Platform). We should treat *our* equivalent spec as a first-class artifact.

---

## 2. Voting & consensus

**What they do.** Each module independently builds its own model/reference frame; after learning, object
IDs are *associated across modules* (layer 2/3 ↔ layer 2/3 lateral connections). Voting is an *inference*
process that **narrows existing hypotheses, never transfers features** (video: 2026/04 - Brainstorming on
Voting). Crucially, columns do **not** share an SDR for the same object — they learn associations between
their *different* codes, and a single cell connecting to a random ~20-cell subsample anywhere is enough for
the network to settle (video: 2025/02 - Review of the Cortical Circuit; video: 2024/09 - Q&A).

**What voting is *for*.** Primarily to **cut sensing steps**, not raise accuracy: 5 modules roughly halve
the steps; an episode can terminate when as few as 1-of-5 modules converges, making the ensemble faster
than a single module (video: 2024/01 - Current Capabilities). Monty is *not dependent* on voting — a
single module works "like feeling a mug with one finger" (video: 2025/07 - Thousand-Brains Systems).

**The thing they got wrong, then fixed, then questioned again.** Early voting (the "columns paper") was a
**bag of features** — object-ID-only, no relative sensor geometry — which gives texture bias and no shape
robustness (video: 2025/07 - Thousand-Brains Systems; video: 2021/11 - Intro to the AI Bus). The fix:
account for the *sensor displacement* between modules and transform incoming pose hypotheses into the
receiver's frame before integrating (video: 2023/01 - Comprehensive Overview). But this transform-per-pair
scheme is the problem child:

- It's **O(n²)** — each receiver computes a *distinct* displacement+rotation for every other module's
  votes (video: 2025/07 - Flash Inference; video: 2025/08 - Review of Reference Frame Transforms).
- Jeff repeatedly flags it as **message-passing, not biological "voting"** (spreading activation that
  settles): associative L2/3 (and L6) connections "have nothing in between" to do a reference-frame
  transform, and each pair needs a *different* one (video: 2026/03 - New Insights Around Deformations).
- **ID and orientation voting are fine** (same message broadcast to everyone); only *relative-location*
  sharing is implausible (video: 2026/03 - New Insights).

**Proposed escapes (none shipped):**
- *Centroids* — each module emits one fixed point per object (e.g. centre of mass) in a shared/global
  frame; votes are cast on the centroid. O(LMs) linear, transform computed once per outgoing vote
  (video: 2025/08 - Review of Reference Frame Transforms).
- *Rapid replay* — after each sensation, replay ~3–10 salient locations/displacements to *all* columns
  (even non-sensing ones) via a shared buffer; timing matches hippocampal replay (50–100 ms) vs saccades
  (3–5/s). Same mechanism also does shared learning (video: 2026/03 - New Insights).
- *Associative forcing* — once one column locks an object ID, that ID *forces* the equivalent reference
  frame in all other columns, enabling live morphology learning without replay (video: 2026/04 - Focus
  Week Recap on Temporary Models).

**Concrete mechanics (as built).** Sending module transforms its pose hypotheses by sensor displacement
and *also sends its own pose* so the receiver can compute the displacement; receiver maps votes into its
model frame, looks up points in a *search radius*, distance-weighted-averages, scales votes to [−1,1] via
max/min, and thresholds (top ~10–20%, e.g. >0.8) before adding to evidence (video: 2022/09 - Initial
Evidence-Based LM; video: 2023/03 - Speedup Discussions; video: 2023/01 - Comprehensive Overview). Monty's
*managing class* (not the modules) routes votes through a central hub; modules don't know each other's
sensor positions. Inter-module bandwidth is *tiny* relative to intra-module compute — the reason scaling
stays tractable, but real scale needs *local* voting among nearest modules (video: 2026/02 - A Thousand
Brains on a Thousand Chips). A connectivity matrix lets voting be made sparser/attentional (video: 2025/07 -
Flash Inference). Two *qualitatively different* voting regimes exist: within-modality (pose-transformed)
vs cross-modality (pure object-ID, no relative pose) (video: 2023/03 - Monty Compared to Transformers).

**Open/unfinished.** Cross-LM *pose* voting is on the to-do list but unimplemented (video: 2024/10 -
Literature Review of 6DoF). Object-ID used majority/consensus (an object removed only if minus-votes
outnumber plus-votes) while pose did *not* — a flagged inconsistency (video: 2022/09 - Initial Evidence-
Based LM). The attentional multi-LM case is partly unaddressed: attending different spots activates
*completely different* columns (video: 2026/03 - Burst Sampling).

**→ For kinogaki-cortex:**
- **STEAL:** voting = *associative consensus over private codes* (each concept-node keeps its own sparse
  code; "same concept" = learned co-occurrence / overlap, not a shared dictionary). Sparse random
  subsampling is enough — we do *not* need all-to-all. Disagreement is a first-class signal.
- **STEAL:** the "ID/orientation broadcast is cheap, *relative location* is expensive" split — in text
  there is no metric sensor geometry, so we sidestep the entire O(n²) transform problem. Our voters share
  *what concept/state*, which is exactly the cheap case.
- **ADAPT:** the *centroid* idea → each voter emits one stable summary code per hypothesis into a shared
  field, voted on directly (linear cost). This is precisely our "sparse voting field."
- **ADAPT:** vote scaling to [−1,1] + a percentile threshold (only the top-K hypotheses count) maps onto
  biased-competition / divisive-normalization from the melting-pot gating finding.
- **AVOID:** transform-per-pair message passing (their own rejected scheme); majority-removal rules that
  forbid a confident voter from winning (Hawkins pushed back on this).

---

## 3. Gating & attention

**The settled core model (stable across 2025–2026).** Attention specifies a **region of 3D egocentric/body
space** (possibly conceptual space), broadcast *globally*, that **gates voting**: only columns sensing
*inside* the region vote and propagate up; others are silenced even when receiving input (video: 2025/11 -
Discussing Attention; video: 2025/10 - Using Episodic Memory Replay; video: 2025/12 - Off-Object
Observations). Recognition tries the **largest** object in the region first; on failure it **narrows**
(word → letters → strokes). The region is a **volume that can collapse to a point** (a sudden sound), not
a coordinate — the mechanism must be able to specify and detect a *boundary* (video: 2025/11 - Discussing
Attention).

**Two regimes.** *Model-free* (saliency/bottom-up, hard to resist, can't be initiated top-down) vs
*model-based* (a higher region directs a lower region where to attend — only possible *after* a
compositional object is learned) (video: 2025/07 - Hypotheses Resampling). Saliency *alone* can't hold
attention on a region (it over-fires on high-contrast boundaries), so the working recipe is **"one
saliency-driven move, then switch to model-based"**: peripheral columns with high prediction error emit a
goal (a 3D point) telling the fovea where to go (video: 2026/01 - Using Saliency). A **goal-state selector**
was added as an explicit arbitration checkpoint between competing goals from learning modules vs sensor
modules (video: 2025/12 - The Legend of Monty).

**Attention ≠ fixation.** A column can covertly attend a body-centric location before the eyes move; moving
fixation doesn't itself change attention (video: 2025/12 - Off-Object Observations). Whether the attended
location stays on the same object or switches decides whether the prior object is "forgotten" everywhere.
Uniform regions emit *no* signal (center-surround cells only report *change*).

**Evolution / tension.** "Region vs point" recurs; Jeff lands on *fundamentally a volume*. There's a
unification hypothesis that attention is *several things at once* — L5 motor → superior colliculus AND L6a
location cells → thalamus — and an attended *region* may just be the joint extent of many columns'
locations, not a single projection (video: 2026/02 - Top-Down Connections). For behaviors, the **attended
area is unsolved**: it must be dynamically sized, defined in body space (not receptive-field terms), and
both bottom-up and top-down; **end-stopping** (cells that stop firing when a feature exceeds a size) is the
candidate mechanism that restricts motion-measurement to a sub-region (video: 2026/05 - Deeper Dive into
Trajectory Memory; video: 2026/04 - Summary of New Ideas). Input *gating* of off-object/parent input into a
child module is explicitly **missing** today (video: 2025/09 - Modeling Compositional Representations).

**Learning constraint.** During *learning* you attend/bind **one location at a time**, even with multiple
sensors — multi-sensor parallelism is *inference-only*. This forces shared learning to be serial/replay
(video: 2026/04 - Brainstorming on Voting).

**→ For kinogaki-cortex:**
- **STEAL:** attention = a **broadcast region that gates who votes**, with default-deny — this is exactly
  the global-workspace ignition + relay-hub gate from our brain-thread finding. The "attended region" in
  text = the current span/scope; only voters inside it commit.
- **STEAL:** the **largest-first, then narrow** recognition policy → coarse concept first, refine to
  sub-structure on failure. Maps to our shallow-timescale stack.
- **STEAL:** "one model-free move, then model-based" → spend cheap surprise-driven exploration once, then
  let high-prediction-error voters *request* where to look next (verb/operator-driven movement).
- **ADAPT:** attention as a *volume that can collapse to a point* → an attention scope with a width/blur
  parameter (mirrors our goal-state "blurriness" Gaussian).
- **AVOID:** pure saliency loops (they don't converge on meaning); defining attention in terms of fixed
  receptive fields (they explicitly reject this).

---

## 4. Messaging protocol (the CMP)

**The contract (most stable, best-specified theme).** The CMP message = a **pose** (location + orientation
in a *common* reference frame) + a list of **features** + routing info (confidence 0–1, `use_state` flag,
unique `sender_id`, `sender_type` SM-vs-LM). Implemented as a single **`State` class** that *every*
component consumes and emits; the same attributes are reinterpreted by context (observed state from a
sensor, hypothesized state from a module, list-of-states for votes, target state from motor) (video:
2023/06 - The Cortical Messaging Protocol). A module **can't tell** if input came from a sensor or another
module — both are features-at-poses, which is what enables compositional hierarchy and cross-modal voting
(video: 2024/12 Overview of the TBP).

Key structural facts:
- **Orientation** is three orthonormal pose vectors + a `pose_fully_defined` flag for symmetry. Non-
  morphological features (color, temperature, curvature) are optional, modality-specific, pose-invariant
  (video: 2023/06 - The Cortical Messaging Protocol).
- The space need **not be 3D Euclidean** — only independent path-integrable dimensions (2D or abstract
  spaces work identically). *This is their explicit route to abstraction/language* (video: 2023/06 - The
  Cortical Messaging Protocol).
- Communication is in a **shared body-centric frame** — every module knows its sensor's location relative
  to the body, so it can output object-location-relative-to-body by simple addition; sidesteps what/where
  pathways (video: 2025/04 - Behavioral Models and Predictive Learning).
- The input to a module is **just location + orientation + features of where the sensor is sensing** —
  *not* the agent/sensor body pose (video: 2025/12 - Processing Off-Object Observations).
- An LM outputs 0, 1, or 2 messages/step: a **GOAL** (down, to a lower module) and/or a **PERCEPT** (up,
  to a higher module) (video: 2025/08/28 GPU Support Meeting).
- Treated as **zero-delay** with a global clock (PyTorch-distributed style); fine temporal structure
  stays inside modules (video: 2021/11 - Research Questions About the AI Bus).
- Motor output is a **location signal**, never muscle commands (nothing in cortex projects to muscle);
  subcortex translates location → movement (video: 2022/02 - Object Representation).

**The one hard rule.** *Models are never passed.* Only objects-at-poses, votes, and goal-states — always
in the format of *states of the world* (video: 2024/11 - Compositional Policies Part 3). The CMP carries
**states, not actions** — a higher module emits a goal-state, the lower module decomposes it into a
sub-state-goal, and only the lowest level turns state into muscle movement subcortically (video: 2024/11 -
Compositional Policies Part 1).

**Evolution on what goes UP the hierarchy (this reversed twice):**
- *2022/03:* don't send the *stable* pooled ID up; send the *unique, changing, location-specific* SDR;
  the parent associates it with a location (video: 2022/03 - How Compositional Models are Constructed).
- *2025/02:* child ID is **NOT** passed up as a feature — the parent never knows what the child is; child
  sends only *orientation* and votes laterally on ID; the parent learns a backward L6a association telling
  the child what to expect (video: 2025/02 - Review of the Cortical Circuit).
- *2024/08:* what *does* go up is a **similarity-encoding SDR** (not the full graph, not a bare integer) —
  and they want *two* output SDRs: a generic/overlapping one (generalization) and a pattern-separated
  random one (telling near-identical objects apart, the chipped-mug problem) (video: 2024/08 - Encoding
  Object Similarity in SDRs).
- *2025/03–2026/02:* top-down feedback carries object ID + a learned location→location association so the
  lower region can **re-anchor instantly** when re-entering a sub-object, with no need for the parent to
  know the child's internal frame (video: 2026/02 - Top-Down Connections).

The same L2/3 output **axon splits** to two destinations: lateral (voting) and hierarchical (compositional
association) — *one output serves both* (video: 2025/11 - More Discussions Around Learning Causality).
They adopt **Sherman & Guillery**: the dominant inter-region path is cortico-*thalamo*-cortical, with the
thalamic relay doing the orientation transform; direct cortico-cortical links are weak/modulatory (good for
voting and predictions) (video: 2023/01 - Hierarchy in the Neocortex). A *driving* input (center-surround,
rotatable by thalamus) is distinguished from a *biasing* SDR input (object/behavior ID — nothing to rotate)
(video: 2025/03 - Thalamocortical Circuitry).

**Layer 1 as a context bus.** Beyond specific projections, L1 carries "noise going past all the time" —
other regions, neuromodulators/saliency, day-of-week-like context — and columns *associatively pick*
whatever is predictive. This is the proposed substrate for external state/context and for causal
broadcast (video: 2025/09 - How Does the Neocortex Split and Combine Models; video: 2025/11 - More
Discussions on Causality).

**Honest admission.** The CMP is currently **unspecified in the implementation** and listed as a v1.0
blocker; they want a real *spec* (message format + module behaviors + what happens during voting/goal
emission) so community modules interoperate by matching the protocol (video: 2025/12 - Our Path to an
Easy-to-Use Platform; video: 2025/09 - Brighton Retreat Kickoff). The likely future change: from
**locations to movements/displacements** (video: 2024/09 - Q&A).

**Cross-modal transfer is real and demonstrated.** Train a module on touch, swap in a vision sensor
module, recognize the same objects at similar accuracy — only touch-absent features (color) are lost
(video: 2024/01 - Current Capabilities).

**→ For kinogaki-cortex:**
- **STEAL — the biggest single transfer:** *one uniform message type that every component consumes and
  emits*, and *never pass internal models, only states/poses-of-the-world*. This is `.prism` Elements as
  the wire AND the store: concepts/states/movements share one Element grammar (matches the melting-pot
  "frames = Elements" convergence). Our `State` analogue = a sparse code + a position-in-concept-space +
  feature slots + confidence + sender id.
- **STEAL:** the explicit "**the space need not be Euclidean — this is the route to language**" stance.
  This directly licenses our *learned discourse/semantic coordinate* bet (melting-pot Part 2c).
- **STEAL:** zero-delay global clock — simplify our message bus the same way; keep fine temporal structure
  inside voters.
- **STEAL:** two output codes — a **generic/overlapping** code (class similarity) + a **pattern-separated
  unique** code (instance distinction). Our concept nodes should emit both; "same concept" uses generic,
  "this exact instance" uses unique.
- **STEAL:** **driver vs modulator edge separation** (content edges vs control/bias edges) — exactly our
  brain-thread finding; lands as two Connection types in the document.
- **ADAPT:** goal-state-down / percept-up as our two message directions; verbs/operators are the
  goal-down "movement" operators.
- **AVOID:** their own admitted mistake — leaving the protocol informal. **Write our CMP-equivalent spec
  as a first-class, versioned artifact early.** And note their hierarchy-direction churn: keep our
  "what propagates up" decision *configurable*, not hardcoded.

---

## 5. Object, state & behavior models

**The unifying claim (stable since 2025).** Morphology and behavior models use the **identical learning-
module mechanism**; the only difference is the input — morphology gets *features at locations* (static),
behavior gets *changes at locations* (video: 2025/08 - Recap of Behavior Solutions). "An amazingly simple
switch from what the inputs are." They consider behavior models the **genuinely novel TBP idea** and the
most exciting (video: 2025/12 - 2026 and Beyond; video: 2026/01 - Brainstorming Compositional Benefits).

**Three (then more) model types.** Morphology (static 3D shape), behavior (changes over time), and
*feature* models (a 2D color/texture surface that wraps around an object, with 2D movement vectors). Some
materials (water, fire, clay) have *essentially no morphology* — they're defined by behavior + low-level
features (video: 2026/02 - Modeling Materials; video: 2025/07 - Proposal for Separating Behavior). Morphology
strictness is arguably a *continuum* (rigid mug → folded shirt → crumpled pile) handled by one module with
looser-to-tighter frames, recognized via local features still in correct *relative* position (video:
2026/02 - Modeling Materials).

**Separate columns — a reluctant but firm decision.** Keeping morphology + behavior in one column would
need *two simultaneous reference frames* — judged near-impossible. So: separate columns, behavior applied
to morphology *via hierarchy* (video: 2025/09 - HTM Sequence Memory; video: 2025/07 - Proposal for
Separating Behavior). Behavior models have **no orientation of their own** and are **not rotation-invariant**
(an arc one way ≠ the same arc rotated); orientation "tags along" from the paired morphology column (video:
2026/05 - Trajectory Memory). Behavior models **cannot be compositional** — multiple independently-moving
parts must be tracked as *separate* behaviors (video: 2026/05 - Deeper Dive into Trajectory Memory). A
behavior is *always paired* with a child morphology model; it carries a single movement vector and doesn't
know which child it is (reusable across children) (video: 2026/04 - Summary of New Ideas).

**What "state" means (this is still moving in 2026).** A behavioral *state* = a **view/snapshot of an
evolving process** ("the hinge is *opening*", not "open") = a morphology key-frame with changing input
(video: 2025/12 - Focus Week Recap). Behaviors are **not necessarily high-order sequences** — most are
externally driven and you can move through states in *any* order (open the stapler partway, reverse)
(video: 2025/11 - Exploring How Columns Learn Behavior). The most recent (2026/05) and most radical move:
**remove state variables from morphology entirely** — a morphology model becomes a *union of possible
features at a location*; the **observed feature at a location IS the state**, and it doubles as an
*affordance* that invokes the right behavior at the right point (video: 2026/05 - Discussing State in
Morphology). A child's state is defined *relative to the parent* (a rocker is up/down only relative to the
switch plate) (video: 2026/05 - Discussing State in Morphology).

**Causality.** Working definition: "*a change in state in one object invokes a change in another*"; **only
behaviors cause behaviors** (video: 2026/02 - Summary of Brainstorming Week; video: 2025/11 - More
Discussions on Causality). Learned *very fast* (one co-occurrence) via STDP (itself a causal rule).
Multi-object interactions (scissors cutting paper) = two separate behaviors + a causal link, **not** a
learned model of the pair; invoking the second behavior needs *both* the causal input (other behavior
active) *and* the morphology/material being present (video: 2026/02 - Modeling Object Interactions). A
behavior model is sensitive *only* to change and is "done" the moment change stops (video: 2026/02 -
Summary of Brainstorming Week).

**Variations & generalization.** Object variations (logo/no-logo, stapler open/closed) were first handled
by an **external state/context signal broadcast on L1** rather than internal state (a column can't hold many
state variables without SDR mixing) (video: 2025/09 - How Does the Neocortex Split and Combine Models). The
favored scheme: a **generic/average base model** + each state storing only *deviations*, falling back to
generic where nothing's stored — though Hawkins dislikes "average" and any privileged master/parent object
(learning order shouldn't matter) (video: 2025/09 - Split and Combine). Compositional models give a "happy
balance" for generalization — tolerant *within* each child while constrained by the spatial *arrangement* of
children (flat models force one bad tolerance knob: strict misses variation, loose becomes a bag of
features). Their named ML benchmark is **Omniglot**, handled best as *strokes drawn in order* (a sensorimotor
path), not images (video: 2026/01 - Demonstrate Benefits of Compositional Models).

**Hypothesis representation (as built).** Each hypothesis encodes exactly three things: object identity,
sensor location on the object, object orientation relative to the stored model. Planned additions: scale,
morphological state, position-in-behavior-sequence (video: 2026/03 - Burst Sampling). Two hypothesis *types*:
*informed* (depends only on current observation — good for *switching* objects) vs *offspring* (seeded near
a high-evidence hypothesis, particle-filter-style refinement — good for honing the *same* object's pose)
(video: 2026/03 - Burst Sampling). Symmetric objects correctly retain many equally-good poses (a strength,
validated via Chamfer distance, not single-ground-truth pose error) (video: 2026/03 - Burst Sampling).

**Shipped (v0.33).** A **2D sensor module** that detects dominant edge orientation in a patch and learns a
2D surface model by accumulating a moving tangent frame ("unrolling" developable surfaces), letting the same
logo be recognized across flat/cylinder/sphere/mug (100% logo-only transfer; sphere hardest); color
invariance via RGB→grayscale up front (video: 2026/05 - The 2D Sensor Module).

**Temporary vs permanent models.** ~90% of waking cognition is building *temporary* compositional
arrangements (hippocampal — "where did I put my glasses") on top of pre-learned static models; rapid
one-shot learning stores only a *coarse category tag* per location, detail proportional to familiarity
(video: 2026/01 - Demonstrate Benefits; video: 2025/10 - Episodic Memory Replay).

**→ For kinogaki-cortex:**
- **STEAL — the central architectural transfer:** *same machinery, different input* → static structure
  (concepts/frames at positions) and dynamic structure (verbs/operators = changes at positions) are the
  **same learning unit**. "Behavior = a twist in the input" is exactly our "verb-as-movement" bet.
- **STEAL:** **"only behaviors cause behaviors," learned in one shot via spike-timing** → causal links
  between *transitions/operators*, not between static concepts; learnable from a single co-occurrence;
  fits online learning with no retrain.
- **STEAL:** the 2026/05 collapse — **the observed feature/value at a position IS the state** (no separate
  state variable) and **doubles as an affordance** → in text, the word/token at a position is the state and
  cues which operator applies. Elegant and directly implementable.
- **STEAL:** generic-base + store-only-deviations as our online-learning + anti-forgetting representation
  (matches ACT-R activation / store-the-path findings). Variance-at-a-location = a confidence signal.
- **ADAPT:** *informed vs offspring* hypotheses → fast concept-*switching* candidates vs refinement
  candidates for the *current* reading state; resample by evidence slope (particle filter over meaning).
- **ADAPT:** Omniglot-as-strokes → treat language as a *path/sequence of moves*, not a bag of tokens.
- **NOTE/CAUTION:** behavior models being *non-compositional* and *orientation-free* is a real limit — if
  our verbs need composition (nested clauses) we are *past* what Monty's behavior model does; that's our
  research frontier, and we shouldn't assume their mechanism extends.
- **AVOID:** forcing two reference frames into one unit (their rejected design); a privileged "master"
  concept (Hawkins' objection — order of learning must not matter).

---

## 6. What they admit is unsolved (and where they hedge)

**Language & abstraction.** There is *no built language capability* — only the licensing claims that the
CMP space "need not be 3D Euclidean" and is "the route to abstraction" (video: 2023/06 - CMP), and the open
speculation that a column may model the *behavior of another column's output* with no relation to physical
movement, raising whether all abstract concepts ride on grid-cell-like spaces or on movement-from-other-
regions (video: How Embodied Movements Might be Learned). **This is exactly the gap kinogaki-cortex is
trying to fill — Numenta has *not* done it.** Take their reference-frame framing as inspiration, not proof.

**Hierarchy.** Direction of information flow up the hierarchy *reversed multiple times* (2022 send-changing-
SDR → 2025 send-only-orientation/never-send-child-ID → similarity-SDR). Whether a child passes its *state*
up is explicitly open (current lean: **no** — parent associates its own location with the child's state via
back-projection) (video: 2025/01 - Compositional Policies Part 7). Hawkins repeatedly pushes back on *strict*
hierarchy; the dominant pathway is thalamic, and "hierarchy or heterarchy" is an open question (video:
2025/07 - Hierarchy or Heterarchy). **Treat hierarchy as emergent and configurable, not load-bearing.**

**Scaling.** Multiple columns are admittedly *slow*, "partly fundamental, partly engineering" (video:
2022/10 - Action Policies & Hierarchy). Real inference benefit at scale needs *local* voting among nearest
modules + custom interconnects respecting sparse connectivity + pipeline-parallelism across levels — *not*
the current central-hub all-collect-redistribute scheme (video: 2026/02 - A Thousand Brains on a Thousand
Chips). Voting scaling and tie-breaking had to be *newly built* and are non-trivial (video: 2025/07 - Q3
Roadmap).

**The protocol itself.** CMP is unspecified/informal and a named source of API instability; "CMP v1
published" is an unmet v1.0 blocker (video: 2025/12 - Our Path to an Easy-to-Use Platform).

**Location voting / reference-frame transforms.** The whole O(n²) per-pair transform problem (Section 2) is
unsolved; centroids and replay are *proposals*, not shipped. "We've never worked out the details" of how the
shared frame is established (video: 2024/09 - Q&A).

**Attention for behaviors.** The dynamically-sized, body-space attended area for measuring whole-region
motion has *no mechanism yet* (video: 2026/05 - Deeper Dive into Trajectory Memory). Input gating of
off-object input into a child module is missing (video: 2025/09 - Modeling Compositional Representations).

**State representation.** Whether state is a 4th grid-cell dimension (x,y,z,state) or a *separate*
representation feeding L1 is an open fork (video: 2025/09 - HTM Sequence Memory); and the 2026/05
"remove-state-entirely" proposal contradicts earlier "external state on L1" — *actively in flux*.

**Object class/ID.** How class/ID is assigned is "admitted unsolved" and may be *functional/affordance-
based* (a gym ball = chair/table/equipment by use), not morphology-based (video: 2026/01 - Learning
Efficiently in a Hierarchy).

---

## 7. How this updates our design

Monty's corpus is strong corroboration for several melting-pot commitments and a useful warning on others.
The **sparse voting field** is well-grounded: their voting is *associative consensus over private sparse
codes* with random subsampling sufficient to settle — so we keep voters' codes private, treat overlap as
"same concept," and use disagreement as a boundary/uncertainty signal; we *inherit their cheap case* (vote
on ID/state, which in text is all there is) and *dodge their hard case* (relative-location transforms,
which have no text analogue), optionally adopting their **centroid** trick (one summary code per hypothesis
into a shared field, linear cost) and their **scale-to-[−1,1] + percentile threshold** as our biased-
competition combine. **Frames-as-Elements** is their single biggest transfer made literal: one uniform
message type that every component emits and consumes, *models never passed — only states/poses-of-the-world*,
so concepts, states, and movements share one Element grammar in the `.prism` document, and we should heed
their explicit regret and **write the protocol spec as a versioned artifact, not an afterthought**.
**Gating** lands cleanly: attention as a *broadcast region that default-denies who may vote*, largest-first
then narrow, "one model-free move then model-based" — i.e. global-workspace ignition + relay-hub gating from
our brain thread, with their **driver-vs-modulator edge split** mapping to two Connection types.
**Verb-as-movement** gets its strongest external endorsement: "behavior = the *same* machinery with changes
instead of features," causality as one-shot behavior-to-behavior STDP, and the elegant 2026/05 collapse where
*the value at a position is both the state and the affordance that invokes the next operator* — but with a
sober caveat that *their* behavior models are non-compositional and orientation-free, so nested-clause
composition is genuinely **our** frontier, not borrowed. Finally, **online learning** is supported by their
generic-base + store-only-deviations scheme, bounded-decaying-EMA evidence, and slope-triggered burst
resampling (add hypotheses, prune by slope, never global retrain) — all of which sit naturally on Core's
snapshot/diff/overlay model. The honest bottom line: Monty validates our *substrate and mechanisms*, but on
the two things we most need — **language/abstraction and deep hierarchy** — Numenta is hedging or empty, so
those remain hypotheses for Experiments A/B to kill or confirm, not results to inherit.
