# kinogaki-cortex — architecture spec v1 (the consolidation) — 2026-06-25

The capstone design, fusing everything gathered: the verdict (`KINOGAKI_CORTEX_RESEARCH.md`), the melting pot
(`IDEAS_MELTING_POT.md`), the Numenta-language synthesis, the three research threads (grounding / brain-gating /
GOFAI), and how Monty was actually built (`MONTY_ENGINEERING_DISTILLED.md`). This is a *hypothesis to build and
test*, not a belief — the experiments at the end are designed to kill it fast.

## 0. The thesis (what makes this not-just-an-LM)

A continually-learning, **non-forgetting**, **inspectable** model of text: a wide field of small **sparse**
models that read raw characters, **vote** to a consensus state, grow **concepts** as durable, human-readable
Elements in a `.prism` document, and thread state through that concept space via **verb-like movement
operators** — learning **online**, locally, with no global retrain. The value proposition is *not* perplexity;
it's *online + non-forgetting + cheap + transparent*. The bet is that this reaches useful language behavior;
HTM-for-text never did, so we build experiment-first.

## 1. The State protocol — spec this FIRST (Monty's hardest-won lesson)

Monty's #1 unmet blocker was never stabilizing its Cortical Messaging Protocol. We make our wire format a
first-class, frozen-early artifact. **One message type; models are never passed, only states-of-the-world.**

- **`State`** = `{ location: vector in the current reference frame, features: sparse code (SDR), confidence,
  sender: {id, type, timescale}, t }`. Every unit consumes and emits `State`; a consumer cannot tell whether a
  `State` came from the input stream or another unit. (Steal directly from Monty.)
- Only `State`s, votes, and goal-states cross the wire — never a unit's internal model.
- This uniformity is what buys compositional stacking + parallelism + cross-view voting. Freeze it in week one.

## 2. The unit — a "column" (sparse, private, online)

- Small model holding a **private SDR** vocabulary over a large sparse space + a local predictor. Sees one
  *view* of the stream (a timescale, a direction, a feature projection).
- **Sparsity (idea #1):** representations are sparse/high-dim/low-overlap → overlap = similarity, novelty =
  low overlap with everything known, and (critically) **low interference between concepts → no catastrophic
  forgetting**.
- **Online learning (idea #2):** local, incremental updates per observation; **ACT-R-style activation** on each
  concept (base-level = recency×frequency with decay + spreading activation) as the inspectable strength/forget
  rule. New concepts are *added* (new SDRs / new Elements), never gradient-smeared over old ones.
- Each unit carries a **precision/confidence** that scales both its vote weight and its learning rate.

## 3. The field + voting (idea #4; grounded in Monty)

- **Thousands** of units (not a few experts — MoE is the wrong frame). Voting = **associative consensus over
  private codes**: units learn co-occurrence associations between their own SDRs and others' — *no shared
  dictionary, no transmitted embeddings*; a sparse subsample (~20–30 active cells) settles the population.
- Voting **cuts inference steps and adds robustness; it is not required for recognition** (Monty). 
- **Disagreement is a first-class signal** (uncertainty / candidate boundary). Voting only beats averaging when
  units **err differently** → engineer diversity: timescale, direction (fwd/back), char-vs-morpheme,
  syntactic-vs-distributional, sub-span.
- **We inherit the cheap case and dodge the hard one:** sharing concept *identity/consensus* is the same
  message to everyone (cheap); Monty's unsolved O(n²) pain is sharing relative *location* (per-pair
  reference-frame transforms) — which **has no text analogue**. Skip it.

## 4. Gating — how a winner is chosen (brain thread → 4 mechanisms)

Thousands of voters need a principled gate. Four transferable mechanisms, composed:
- **Default-deny attention region + global-workspace ignition:** an attention signal (a region/scope, broadcast
  globally) gates *who may vote*; a winning coalition crosses an ignition threshold and is **broadcast** back to
  all units as shared context (the consensus/commit). (Monty's attention = a broadcast region that default-denies
  who votes — same idea.)
- **Biased-competition + divisive normalization** = the vote-combination math: a shared suppressive pool (strong
  vote auto-suppresses the rest) with a **multiplicative pre-normalization gain per unit** = the promote/suppress
  operator.
- **Neuromodulatory meta-gating = the anti-forgetting lever:** learn fast only on *large surprise* AND a
  *detected regime shift* (new topic/doc); otherwise hold; on a shift **allocate new capacity rather than
  overwrite**. Separate "surprise within a model" from "surprise about the model."
- **Driver vs modulator edges:** keep *content* edges (what a unit is about) physically separate from *control*
  edges (gain/precision). Error-up / prediction-down so only surprise propagates (sparse by construction).

## 5. Boundaries & the timescale stack (idea #3 boundaries; verdict)

- **Two candidate boundary signals — test both:** (a) **transient Bayesian surprise** = normalized KL belief-shift
  vs a low-error background (the validated event-boundary signal — beats raw surprisal/entropy); (b) Monty's
  **slope-of-best-hypothesis → burst-resample** (when the leading hypothesis's evidence slope drops, spawn new
  hypotheses). Plus **view-disagreement** as a third.
- Boundaries **emerge** (HM-RNN learned detector gating COPY/UPDATE/FLUSH); on FLUSH, promote the finished
  segment's summary one level up.
- **Shallow timescale stack** (letters→words→phrases→sentences→…): a *few* levels, because chunking across
  timescales is real — but the mass of the model is in **parallel voting**, not depth (idea #6, below).

## 6. Concepts as frames; hierarchy emerges (idea #6; GOFAI)

- **A concept = a Minsky frame = a typed `.prism` Element**: a type, named slots with **default-value priors**,
  attached procedures (IF-NEEDED = derived slots, IF-ADDED = propagation). KL-ONE adds **subsumption /
  auto-classification** (place a new concept under its generalizations; flag contradictions) + the **TBox/ABox**
  split (concept schema vs. observation stream).
- **Hierarchy emerges from parallelism (resolves #6):** Society-of-Mind — a **settled coalition becomes a
  higher-level concept node**. Hierarchy is *grown*, not imposed (matches TBT's parallel stance + the brain's
  emergent timescale gradient + Monty's repeated hierarchy reversals → don't hard-wire depth).
- **Learning = promote a settled coalition to a durable node** (K-lines / chunking), storing the *path* that
  reached it for analogical reuse.

## 7. Movement — verbs as learned transition operators (idea #3; the frontier)

- A **movement** = `(operator, {agent, patient, source, goal, instrument, force, …})` — Montague's function +
  Fillmore's typed source/goal roles — *the same Element grammar as a frame*, so concepts and movements share
  one document shape. Backed by 4 converging traditions (Montague / Fillmore / Schank / Talmy).
- Monty endorses the core: **behavior = changes-at-locations**, *same machinery* as static features;
  "value-at-a-position is both the state and the affordance." **But Monty's behaviors are non-compositional and
  orientation-free → nested/compositional movement is OUR frontier** (frames give us the composition they lack).
- Caveats baked in: stative verbs need *non-spatial* learned dims; force ≠ displacement (carry a force/tendency
  param); **operators are meaningless until the space has settled** (§8).

## 8. The reference frame for text — learned, not cursor (the crux; §grounding)

- The cursor scanning characters is the *trivial* motion. The load-bearing reference frame is a **learned,
  low-dimensional conceptual coordinate space** the system *settles into* over a corpus — verbs/operators are
  the learned transitions through it (Hawkins' exact claim for math; TBP's "space need not be Euclidean").
- **How it settles:** don't hand-define axes. Treat candidate operators as proposed moves, predict their
  consequences, and let the geometry **emerge** from minimizing prediction error over a large corpus (slow-feature
  / self-organization applied to *learning the frame itself*).
- **Three candidates to prototype + compare:** (a) positional (too thin), (b) syntactic-tree coordinates
  (movement = traversing a parse), (c) learned discourse/semantic coordinates (the bet; possibly scaffolded by b).

## 9. Grounding — invented internally, but conditionally (§grounding thread)

- Helen Keller proves a *missing modality is bridgeable by language*, **not** that *no world is needed*.
  Othello-GPT proves grounding can **emerge from prediction forcing a latent world model** — *iff the
  environment is built so the only way to cut prediction error is to reconstruct a hidden structure*.
- **Requirements:** (1) an **agency/Markov-blanket** loop — the reader *acts* (chooses what to read/predict/jump)
  and the corpus responds; reading-order/navigation is the action space. (2) A corpus that is
  **unpredictable-without-a-world-model** (raw web text fails — too surface-predictable). (3) A **MinSet** of
  symbols bound to something *checkable* (a task outcome, code that runs, a consistent game/sim) — the
  non-symbolic floor. Then everything else grounds indirectly by definition.
- **Consistency + agency beats sensor richness** (multimodal adds <5%). A cheap, internally-consistent text/nav
  environment can out-ground a passively-consumed video feed.

## 10. Mapping to Core (where it fits 1:1, where it doesn't)

- **Elements / paths** = frames/concepts/States, in an ordered tree (letters→…→themes; each discovered concept an
  Element). **Connections** = driver (content) + modulator (control/gain) + lateral **vote** edges. **Time-sample
  axis** = activation/belief trajectories as the cursor scans (state-over-time is first-class). **Evaluator** =
  the gated consensus / message-passing resolve. **The document** = the durable, diff-able, **inspectable**
  continually-grown model — persist deltas after each observation (no checkpoint-the-net).
- **Outside Core:** the numeric learners (per-unit predictors, the boundary detector, SDR updates) are ML, not
  document evaluation — the learner mutates the document through one surface (the Atlas pattern). Core stores +
  serves structure; it does not train.

## 11. What we steal / adapt / avoid (vs Monty)

- **Steal:** the uniform `State` protocol + "never pass models"; private-code associative voting; default-deny
  attention gating; motor-first (emit target *locations* in concept space); leaky-integrate predict-compare with
  burst resampling.
- **Adapt:** their behavior=changes machinery → our compositional verb-operators; their 3D pose → our *learned
  non-Euclidean* concept frame.
- **Avoid / dodge:** location voting + per-pair reference-frame transforms (no text analogue); hard-wired
  hierarchy depth (they keep reversing it — let it emerge); an under-specified protocol (spec ours early).
- **Our frontier (unsolved by them):** compositional/nested movements, language/abstraction, deep emergent
  hierarchy.

## 12. Experiment plan (ready to code) — kill the bet cheaply

- **A — boundary kill-test (days):** byte predictor → transient Bayesian surprise segmentation of raw chars.
  Metric: boundary P/R/F1 vs word/sentence boundaries on space-stripped text. Baselines: surprisal, entropy, BPE,
  HM-RNN detector, and Monty-style evidence-slope. **Kill if** transient surprise doesn't beat surprisal/entropy.
- **B — the actual thesis (online, non-forgetting):** train continually across text domains in sequence; measure
  **catastrophic forgetting** + update cost vs a streaming byte-LM baseline. This proves/kills the core value prop.
- **C — does the hierarchy earn its keep:** condition next-char prediction on the resolved higher-level concept;
  metric: perplexity reduction vs flat baseline. If none, the hierarchy is decorative.
- **D — does voting beat the best single view:** add diverse views + accuracy-weighted consensus; metric: does
  disagreement add boundary signal and does consensus beat the best single view.
- Datasets: enwik8/text8, PG19, a children's-story corpus; space-stripped Brown/Wikipedia + human
  event-boundary norms for boundary eval. The `.prism` document is the artifact a human inspects throughout.

## 13. Top risks / open questions

1. Does transient Bayesian surprise transfer *down* to letter/word boundaries (validated only at narrative-event
   granularity)? (Experiment A settles it.)
2. Does the conceptual reference frame actually **settle**, or collapse / never converge?
3. Does online sparse local learning reach language-level richness, or hit HTM-for-text's wall?
4. Beyond inspectability + no-forgetting, is there a task where this **outperforms** a byte-LM, or is the honest
   value prop "transparent, editable, continually-learning model of text"? (Experiments B/C settle it.)
5. Can we build a corpus/environment that is genuinely unpredictable-without-a-world-model + a checkable MinSet?

---
*Build order: (1) freeze the `State` protocol + Core schema; (2) Experiment A; (3) if A lives, the unit + field +
voting + ACT-R activation; (4) Experiment B (the real thesis); (5) gating + emergent concepts; (6) movement
operators + the settling frame. Everything gated on the prior experiment surviving.*
