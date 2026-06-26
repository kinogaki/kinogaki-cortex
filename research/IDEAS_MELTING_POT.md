# kinogaki-cortex — the melting pot (our-side ideas + open bets) — 2026-06-25

Our own hypotheses poured in alongside the Numenta corpus. This is the design-thinking layer: what we
*commit* to, the one *hard open bet*, and the *research threads* we're grounding in (filled by saved sources
under `research/sources/{grounding,brain,gofai}/`). Read with `KINOGAKI_CORTEX_RESEARCH.md` (the verdict) and
`NUMENTA_LANGUAGE_SYNTHESIS.md` (what Numenta actually says).

## Part 1 — committed design principles (the non-negotiables)

1. **Sparsity (SDR-style).** Sparse, high-dimensional, low-overlap representations are the substrate.
   *Why:* they're the mechanism behind the next principle (low interference → no forgetting), they're robust
   to noise, and union/overlap is a cheap similarity + set-membership operation. *Lands as:* concept/state
   vectors are sparse binary (or sparse-real) over a large space; "is this the same concept" = overlap; novelty
   = low overlap with everything known. *Risk:* sparse local codes historically trade representational
   richness — the open question is whether sparsity + the right structure reaches language-level abstraction.

2. **Online learning without catastrophic forgetting.** Learn continually from the stream; never a global
   retrain; old knowledge isn't overwritten by new. *Why:* this is the actual value prop vs transformers — a
   model that grows forever and is cheap to update. *Lands as:* local, incremental updates per observation;
   new concepts are *added* (new sparse codes / new Elements in the `.prism` doc), not gradient-smeared over
   old ones. This is THE property to measure (Experiment B in the verdict). *Substrate fit:* Core is ideal —
   snapshot/diff/overlay the growing model, persist after each observation.

3. **Many voting models — a LOT of them.** Not a handful of experts (MoE is the wrong frame): thousands of
   small, redundant, independently-updatable models that reach consensus. *Why:* robustness (no single point
   of failure), graceful degradation, each unit cheaply online-updatable, and *disagreement is a first-class
   signal* (uncertainty / boundary). *Lands as:* the unit of the system is a small "column" model; the system
   state is a weighted consensus; voting + disagreement drives both inference and segmentation. *The catch
   (verify):* voting only beats averaging when members err *differently* — so the models must see genuinely
   different things (different timescales, directions, feature-views, sub-spans). Engineering diversity is the
   real work.

4. **Parallelism over hierarchy (TBP's actual stance — idea #6).** TBT grounds intelligence more in *massive
   parallelism* (thousands of columns modeling the same thing from different vantage points and voting) than
   in a deep feature hierarchy — Hawkins explicitly pushes back on strict hierarchy. *Our resolution of the
   tension with the HM-RNN multi-timescale story:* use a **shallow stack of timescales** (letters→words→
   phrases→… as a few levels, because chunking across timescales is real and validated) but put the **mass of
   the model in parallel voting *within/across* levels**, not in depth. Wide, not deep. *To verify against the
   scraped channel corpus — Numenta says this repeatedly; capture their exact reasoning.*

## Part 2 — the one hard open bet: reference frames & "movement" (idea #3)

This is the crux the whole project lives or dies on, and the part nobody (incl. Numenta) has filled.

- TBP's own operational stance: *"reading and producing language can be framed as a sensorimotor task where
  the sensor moves through the sentence space"* — so "motor = moving through text" is sanctioned, but
  **undefined** (what is "sentence space"? what generates the move?).
- **Our sharper hypothesis (the user's):** maybe *the cursor scanning characters is the trivial/decorative
  motion, and the load-bearing "movements" are linguistic operators — verbs (and prepositions, conjunctions,
  morphology) act as transformations that move the current state through a learned concept space.* "The cat
  *sat on* the mat": `sat-on` is a movement operator applied to a location. Reasoning = path through that space
  (Hawkins' exact claim for math).
- **How the space *settles* (the key idea):** don't hand-define the coordinates. While reading a large corpus,
  treat candidate operators (verbs/functions) as *proposed moves*, predict their consequences, and let the
  geometry of the concept space **emerge** from minimizing prediction error over the corpus — the dimensions
  and the operators co-adapt until they're consistent. (This is slow-feature / self-organization / predictive
  learning, applied to *learning the reference frame itself*, not just filling a known one.)
- **Three candidate "reference frames for text" to prototype + compare** (none inherited — all our design):
  (a) *positional/sequence* coordinates (trivial, likely too thin); (b) *syntactic-tree* coordinates (position
  in a dependency/constituency structure — movement = traversing the parse); (c) *learned discourse/semantic*
  coordinates (a low-D space the system discovers, with verbs/operators as learned transition functions). Bet
  is on (c), possibly scaffolded by (b).
- **Open question to settle empirically:** do "mental moves on verbs while reading" + "settle after a large
  corpus" actually yield a usable, stable conceptual coordinate space — or does it collapse / never converge?

## Part 3 — research threads being grounded (sources saved per thread)

- **#5 Grounding & Helen Keller — `sources/grounding/`.** The bet that the symbol-grounding problem may be
  *softer than assumed*, or that grounding can be *bootstrapped/invented* as part of the world model rather
  than requiring real-world sensorimotor data. Helen Keller as an existence proof (rich conceptual world from
  radically impoverished sensory channels). Whether an *internal* environment + something like **active
  inference** can let representations "settle," and whether cheap proxies (reading order, document structure,
  even watching video) can serve as the "world" to move through. *What we want: is grounding-without-the-world
  defensible, and what's the minimal "environment" our text-reader needs to settle meaning?*
- **#7 Brain architecture & information gating — `sources/brain/`.** The big one: how connectome/anatomy gives
  rise to perception/thinking, and especially **how information is gated** — thalamo-cortical loops & the
  thalamus as relay/gate, basal-ganglia action selection (go/no-go), attention as gating, neuromodulation
  (ACh/DA/NE) gating learning rate & precision, predictive routing, global-workspace broadcast. *What we want:
  a concrete gating mechanism for kinogaki-cortex — what gets promoted, what's suppressed, what controls when
  a concept "wins" and propagates — because thousands of voting models need a principled gate.*
- **#8 GOFAI / Minsky (+ verbs-as-operators) — `sources/gofai/`.** The deliberately-forgotten symbolic ideas
  worth resurrecting now that everyone's on deep learning: Minsky's *Society of Mind* (mind as many small
  agents), *frames* (Minsky's structured slot/filler knowledge — strikingly close to our `.prism` Elements),
  *K-lines*, Schank's *scripts*, semantic networks, SOAR/ACT-R. Plus the linguistic angle: verbs/functions as
  operators over arguments (formal/operator semantics). *What we want: structured-knowledge mechanisms that
  pair naturally with an inspectable concept document, and the steel-man for symbolic structure inside a
  brain-inspired learner.*

## Part 4 — how it composes (provisional)

A wide field of **sparse**, **online-learning**, **voting** small models reads raw text; **transient Bayesian
surprise** + view-**disagreement** carve boundaries across a **shallow timescale stack**; discovered concepts
persist as Elements in a **`.prism` document** (inspectable, diff-able, never retrained-from-scratch); a
**gating** mechanism (from thread #7) decides what's promoted; **verbs/operators as learned movements** (Part
2) thread state through a **concept reference frame the system settles into** over a corpus; grounding is
**invented internally** and settled via prediction/active-inference (thread #5) rather than inherited from the
physical world. Structured-knowledge scaffolding borrows from **frames/Society-of-Mind** (thread #8).

Whether this is a breakthrough or a seductive dead end is decided by the experiments in the verdict doc —
especially: does online learning actually avoid forgetting at language scale, and does the concept space
*settle*. Everything here is a hypothesis to test, not a belief to defend.

## Part 5 — what the research returned (distilled transfers, 2026-06-25)

The three threads landed (sources in `sources/{grounding,brain,gofai}/`). The striking result: **they converge
on one coherent architecture, and it maps onto Core.**

**Grounding (#5) — defensible, but conditional.** Helen Keller / blind-color work proves a *missing modality is
bridgeable by language* — NOT that *no world is needed* (she had touch, agency, emotion, and a corpus written
by grounded humans). The decisive evidence FOR our bet is **Othello-GPT**: a model fed only move transcripts
built an internal board model — grounding emerged from *prediction forcing a latent world model*, no sensors.
So the bet survives **iff** we engineer the environment so that *the only way to lower prediction error is to
reconstruct a consistent hidden structure*, plus a **MinSet** of symbols that bind to something *checkable*
(a task outcome, code that runs, a consistent game) — not just text agreeing with text. **Design lever: a
consistent latent world + agency beats sensor richness** (multimodal adds <5% and doesn't confer reference).
Reading-order/navigation = the action space; the corpus must be *unpredictable-without-a-world-model*.

**Brain & gating (#7) — four mechanisms that answer "how do thousands of voters get gated":**
- *Relay/router hub + global-workspace ignition* — don't let models talk all-to-all; route through a
  **default-deny** gate; a winning coalition crosses a threshold and **broadcasts** to all (the consensus/commit).
- *Biased-competition + divisive normalization* — the actual **math for combining votes**: a shared suppressive
  pool (strong vote auto-suppresses the rabble) with a **multiplicative pre-normalization gain per model** =
  the promote/suppress operator.
- *Neuromodulatory meta-gating* — the **anti-forgetting lever**: learn fast only on *large surprise* AND a
  *detected regime shift*; otherwise hold; on a shift **allocate new capacity rather than overwrite**
  (separates "surprise within a model" from "surprise about the model"). This is principle #2's mechanism.
- *Error-up / prediction-down + driver/modulator separation + timescale gradient on a rich-club hub* — only
  surprise propagates (**sparse by construction**); keep **content edges separate from control edges**; the
  timescale hierarchy *emerges from connectivity*, it isn't imposed.

**GOFAI / Minsky + verbs (#8, #3) — structured runtime to graft onto the inspectable document:**
- **Frames = our typed `.prism` Elements** with default-slot priors + attached procedures (IF-NEEDED/IF-ADDED);
  KL-ONE adds subsumption / auto-classification + the TBox/ABox (schema vs. observation stream) split.
- **Society of Mind = the voting field**, and — *this resolves the hierarchy-vs-parallelism tension (#6)* — a
  **settled coalition becomes a higher-level concept node**, so hierarchy *emerges from parallelism* rather
  than being imposed (matches TBT's parallel stance AND the emergent-timescale finding).
- **ACT-R activation** (base-level recency×frequency decay + spreading activation) = a ready-made, inspectable
  **online strength + forgetting rule** to put directly on concept nodes.
- **K-lines + chunking** = "learning = promote a settled coalition to a durable node, storing the *path* that
  reached it" (analogical reuse). **Production conflict-resolution** (Soar preferences / ACT-R utility) = the
  consensus rule, beats naive argmax. **Impasse → subgoal** = decline/deliberate instead of guessing.
- **verb = learned transition operator** is *sound* — four traditions converge: Montague (verb = function over
  args), Fillmore (typed **source/goal** roles), Schank (small **primitive-operator** basis), Talmy (**Path** +
  **force**, change-of-state = change-of-location). A movement = `(operator, {agent, patient, source, goal,…})`
  — *the same Element grammar as a frame*, so concepts and movements share one document shape. Caveats: stative
  verbs need non-spatial dims; force ≠ displacement (needs a force/tendency param); operators are meaningless
  until the space has **settled**.

**The convergence:** frames(=Elements) + Society-of-Mind(=voting field) + ACT-R activation(=online strength) +
neuromodulatory gating(=when/how-much to learn) + biased-competition(=how to combine) + global-workspace(=commit
/broadcast) + verb-as-operator(=movement) + transient-Bayesian-surprise(=boundaries) compose into ONE design —
and it lands on Core almost 1:1: **Elements = frames/concepts; Connections = driver/modulator + vote edges;
time-samples = activation trajectories; Evaluator = the gated message-passing/consensus; the document = the
durable, inspectable, continually-grown model.** That coherence is the strongest "there's something there"
signal so far — though it's a coherent *hypothesis*, still to be killed-or-confirmed by Experiments A/B.
