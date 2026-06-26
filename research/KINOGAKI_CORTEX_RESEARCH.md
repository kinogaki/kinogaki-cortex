# kinogaki-cortex — research verdict & build design (2026-06-25)

A deep, source-verified assessment of the hypothesis: *a .prism-described hierarchical learning network
that grows layers of abstraction by using prediction-error/ambiguity as concept boundaries, reads text at
multiple timescales, runs multiple world models that vote, and later generates — with "motor" = moving
through text.* Built on Core (the Prism document model). Method: fan-out web research, 21 primary sources,
25 falsifiable claims verified by 3-vote adversarial check (22 confirmed, 3 killed). Citations inline.

## Verdict in one paragraph

**There is something there — but it is narrower than the framing, and the value proposition is NOT "beat a
transformer on perplexity."** Three mechanisms are well-founded: (1) prediction-error as a boundary signal
*works, but only in a refined form* — **transient Bayesian surprise** (a change in the *shape/confidence* of
the next-token distribution), not raw surprisal or entropy, predicts human boundaries [Kumar/Zacks, Cog Sci
2023]; (2) **emergent multi-timescale hierarchy** is a solved, validated mechanism — the HM-RNN's learned
binary boundary detector gating COPY/UPDATE/FLUSH [Chung et al., ICLR 2017] is *exactly* letters→…→themes
with boundaries that emerge; (3) reading raw characters and recovering structure is feasible [Van Aken 2011;
MambaByte 2024]. The **weak link is the Thousand Brains borrowing**: "motor = navigating text" is a likely
**category error** — TBT's load-bearing claim is that *self-generated movement through 3-D space* makes a
sensor's next input predictable via metric reference frames, and Numenta's own reference implementation
(Monty) **explicitly does not work on static datasets and has never implemented or tested language**, with
the cross-level mapping "outstanding." So TBT is *inspiration, not a blueprint you inherit*. **The honest
differentiator is not prediction (every LM generates from prediction) — it is the combination the rest of
this doc builds on: online/continual learning without catastrophic forgetting + an explicit, inspectable,
persisted concept hierarchy + multi-view disagreement as an uncertainty signal.**

## What the research established (verified claims)

- **Transient Bayesian surprise is the right boundary signal.** KL divergence between *successive* next-word
  belief distributions (from a predictor) matched human narrative event boundaries at the 95–99.8th
  percentile; plain surprisal (−log p of the observed token) and entropy *failed* (≈0 correlation). It works
  because it "factors in the shape of the entire distribution" and detects *confident→uncertain* shifts, and
  because it is **transient** — errors trigger boundaries only when they *stand out against a low-error
  background*. This is precisely how to operationalize your "ambiguity," and it cleanly separates *surprising*
  from *meaningful*. [PMC11654724; cogs.13343] **(high confidence; caveat: shown on 3 spoken stories at
  event granularity with GPT-2 — transfer DOWN to letter/word/phrase boundaries is untested.)**
- **Emergent boundaries are a solved mechanism.** HM-RNN learns hierarchical structure with no boundary
  supervision via a per-layer parametrized binary detector; FLUSH summarizes a finished segment up a level
  and reinitializes; higher layers update far less often (270/56/9 updates across 3 layers on a 270-char
  span). Adopt this, don't reinvent it. **Limitation: its learned boundaries land on spaces/bigrams/prefixes
  — not cleanly linguistic — which is why you pair it with the transient-surprise signal above.** [arXiv 1609.01704]
- **The bottom of the stack is well-trodden (so novelty can't live there).** Pure letter statistics recover
  word boundaries from space-stripped text (Harris successor-variety → minimum-entropy → Van Aken Viterbi);
  byte/tokenizer-free LMs are *competitive* (MambaByte 33.0 vs 36.3 PPL on PG19, compute-matched — "competitive,"
  not definitively better; 2-1 verify split). [arXiv 1105.6162; arXiv 2401.13660]
- **Predictive coding is the principled bridge — and the novelty trap.** A simple unsupervised
  predict-the-next-input loss is argued sufficient to grow hierarchically rich representations — "an argument
  which has found recent support in modern autoregressive models." Active Predictive Coding gives a concrete
  template: hierarchical world models with *task-invariant state-transition networks + task-dependent policy
  networks at multiple levels* (a real "state/motor split"). **But this is also why "generation emerges from
  prediction" buys you nothing a transformer lacks.** [arXiv 2107.12979; Rao APC, Neural Computation 2024]
- **Numenta already tried text** (Cortical.io Semantic Folding: word-SDRs on a 2-D topographic map via
  competitive Hebbian/inhibition) and it did **not** become competitive language tech. HTM learns by a
  *local, unsupervised Hebbian rule, sparse binary units, no backprop*. [Semantic Folding white paper;
  numenta.com/blog 2019] **(Refuted and must NOT be repeated: HTM is "discontinued" — false; Semantic Folding
  is a clean "distributional-only TBT substitute" — false; APC is "first unified Hinton+Hawkins+RL" — false.)**

## Your four advantage axes — honest status

These are the real thesis (a *thinking* LM that's cheap, online, non-forgetting, massively parallel). The
research **confirmed the mechanism but did not substantiate the value-prop claims at language scale** — so:

| Claim | Status | Honest read |
|---|---|---|
| **No catastrophic forgetting / online learning** | **Plausible-strong, unproven at language** | The most defensible edge. Sparse distributed reps have low overlap → low interference; local Hebbian learning is online (no replay, no full retrain). Verified that HTM works this way. **But demonstrated only on streaming/sequence/anomaly tasks — never on competitive language.** This is the property most worth *measuring directly* (see Experiment B). |
| **Orders-of-magnitude faster train/infer** | **True per-update, misleading as stated** | Local updates + no backprop-through-time + sparse activation are cheap per step. But "faster" silently means "faster to *competitive language quality*," and HTM never reached that. Speed is real; speed-to-good-LM is unproven and is the open bet. |
| **Thousands of models, not a few experts (MoE is wrong frame)** | **Architecturally legitimate, diversity is the catch** | Agreed MoE ≠ this: thousands of small, redundant, online-updatable voting units is a different design. **But voting helps only when members err *differently*** — for a single text stream you must engineer genuine diversity (timescale, direction, char-vs-morpheme, syntactic-vs-distributional views), or thousands of models just average to the mean. |
| **A "thinking" LM breakthrough** | **Aspirational; rests on the above transferring to language** | If online + non-forgetting + cheap + inspectable *all hold AND reach language competence*, that's a real breakthrough no transformer offers. The brutal-honesty counter: **no one has shown HTM/TBT producing language-competitive behavior** — the entire bet is whether your specific design closes the gap HTM-for-text hit. |

**Reframed verdict for your thesis:** the prize is not perplexity, it's a **continually-learning,
non-forgetting, cheap-to-update, human-inspectable** model of text. That value prop is legitimate and
*underexplored*. The risk is HTM's history — sparse local learning may never reach the representational
richness language needs. So the project should be **experiment-first**, designed to kill itself fast.

## Where Prism/Core fits (genuinely well) — and where it doesn't

**Core is an excellent substrate for the *state and structure*, precisely because the model learns online.**
A continually-learning network wants exactly what Core gives: a durable, value-typed, diff-clean graph you
mutate incrementally and persist after every observation — snapshot/diff/overlay the growing model, never
"checkpoint the whole net."

- **Elements / paths** = layers and discovered concept nodes. An ordered tree at hierarchical paths *is*
  letters→words→phrases→sentences→paragraphs→themes; each discovered chunk is an Element under its layer.
- **Connections** = the wiring: bottom-up evidence (child chunk → parent concept), top-down prediction
  (parent → expected next child), and lateral **vote** edges between competing views.
- **Time-sample axis** (Core's first-class per-property animation) = the sequence index. A property's value
  over time-samples is the activation/belief trajectory as the cursor scans — the natural home for "state over time."
- **Evaluator** = inference / message-passing: resolve a node's value through its connections + current
  time-sample (bottom-up evidence ∘ top-down prediction) — the predictive-coding resolve.
- **Persistence** = the grown hierarchy *is* the `.prism` document: durable, diff-able, **human-inspectable
  and human-editable** — the genuine differentiator vs an opaque LM.

**Where Core does NOT fit:** the numeric learners — the per-view predictor models and the boundary detector —
are gradient/representation ML, not document evaluation. **Keep the learner OUTSIDE Core**; it writes
discovered concepts/edges INTO the Document (exactly the Atlas pattern: an external process mutates the live
model through one surface). Core stores and serves the structure; it does not train it.

## Proposed architecture (grounded in the verified science)

1. **Substrate:** raw bytes/characters in; a small per-view predictor (e.g. a byte-level SSM/RNN) produces a
   next-symbol *distribution* at each step.
2. **Boundary signal:** compute **transient Bayesian surprise** = normalized KL between successive predictive
   distributions, thresholded *relative to a local background* (transience is load-bearing). This gates an
   HM-RNN-style binary boundary → COPY / UPDATE / FLUSH.
3. **Hierarchy:** on FLUSH, write the finished segment's summary as a new Element one path-level up; higher
   levels update less often by construction. Boundaries *emerge*; they are not hand-set.
4. **Multiple world-model views + voting:** several predictors over the *same* stream but genuinely diverse —
   different timescales, forward/backward, char-vs-morpheme, syntactic-vs-distributional. Their **disagreement
   is itself an ambiguity signal** feeding boundaries; their **accuracy-weighted consensus** sets current
   state. This is the only sound analogue of cortical voting for one text stream (TBT voting needs different
   sensor patches; here the "patches" are different feature/timescale views).
5. **Online learning:** local, incremental updates after each observation; persist deltas into the `.prism`
   Document. No global retrain — this is where the no-forgetting / cheap-update thesis lives and must be measured.
6. **Generation:** sample forward from the resolved top-down predictions.
7. **TBT framing:** keep as *inspiration only*. If you want a real "reference frame for text," it must be a
   **discourse-structural / syntactic-tree / positional coordinate system** in which movement generates
   predictable consequences — that is a design you invent, not one TBT hands you. Don't make it load-bearing in v1.

## Phased build plan — designed to kill the bet cheaply

- **Experiment A — the boundary kill-test (days).** On raw characters, a small byte predictor; compute
  transient Bayesian surprise; segment. **Metric:** boundary precision/recall/F1 vs true word boundaries on
  space-stripped text (Van Aken's task) + sentence boundaries. **Baselines:** raw surprisal, entropy, BPE
  merges, HM-RNN's learned detector. **KILL IF** transient Bayesian surprise doesn't beat surprisal/entropy at
  recovering boundaries (the 2023 result predicts it should; if it doesn't transfer below event-granularity,
  the central "ambiguity = boundary" bet is weak — stop).
- **Experiment B — the *actual thesis* test (your four axes).** Train continually across several text domains
  in sequence; **measure catastrophic forgetting** (does domain-1 performance degrade after learning domains
  2..N?) vs a backprop byte-LM baseline trained the same streaming way, and **measure update cost**. This is
  the experiment that proves or kills the "online, non-forgetting, cheap" value prop — the part the research
  could *not* confirm from the literature.
- **Experiment C — does the hierarchy earn its keep?** Stack 2–3 levels, write FLUSH summaries as Elements;
  **metric:** does conditioning next-char prediction on the resolved higher-level concept *reduce perplexity*
  vs a flat byte baseline? If not, the hierarchy is decorative.
- **Experiment D — does voting beat the best single view?** Add views + accuracy-weighted consensus; **metric:**
  does disagreement add boundary signal, and does consensus beat the best single view (it only helps if views
  err differently)?
- **Datasets:** enwik8/text8, PG19, a children's-story corpus (matches the event-segmentation domain);
  space-stripped Brown/Wikipedia + human event-boundary norms for boundary eval.
- Throughout, the **`.prism` Document is the artifact a human inspects** — that inspectability is the real deliverable.

## Risks / open questions

1. Does transient Bayesian surprise transfer *down* to letter/word/phrase boundaries, or is it a high-level
   semantic-event phenomenon? The whole bottom of the stack depends on this and it is untested sub-word.
2. Is there a defensible, load-bearing "reference frame for text" that preserves what makes TBT movement
   predictive — or is the framing decorative? (Design burden, not inherited.)
3. Do multiple voting models add anything beyond an ensemble for a single stream? Need genuinely diverse views.
4. **Beyond inspectability + the online/no-forgetting properties, is there a task where this OUTPERFORMS a
   plain byte-LM — or is the honest value prop "a transparent, editable, durable, continually-learning model
   of text" rather than a raw-capability gain?** Experiments B and C settle this.

## Annotated reading list

- **Hawkins, *A Thousand Brains* (2021)** + **Thousand Brains Project / Monty docs** — the theory and its
  *admitted* limits (no static data, language unimplemented). Read for inspiration + to see the category-error risk.
- **Chung, Ahn, Bengio — Hierarchical Multiscale RNN (ICLR 2017, arXiv 1609.01704)** — *the* mechanism for
  emergent multi-timescale boundaries (COPY/UPDATE/FLUSH). Adopt directly.
- **Kumar, Goldstein, Zacks et al. — Bayesian surprise & event boundaries (Cognitive Science 2023, PMC11654724)**
  — proves *transient Bayesian surprise* (not surprisal/entropy) is the boundary signal. The crux paper.
- **Zacks & Tversky — Event Segmentation Theory** — prediction error as event boundaries; the cognitive basis.
- **Rao & Ballard (1999)** + **Millidge/Seth/Buckley predictive-coding review (arXiv 2107.12979)** + **Rao,
  Active Predictive Coding (Neural Computation 2024)** — the principled predict-to-learn bridge and the
  state/policy-split template.
- **Numenta HTM papers + Cortical.io Semantic Folding white paper** — the prior text attempt and the
  local-Hebbian/SDR learning rule (the online/no-forgetting properties). Study why it didn't scale.
- **Ha & Schmidhuber, World Models (2018)** + **LeCun, JEPA position paper (2022)** — predictive world-model
  framing; JEPA's "predict in representation space, not pixels/tokens" is relevant to your hierarchy.
- **MambaByte (arXiv 2401.13660)** + **Van Aken statistical word segmentation (arXiv 1105.6162)** — the
  tokenizer-free substrate and the classic distributional-segmentation baseline you must beat.

---
*Sources verified 3-vote adversarial; full claim ledger in the research run. Architecture/plan sections are
synthesis (medium confidence) — reasoned from the verified science, to be proven by Experiments A–D.*
