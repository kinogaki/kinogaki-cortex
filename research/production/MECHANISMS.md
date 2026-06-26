# MECHANISMS — the surviving count-native production organs

Every mechanism that survived the two adversarial lenses (rule-compliance + reinvention; evidence
honesty + category errors), with the **revise** fixes folded in and the **cuts/merges** recorded at
the bottom. Grouped by the **loop** each serves: **production / world-model / metacognition /
social-ToM / motivation** (several inform two — marked).

Each entry carries:
- **Rules honored** — online-only / bounded-memory / fragile-ideas / cognition-as-guide.
- **Refines/extends** — the experiment id(s) it builds on (see [PROVENANCE](../PROVENANCE.md)).
- **Novelty** — honest: NEW, refines-X, or reinvents-X-on-a-new-axis.
- **Experiment** — corpus / metric / baseline / kill-condition, with a queue letter (**BN onward**,
  continuing A…BJ).

The cog-sci behind each is in [SCIENCE.md](SCIENCE.md); the acquisition organs these compose with
(AU chunk lexicon, BD/BL producer, AV grounding, AY/BF gates) are in the
[acquisition library](../acquisition/MECHANISMS.md) and its [BUILD_QUEUE](../acquisition/BUILD_QUEUE.md).

> **Reconciliation note.** The recurring proposal across the angles was *"a second count table
> approximating the listener."* It is rule-legal and is **not** a reinvention of Exp AM iff scored
> on referential/communicative **success**, not bpc. The reconciliation kept the **first** clean
> instance of each new organ, revised the ones pinning a distinct angle on the same idea, and **cut**
> four duplicates and one true category error. The count and the cuts are at the end.
>
> **Status honesty.** Exp BE (contingency teaches, +0.45 bpc over yoked), AV (word→referent
> grounding), BD/BL (the producer), BK (chunk-not-a-predictor), and AG/AB (the confidence scalar)
> all **ran**. **BM** (the live-Haiku reactive loop) is **blocked-on-credentials**: the contingency
> win was reproduced on a *scripted* partner only. Every mechanism whose strongest test needs a
> genuinely contingent partner inherits that unfinished state — its headline numbers are
> scripted-fallback numbers until a key is set. Flagged per organ.

---

# PRODUCTION — say a thing for its effect, not for fidelity

## P1 — L0-listener table + S1 speaker (count-native RSA, depth-1) — queue **BN**
**The headline social organ.** Add a `ListenerColumn`: a count table indexed utterance→referent,
`L0(s|u) ∝ count(u-seen-with-referent-s)·prior(s)`, built online as the **inverse-read of AV's
word→referent matrix** — the same increments, a second index. It is **not** a new representation; it
is AV read backwards. Production becomes a rerank over the BD/BL producer's candidates: for the
intended referent `s`, score each candidate continuation `u` by `log L0(s|u) − λ·cost(u)` (cost =
token length or −log prior-frequency), order by **AJ** take-the-best (noncompensatory, early-stop),
emit. Close the loop on AT: the InterlocutorEnv responder decodes `ŝ = argmax L0(s|u)`; reward = 1
if `ŝ == s` (or partner's next-turn referent-overlap rises), else 0. On reward, increment the
winning `(u,s)`; on failure, **AB**-split a MISS onto `(u,s)` and increment the repair alternative —
communicative breakdown→repair in counts. Depth-1 only (skip L1 — bounded memory); RSA's α
temperature is replaced by AB's knob-free `f·c` sharpening. Because `P_L0` is the *listener's* table,
an ambiguous `u` (high **AO** fan over the listener's referents) scores low even when the speaker is
sure — implicature falls out of the scoring.

- **Rules:** online (additive `(u,s)` counts + Bayes lookup + one increment/turn), bounded (shares
  AV's bounded referent set; LFU/AR eviction; the listener table is *smaller* than the speaker's),
  no-backprop (counts + AB sharpen + AJ argmax), cognition-as-guide (RSA, scoped to decode).
- **Refines/extends:** extends **AV** (inverse direction), refines **G6** (margin source = listener,
  not speaker), reuses **AB**/**AJ** as the reranker. Not a reinvention of G1 (which reads only the
  speaker table).
- **Novelty:** NEW (the inverse-read second table + decode-reward loop).
- **Experiment.** *Corpus:* Yu & Smith referential-uncertainty scenes through AT's scene-bearing
  InterlocutorEnv — name a target among N competitors; responder (echo first, Haiku once keyed)
  reports the recovered referent. *Metric:* **referential success** (% of turns the responder's `ŝ`
  equals the intended `s`); utterance cost at matched success; the implicature signature (does S1
  prefer the costlier *disambiguating* token when the cheap one is ambiguous?). Report referential
  success, **not** bpc. *Baseline:* (1) ship-the-argmax speaker (no listener model); (2) the
  **YOKED-listener** control — rerank against a *scrambled* L0 with the same count mass, wrong `u→s`
  alignment (the BE template, registered before running) — the load-bearing baseline; (3) AM static
  prior. *Kill:* listener-rerank does not beat speaker-only on referential success AND matches the
  yoked control — then the second table buys nothing over reading the speaker table or random
  reindexing. FRAGILE budget (≥10 settings of λ and the f·c gate); a first weak *bpc* result is
  expected and is **not** a kill (judge on the ambiguous/competitor slice — AF's right-axis rule).
  → queue **BN**.

## P2 — Function-tagged emission (echoic / mand / tact over one form) — queue **BO**
Keep **three** bounded count tables keyed by the **same** emitted pattern (an AU chunk), separated by
**antecedent/consequence class**, not surface form. (a) *echoic*: pattern × high Jaccard/skip-gram
overlap with the immediately-preceding input window — the form was just heard. (b) *tact*: pattern ×
current scene/world-model state token from AV's referent counts — the form names what is present.
(c) *mand*: pattern whose emission was historically followed by a contingent reply that raised P13's
reward accumulator — the form got something. At production, G6 gathers margin-passing candidates,
then routes by which table's antecedent **matches now** (AJ take-the-best over the three
antecedent-match cues). The same string plays three operants; selection is by live context — Skinner's
**functional, not topographic** split.

- **Rules:** online (three additive count tables, incremented per emission with its observed
  antecedent), bounded (keyed on the AU lexicon × a small antecedent code; tact reuses AV's store;
  mand reuses P13's accumulator), no-backprop (counts + Jaccard + AJ argmax), cognition-as-guide
  (Skinner's operant split).
- **Refines/extends:** extends **G6** (a third routing axis among margin-passing candidates), depends
  on **AV** (the tact channel), reuses **AJ** (take-the-best), composes with **AY/M17** (M17 decides
  *if* emittable; P2 decides *as which operant*).
- **Novelty:** NEW (the antecedent-keyed three-view routing).
- **Revise (applied):** the *tact* table **requires AV's scene-bearing env** (built, Exp AV) — state
  that dependency up front; until a scene is rich, run **echoic + mand only** (which need just
  input-overlap and the contingency counter). Judge on routing accuracy per class, never bpc.
- **Experiment.** *Corpus:* InterlocutorEnv over the AV scene-bearing env (so tact has a perceptual
  antecedent); scripted-reply fallback where needed. *Metric:* **functional-appropriateness** — for
  held-out (antecedent, form) triples, does the routed operant match the gold function (echo after
  repeat, tact when referent present, mand before reward)? Routing accuracy per class + a confusion
  matrix; secondary: does mand-routing raise contingent-reply rate over ungated G6 emission. *Baseline:*
  G6 with ONE undifferentiated table (no function dissociation); plus an echoic-only ablation (collapse
  mand/tact into echoic). *Kill:* the three antecedent cues do not separate the same form into distinct
  operants above the single-table baseline after the FRAGILE budget (≥10 antecedent-window / overlap-
  threshold variations) — then function is not recoverable from antecedent counts here; park (may need
  a richer scene). → queue **BO**.

## P3 — Dual goal-conditioned production tables (GET vs SHOW) with perlocutionary reward — queue **BP**
Keep **two** production tables keyed by **goal** (Bates proto-imperative vs proto-declarative). *Table_GET*:
counts of (goal-token g, emitted u, observed listener-action a) giving `P(a|g,u)`; act picks `u`
maximizing `P(desired_action | g, u)`. *Table_SHOW*: counts of (referent r, emitted u, listener-attended
a) giving `u` maximizing listener-oriented-to-r. Reward is **perlocutionary**: did the listener deliver
the object (GET) or shift attention (SHOW). The tables are **separate**; transfer is an *explicit counted
bridge* (a cross-table offset that aliases an entry only once it has paid off in both) — functional
independence built in, not assumed. Skip-gram backoff answers Chomsky's novelty objection (a novel `u`
inherits reward from its sub-units).

- **Rules:** online (two additive tables + leaky decay + one bridge counter), bounded (two bounded
  tables under LFU/AR; bridge O(shared entries); skip-gram backoff needs no new rows — the separation
  cost vs one table *is* the claim, at equal budget), no-backprop (count increment on goal-satisfaction,
  the Rescorla-Wagner shape from BC), cognition-as-guide (mand/tact functional independence).
- **Refines/extends:** reuses **P1**'s decode-reward as the perlocutionary signal (not a third reward
  path); composes with **P2** (the operant tables P2 routes into).
- **Novelty:** NEW (goal-conditioned tables + the counted transfer bridge).
- **Revise (applied):** build **only after P1 lands**; reuse P1's decode-reward; cap the bridge counter;
  register the functional-independence (no-free-transfer) test as the **sole** load-bearing axis. A clean
  "English transfers freely" negative is the publishable outcome.
- **Experiment.** *Corpus:* InterlocutorEnv with a listener that takes one of M actions (deliver object i
  / attend referent j); modes GET and SHOW/NAME; train GET-first, then probe NAME transfer. *Metric:* GET
  success rate and NAME accuracy over turns; the **functional-independence signature** — a GET-mastered
  form not immediately available in NAME above the bridge's transfer rate. *Baseline:* (1) ONE shared
  production table (no functional independence); (2) BE single table (fidelity reward); (3) increment-only
  no-reward producer. *Kill:* the shared table matches the dual on **both** GET and NAME at equal budget
  (cut to one table); OR GET-first does not speed GET vs NAME (goal-conditioning inert). FRAGILE budget on
  bridge-threshold/decay; English-transfers-freely is a publishable negative. → queue **BP**.

## P4 — Message-tuple conceptualizer: select what to say before formulating (the Levelt split) — queue **BQ**
Insert a preverbal-message stage between "have something to say" and BD/G6 formulation, so the loop stops
**continuing the text** and starts **saying a thing**. Each turn, score every slot-tuple `m = <who, relation,
what>` readable from AH's explicit registry by `salience(m) = updateCost(m) × audienceGap(m)`, where
**updateCost** = the per-channel KL belief-jolt from **AC** (which dimension just moved most — Zacks/Zwaan
"what is worth saying") and **audienceGap** = the speaker-minus-listener divergence from P7. Pick
argmax-salience as the message; **only then** run BD/G6 to realize it. Gate the whole stage behind **AG**'s
dual trigger: run the (more expensive) conceptualizer race only when the cheap continuation is low-confidence
or when conceptualizer and continuation disagree — else ship the fast BD continuation (AG's no-harm contract).

- **Rules:** online (a product of two existing online signals — AC per-channel KL, P7 divergence — argmax
  over a bounded slot set), bounded (AH's bounded registry × P7's bounded per-dialogue table; one
  message-tuple at a time — the conceptualizer is a capacity-1 bottleneck by design), no-backprop, cognition-
  as-guide (Levelt conceptualizer, thinking-for-speaking).
- **Refines/extends:** reuses **AC** (KL belief-jolt as salience), **AH** (the slot-tuple operands), **AG**
  (the no-harm gate), **P7** (the audience-gap term).
- **Novelty:** NEW (a preverbal message-selection stage; the spine only ever continued the text).
- **Revise (applied):** run the **updateCost-only (AC)** half **first** against a trivial "report the
  most-surprising token" heuristic — if AC-salience alone can't beat that, the audienceGap term is moot.
  Only add audienceGap once **P7** has cleared its static-prior bar.
- **Experiment.** *Corpus:* the P1/P7 referential corpus plus a **state-change probe set** (scripted
  mini-narratives where exactly one situation dimension changes per turn, with the gold "most-reportable"
  tuple annotated). *Metric:* (1) does the selected message-tuple match the gold most-reportable change
  (precision@1)? (2) is BD-formulation-of-a-selected-message more on-target than BD-continuation? (3) AG-gate
  efficiency (fraction of turns the conceptualizer fires; no-harm where it doesn't). *Baseline:* BD/G6 with
  no conceptualizer (continue-the-text); ablate the two salience terms (updateCost-only vs audienceGap-only
  vs the product — the product must beat both halves); a trivial most-surprising-token heuristic. *Kill:*
  selected tuples are no better than chance / most-frequent-slot at matching the gold change, OR conceptualizer-
  then-formulate ties the most-surprising-token heuristic. FRAGILE: tune the weighting ≥10 steps and confirm
  the AG gate fires on the right turns first. → queue **BQ**.

---

# METACOGNITION — read the confidence scalar as a control signal

## P5 — Three-way production controller (emit / deliberate / ask), routed on goal-vs-form conflict — queue **BR**
Make `act()` consult a scalar confidence `c` **before** emitting and branch three ways instead of sampling.
Compute `c` at emit time from counts the agent already has, at **two altitudes** (Nozari): **FORM conflict**
= the margingen margin over char/word n-gram experts (top1/top2 vote-gap, branching-entropy of the
continuation); **GOAL conflict** = the same margin over the content/skip-gram experts that should realise the
current beat/topic (AG's long-order {5,6} voter as the goal drive). Combine to `c` = AB's `f·c` on the winning
candidate, attenuated by normalized branching-entropy. Then: (1) `c` high → System-1 single-pass emit; (2) `c`
low AND the conflict is **form-side** (knows *what*, competes over *how*) → enter **AL**'s bounded deliberate
loop and emit the deliberate winner; (3) `c` low AND the conflict is **goal-side** (the goal voter itself is
flat — missing context) → emit the **ASK** action (P6) instead of a token. The routing key — *is the gap in the
goal or the form* — is the one genuinely new dial; everything else is AG/AB/AL/margingen rewired into `act()`.

- **Rules:** online (`c` read off existing leaky accumulators at emit time; no second pass), bounded (adds no
  table; `c` is a scalar per candidate; the deliberate loop has AL's capped step budget), no-backprop (reuses
  `cortex.vote`, `confidence.truth_of`, `margingen.produce`, `deliberate.py` verbatim), cognition-as-guide
  (feeling-of-rightness recruits System-2; Goupil's third "ask" outcome).
- **Refines/extends:** refines **AG** (two-way gate → three-way control); reuses **AB**, **AL**, **G6/margingen**.
- **Novelty:** refines AG (the goal-vs-form routing key is new).
- **Experiment.** *Corpus:* InterlocutorEnv over a battery where some prompts are answerable (form-only
  ambiguity) and some are referentially under-specified (goal-side gap), scripted-or-Haiku responder; text8
  next-word for the no-ask floor. *Metric (three, reported separately — FRAGILE):* (a) on answerable prompts,
  well-formedness/over-generation of emitted spans (the deliberate branch should beat flat sampling); (b) on
  under-specified prompts, **ASK-precision** = fraction of asks on genuine goal-gap items vs answerable items
  (Goupil selectivity); (c) calibration of `c` against actual emit-correctness (ECE, the AB axis). *Baseline:*
  flat-sampling `act()`; AG-defensive gate that only deliberates and never asks (proves the ask branch adds
  something); a confidence-blind "always ask when margin<θ" ignoring the goal-vs-form split (proves the routing
  key is load-bearing). *Kill:* the goal-vs-form routing does not separate ask-appropriate from deliberate-
  appropriate prompts above the confidence-blind baseline (ask-precision no better than chance) after the
  FRAGILE budget (≥10 settings of θ, altitude weights, entropy attenuation) — then collapse to AG's two-way
  gate, keeping the ask as an unconditional low-margin fallback. → queue **BR**.

## P6 — The clarification ASK as a first-class, accumulator-gated action with contingency reward — queue **BS**
Add a distinguished **ASK** token to `act()`'s emission vocabulary that, when emitted, makes the next Turn
answer it (the maximally-contingent utterance — Goupil's "ask when you know you don't know"). Three honest
pieces. (1) **Trigger:** the ask competes with token emission only when goal-side `c < θ_ask` (from P5) AND the
gap is a **true gap**, not a tip-of-the-tongue. Distinguish via the FOK/TOT signature: leader margin low BUT
**neighborhood mass** high (aggregate AB `f·c` over co-active content features — first-letter/semantic-class/
related-skip-gram counts) ⇒ TOT ⇒ persist/expand search (lower the emit threshold for near-target candidates,
spend AL steps), **not** ask; neighborhood mass near-zero ⇒ true gap ⇒ ask. (2) **Cost gate:** asking is
effortful — gate it behind a leaky accumulator over *sustained* uncertainty so only persistent/important gaps
cross (infants ask selectively). (3) **Reward** on the AT contract: communicative reward = drop in downstream
conflict/obs_surprise after the answer arrives (the surprise_probe delta), credited via BE's contingency dial.
Register the **YOKED** control before running: an "asked-but-answer-timing-scrambled" condition (BE's exact
shape). A *forward-sketch near-zero-count* trigger (the partner hasn't met a referent the agent is about to use)
is folded in here as an audience-aware ask source.

- **Rules:** online (trigger reads existing counts; leaky accumulator + reward credit are per-step scalars —
  BE's gain machinery), bounded (one extra token id, one accumulator scalar, one neighborhood-mass read as a
  **bounded distinct-counter / HyperLogLog sketch**, never an exact per-context set), no-backprop (count reads +
  leaky accumulator + Rescorla-Wagner reward credit), cognition-as-guide (selective infant asking; FOK/TOT).
- **Refines/extends:** reuses **BE** (the contingency reward + yoked control), **AB** (neighborhood-mass read),
  **AT** (ask = a special emission; answer = next observation), routed-to by **P5**.
- **Novelty:** NEW (ask-as-first-class-action + the TOT-vs-true-gap split + the audience-aware near-zero-count
  trigger). Distinct from P5 (which *routes* to ask; P6 *builds* the ask + its trigger).
- **Experiment.** *Corpus:* InterlocutorEnv with a staged responder that (a) answers asks informatively on
  goal-gap items, (b) on a yoked control delivers the same answer text with scrambled timing. Items span
  true-gap, TOT, and answerable. *Metric:* (1) **ASK selectivity** = P(ask | true-gap) vs P(ask | answerable) vs
  P(ask | TOT) — concentrate on true-gap, NOT fire on TOT (where persist fires) nor on answerable; (2) post-ask
  conflict/surprise drop on contingent vs YOKED timing — must beat yoked; (3) ask rate stays bounded. *Baseline:*
  no-ask agent (P5 deliberate-only); always-ask-when-uncertain (no accumulator gate, no TOT/gap split — predicted
  to over-ask); ask-with-no-reward. *Kill:* asking does not concentrate on true-gap items above always-ask AND the
  post-ask surprise drop does not beat YOKED — then the ask is a free reflex with no metacognitive content. Per
  FRAGILE, check the accumulator threshold and the sketch resolution first; park the TOT/gap split if only the
  bare ask survives. → queue **BS**.

---

# SOCIAL-ToM — model the listener, scoped honestly as recipient design

> Every organ here is scoped to **recipient design / audience exposure**, never belief tracking. A count
> table over partner surface tokens cannot represent the listener knowing something the speaker knows is
> false — that is the real false-belief test and it is **out of scope**, not solved. Each carries the
> egocentric (self-model / speaker-surprisal) control as the load-bearing baseline; if the listener table
> never beats "assume the listener knows what I know," the ToM framing is empty and the organ is parked.

## P7 — Recipient-design dissociation: the verbosity gap, riding the L0 table — queue **BT**
The one load-bearing recipient-design test P1 lacks. Per dialogue, maintain a leaky listener-divergence
overlay on top of P1's L0 table: `audience_gain(u for message m) = max(0, speaker_activation(m) −
listener_activation(m))` retrieved by **AO** cue. Rerank BD's candidates by `audience_gain`; **fall silent**
when the gain is below θ (the listener already knows it — Gricean quantity, for free). The deliverable is the
**verbosity dissociation**: more tokens for listener-novel referents than listener-known ones — the audience-
sensitivity signature, measured against AM's static frozen-marginal prior.

- **Rules:** online (a per-dialogue leaky overlay on P1's counts; divergence is an L1 over AO-retrieved
  activations on demand), bounded (per-dialogue, discarded at conversation end — AQ; keyed on the bounded set
  of mentioned constructions; shrinks toward the speaker's marginals so unmentioned items cost nothing),
  no-backprop (increments + L1 + a multiplicative reweight, **not** REINFORCE), cognition-as-guide (least-
  collaborative-effort).
- **Refines/extends:** refines **G4** (from a passive "what was acknowledged" record to an active divergence
  scorer); rides **P1**'s table (no third table); reuses **AO** retrieval.
- **Novelty:** refines G4 (the predictive-divergence half G4 left out).
- **Revise (applied):** **drop the separate table** — reuse P1's L0 inverse-read; add only the per-dialogue
  `audience_gain` + the fall-silent rule. Make the **verbosity dissociation** the *sole* kill axis;
  "wins success but not dissociation" is the honest PARTIAL.
- **Experiment.** *Corpus:* a multi-turn referential-communication task (director-matcher / TANGRAM naming
  game) through InterlocutorEnv; build a contrived asymmetry — half the targets are listener-known
  (acknowledged earlier), half novel. *Metric:* the **verbosity dissociation** — tokens-per-referent HIGHER for
  listener-novel than listener-known items (load-bearing); secondary: referential success; does emitting the
  highest-`audience_gain` chunk reduce next-turn listener surprise more than the highest coverage×frequency
  chunk (BD baseline)? *Baseline:* BD producer with no listener table (predicts equal verbosity — the null); a
  **static** listener prior = frozen marginals with no per-dialogue divergence (AM's control); G4 acknowledged-
  overlay-only. *Kill:* the dissociation is absent AND the live overlay does not beat the static prior on
  referential success, across two seeds. Per FRAGILE, ≥10 nurture steps on the divergence-weight and the
  shrink-strength first; success-without-dissociation is a reported PARTIAL. → queue **BT**.

## P8 — Listener-divergence informativity gate (redundancy suppression + INFORM news) — queue **BU**
Approximate audience design as a **divergence** between two count tables, scoped to **Gricean-quantity
redundancy suppression and reference resolution** — *not* a general production scorer (P1 owns that). Keep a
second bounded n-gram table updated **only** from the listener channel = "what the listener has been
saying/hearing." At production, for each margin-passing candidate compute `informativity = surprisal under the
LISTENER table − surprisal under a STATIC listener prior` (AM's law: never vs no-prior). Prefer candidates that
are non-redundant given listener common ground (high listener-surprisal where the goal is to **inform**) but
still recoverable (not impossibly high). The proto-imperative→proto-declarative split is operationalized as
*which reward is firing*: a world/scene-state-change reward (mand, P13) vs a listener-state-change reward (the
divergence rose because the listener's table lacked it). **(Folds in the former per-listener INFORM news-gate;
the REQUEST/SHARE reward-shape trichotomy is parked as inspection-only until INFORM beats the egocentric gate —
the motive taxonomy is not smuggled in as established.)**

- **Rules:** online (one extra n-gram table from listener tokens; informativity is a closed-form surprisal
  difference per candidate), bounded (shares the agent's bounded Column band shape + context-tail trim; the
  per-dialogue overlay decays/resets between conversations — AQ), no-backprop (count-based surprisal difference +
  threshold), cognition-as-guide (informative pointing / Liszkowski).
- **Refines/extends:** refines **G4** (the non-redundancy half G4 left out); honors **AM** (scored vs a static
  prior, never a long-span predictor); complements **P7** (P7 = verbosity dissociation; P8 = redundancy
  suppression).
- **Novelty:** NEW for the redundancy/informativity axis.
- **Revise (applied):** narrowed strictly to redundancy suppression + reference resolution; cede the general
  listener-rerank to P1. Run **only the INFORM news-gate** as the falsifiable core (listener-surprisal vs
  speaker-surprisal egocentric); park REQUEST/SHARE.
- **Experiment.** *Corpus:* multi-turn InterlocutorEnv where referents/constructions recur and the partner's
  prior turns establish common ground (some info already "known," some new). *Metric:* recipient-design — does
  the gate suppress redundant (listener-known) forms and prefer informative (listener-novel-but-recoverable)
  ones? % redundant-mention suppressed + reference-resolution accuracy; informativity measured as listener-
  surprisal **minus** the static-listener-prior surprisal. *Baseline:* AM's **static** frozen listener prior
  (mandated, never no-prior); speaker-surprisal egocentric gate; speaker-table-only production (predicts
  redundant over-mentioning). *Kill:* the listener-divergence gate does not beat the static prior on redundancy-
  suppression OR reference resolution, OR does not beat the speaker-surprisal egocentric gate — then it has
  landed on AM's dead 0.9% slice; park inspection-only, keep G4's grounded-form bias only. → queue **BU**.

## P9 — Cue-gated egocentric↔common-ground blend (the Keysar default) — queue **BV**
Don't run the expensive listener rerank every token (psychologically wrong and unbounded). Keep an egocentric
**default** = G1's plain speaker table. Maintain a small per-partner common-ground table (referents/constructions
both emitted by the agent AND acknowledged — reappeared in the partner's turn via AO cue retrieval). Produce by
a weighted vote: `score = w_ego·speaker + w_cg·commonGround`, where `w_cg` is a leaky accumulator that **rises**
on cues (referent just mentioned, partner confirmed/repeated) and **decays** otherwise. Trigger the full
listener-decode rerank (P1) **only** when a cheap AC branching-entropy/ambiguity check on the leading candidate
fires — Keysar's resource-gating made count-native. The overlay resets/decays between conversations (AQ).

- **Rules:** online (one leaky `w_cg`, additive common-ground counts, AC entropy already computed; the gate is a
  threshold), bounded (per-conversation table discarded at end — AQ; `w_cg` is O(1); the expensive rerank fires
  on a small ambiguous fraction), no-backprop (leaky counts + threshold), cognition-as-guide (Keysar egocentric
  default).
- **Refines/extends:** refines **G4** (adds the egocentric default + the cue-gated blend weight + the entropy-
  gated escalation G4 lacked); refines **AC** (KL/entropy as an escalation trigger); the **compute-gate on top of
  P1**.
- **Novelty:** refines G4 (egocentric default + cue-gated escalation).
- **Revise (applied):** pair it explicitly as the compute-gate **on top of the kept P1**; make **compute-cost-at-
  matched-accuracy** the headline (always-on rerank = the vindicating baseline); keep the AM-static-prior test as
  the secondary common-ground claim.
- **Experiment.** *Corpus:* multi-turn InterlocutorEnv mixing unambiguous turns (ego default should suffice) and
  ambiguous-competitor turns (escalation should fire). *Metric:* referential success AND **compute cost** (fraction
  of tokens that triggered the full rerank — should stay low, Keysar-egocentric); does `w_cg` track acknowledgement;
  reference-resolution accuracy. *Baseline:* always-on listener rerank (P1, no gate) — same accuracy at much higher
  cost vindicates the gate; AM's **static** common-ground prior. *Kill:* the gate costs as much as always-on for no
  accuracy preserved, OR the common-ground blend does not beat AM's static prior on reference resolution — then it
  landed on AM's dead slice; park inspection-only. FRAGILE budget; a clean negative on the gate alone is acceptable
  if the blend survives. → queue **BV**.

## P10 — Audience-model EFE selector (other-directed control) — queue **BW**
Goal-directed **control**: emit to make the input come back as wanted (Friston). Add a second count table `L`
predicting the partner's **next** utterance/move (skip-gram counts of "what the partner tends to emit after
context X," built online from the InterlocutorEnv reply stream with the **same** Column machinery the agent reads
with — Pickering-Garrod prediction-by-simulation). Turn `act()` from a passive sampler into a controller. For each
candidate `c` (the top few from P1/M17's self-gated pool), look up `L`'s predicted partner-response and score `c`
by a count-native heuristic in the **shape** of Expected Free Energy: (1) **pragmatic** = utility(predicted partner
response) from a leaky reward count table; (2) **epistemic** = predicted entropy-drop in `L` after assuming `c`
grounded. Emit `argmax(pragmatic + epistemic)`, gated by own-confidence `f·c` (AB): below threshold, stay silent or
hedge. `L` updates with BE's hot/cold gain.

- **Rules:** online (additive/leaky n-gram counts off the partner stream; selection is a forward lookup + argmax),
  bounded (`L` is one extra Column band under the same trim + LFU/AR; reward table keyed on bounded response
  classes; scoring over the few P1/M17 candidates), no-backprop (two table lookups summed + an AB-gated argmax),
  cognition-as-guide (other-directed control).
- **Refines/extends:** reuses **P1** (the partner table is shared — not a third table), **AB** (the own-confidence
  gate), **BE** (hot/cold update); composes-above **G1** (selection above realization); honors **AM** (static
  partner-marginal baseline).
- **Novelty:** NEW (utterance-selection-by-predicted-effect; the EFE-shaped decomposition over counts).
- **Revise (applied):** **GOAL-attainment rate is the ONLY headline kill axis**; demote partner-next-token-prediction
  accuracy to a sanity check (it will land on AM's slice). Build **after P1** so the predictor table is shared.
  Report it as a **count-native heuristic motivated by the EFE shape**, never as the free-energy functional. Run the
  headline test on a **grounded micro-task with an external target referent** so GOAL is not self-defined (avoid the
  circularity of rewarding the partner for what it already tends to do); success = task-completion, never an LLM
  quality judgment.
- **Experiment.** *Corpus:* InterlocutorEnv, scripted contingent responder for the free run, live Haiku once a key
  exists; a grounded micro-task — agent must get the partner to emit/acknowledge a target referent. *Metric:* (1)
  **GOAL-attainment rate** (fraction of turns the partner's reply matches the intended target — did speaking *change*
  the listener as wanted); (2) partner-next-token prediction of `L` vs the static partner-marginal (AM's law, sanity
  only); (3) epistemic/pragmatic ablation (each term alone vs summed). *Baseline:* P1+M17 self-gated production
  WITHOUT the audience model — does modeling the listener raise GOAL-attainment above speaking-to-be-fluent? plus a
  static partner-marginal; plus EFE-pragmatic-only and EFE-epistemic-only. *Kill:* audience-model selection does not
  raise GOAL-attainment above self-fluent production at matched token budget, OR `L` predicts no better than the
  static marginal on the 99% slice — then it is tail-smoothing (collapses into AM/G4), park. FRAGILE budget (≥10
  variations of GOAL definition, EFE weighting, `L` order); a clean negative on a thin scripted partner is
  acceptable, retest on live Haiku. → queue **BW**.

---

# WORLD-MODEL — predict how the listener moves, not just where they are

## P11 — One-step audience rollout to score an utterance before saying it (likely-negative, built last) — queue **BX**
The JEPA / world-model move, count-native and **bounded to one step**. Maintain register-**transition** counts:
`c(listener_state → listener_state after utterance-chunk u)`, accumulated online (when the agent said `u` and the
listener's next turn revealed a state, count the transition). To score a candidate utterance for the chosen message
`m`, do a **one-step** rollout: predict the listener's post-utterance state from the transition counts, then score by
predicted divergence-to-target (P7). Pick the chunk whose predicted listener-state-change most reduces the gap —
"say the thing I predict will move them toward where I want them." A soft active-inference twist: prefer chunks the
agent predicts will make the listener's next reply match the agent's prediction (low predicted obs_surprise — AT's
existing probe, read forward). The smallest possible "speak with purpose" organ — a single count lookup of "what
usually happens to a listener when I say this," **not** a deep dream (AN's blind resonator failed; deeper rollout is
forbidden).

- **Rules:** online (transition counts additive; rollout is a single lookup + L1 on demand), bounded (keyed on
  (utterance-chunk-cluster, listener-state-channel) under LFU; depth-1 holds one predicted state; **deeper rollout
  explicitly forbidden**), no-backprop (transition counting + lookup + L1 + the existing surprise probe), cognition-
  as-guide (predict-then-act).
- **Refines/extends:** depends on **P7/P1** (depth-0 gap-scoring is its load-bearing baseline); reuses **AT**'s
  surprise probe.
- **Novelty:** NEW (the transition table), with a **pre-declared likely negative** — "where the listener IS matters;
  how they MOVE doesn't" is the production-side echo of AM.
- **Revise (applied):** **gate behind a positive P7/P1 result** — build only if depth-0 gap-scoring already wins,
  then test strictly whether depth-1 transition counts beat depth-0 at equal memory. Park (don't delete) the
  transition table on the expected negative. Cut/downplay the "active-inference" label — it is a one-step count
  lookup.
- **Experiment.** *Corpus:* InterlocutorEnv referential game with a responder whose state is **observable** in its
  reply (it names what it now believes / picks a referent), so utterance→state transitions are countable; held-out
  (message, candidate-utterance) pairs with a known gold "best utterance to move this listener." *Metric:* (1) does
  choosing the highest-predicted-divergence-reduction utterance beat P7's depth-0 gap-score on referential success
  (does modeling HOW the listener moves add anything)? (2) is obs_surprise on the listener's actual next reply LOWER
  when the rollout-scored utterance was picked? *Baseline:* P7 divergence scorer with NO transition model (depth-0 —
  the load-bearing baseline); a frequency baseline (most-common utterance for this message). *Kill:* depth-1 rollout
  does not beat depth-0 gap-scoring at equal memory across two seeds — keep the simpler P7 (the likely, publishable
  negative); park the transition table for inspection. → queue **BX**.

---

# MOTIVATION — what makes a reader start to speak

## P12 — Learning-progress vocal-play drive (endogenous emission, no listener) — queue **BY**
The one organ that needs **no** social foundation — it breaks AT's cold-start with an intrinsic drive so `act()`
produces *before* any reply exists (>90% of protophones are to no one). Per context-region bucket (a coarse
leader-cluster over the K-tail signature, bounded, AI/AR-evicted), keep two leaky next-token-accuracy estimates: a
**short**-window `acc_s` and a **long**-window `acc_l`. Define learning-progress `LP = acc_s − acc_l` (Oudeyer
competence-progress, count-native). Keep an interest table = a vote over regions weighted by `|LP|`. On a
self-initiated turn (no pending reply), `act()` samples which region to babble in from the interest table, then
emits via the BD/BL producer seeded there — generating self-experience exactly where the counts are still
sharpening, reproducing the automatic curriculum (abandons mastered AND impossible regions).

- **Rules:** online (two additive leaky accumulators per bucket + a vote table; leader-clustering for the bucket
  key — similarity only, never relation directions), bounded (O(#regions) buckets, capped and power-law evicted;
  mastered/impossible regions self-prune as `|LP|→0`), no-backprop (`LP` is a difference of two count-derived
  accuracies; sampling is a vote draw), cognition-as-guide (Oudeyer learning-progress, endogenous play).
- **Refines/extends:** extends **AT** (answers its cold-start note — generation is gibberish); emits through
  **BD/BL**; reuses the **AU/M1** leader-cluster.
- **Novelty:** NEW (the project never had an intrinsic emission drive; `act()` samples at fixed cadence).
- **Experiment.** *Corpus:* text8 via CorpusEnv for the curriculum shape; then InterlocutorEnv/BM with the drive
  choosing self-turns; babble through BD/BL. *Metric:* (a) the developmental-stage signature — does the
  entropy/complexity of self-emitted forms rise over stream-time as buckets master (Oudeyer ordered stages)? (b)
  per-token learning efficiency — bpc reached at fixed observe-budget, curiosity-targeted self-practice vs
  uniform-random vs no self-practice; (c) does it visit under-sampled regions more than a frequency-matched random
  prober? *Baseline:* (i) `act()` at fixed cadence sampling the flat pooled dist; (ii) **novelty-only** drive
  (1/count, no progress term) — the control that isolates **progress** from raw novelty; (iii) read-only. *Kill:*
  curiosity-targeted self-practice gives no better per-token bpc than uniform-random AND produces no ordered-stage
  curve across ≥2 seeds — the LP term is inert; fall back to novelty-only or read-only. FRAGILE budget (≥10
  short/long window-ratio settings); near-zero early is expected. → queue **BY**.

## P13 — Mand-reward accumulator (which emitted pattern the reply answered) — queue **BZ**
Refine BE's whole-turn contingency dial down to the **producible unit**. Per emitted AU chunk, keep a leaky
accumulator integrating reward = 1 if a contingent reply (any downstream listener-channel activity) follows within
a short window, 0 otherwise, decayed with a half-life (Exp R leaky shape). A chunk whose emission reliably draws an
answer has its accumulator raised, raising its future emission weight AND feeding P2's mand table. Bias the reward
toward "more speech-like" = higher likelihood under the read-model's own n-gram vote (canonical-babbling salience).
This is BE's contingency made **per-pattern** instead of per-turn, so **τ finally has graded per-chunk Δt to bite
on** — directly addressing BE/BM's "τ inert because warmth is binary" caveat.

- **Rules:** online (one leaky accumulator per chunk, updated with the reward bit), bounded (one scalar per chunk
  in the bounded AU lexicon; accumulators decay and evict with their chunk under LFU/AR), no-backprop (leaky
  additive reward integration, Rescorla-Wagner shape), cognition-as-guide (mand reinforcement).
- **Refines/extends:** refines **BE** (per-turn → per-chunk credit); feeds **P2** (the mand table).
- **Novelty:** refines BE.
- **Revise (applied):** relabel from "refines BE" to **"BE-and-this are one experiment"** — build the per-chunk
  accumulator and the per-turn dial as the **same** run with the yoked/scrambled control registered first, so the
  per-chunk-vs-per-turn comparison is the deliverable. **Lock the kill to the graded-delay τ-discrimination test**;
  require a per-chunk-contingent responder (some chunks reliably answered, some ignored) as a precondition, else it
  is BE re-run.
- **Experiment.** *Corpus:* BM's live/fallback reactive loop with a responder whose reply probability genuinely
  depends on **which chunk** the agent emitted. *Metric:* does the per-chunk accumulator concentrate emission mass on
  the answered chunks over turns (mand-uptake: rise in P(emit answered-chunk) vs ignored-chunk)? Plus the graded-
  delay sweep — does the exponential τ window now discriminate (the test BE/BM could not run)? *Baseline:* BE's
  per-TURN contingency dial (predicts uniform up-weighting of all warm chunks); plus the **YOKED** scrambled-timing
  control (registered before running). *Kill:* per-chunk credit gives no concentration on answered chunks beyond
  BE's per-turn dial AND τ stays inert under graded delays — then mand-reward is subsumed by BE's whole-turn gain;
  keep the simpler turn-level dial. → queue **BZ**.

## P14 — Listener-model counter: communicative-success reweight, reaction-typed, with extinction burst — queue **CA**
A second count table approximating the listener: `c(cue → reaction)`, where cue = the message/role bundle + form the
agent emitted (AO feature key + fan), `reaction ∈ {responded-in-kind, responded-off-channel, ignored}` extracted from
the next Turn (in-kind = the reply re-uses the agent's channel/lexical material; ignored = no reply within the leaky
window). Each cell is AB-split into (hits, misses) so the audience prior is calibrated. At `act()`, among BD/BL's
well-formed candidates, score each by the listener-model's predicted responded-rate and emit by AJ take-the-best over
{self/language validity, listener validity}, weighting in-kind above off-channel (Zhang). The **extinction-burst**
controller rides on top: a streak counter raises emission willingness while in-kind replies arrive; when observed
responded-rate falls below the form's expected rate (prediction error > 0), **temporarily raise exploration
temperature** (the still-face extinction burst, Franklin), recovering when replies resume.

- **Rules:** online (one additive/leaky counter per (cue,reaction) cell, AB split, a leaky response-rate expectation,
  a streak scalar), bounded (keyed on bounded AO cue store + top-k forms; per-conversation overlay decays/resets — AQ;
  fan-division caps per-cue mass), no-backprop (multiplicative count reweight of emission candidates, AF-preemption
  style, NOT policy gradient; the burst is a threshold on a count-derived prediction error), cognition-as-guide
  (communicative success; still-face/extinction).
- **Refines/extends:** refines **BE** (per-form, reaction-typed, calibrated audience counter beyond a global timing
  gain); reuses **AB** (calibration), **AO** (cue retrieval/fan), **AJ** (take-the-best); steers **BD/BL**; honors
  **BK** (production-side, judged on uptake, never the bpc pool).
- **Novelty:** refines BE — the **reaction-typing** (in-kind/off-channel/ignored) and the **extinction-burst**
  controller are the genuinely new content; the bare reweight overlaps P1/P8.
- **Revise (applied):** recast "refines BE" as **"BE-timing-reweight vs this per-form reweight is the experiment"**
  (BE being the load-bearing baseline the per-form table must beat); **scope the new content to reaction-typing +
  extinction dynamics**, ceding the plain form-reweight to P1; report turn-overlap/uptake **separately from bpc** (BK);
  demote the extinction burst to a **secondary qualitative** signature (no curve-matching a borrowed exponent).
- **Experiment.** *Corpus:* BM live loop (Haiku once keyed) + scripted child-register fallback; reaction labels from
  the reply (in-kind by lexical/channel overlap; ignored = empty/timeout); matched passive CorpusEnv of equal tokens.
  *Metric:* (a) **uptake** — does emitting listener-high forms lengthen turn-taking bouts vs self-high forms (Zhang
  in-kind bout effect)? (b) communicative-success rate (in-kind reply) reweight ON vs OFF; (c) the extinction-burst
  signature (when replies stop, emission rate/variance spikes then extinguishes) — secondary. *Baseline:* **BE/G2**
  contingency-gated counting (timing only, no per-form table) — does a per-form audience counter add anything? plus
  AM's **static** listener prior; plus fixed-validity AJ over the same cues. *Kill:* listener-reweight gives no
  uptake/communicative-success gain over BE timing-reweight AND no gain over a static prior on the 99% production
  slice — cut (E3 was pre-flagged for this); if the extinction burst doesn't appear across ≥2 seeds, drop the burst
  controller but keep the reweight if uptake holds. → queue **CA**.

## P15 — Interpersonal-foraging cadence (bursty emission gated by recent payoff) — queue **CB**
Make the loop proactive instead of input-starved. A leaky accumulator `A` over steps sets the next-emit interval; a
contingent in-kind reply gives `A` a downward kick (vocalize-again-**sooner**, the interpersonal-foraging result), so
a hit → talk more now, a dry spell → back off. Threshold `A` into two states: **EXPLOIT** (short intervals — the
listener-model and BE hot-gain are paying out, stay in this patch, high learning gain) and **EXPLORE** (long
intervals — responded-rate in the current patch has dropped, switch patches: raise temperature and let P12's
curiosity drive sample a higher-entropy continuation). The inter-event interval becomes a cheap online signal of
whether the current form/topic is still paying out — and binds the two drives: EXPLOIT feeds P14, EXPLORE hands
control to P12.

- **Rules:** online (one leaky accumulator + a two-way threshold; the patch-payout estimate is P14's per-form
  responded-rate), bounded (O(1) cadence state; reads existing counters), no-backprop (threshold on a leaky scalar),
  cognition-as-guide (interpersonal foraging; explore/exploit).
- **Refines/extends:** the explore/exploit **coupler** between **P12** (explore) and **P14** (exploit); distinct from
  **G5** (a fixed pacing oscillator — this is payoff-kicked).
- **Novelty:** NEW as a turn-control coupler (payoff-kicked intervals, not a fixed oscillator).
- **Revise (applied):** run it **only as the coupler** tying P12 (explore) to P14 (exploit), not a standalone learner;
  **drop the borrowed burstiness-β curve-match** (curve-matching a borrowed exponent proves little); make matched-token
  learning-gain the **sole** success axis, cadence-realism reported up front as non-kill; differentiate from G5
  explicitly. Do not run until P14 clears its static-prior bar.
- **Experiment.** *Corpus:* BM InterlocutorEnv with varied reply rate/contingency (vary how often the responder
  answers). *Metric:* **learning gain** (Δbpc or Δreply-surprise per token) in EXPLOIT bursts vs EXPLORE at MATCHED
  token count — the only success axis; cadence-realism (burstiness of inter-event intervals) secondary, reported up
  front, not a kill axis. *Baseline:* fixed-cadence `act()` (current AT); a random-cadence control with the same
  emission budget (isolates payoff-gating from mere variable timing). *Kill:* EXPLOIT-burst tokens yield no higher
  learning gain than EXPLORE tokens at matched count — cadence buys nothing, park (keep only if it wins on cadence-
  realism, and say so up front). → queue **CB**.

---

# INNER SPEECH — a second, self-addressed channel (production + metacognition)

> Honesty constraint (anendophasia, 2024): the self channel is **recruited and gated, never load-bearing** for every
> path. Non-verbal fallbacks (outward band alone) are preserved; kill if it doesn't measurably lift outcome counts
> where recruited. And per AM: a persistent self-channel prior must beat a **static** prior on the 99% non-backoff
> slice or be parked as a backoff prior, not the System-2 organ claimed.

## P16 — SELF channel: a second self-addressed count table, difficulty-gated — queue **CC**
Add a second token register to `CortexAgent` sharing ONE vocabulary with the outward stream but keyed `author='self'`:
a third band of Columns (`cols_self`, same orders) plus a SELF context buffer interleaved into `self.buf` with a 1-bit
channel tag. Before each `act()` the agent emits a short self-trace into `cols_self` and appends it to its own context
(AT's `act()` output fed back as the next context, but **withheld from `env.step`** — "going inward" is literally
`act_len>0` into `self.buf` with the utterance not externalized). The SELF band is **seeded, not invented**: when the
InterlocutorEnv reply contains a regulatory n-gram aimed at the agent (an imperative/correction detected as a
high-overlap restatement of the agent's last act — the repair trigger), copy that n-gram into `cols_self` with
`author='self'` (Vygotsky's other-to-self transfer = copy the regulator's n-gram into the self table). Volume is gated
on **difficulty = branching-entropy of the current outward dist** (Exp A signal): high entropy → emit more self-tokens,
low entropy → none. Prediction pools outward+self bands via the existing geometric-mean `vote()`, self weighted by its
own AB `(f,c)` so an unreliable self-channel can't dominate. This is the substrate the next two organs read/write.

- **Rules:** online (additive counts into a second band + a per-step entropy scalar; identical update path to
  `observe()`), bounded (same trim + LFU/AI eviction; difficulty-gating means most low-entropy steps emit **zero**
  self-tokens, so the table grows only where outward prediction is genuinely uncertain), no-backprop (counts + leaky
  accumulators; seeding = copying observed n-grams; gating = thresholding an already-computed entropy), cognition-
  as-guide (Vygotsky private speech; difficulty-gated).
- **Refines/extends:** extends **AT** (act-into-self), reuses **A** (the entropy gate), **AB** (the self-band weight),
  the **InterlocutorEnv** regulatory channel (the seed).
- **Novelty:** NEW (a self-addressed band seeded by the partner's regulatory n-grams).
- **Experiment.** *Corpus:* InterlocutorEnv over CDS (or text8+babble substitute as in BE) with a responder that issues
  occasional imperatives/corrections; held-out text8 for bpc. *Metric:* (1) held-out bpc of outward+self pool vs
  outward-only; (2) **self-talk volume vs outward branching-entropy correlation** (must rise with difficulty — the
  private-speech-rises-with-difficulty signature); (3) does seeding from regulatory replies lift bpc MORE than the same
  tokens read as ordinary outward input (the BE contingency comparison, reused)? *Baseline:* outward-only `CortexAgent`
  (today's agent, `act_len=0`); plus a **NULL-self** ablation (self band fed RANDOM tokens at the same volume — isolates
  self-CONTENT from extra counts). *Kill:* the self band added to the vote does not beat outward-only on held-out bpc
  AND self-volume shows no positive correlation with outward entropy after the FRAGILE budget (≥10 settings of gate
  threshold × self weight). If it only helps the rare/backoff slice, park it as a backoff prior, not a general organ.
  → queue **CC**.

## P17 — Condensation by self-MI: self-utterances compress as their counts mature — queue **CD**
Make the SELF channel **shorten** with mastery (Vygotsky predication/abbreviation, on counts). For each recurring
self-trace, track its leaky count and its self-internal predictability (branching-entropy of its own continuation under
`cols_self`). When a trace's count clears a threshold AND its continuation is low-entropy, do two count edits: (a) drop
the lowest-**mutual-information** token in the trace — the "already-known subject," the token whose presence barely
changes the rest of the self-utterance's distribution (MI≈0 against the remainder, read straight off the self counts) —
keeping the high-MI "operative predicate"; (b) leader-cluster the recurring condensed trace into ONE centroid token
(spawn-on-novelty, the AU chunk-minting move). Keep a bounded expansion table (condensed-token → full phrase) so when
outward branching-entropy **spikes** the agent re-externalizes: decode the condensed token back and emit it as overt
private speech. Flexible expanded↔condensed switching as a count+entropy gate.

- **Rules:** online (per-trace count + leaky entropy + an MI ratio off existing self-counts; leader-clustering is the
  allowed online primitive), bounded (a **net memory WIN** — a long recurring self-phrase collapses to one centroid;
  the dropped low-MI token frees its row; the expansion table is bounded, LFU-evicted), no-backprop (count threshold +
  MI comparison + leader-cluster + a table lookup), cognition-as-guide (Vygotsky abbreviation).
- **Refines/extends:** parasitic on **P16** (the SELF channel); reuses **AU/M1** leader-clustering for the centroid;
  the expansion table writes/reads **AH** slots backwards.
- **Novelty:** NEW (the self-MI drop-the-known-subject rule).
- **Revise (applied):** strip the Vygotsky overlay to a one-line motivation; state the falsifiable core flatly
  (**MI-drop beats frequency-drop AND outward bpc doesn't degrade**); make it explicitly **contingent on P16 surviving
  its own kill first** (a condensation curve on a dead self-channel proves nothing).
- **Experiment.** *Corpus:* a repetitive InterlocutorEnv task where the same self-regulatory phrase recurs, then a
  difficulty spike injected mid-stream. *Metric:* (1) mean self-trace **length** over stream time per recurring pattern
  — must fall as count matures (the condensation curve); (2) which tokens survive — must be the high-MI "predicate"
  tokens, not the high-frequency "subject" tokens; (3) re-externalization rate vs outward entropy — condensed traces
  re-expand exactly on difficulty spikes; (4) outward bpc must NOT degrade as self-traces shorten (lossless for control).
  *Baseline:* fixed-length self-traces (P16 with condensation OFF); plus a **frequency-only** ablation that drops the
  most-frequent token instead of the lowest-MI token (must lose — proving it's MI, not frequency, that identifies the
  droppable "known subject"). *Kill:* self-trace length does not fall with maturity, OR condensation degrades outward
  bpc, OR the lowest-MI-drop is no better than frequency-drop. Park with the step it died at (FRAGILE). → queue **CD**.

---

# ACTIVE-INFERENCE CONTROL — turn-taking and anticipation as one gain scalar (production)

## P18 — Precision turn-taking gate (read-vs-speak as one gain scalar) — queue **CE**
Replace AT's fixed `act_len` cadence with a precision/gain gate that decides **when to speak**. Two leaky accumulators:
a **LISTEN** accumulator charged by incoming-token surprise / branching-entropy (Exp A) and AC's KL boundary spike on
the partner stream (high surprise = the world has something to teach, keep listening); a **SPEAK** accumulator charged
when the agent has a high-confidence emission ready (own `f·c` from AB clears a bar AND, via P10, a candidate with
positive predicted partner-effect is staged). A single scalar gain `g` multiplies input influence: HIGH `g` = listen
mode (input updates counts strongly via BE's hot register, suppress own output); LOW `g` = speak mode (down-weight the
tokens the agent is producing so it does **not** re-ingest its own words as evidence — the **self-audio attenuation**
Friston names). Flip `g` when SPEAK crosses LISTEN, with hysteresis to avoid chatter.

- **Rules:** online (two leaky accumulators + one threshold flip per step; charges from Exp A entropy / AC KL / AB
  `f·c`, all already online), bounded (O(1) state; reuses BE's hot/cold tables for the attenuation), no-backprop
  (AG's count-native gate shape — compare accumulators, threshold, flip), cognition-as-guide (precision/turn-taking;
  self-audio attenuation).
- **Refines/extends:** re-points **AG**'s gate machinery (when-to-think → when-to-speak); reuses **AB**, **A**, **AC**,
  **BE**; distinct from **G5** (a pacing oscillator that doesn't gate read-vs-speak on input-surprise vs own-confidence
  and has no self-attenuation).
- **Novelty:** NEW as a turn-control organ — the **self-audio attenuation** term is the isolable new piece.
- **Revise (applied):** **lead with the falsifiable self-ingestion-error claim** (gate ON vs OFF cuts the bpc
  contribution from re-counting own output) as the load-bearing test; treat the Friston precision framing as motivation
  only; verify the AT reactive loop runs before claiming the read-vs-speak win. Attenuate only the **evidence-from-world**
  channel for self-tokens, keep a separate self-monitoring update (don't conflate "don't treat my output as the world's
  reply" with "don't learn from my output at all").
- **Experiment.** *Corpus:* InterlocutorEnv (scripted, then Haiku) over CDS-flavoured dialogue / text8-as-reply; vary
  how informative the partner's turns are. *Metric:* (1) held-out bpc and turn-overlap under the precision gate vs fixed
  cadence; (2) **self-ingestion error** — bpc contribution from the agent re-counting its own emitted tokens, gate ON vs
  OFF (the attenuation should cut it — the load-bearing axis); (3) flip behaviour — does `g` enter speak-mode
  preferentially when own-confidence is high AND incoming surprise is low? *Baseline:* AT fixed `act_len` cadence
  (gain≡1, no attenuation) at matched tokens; plus a **no-attenuation** ablation (flip `g` but keep counting own output)
  to isolate the self-audio term. *Kill:* precision gating matches fixed cadence on bpc AND turn-overlap at matched
  tokens, AND self-attenuation removes no measurable self-ingestion error — the gain knob is inert, revert to fixed
  cadence. FRAGILE budget on charge-rates and hysteresis. → queue **CE**.

## P19 — Anticipatory completion-launch buffer (predictive, not reactive) — queue **CF**
Levinson's timing constraint (~200ms gaps vs ~600ms production latency) forces launching production **while** the
incoming turn is still unfolding. Don't wait for end-of-input to compose. Run an incremental completion-point detector
on the partner stream: a branching-entropy estimate (Exp A) of how close the incoming utterance is to a likely
completion — low branching entropy = predictable ending soon. In parallel, as soon as the partner's intent is guessable
(P10's running prediction of partner tokens sharpens), **pre-stage** candidate responses by pulling them from P10/G1
into a small staging buffer. The launch is then: when completion-probability crosses a threshold AND a candidate is
staged, release it (the SPEAK accumulator of P18 is armed by this detector). Emission fires AT the turn boundary instead
of one full composition-latency after it.

- **Rules:** online (branching-entropy on the incoming stream + a running P10 prediction; staging is a copy of the few
  top candidates), bounded (staging buffer holds a fixed small k candidates; the detector is O(1) per token; no new
  persistent table), no-backprop (threshold on a count-derived entropy + argmax staging), cognition-as-guide (Levinson
  anticipation).
- **Refines/extends:** reuses **A** (the completion detector); depends on **P10** (the partner predictor) and **P18**
  (the SPEAK accumulator it arms).
- **Novelty:** NEW (nothing pre-stages production mid-incoming-turn; `act()` fires only after `observe()` returns).
- **Revise (applied):** **sequence it last, downstream of P10/P18 actually working**; sharpen the metric to **launch
  accuracy** (early-launched response still appropriate at the true boundary vs waiting) since the token-latency claim
  is hard to make load-bearing in a harness with no real-time clock — the accuracy-preserved-under-anticipation result
  is the honest deliverable; keep the launch-early-but-random-candidate ablation as load-bearing (proves staging, not
  the early trigger, carries appropriateness).
- **Experiment.** *Corpus:* InterlocutorEnv with turns long enough to have a mid-turn predictable tail (multi-token
  replies); scripted then Haiku. *Metric:* (1) **launch accuracy** — when launched early, is the staged response still
  appropriate at the actual boundary (must not degrade well-formedness vs waiting)? (2) response latency (tokens of dead
  air between the partner's true turn-end and the agent's emission), anticipatory ON vs reactive — secondary; (3) does
  the branching-entropy completion detector fire within the early part of the incoming turn? *Baseline:* reactive launch
  (compose only after `observe()` returns the full turn — the AT default); plus a **launch-early-but-random-candidate**
  ablation (shows the staging, not just the early trigger, carries accuracy). *Kill:* early launch degrades response
  appropriateness (early-staged responses worse at the true boundary than waiting), OR the completion detector fires no
  earlier/no more accurately than end-of-turn detection — anticipation buys nothing, revert to reactive. FRAGILE budget
  on the completion threshold. → queue **CF**.

---

# Cuts & merges (the reconciliation ledger)

Four mechanisms were **cut** as duplicates or a category error; several were **folded** into a surviving organ.

- **CUT — Verb-frame complement routing into per-entity BELIEF tables (mental-verb-count-gated).** The one true
  **category error**: it reifies "an embedded proposition can be false while the sentence is true" as a per-entity count
  table routed by a verb frame. The load-bearing part of false belief is tracking a divergence between an agent's
  representation and the world and using it to predict that agent's action; a complement-clause n-gram conditioned on a
  subject token is topic-conditioned backoff — it matches a Sally-Anne template by surface co-occurrence with no belief
  content, and the "NSL emergence curve" is a count-threshold artifact. Also strains bounded memory (unbounded per-entity
  belity-table spawn). Parked for a richer grounded env; if salvaged, only as a pure verb-frame **topic-routing** test
  with **no** false-belief or ToM claim.
- **CUT — Per-listener news-gate (request/inform/share reward shapes).** Re-skin of P8/P1: "gain each candidate by
  surprisal under a per-listener known-tokens table" is the listener-divergence gate again, and the three-motive reward
  shapes duplicate the GET/SHOW split. **The INFORM news axis is folded into P8**; the REQUEST/SHARE trichotomy is parked
  as inspection-only (not smuggled in as established).
- **CUT — "The audience model: a second listener table + speak→decode→reward loop."** The **same** organ as P1
  (explicitly "AV read backwards" + decode-reward + AB-split MISS repair + yoked control). Two near-identical proposals;
  the depth-1-RSA framing (P1) is kept, this duplicate cut.
- **CUT — "Count-native audience model: forward-sketch re-vote + ask-trigger."** Another second-listener-table-as-
  generation-bias scored vs AM's static prior; its win axes are covered by P8 (redundancy) and P6 (audience-aware ask).
  Its **near-zero-count forward-sketch ask trigger is folded into P6**.
- **CUT — "Audience-model critic over self-traces."** A second partner-trained listener table scoring SELF-channel traces
  as a recipient-design re-vote + utility bandit; both uses are covered by P1/P8 (recipient design) and P5/P6 (the
  AG-gated controller). As a critic over self-traces it is a thin add. Its self-as-listener control = P1's yoked control.

**Merges:** M-AUDIENCE's recipient-design scorer was merged into **P7** (drops its separate table, rides P1's L0). The
INFORM news-gate merged into **P8**. The forward-sketch ask trigger merged into **P6**.

**Tally:** 19 organs kept (12 KEEP + 7 REVISE), **5 cuts** (1 category error + 4 duplicates, two of which contributed a
folded sub-mechanism).

---

# How these compose (the build order)

The fastest path from the BD/BL producer to a speaker that means it:

1. **P1 (BN)** — the listener table (AV read backwards) + decode-reward loop. Unblocks every social organ; judged on
   referential success against a yoked listener.
2. **P7 (BT)** rides P1 for the verbosity dissociation; **P8 (BU)** adds redundancy suppression; **P9 (BV)** gates the
   whole rerank to a small ambiguous fraction (Keysar).
3. **P2 (BO)** tags emission by function (echoic/mand/tact); **P13 (BZ)** feeds its mand table the per-chunk reward; **P3
   (BP)** splits GET vs SHOW once P1 lands.
4. **P5 (BR)** + **P6 (BS)** make `act()` three-way (emit / deliberate / ask) on the existing confidence scalar.
5. **P12 (BY)** is independent — the intrinsic drive that needs no listener; **P14 (CA)** + **P15 (CB)** add the
   audience-conditioned reward and the foraging cadence once the loop is live.
6. **P16 (CC)** + **P17 (CD)** add the self channel; **P18 (CE)** + **P19 (CF)** the precision turn-taking and
   anticipatory launch.
7. **P4 (BQ)** (conceptualizer) and **P10 (BW)** (EFE selector) sit above the producer once P1/P7 have cleared their
   static-prior bars; **P11 (BX)** is built last and is expected to be a clean negative.

The whole library inherits AT's central kill-test — **does reactive dialog teach more per token than passive reading?**
— and the live-Haiku confirmation that BM still owes once a key is set.
