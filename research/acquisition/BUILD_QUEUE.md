# BUILD_QUEUE — the next experiments (AU onward)

Continuing the A…AT naming (see [PROVENANCE](../PROVENANCE.md)). Ordered **not** by
developmental rung but by **what unblocks the generation organ and the live reactive loop
fastest**. Each entry: one sentence + its kill-condition, cross-linked to
[MECHANISMS.md](MECHANISMS.md).

The through-line: **AT** gave us a harness whose `act()` is gibberish. The fastest path to
real generation is (1) a chunk lexicon so `act()` emits whole units, (2) a constructicon that
drives slot-filling, (3) the comprehension>production gate that makes generation *selective*,
then (4) close the reactive loop so a reply can *teach*. Acquisition-only refinements and eval
organs interleave where they unblock a generation step.

---

## Tier 1 — make `act()` emit real units (unblocks all generation)

**AU — Chunk lexicon with sub-unit interference** ([M1](MECHANISMS.md#m1--chunk-lexicon-with-sub-unit-interference-the-parserisbilen-organ)).
Build a `ChunkColumn` that greedily covers the buffer, mints concatenations, and **leaks weight
from sub-units as the whole commits**; its chunks become `act()`'s emission vocabulary.
*Kill:* the spliced B–C transition does not decay below pure-TP **and** the chunk agent's
generation/bpc does not beat the fixed-order n-gram agent — after the FRAGILE budget (≥10 decay-
rate/cover-policy variations; first weak result is **expected**).

**AW — Streaming-association slot strength (ΔP/PPMI)** ([M5](MECHANISMS.md#m5--streaming-association-slot-strength-p--ppmi-as-the-construction-substrate)).
Swap AF's raw token/type slot scoring for streaming ΔP/PPMI marginals so the constructicon
gates over-generation before G1 produces.
*Kill:* association does not lower over-generation below AF's −39.5% AND does not beat raw-count
on held-out compositional perplexity at equal memory (after 10–20 steps, both dials) — park as
"raw counts suffice for English," not killed.

**BD — Coverage-competition production (the generation turn)** ([G1](MECHANISMS.md#g1--coverage-competition-production-the-open-slot-drives-act-merged-organ)).
Replace flat sampling: retrieve constructions (AO cue+fan), score by coverage×frequency, compete
by AJ take-the-best, fill the slot, emit — with the merged Levelt three-buffer decoder scaffold.
*Kill:* not measurably more well-formed / less over-generating than the flat sampler on the
**constructional** battery (its right axis, not raw perplexity) — park as "needs the situation
model (AM frontier)," do **not** kill the constructicon.

---

## Tier 2 — make generation selective + grounded

**AY — Two-threshold comprehension/production gate** ([M17](MECHANISMS.md#m17--two-threshold-comprehensionproduction-gate-the-cp-lag-organ)).
Gate `act()`'s sampling on an argmax + high-confidence production threshold while comprehension
reads at a low threshold — "understands but won't say it yet."
*Kill:* the C-before-P lag is absent or does not widen with competitor density across two seeds
(check it at the form/grammatical level before killing).

**BF — Margin-gated production** ([G6](MECHANISMS.md#g6--margin-gated-production-read-the-counts-the-hard-way)).
Read the same counts in the production direction; emit only when the runner-up **margin** clears
θ, else back off or stay silent.
*Kill:* gated precision not above ungated at matched recall, OR the gap does not appear/shrink
with evidence.

**AV — Cross-situational word→referent learning (dual-variant)** ([M2](MECHANISMS.md#m2--cross-situational-wordreferent-learning-dual-variant-in-a-scene-bearing-env)).
Build the scene-bearing env (referent-ids the agent did **not** get from the token stream); ship
dense-PMI vs bounded propose-but-verify at equal memory; grounds why `act()` says a word.
*Kill:* neither variant reaches above-chance from co-occurrence alone, OR variant B fails the
at-chance-after-disconfirm signature — keep both on the FRAGILE budget; pick by harness grounding,
not raw accuracy.

---

## Tier 3 — close the reactive loop (a reply teaches)

**BE — Contingency-gated learning rate** ([G2](MECHANISMS.md#g2--contingency-gated-learning-rate-the-temporal-contingency-dial)).
Multiply each count increment by `exp(−Δt/τ)` since the last self-emission, with hot/cold
registers; pair with **G5** two-state cadence as one experiment.
*Kill:* contingency-ON matches the **YOKED** ablation (scrambled timing) on bpc AND turn-overlap
at matched tokens — register the yoked baseline **before** running; surface the honest negative.

**BC — Rescorla-Wagner recovery loop** ([M20](MECHANISMS.md#m20--rescorla-wagner-recovery-loop-recovery-without-feedback)).
Predict the inflected form, then on observing **decrement** the predicted-but-absent form — the
first acquisition use of AT's reactive contract; recovery without correction.
*Kill:* increment-only passive reading recovers just as fast (recovery is just frequency) — keep
the simpler increment-only loop.

---

## Tier 4 — eval bars (judge the acquisition phase honestly)

**BH — BLiMP / minimal-pair Probe + impossible-language ablation** ([E1](MECHANISMS.md#e1--blimp--minimal-pair-scoring-probe--impossible-language-ablation)).
Score grammaticality the field's way over the existing vote; ablate natural vs scrambled English
(control for scramble complexity).
*Kill:* the count band ≤ bigram on BLiMP, OR natural and scramble learned equally easily — do
**not** kill on the gap to transformers; report per-phenomenon (agreement/interrogatives are
expected weak slices).

---

## Tier 5 — acquisition refinements (interleave as they unblock)

**AX — Function-word anchor voter** ([M7](MECHANISMS.md#m7--function-word-anchor-voter-free-top-k-frequency-bootstrap)).
Mine the top-frequency band as a free closed-class anchor; adjacency-vote a category cue into AJ.
*Kill:* adds <2% purity over AF on English AND degrades the AJ-combined result on either language
(confirm it isn't just mis-validitied before killing).

**AZ — Reliability-gated boundary detectors → head-final drift** ([M3](MECHANISMS.md#m3--reliability-gated-boundary-detectors--the-head-final-drift-scoped-down)).
Build the forward/backward-TP + entropy detectors with AB hit/miss tallies; the deliverable is the
**forward→backward-TP drift on a head-final corpus**.
*Kill:* the gate does not show the drift on head-final text after the FRAGILE budget (a clean
negative is acceptable; the ~13mo drift is thin).

**BA — Dual-route inflection head** ([M19](MECHANISMS.md#m19--dual-route-inflection-head-with-fc-blocking-words-and-rules-as-one-knob)).
Fuse AB f·c + AF token/type + AJ take-the-best into one tunable dual-route gate; reproduce the rare
item-specific micro-U.
*Kill:* no gate setting reproduces a low-constant item-specific rate, OR dual-route matches child
error-TYPE no better than single-route.

**BB — Variation-set minimal-pair miner** ([M16](MECHANISMS.md#m16--variation-set-minimal-pair-miner-adjacent-utterance-diffing)).
Diff adjacent utterances over a ring buffer to harvest (slot, filler) evidence for AF and boundaries
for M.
*Kill:* does not improve compositional-generalization OR boundary F1 on **any** slice (Haga's
syntax-only help is a PASS) — kill only if it loses on syntax too.

**BI — Goldilocks learning-rate gate** (fragile; budget-efficiency).
Make the write-weight an **inverted-U** on surprisal (skip the already-known low-s and the
unparsable high-s); fold the N400/cloze **read-out** validation in here (kept distinct from the
write-gate). *(This subsumes the cut "surprise-as-learning-gate"; the monotone version was the
naive form this corrects.)*
*Kill:* at equal table-size the gate does not improve bpc, OR improves it only by storing more (no
budget win) — check the high-s rare-context slice it skips before killing.

**BJ — Structure-graded recursion exposure** (the one defensible curriculum; **fragile**,
recursion-only). Order by **embedding depth**, self-gated on the agent's own branching entropy
(admit depth d+1 only after depth-d transition entropy stabilizes) over **AO** content-cue tables —
the exact axis Exp AK did not test (it killed age/length/memory-budget curricula).
*Kill:* self-gated depth ordering does not beat full-input-from-start (AK's winner) on depth-2/3
center-embedded agreement — given AK's strong prior this is **expected** to be a hard sell; gate
fragile, 10–20 steps on the recursion-only axis. A loss confirms AK extends to structural ordering
(a clean, publishable negative).

---

## Tier 6 — consolidation & sleep (durability, runs across rungs)

**BG — Inverse-count / surprise-prioritized spindle replay** ([M24](MECHANISMS.md#m24--inverse-count--surprise-prioritized-spindle-replay-selective-frequency-stratified)).
Replace AA's uniform replay with an inverse-count heap that protects the rare tail; dual
item-vs-regularity budgets + a developmental anneal.
*Kill:* does not beat uniform replay on **rare-context** bpc at equal offline budget, OR harms
common-context more than it helps rare.

**Also queued (lower priority, cross-linked):** **M23** two-rate memory + lexical-competition delay
(refines AE/AA; pre-sleep-competition~0 marker); **M25** schema-consistency gate (the McClelland-2020
fast-merge route, with the Coutanche one-shot path deliberately not built); **M26** TMR cueing +
conditional early-exit sleep (strong Atlas fit — entity names as the cue); **M13** spacing-sensitive
fast-map (massed-vs-spaced at the word-referent grain); **M21** mastery-mines-the-rule (per-item
micro-U). Each carries its own kill-condition in MECHANISMS.md.

**Acquisition-only stragglers:** **M6** two-sided frames + cross-anchor merge; **M8** seeded label
propagation; **M9** dual-routing (the sharpest falsifiable claim — promote if a dissociation result
would change the angle's confidence); **M10** agreement-gated validity update (must beat
**fixed-validity AJ** on the non-backoff slice, else cut); **M11** ME fan-reservation; **M12**
shape-bias meta-counter (inspection-only without a perceptual channel); **M14** graduation accumulator
(its AoA curve doubles as **E2**'s POS-split companion); **M22** exemplar-chaining for overextension;
**E2** AoA tracker; **E3** communicative-success reweighting (run **after** BE, with G2 as baseline —
cut if it adds nothing beyond contingent timing); **M4** identity/repetition channel (graveyard-bin).

---

## Why this order

1. **AU/AW/BD** turn `act()` from gibberish into whole-unit, construction-driven emission — the
   one thing AT explicitly deferred and the prerequisite for *any* generation claim.
2. **AY/BF/AV** make that emission **selective** (won't-say-it-yet) and **grounded** (about a
   referent) — the difference between fluent nonsense and a first word.
3. **BE/BC** close the reactive loop so a reply *teaches* — the AT kill-test ("does reactive
   dialog teach more per token than passive reading?") becomes answerable.
4. **BH** is the honesty bar that keeps every claim above from being scored on the wrong axis
   (bpc) — required before promoting any of them.
5. Everything else refines or hardens, gated FRAGILE, judged on its own winning axis.
