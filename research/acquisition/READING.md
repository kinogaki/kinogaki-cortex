# READING — annotated bibliography

The key sources across all twelve angles. One line each: why it matters and what to take.
Grouped by the [SCIENCE.md](SCIENCE.md) angle. Where a source directly grounds a mechanism,
the [M#/G#/E#](MECHANISMS.md) is named.

---

## 1. Statistical & distributional learning

- **Saffran, Aslin & Newport 1996** — TP segmentation in 8-month-olds; the frequency-matched
  control. *Take:* the clean kill-test corpus for M1/M3; proves conditional structure, not
  frequency. https://www.science.org/doi/10.1126/science.274.5294.1926
- **Harris 1955 / successor variety** — branching entropy rises at boundaries. *Take:* this is
  Exp A; the segmenter we already won with. https://doi.org/10.1080/00437956.1955.11659552
- **Isbilen et al. 2023** — chunk-based segmentation; a learned ABC weakens B–C (the splice
  test). *Take:* **the** result M1 must reproduce; pure-TP is the loser.
  https://doi.org/10.1111/cogs.13193
- **Siegelman et al. 2017** — SL benchmark scores have poor reliability. *Take:* never validate a
  capability on a single F1; require ensemble convergence. https://doi.org/10.1016/j.jml.2017.01.001
- **Benjamin et al. 2023 (neonates)** — tracking the statistic and committing a boundary are
  dissociable. *Take:* keep the entropy signal and the commit threshold as two knobs (M1/M3).

## 2. Usage-based & construction grammar

- **Tomasello 2003, *Constructing a Language*** — item-based → abstract; no verb is an island.
  *Take:* the developmental trajectory M6 must show; transfer = the cross-anchor merge.
  https://www.hup.harvard.edu/books/9780674017641
- **Stefanowitsch & Gries (collostructions)** — association beats raw frequency for slot
  membership. *Take:* the substrate swap in M5 (ΔP/PPMI). https://doi.org/10.1075/ijcl.8.2.03ste
- **Goldberg, coverage/competition** — creative use licensed by coverage, blocked by a competitor.
  *Take:* G1's coverage×frequency heuristic (presented as inspired-by, not the human mechanism).
- **"LLMs learn constructions humans don't know" (2025)** — raw co-occurrence over-generates.
  *Take:* association + preemption is the constraint that keeps the inventory human-like (M5).
- **Royal Society Open Science 2024 (CDS construction model)** — the holophrase→abstract
  trajectory needs CDS-scale data + an exposure-order axis. *Take:* don't claim a trajectory from
  text8 alone (M6 caveat).

## 3. The bootstrapping problem

- **Mintz 2003, frequent frames** — two-sided A_x_B frames, 91–98% same-category on top-k CDS
  frames. *Take:* M6's two-sided frame; bind the figure to top-k CDS only.
  https://doi.org/10.1016/S0010-0277(02)00247-3
- **Syntactic-bootstrapping meta-analyses (d≈0.24; Cao & Lewis-style)** — real but small,
  English-favoring. *Take:* weight the verb-frame cue modestly, never sole (M9).
- **2025 word-order ablations** — order hurts verbs, co-occurrence hurts nouns. *Take:* M9's
  dual-representation dissociation — the sharpest falsifiable claim in the library.
- **Cassani et al. (seeded categorization)** — a handful of labeled seeds bootstrap noun/verb.
  *Take:* M8's seed-injection + precision-first abstain.

## 4. Word learning, reference & constraints

- **Yu & Smith 2007** — cross-situational word learning. *Take:* M2's scene paradigm.
  https://doi.org/10.1111/j.1467-9280.2007.01915.x
- **Trueswell et al. 2013** — propose-but-verify; at-chance after a disconfirmed trial. *Take:*
  M2 variant B's human signature (the kill-condition). https://doi.org/10.1016/j.cogpsych.2012.10.001
- **Frank, Goodman & Tenenbaum 2009** — a *Bayesian* cross-situational model with ME. *Take:* PMI
  **approximates** this pressure; do not claim ME "for free" (M2/M11 honesty fix).
- **Smith et al., shape bias** — emerges after ~50 nouns, not innate. *Take:* M12's emergence
  curve is the test.
- **Vong et al. 2024, CVCL (*Science*)** — grounded word learning from a child's head-cam via
  simple online association + contrast. *Take:* the existence proof that count-native grounding
  works; the centroid/contrast read-out M2/M11 lean on.
  https://www.science.org/doi/10.1126/science.adi1374

## 5. Developmental sequence as curriculum

- **McMurray 2007** — the spurt is parallel accumulation over a Zipfian difficulty distribution.
  *Take:* almost exactly our Column bank; the S-curve is a sanity check, not a feature (M14, M2).
  https://doi.org/10.1016/j.cognition.2007.07.015
- **Brown 1973 (14-morpheme order)** — real order, debated determinants. *Take:* M18 reports
  correlation, does not claim to explain it.
- **Chang 2025 / BabyLM** — low surprisal ≠ learned; distributional trajectories correlate poorly
  with AoA. *Take:* the **hard rule** — never define acquisition by held-out bpc (M14, E2).

## 6. Input, child-directed speech & plausible training

- **Feng et al. 2024** — at ≤100M words the algorithm is the lever, not CDS content. *Take:* test
  mechanisms as algorithm differentiators (per-word extraction), not "train on CHILDES and win."
- **BabyLM curriculum results (β≈−3.6)** — curriculum hurts. *Take:* confirms Exp AK externally;
  gate, don't schedule (CURRICULUM.md).
- **Onnis / Waterfall, variation sets** — adjacent near-duplicates give minimal-pair structure.
  *Take:* M16's miner. **Haga et al.** — variation sets help syntax, not world knowledge. *Take:*
  M16's honest split.
- **Ferry / Thiessen (IDS)** — the benefit is attentional/prosodic, not textual. *Take:* M15
  models IDS as a salience multiplier, refusing the IDS-as-text category error.
- **Lai & Poletiek (embedding-depth curriculum)** — the one curriculum survivor; thin. *Take:*
  the fragile, recursion-only, self-gated exception (BUILD_QUEUE BJ).

## 7. Social-pragmatic & interactive grounding

- **Goldstein & Schwade** — contingent vs yoked replies dissociate learning with identical text.
  *Take:* the **yoked ablation** is G2's required control. https://doi.org/10.1111/j.1467-9280.2008.02117.x
- **Clark (reformulation; least-collaborative-effort)** — repair carries both negative and
  positive evidence; ground common information. *Take:* G3's paired count edit; G4's grounded-form
  bias.
- **Ambridge 2018** — preemption and entrenchment are collinear. *Take:* G3 implements them as
  **one** expectation-violation signal; don't claim to dissociate.
- **Salhan 2025, ContingentChat (BabyLM)** — contingency = modest gains on a competent base.
  *Take:* report contingency on its own metric (turn-overlap), not as a bpc breakthrough.
- **Akhtar & Gernsbacher (joint attention contested)** — blind/overhearing learners acquire.
  *Take:* G2's gain stays **soft**; cold input still counts.

## 8. Errors, U-shaped learning & words-and-rules

- **Marcus et al. 1992** — overregularization is rare (~2.5–10%) and constant; micro-U not macro-U.
  *Take:* M19/M21's target and the macro-U kill-condition. https://doi.org/10.2307/1166115
- **Ramscar et al. (recovery without feedback)** — implicit negative evidence; R-W blocking. *Take:*
  M20's predict-then-decrement loop.
- **Weissweiler et al. 2025 (graded productivity)** — "spling"→"splung"; production is graded, not
  rule-perfect. *Take:* M19's default vote stays continuous in suffix entropy.
- **Ferreira & Xu (exemplar chaining)** — overextension 55% top-5 vs 12% baseline. *Take:* M22's
  benchmark; report recall primary, the comprehension>production asymmetry exploratory.

## 9. Prediction & predictive processing

- **Hale 2006 (entropy reduction)** — a difficulty signal distinct from surprisal. *Take:* G7's
  commit/stopping channel; the UID band target. https://doi.org/10.1207/s15516709cog0000_64
- **Huettig & Mani 2016** — prediction is a helping hand, not necessary. *Take:* every
  prediction-error gate ships with an ablation (the base learner must still acquire).
  https://doi.org/10.1080/23273798.2015.1072223
- **Reuter et al. 2019 (predict-and-redirect)** — encoding gated conjunctively. *Take:* the
  two-factor encoding gate (folded into BUILD_QUEUE BI's family).
- **Gambi et al. 2023** — the error-memory boost fails in children (immature episodic binding).
  *Take:* the live-vs-consolidated two-store split (M23/M25).
- **Kidd et al. "Goldilocks" (attention)** + **2025 VOE pupillometry null** — Goldilocks is about
  attention/look-away, not memory write-rate. *Take:* BI's inverted-U is a budget-efficiency
  engineering choice with an attentional analogy, not a "learning law."

## 10. Memory, consolidation & sleep

- **McClelland, McNaughton & O'Reilly 1995 (CLS)** — fast hippocampal / slow neocortical. *Take:*
  the parent of Exp AE; M23's two-store. https://doi.org/10.1037/0033-295X.102.3.419
- **Gaskell / Davis (lexical competition needs sleep)** — novel-word competition emerges after
  sleep. *Take:* M23's pre-sleep-competition~0 marker.
- **Schapiro et al. 2018** — replay biased toward weak/infrequent items. *Take:* M24's
  inverse-count replay — the single most-buildable signal AA/AS didn't implement.
  https://doi.org/10.1098/rstb.2016.0049
- **McClelland 2020 (schema-consistency)** — consistent items integrate fast (vs a rigid
  time-lock). *Take:* M25's gate. **Coutanche 2023 (fast-mapping non-replication)** — *Take:* no
  one-shot direct-to-cortex shortcut (M25's deliberate negative control).
- **Ball et al. 2024–25** — novel-word semantic priming is fragile and dissociates from form.
  *Take:* keep form and meaning consolidation on separate schedules (M24's two budgets).

## 11. Production is not comprehension backwards

- **Levelt (conceptualize/formulate/articulate)** — the production pipeline; seriality debated.
  *Take:* G1's decoder scaffold (one modular instantiation, not "the" pipeline).
- **Levy & Jaeger (Uniform Information Density)** — "that"-omission tracks predictability. *Take:*
  G7's omission-tracks-predictability kill-test.
  https://papers.nips.cc/paper/2006/hash/c6a01432c8138d46ba39957a8250e027
- **Bock & Griffin (structural priming)** — transient lexical boost + persistent abstract. *Take:*
  G8's two-timescale counters.
- **Pickering & Garrod (one-shared-engine, forward-model monitor)** — moderate evidence. *Take:*
  the self-monitor is engineering convenience with a cognitive analogue, not a settled claim.

## 12. Cognitive-neuro / BabyLM learnability

- **BabyLM Challenge** — ~80–85% BLiMP on 10–100M words (heavily epoch-engineered transformers).
  *Take:* E1's honest bar; don't quote 97%/0.85 as ours.
- **Kallini et al. 2024 (impossible languages)** — natural vs scrambled; gap may be
  complexity-driven. *Take:* E1's ablation, with the complexity confound controlled.
  https://arxiv.org/abs/2401.06416
- **Michaelov et al. (surprisal predicts N400)** + **Warstadt & Bowman (functional ≠ mechanistic)**
  — *Take:* validate −log P_count vs N400 as a **read-out correlation**, never claim mechanism
  match (BI's folded validation).
- **Ficarra 2025 (POS-stratified AoA)** — function words stabilize first; nouns need wider context.
  *Take:* E2's POS-split deliverable, recoverable with Exp S offset Columns.

---

## Project-internal anchors (read these first)

- [PROVENANCE.md](../PROVENANCE.md) — the A→AT lineage; how each experiment led to the next.
- [LAB_NOTEBOOK.md](../../experiments/LAB_NOTEBOOK.md) — the running results.
- [FRAGILE_IDEAS.md](../FRAGILE_IDEAS.md) — don't prune weak ideas at the baseline gate; nurture
  10–20 steps; keep a graveyard.
- [MEMORY_CONSTRAINT.md](../MEMORY_CONSTRAINT.md) — the bounded-memory rule; cope via
  generalization, sleep, external memory.
- [COGNITION_AS_GUIDE.md](../COGNITION_AS_GUIDE.md) — human cognition (flaws included) is the
  guiding model; System 1 vs System 2; biases-as-features.
