# Experiment AS — what survives a budget — 2026-06-26

**The setup.** The capstone "what survives scale" (overnight batches) measured every candidate mechanism with
**unbounded memory** — `np.unique` over the whole stream, every count kept forever. Under infinite storage the
verdict was blunt: **more data subsumes any mechanism that merely re-predicts what raw counts already hold.**
The top-down topic prior (ignition), word-concept generalization, and consolidation/sleep all read out ≈0 gain
at 50–150 MB. They were flat because nothing was ever thrown away.

**The prediction under test.** The bounded-memory rule says this should **flip**. Once you cannot keep every
count, you must discard most of them, and the only way to stay good after discarding is to have **generalized
first** — turned a million specific high-order counts into a few reusable abstractions. So under a *fixed entry
budget* the "vanished" mechanisms should **earn their keep** (improve bpc/bits at equal memory) where they were
flat unbounded.

We re-ran three of the vanished mechanisms at a real data scale (**text8, 60 M train chars / 2 M held-out**,
single streaming pass, fixed seed 0) and measured, for each, two deltas:

> **Δ_unbounded** = quality(mechanism ON) − quality(mechanism OFF), full tables  → expect ≈ 0 (the prior finding)
> **Δ_bounded**   = the same, but BOTH sides capped to the SAME stored-entry budget → predicted > 0 if it flips

A **flip** = Δ_unbounded ≈ 0 AND Δ_bounded > 0. The cap is a heavy-hitter keep-top-B (count-min style): keep the
highest-evidence contexts, drop the long sparse tail — exactly the entries an unbounded model leans on. All three
mechanisms reuse the existing online substrate (`lib/consolidate`, `lib/ignition`, `lib/constructions`,
`lib/jepa`) through `lib/budget_dichotomy.py`. Whole run ≈ 3 min on CPU.

## Headline — the dichotomy, at 60 M chars

| mechanism | metric | Δ_unbounded | Δ_bounded | verdict |
|---|---|---:|---:|:--|
| **consolidation / sleep** | bpc | **−0.0056** | **+0.144** | **FLIP** — flat unbounded, large win under budget |
| top-down topic prior (ignition) | bpc | +0.0000 | −0.0000 | neutral — earns no budget slice |
| word-concept generalization | bits/word | +0.0535 | −0.079 | no flip at this scale (flips only when the cap binds hard relative to the data — see below) |

**One mechanism flips cleanly, one is neutral, one is scale-conditional.** The bounded-memory rule's core claim
— *generalization is invisible with infinite storage and decisive under a budget* — is vindicated by
consolidation in the cleanest possible form, refuted for the ignition prior, and qualified for concepts.

## Mechanism 1 — consolidation / sleep: the flip, and it is a CURVE

OFF = raw order-5 char counts. ON = one count-based **sleep pass** (`lib/consolidate.sleep`): prune the
untrustworthy tail, **distill** specific contexts whose distribution equals their backoff (lossless), and
**promote** recurring high-order contexts into shared **concepts** (online leader clustering). Then both sides
are capped to the *same* stored-entry budget and scored on held-out bpc.

- **Unbounded: Δ = −0.0056 bpc.** Sleep is near-lossless by design, so with full tables it changes nothing — the
  prior finding, reproduced at scale. (It is even very slightly negative: distillation drops a few contexts that
  carried a sliver of signal.)
- **Bounded (20 000-context cap, ~296 k entries vs ~2.3 M full): Δ = +0.144 bpc.** A large win. At equal memory,
  the model that *distilled-and-promoted before discarding* keeps reusable generic structure where the raw model
  keeps only frequent literals and falls off a cliff on everything else.

The flip is not a single lucky point — it is a **monotone curve in the budget** (sweep at 60 M):

| context budget | Δ_unbounded | Δ_bounded | entries kept (of 2.3 M full) |
|---:|---:|---:|---:|
| 5 000   | −0.0056 | **+0.307** | 95 k |
| 20 000  | −0.0056 | **+0.144** | 296 k |
| 60 000  | −0.0056 | **+0.052** | 702 k |
| 200 000 | −0.0056 | +0.0015 | 1.56 M |

Δ_unbounded is pinned flat; Δ_bounded **grows as the cap tightens and vanishes exactly when the budget stops
binding** (200 k contexts ≈ full table → back to ≈0). This is the bounded-memory rule made quantitative: the
value of consolidation is *precisely* the memory pressure it relieves.

## Mechanism 2 — top-down topic prior (ignition): neutral, no flip

A committed global topic **G** (online recency-weighted topic histogram + ignition/hysteresis,
`lib/ignition.commit_G`) is broadcast onto every char; the ON model adds a `(G, ctx)` backoff tier consulted
**only where the literal context is unseen** — a top-down prior filling holes, not overriding good literals.

- **Unbounded: Δ = +0.0000 bpc.** With full tables the literal context is essentially always present, so the
  G-tier never fires. G is fully subsumed — the prior finding, exactly.
- **Bounded (entry-matched): Δ = −0.0000 bpc.** Here is the honest negative. When we split a fixed budget between
  literal contexts and a G-tier, the binary search that equalizes *stored entries* keeps spending the budget on
  **more literal contexts** — a marginal literal context is worth as much as a `(G, ctx)` row, and the
  `(G, ctx)` rows are *denser* (G splits a context's mass across topics), so they cost more memory per row. The
  topic prior earns no slice of the budget. (An earlier context-matched cap made G look strongly negative; that
  was a memory-accounting artifact — G rows cost more — not a real effect. Entry-matched, it is a clean zero.)

The verdict: this crude **hash-word → PPMI-cluster** topic signal is too coarse to beat literal contexts at equal
memory. The mechanism does not flip; whether a *sharper* topic representation would is open.

## Mechanism 3 — word-concept generalization: a flip that depends on how hard the budget binds

OFF = raw word frame→filler counts (`X ___` 1-gram frames). ON = open-slot **constructions**
(`lib/constructions`): predict the next word through its filler **category** — `P(w|frame) = Σ_c P(c|frame)·P(w|c)`
— so a frame can place mass on fillers it has never hosted. Memory is counted in stored **cells** (a raw frame
costs one cell per distinct seen filler; a concept head costs one category vector + a *shared* category lexicon).
The bounded ON is a hybrid: literal where attested, category where not, at equal cells.

- **Unbounded: Δ = +0.054 bits/word.** A small residual help — the category head still smooths a few rare pairs
  the raw counts saw thinly. Smaller than at low data (it was +0.84 at 10 M), shrinking toward 0 as data grows,
  consistent with the subsumption story.
- **Bounded at 60 M (150 k-cell budget): Δ = −0.079 bits/word.** At this scale the raw frame table is so rich
  that 150 k cells buys excellent literal coverage; the over-generalizing category head (it spreads mass across a
  whole category) **loses** to spending those cells on more literal frames. No flip — across 30 k–400 k cells the
  bounded delta stays negative.

**But the flip appears when the budget binds hard relative to the data.** At **10 M chars** the same mechanism, at
the same 150 k-cell budget, **flips positive (Δ_bounded = +0.125)** — there the lexical data is scarce enough that
the cap removes genuinely useful frames, and the category head buys them back by generalization. The concept
mechanism's keep is therefore **scale-conditional**: it earns its place only when memory is tight *relative to*
how much lexical data you have. At 60 M with a few hundred k cells, it is not tight enough.

## Honest reading (fragile-ideas axes)

- **Right axis.** The claim is "does generalization earn its keep under a budget where it was flat unbounded",
  scored on held-out bpc/bits at *equal stored memory* — not bpc-vs-bigram or accuracy-on-the-training-set.
- **The clean win is real and load-bearing.** Consolidation flips, and its gain is a monotone function of how
  hard the cap bites (the sweep). This is the bounded-memory rule's central prediction, confirmed.
- **The negatives are honest.** Ignition is a clean zero — the prior is subsumed and earns no budget. Concepts do
  not flip at 60 M; they flip at 10 M. We report both rather than tuning the scale until all three agree.
- **The throughline.** With infinite memory, more data wins — generalization is invisible. Under a budget,
  *whether* generalization earns its keep is exactly a contest of "abstraction cells vs more literal cells", and
  the winner depends on how tight the budget is relative to the data. Consolidation (which generalizes
  *losslessly* — distill is exact, promote is a fallback) wins broadly; lossy generalizations (a coarse topic
  prior, a mass-spreading category head) win only in the regime where the cap removes things they can rebuild.

## Reproduce

```sh
exp_a_boundary/.venv/bin/python exp_as_budget/run.py            # 60 M default; writes results.tsv + the sweep
AS_TRAIN=10000000 AS_EVAL=2000000 exp_a_boundary/.venv/bin/python exp_as_budget/run.py   # the 10 M concept-flip regime
```

Knobs: `AS_TRAIN`, `AS_EVAL`, `AS_BUD_CONSOL` (ctx), `AS_BUD_IGNIT` (ctx), `AS_BUD_CONCEPT` (cells), `AS_SWEEP`.
Library: `lib/budget_dichotomy.py`. Machine-readable: `exp_as_budget/results.tsv`.
