# Experiment BJ — structure-graded recursion exposure (self-gated embedding depth) — 2026-06-26

**The one curriculum AK did not test.** AK staged the *memory budget* (the per-context leak-horizon `H`) and
returned an honest negative: growing `H` ties full-from-start, because a count learner has **no gradient to
lock** — an early noisy count is simply outvoted by later evidence, never frozen, so Elman's "starting small"
has nothing to rescue. AK's winner was **FULL** (every regularity present from char one). BJ asks the sharper,
*structural* question Elman's original recursion result was about: order the stream by **embedding depth**, and
let the agent **self-gate** the order on its own branching entropy — admit depth `d+1` only once depth-`d`
transition entropy has *stabilized*. Teacher-free: the gate reads the agent's own count tables, no schedule
clock, no labels. This is the exact axis AK left untouched (it killed age/length/memory-budget curricula).

**The test corpus — center-embedded subject–verb agreement** (the textbook recursion stressor):

```
depth 1:  S1            <embedded filler>  run{agree S1}
depth 2:  S1 S2         <embedded filler>  run{agree S2}  run{agree S1}
depth 3:  S1 S2 S3      <embedded filler>  run{S3} run{S2} run{S1}
```

Verbs close **inner→outer** (center-embedding): the LAST `run`'s agreement char depends on the FIRST
(outermost) subject, across the whole nest. Each verb is a **generic** `run` (no key re-emission), so the
agreement char's only cue is the governing subject, back across the embedding. Each of 120 distinct 3-letter
subject keys is permanently number-classed 50/50 (`s` singular / space plural); the middle filler is unique
per sentence so whole sentences can't be memorized. **Difficulty is graded by depth:** the order-`K` window
(K=12) spans the depth-1 subject→verb cue (learnable) but the deeper outer subjects are pushed out of window
by the intervening inner verbs — the center-embedding a windowed count learner has no stack for. Chance on the
agreement char = ppl 2.0 / acc 0.50.

Three regimes on the **same multiset of sentences**, single streaming pass, fixed seed — only the **order**
changes, never the learner (orders 1–12 add-alpha backoff, leak horizon H=200 for bounded memory,
predict-then-update):

- **GRADED** — easy→hard by depth, the depth gate **self-opened** by branching-entropy stability.
- **FULL** — all depths interleaved uniformly from char one (AK's winner).
- **ANTI** — hard→easy (deepest first), the curriculum reversed (ordering control, wrong direction).

12k sentences (4k/depth), 70/30 train/probe split, deep token scored on the held-out probe (`learn=False`).

## Result — OUTER agreement perplexity (the recursion-only axis; lower = better; seed 0)

| depth | GRADED ppl | FULL ppl | ANTI ppl | GRADED acc | FULL acc | what it is |
|---|---:|---:|---:|---:|---:|---|
| 1 | 2.020 | 2.020 | 2.020 | **0.785** | **0.783** | learnable (cue in window) |
| 2 | 2.035 | 2.037 | 2.038 | 0.514 | 0.511 | out of window = recursion test |
| 3 | 2.013 | 2.011 | 2.012 | 0.511 | 0.511 | out of window = recursion test |

Self-gate log (seed 0): opened depth-2 after **242** sentences (Δbe=0.0003), depth-3 after **866**
(Δbe=0.0137) — the gate **fires** (depth-1 entropy stabilizes, then admits the next depth); it is doing its job.

## FRAGILE budget — deep (d2+d3) OUTER ppl across 4 seeds × 4 gate settings (16 variations)

| seed | eps | window | GRADED | FULL | ANTI | G vs F | win? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 0 | 0.02 | 200 | 2.024 | 2.024 | 2.025 | −0.0% | no |
| 0 | 0.01 | 200 | 2.023 | 2.024 | 2.025 | −0.1% | yes |
| 0 | 0.05 | 150 | 2.024 | 2.024 | 2.025 | −0.0% | no |
| 0 | 0.02 | 400 | 2.024 | 2.024 | 2.025 | −0.0% | no |
| 1 | 0.02 | 200 | 2.036 | 2.035 | 2.034 | +0.0% | no |
| 1 | 0.01 | 200 | 2.035 | 2.035 | 2.034 | −0.0% | no |
| 1 | 0.05 | 150 | 2.036 | 2.035 | 2.034 | +0.1% | no |
| 1 | 0.02 | 400 | 2.035 | 2.035 | 2.034 | −0.0% | no |
| 2 | 0.02 | 200 | 2.029 | 2.030 | 2.030 | −0.1% | yes |
| 2 | 0.01 | 200 | 2.027 | 2.030 | 2.030 | −0.2% | yes |
| 2 | 0.05 | 150 | 2.028 | 2.030 | 2.030 | −0.1% | yes |
| 2 | 0.02 | 400 | 2.028 | 2.030 | 2.030 | −0.1% | yes |
| 3 | 0.02 | 200 | 2.033 | 2.033 | 2.032 | −0.0% | no |
| 3 | 0.01 | 200 | 2.035 | 2.033 | 2.032 | +0.1% | no |
| 3 | 0.05 | 150 | 2.033 | 2.033 | 2.032 | −0.0% | no |
| 3 | 0.02 | 400 | 2.033 | 2.033 | 2.032 | +0.0% | no |

**Mean deep ppl: GRADED 2.030 · FULL 2.031 · ANTI 2.030** (chance = 2.000). GRADED edges FULL by >1e-3 in
5/16 variations, but every swing is **≤0.2%** — well inside the seed/gate noise band (0.5%), and the sign
flips both ways across seeds. Mean GRADED-vs-FULL = **−0.02%**.

## Verdict — KILL FIRED (clean, expected negative)

**Self-gated depth ordering does NOT beat full-from-start on depth-2/3 center-embedded agreement** (mean
−0.02%, inside the noise band; ANTI ties too). The kill-condition fired exactly as BJ predicted — and the
*why* is the same mechanism AK exposed, now confirmed on the structural axis:

1. **Depth-1 is learned by every regime (acc 0.78); depth-2/3 stay at chance (acc ~0.51) by every regime.**
   The center-embedding cue is simply out of the order-K window, and a count/backoff learner has **no stack**
   to bridge it. The shallow skill does not *transfer* to the deep span — so there is nothing for a curriculum
   that front-loads the shallow rung to hand off.
2. **The self-gate works but doesn't matter.** It reliably opens depth-2 (~242 sentences) once depth-1's
   branching entropy stabilizes, then depth-3 (~866). The ordering is *real*; its *effect on the deep token*
   is nil.
3. **Ordering DIRECTION is inert.** ANTI (deep-first, the wrong curriculum) ties FULL just as closely as
   GRADED does. For a count learner the order of structural exposure carries no signal at all — counts are
   additive and commutative, so the multiset of sentences fixes the table regardless of arrival order; only
   the *final* counts matter (exactly AK's finding for the memory budget, now for structure).
4. **AK extends to structural ordering.** Where AK showed growing the *memory budget* on a schedule buys
   nothing, BJ shows ordering by *structural complexity* (the one axis AK did not test, the strongest case
   for a curriculum) **also** buys nothing. "Starting small" was a property of the gradient optimizer, not of
   count-native learning — and that now holds for structure as well as memory. A loss here was flagged as
   **expected** (a hard sell vs AK's strong prior); it is a clean, publishable negative.

## Honesty notes / scope

- **Corpus substitution.** BJ names no on-disk corpus; the spec calls for center-embedded agreement (an
  Elman/recursion construct). We synthesize it in `lib/recursion.py` (generalizing AK's single-clause
  agreement corpus to nested depth), as the spec invites ("synthesize toy streams where the spec needs
  them"). No text8/CHILDES slice would carry a controlled, depth-labelled center-embedding signal.
- **The deep token is at chance for all regimes.** This is the load-bearing honest caveat: because the
  windowed count learner cannot represent the depth-2/3 dependency *at all*, the curriculum comparison is
  "does ordering help reach above chance?" and the answer is no for any regime. We report this plainly rather
  than dressing a chance-level tie as a win. The depth-1 rung (acc 0.78) confirms the task is genuinely
  learnable where it is reachable, so the deep failure is the model's missing stack, not a broken corpus.
- **What a follow-up should do.** To give a structural curriculum *any* chance, the substrate would need a
  stack-like organ (a push/pop chunk register, cf. M1's chunk lexicon AU) so the depth-1 skill could compose
  into depth-2. With a pure backoff window there is no composition to bootstrap. BJ's negative is therefore
  also a *pointer*: structural curricula are moot until the learner can represent structure recursively.

## Rules honored

Strictly **one streaming pass**, predict-then-update at every position; **no gradients, no batch optimization,
no k-means/SVD/eigen**; **bounded memory** (per-context leaky accumulator, horizon H=200; finite key/context
vocabulary). The self-gate reads only the model's own online branching entropy. Fixed seed; the verdict holds
across seeds 0–3 and four gate settings. The mechanism lives in `lib/recursion.py`; the experiment in `run.py`.

## Axis

The headline axis is **acquisition of a depth-2/3 center-embedded agreement dependency under a structural
exposure curriculum** — perplexity/accuracy on the OUTER agreement char, not flat bpc. On that axis: an honest
negative for the curriculum hypothesis. Self-gated depth ordering ties full-from-start; the reversed order
ties too; only depth-1 (in-window) is learnable at all. AK's verdict extends from the memory budget to
structural ordering: a teacher-free count learner needs no curriculum, only enough final memory — and a stack
it does not have.
