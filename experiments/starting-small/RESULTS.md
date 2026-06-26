# Experiment AK — memory-budget-as-curriculum ("starting small") — 2026-06-26

**The right axis: acquisition of a LONG-RANGE, accumulation-bound dependency.** Elman 1993 found a recurrent
net learned complex *embedded* structure ONLY if it started small — either the data was staged simple→hard, or
the net's own memory started short and grew. Thrown the full problem at full memory from step one, it failed to
find the long-range structure *at all*. Vygotsky's ZPD says the same from the other side: spend effort at the
edge of the masterable. Our substrate already carries a bounded-memory RULE — we always have a budget. This asks
the sharper question: **is GROWING the budget on a schedule itself a curriculum that beats a fixed budget?** And
it stages MEMORY, not data — the count-native, teacher-free "starting small."

**The dial.** Each (context, next-token) pair is a *leaky accumulator*: on every touch its 27-count row decays
by `lam = exp(-1/H)` before the new token is added, so `H` is a leak-horizon measured in *occurrences of that
context* — the effective number of recent visits that still count. We put `H` on a schedule.

- **GROW** — `H` rises linearly `3 → 600` over the single pass (start short: only fast-recurring local
  regularities survive; grow: longer-range structure composes on top of the now-stable local counts).
- **FULL** — `H = 600` from char one (Elman's "full complexity from the start").
- **FIXED** — `H = 3` the whole pass (a permanently short memory).

**The test corpus** (Elman's embedded-agreement paradigm, count-native). A char stream of
`<subject-key> <embedded clause of varied filler> <key> run{s|·}`. Each of **120 distinct 3-letter keys** is
permanently bound to a number class (50/50); the agreement target after `run` is `s` (singular) or *space*
(plural) — a clean 50/50 with **no local cue**. The key is re-emitted right before the target, so an order-8
char context *can* span `<key> run → target` — the dependency is **reachable**, not out of n-gram range. But
each specific `<key> run` context recurs only **~every 120 sentences**, interleaved with all other keys. So a
SHORT horizon decays a key's class count to nothing before its next visit (→ chance), while a LONG horizon
accumulates it across sparse visits (→ the agreement). *Whether the bound dependency is acquired turns entirely
on the leak-horizon* — exactly the dial the schedule moves. Chance on the target = ppl 2.0 / acc 0.50.

Char-level, orders 1–8, 12k sentences (~502k chars), single pass, fixed seed 0.

## Result — perplexity on the cue-distant agreement tokens (lower = better)

| regime | target ppl | target acc | overall bpc | vs FULL |
|---|---:|---:|---:|---:|
| FIXED (small, constant) | 3.570 | 0.699 | 1.799 | **+30.1%** |
| FULL (large from start) | **2.744** | 0.712 | 1.418 | +0.0% |
| GROW (small → large)    | 2.751 | 0.712 | 1.423 | +0.3% |
| GROW + ZPD (conf-weighted) | 2.903 | 0.711 | 1.502 | +5.8% |

**Robustness (target ppl, seeds 1/2/3, plus a slow-start quadratic GROW):**

| seed | FULL | GROW (linear) | GROW (quadratic) | FIXED |
|---|---:|---:|---:|---:|
| 1 | **2.874** | 2.883 | 2.944 | 3.659 |
| 2 | **2.818** | 2.828 | 2.883 | 3.597 |
| 3 | **2.804** | 2.808 | 2.853 | 3.526 |

## Findings

1. **HONEST NEGATIVE: growing the budget does NOT beat full-from-start.** Across every seed and both schedule
   shapes, FULL edges GROW by a hair (+0.3–0.5% ppl) and never loses. Spending *longer* small (the quadratic
   schedule) makes GROW strictly *worse*, not better — the opposite of what a curriculum should do. **Elman's
   "starting small" effect does not reproduce for a count learner.**
2. **But a permanently small budget DOES lose, decisively.** FIXED is +30% ppl over FULL on the long-range
   token. So the bounded-memory regime is real and the leak-horizon is load-bearing — the dependency genuinely
   *is* accumulation-bound. The schedule simply isn't the lever: what matters is the *final* horizon, and
   reaching it sooner is (very slightly) better.
3. **Why the asymmetry with Elman.** Elman's net failed full-from-start because gradient descent on a high-
   capacity net *locks in* a bad early solution it can't climb out of; starting small kept it plastic until the
   easy structure was in place. A count model has **no gradient to lock** — counts are additive and
   self-correcting, an early noisy high-order count is simply *outvoted* by later evidence, never frozen. So
   there is nothing for a growing schedule to rescue. The interesting reading: **"starting small" was a property
   of the optimizer, not of learning itself.** A teacher-free count learner needs no curriculum; it only needs
   enough *final* memory.
4. **The ZPD overlay hurts (−5.8% ppl).** Confidence-weighting exposures (up-weight near-threshold contexts,
   down-weight mastered) consistently *raised* perplexity. Down-weighting a "mastered" high-confidence context
   starves exactly the keys whose sparse evidence must be hoarded to survive the leak — in a count substrate,
   throttling the count you add to a confident context is just discarding evidence. ZPD pruning assumes a
   capacity-limited learner that benefits from spending effort elsewhere; additive counts aren't capacity-
   limited that way, so the prune is pure loss.

## Online note

Strictly one streaming pass, predict-then-update at every position; no replay, no gradient descent, no batch
optimization. The leak is a per-context-touch lazy decay (a leaky accumulator, the substrate's existing
primitive); `H` is read from the schedule at each step from the fraction of stream consumed. The ZPD weight
reads the predicting context's online NARS confidence (hit/miss top-1 track record) as of its own arrival.
Nothing iterates to convergence; nothing backprops. Fixed seed; verdict holds across seeds 1–3.

## Axis

The headline axis is **acquisition of a reachable-but-sparse long-range dependency under a memory budget** —
measured as perplexity on the cue-distant agreement token, not flat bpc (flat bpc is dominated by local
filler). On that axis the verdict is an honest negative for the curriculum hypothesis: **growing the budget
ties full-from-start; only a permanently small budget loses.** The bounded-memory rule stands; "starting small"
does not transfer from gradient nets to count learners.
