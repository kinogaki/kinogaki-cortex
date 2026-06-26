# Experiment B — catastrophic forgetting (English registers) — 2026-06-25

**Setup.** Char-level next-char prediction streamed through 4 English registers in sequence (austen →
shakespeare → darwin → bible), one pass, no replay, ~120k chars/register. Dense MLP (96k params) vs Sparse
k-WTA readout (61k learned) vs online backoff count n-gram. Eval bpc on held-out from all 4 after each phase.

## Result

**Forgetting (Δ bits-per-char on austen, final − right-after-training-it):**

| model | austen peak bpc | austen final bpc | forgetting Δ |
|---|---:|---:|---:|
| dense | 3.611 | 3.414 | **−0.198** (improved) |
| sparse | 3.736 | 3.636 | **−0.101** (improved) |
| count | 2.508 | 2.353 | **−0.155** (improved) |

**Peak top-1 accuracy per register (did it learn?):** dense 0.28–0.42, sparse 0.27–0.38, **count 0.52–0.63**.

## Findings — what DIDN'T work, and what we learned

1. **NEGATIVE RESULT: no catastrophic forgetting within English.** Every model's austen performance *improved*
   after training on the other registers (positive backward transfer). The planned differentiator demo failed
   — **not because the property is false, but because the test had nothing to forget.** Four English registers
   are ~one stationary distribution at the char level; more English helps predict English. Catastrophic
   forgetting needs genuine task/distribution shift, which "English-only, char-level" does not provide. *To
   demonstrate no-forgetting we'd need either different tasks/mappings or a level where English topics truly
   diverge — char-level English registers are the wrong test.*
2. **KEY POSITIVE INSIGHT: the online associative (count) model beat both gradient nets, decisively** (bpc 2.3
   vs 3.5; acc 0.6 vs 0.3) — *and* it is inherently online, non-forgetting, and backprop-free. The gradient
   nets badly underfit (small, SGD, 120k chars). **This points at the substrate:** our "columns" should be
   **sparse associative memories (counts / Hebbian over sparse codes), not mini gradient nets.** The north
   star (local, online, accumulative, no global loss) and the *better* predictor coincide — that's a strong
   nudge. Gradient/backprop is the wrong tool here; association is the right one.
3. The sparse k-WTA model worked mechanically (localized updates by construction) but, with only a learned
   linear readout over a *fixed random* code, it underfit — a fixed random projection throws away too much. A
   *learned/grown* sparse code (or associative counts over sparse contexts) is the fix.

## Decision / pivot

- **Drop cross-register forgetting as the headline test** (English-only can't show it cleanly; revisit with
  real task shift only if needed).
- **Adopt a sparse-associative substrate** (counts/Hebbian over sparse codes), not SGD nets — it's better here
  AND it's the north star.
- **Chase the north-star-aligned win next (Exp C): does a grown, inspectable CONCEPT hierarchy earn its keep?**
  Use branching-entropy (Exp A) to discover words online, grow a lexicon/concept store, and test whether
  conditioning prediction on the learned concepts lowers bits-per-char vs a flat char model — while yielding an
  inspectable English concept document. That is where a real, useful result most likely lives.
