# Experiment F — scaling dynamics (data × cortex capacity) — 2026-06-25

**Setup.** text8 (100MB clean English, a-z+space). Vectorized fixed-order char count model, add-α, fixed 2M-char
held-out. Two axes: **data** 1→90MB (≈90×) and **capacity** = n-gram order K=2..5 (how much context the cortex holds).

## The grid — bits-per-char (rows = capacity K, cols = train MB; lower better)

```
  K\MB      1       3      10      30      90
  2      2.959   2.927   2.913   2.906   2.902      ← saturated: 90× data buys only −0.057
  3      2.526   2.445   2.395   2.374   2.360
  4      2.321   2.171   2.070   2.025   1.992
  5      2.402   2.153   1.972   1.880   1.816      ← biggest data gains, still dropping (−0.064 last 3× step)
```

## Dynamics — the answer

1. **More data always helps, but how much depends on capacity.** A small cortex (K=2) is *capacity-limited* —
   90× more data buys almost nothing (−0.057 bpc). A large cortex (K=5) keeps improving strongly (−0.587 over
   the range) and is **not yet saturated** at 90MB. → **a bigger cortex extracts more from more data.**
2. **More capacity helps generalization — but only with enough data.** At 1MB, K=5 *overfits* (2.402, worse
   than K=4's 2.321 — data-starved sparsity). At ≥3MB, K=5 wins. → **the optimal cortex size grows with data**
   (the capacity×data interaction — the core scaling law).
3. **Where we are: in the productive regime, NOT saturated.** At our largest point (K=5, 90MB) bpc is still
   falling and optimal-K rose 4→5 as data grew — so K=6+ with yet more data would keep helping. There is
   headroom on both axes.

## Calibration

Order-5 count model = **1.82 bpc** on 90MB text8 — squarely in PPM/n-gram territory (published PPM ≈1.8–1.9).
For reference: char-LSTM ≈1.4, neural SOTA ≈1.0–1.1. The **concept layers add a second capacity axis**: the
word lexicon (Exp C) gave +16–22% on top of a char model, and word-level n-grams compounded (Exp E). So the
full cortex capacity = *context order × concept vocabulary × levels*, and the scaling law says: **to exploit
100× data we must grow the cortex (order + concepts) — and it will keep paying off.**

## Implication for the project

We are **data-and-capacity-scalable, not saturated** — the substrate behaves like a real learner (power-law-ish
data gains, capacity×data interaction). Before adding new mechanisms, the cheapest wins are simply *more data +
bigger cortex* (higher order, larger concept vocabulary, more levels). The architecture has room to climb from
1.82 toward neural territory by scaling capacity, while staying online/associative/inspectable.
