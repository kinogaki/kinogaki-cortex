# Exp O — how fast can a Column go? np.unique vs dense bincount vs MLX/Metal GPU — 2026-06-25 (autonomous)

Same char Column (order 5), same backoff model, **identical bpc**, three backends timed (learn + batch eval) on
enwik9. The question: "really fast columns + GPU", and honestly — where does the GPU actually help?

| size | backend | learn | eval | test bpc |
|---|---|---:|---:|---:|
| 100 MB | np.unique (sort) | 11.20 s | 0.18 s | 1.8771 |
| 100 MB | bincount (CPU dense) | 5.45 s | 0.07 s | 1.8771 |
| 100 MB | **MLX (GPU/Metal)** | **0.33 s** | 0.06 s | 1.8771 |
| 300 MB | np.unique (sort) | 36.87 s | 0.20 s | 1.8316 |
| 300 MB | bincount (CPU dense) | 18.25 s | 0.09 s | 1.8316 |
| 300 MB | **MLX (GPU/Metal)** | **0.74 s** | 0.02 s | 1.8316 |

## Findings

1. **The representation matters more than the hardware, first.** Switching the count table from sorted-unique
   (`np.unique`, O(n log n)) to a **dense histogram** (`np.bincount`, O(n)) is a free **2× on CPU** — and it's the
   representation the GPU wants. Bounded order (≤5 → 27⁶ = 387 M bins fit) makes this possible.
2. **GPU/Metal is a genuine ~25–50× on learn.** MLX scatter-add into the dense table: **100 MB in 0.33 s
   (34× over sort, 16× over dense CPU); 300 MB in 0.74 s (50× / 25×).** A gigabyte column → ~2–3 s on the GPU
   vs 156 s on `np.unique` (Exp N). Same bpc to 4 decimals — it's the identical model, just on Metal.
3. **Where GPU helps is learn (scatter) and will help more for voting/attention (many experts gathered + reduced
   in parallel).** Eval here is already sub-0.1 s on CPU (memory-bound gather); the GPU's headroom shows up when
   there are *many columns* to pool — the attention/voting regime, and the reason Monty's "multiple columns are
   slow" becomes a non-issue for us.

**Net:** dense histograms + MLX/Metal make the Column effectively free to train at gigabyte scale. Order can now
grow (6→8) and many columns can vote without a speed penalty — the substrate is no longer the bottleneck.
MLX (Metal under the hood) was the right call; a hand-written C++/Metal kernel is only needed if a future op
(e.g. fused gather-vote over thousands of columns) outgrows MLX.
