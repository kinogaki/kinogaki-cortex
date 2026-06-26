# Autonomous session plan (2026-06-25 night) — recovery doc

Running unattended while the user sleeps. Mandate: **gigabyte scale + really fast columns (GPU) + discover
boundaries/phrases + attention.** Document everything as I go (LAB_NOTEBOOK.md + per-exp RESULTS.md).

Env: Python 3.14, numpy 2.5, **MLX installed (Apple Silicon GPU live)**, 16 cores / 64 GB. venv at
`exp_a_boundary/.venv/bin/python`. Corpus: text8 (100 MB) in `data/`; downloading **enwik9 (1 GB)** → `data/enwik9`.

## Order of work (each → background run → analyze → document → next)
1. **Infra (fast):** `lib/corpus.py` (bytes→int8 id array, vectorized normalize) + `lib/fastchar.py`
   (vectorized char backoff with BATCH bpc — kills the per-position Python eval that made Exp K slow).
2. **Exp N — gigabyte char scaling:** char model 10 MB → 100 MB → 1 GB (enwik9), id-space, batch eval.
   Headline: does bpc keep falling to GB, and how fast can we learn. CPU first, then MLX GPU benchmark.
3. **Exp O — GPU columns (MLX):** port char count+predict to MLX, benchmark vs numpy at 100 MB / 1 GB.
   (Char order ≤5 so combined id fits int32: 27^5·27 = 3.9e8 < 2.1e9.)
4. **Exp L — associative attention** (already built: `lib/attention.py`): run at 10–40 MB; inspect what it
   learned to attend to; does it beat fixed n-grams on coherence.
5. **Exp M — emergent boundaries:** branching-entropy phrase discovery over the WORD stream (reuse
   `cortex.branch_chunk`), Bayesian-surprise TOPIC boundaries over phrases; scope attention to topic segments
   (un-cheat the space-splitting). Metric: phrase sensibility + does discovered segmentation beat fixed.
6. **Morning summary:** update LAB_NOTEBOOK state section + write a top-level NIGHT_REPORT.md.

## Invariants / guardrails
- Document honestly (what worked AND didn't). Newest at bottom of LAB_NOTEBOOK.
- Don't touch production/AWS/PyPI. Only experiments under `kinogaki-cortex/`. No `pip install kinogaki`.
- The uniform `Column` thesis: same component, wired bigger. Surprise = boundaries + attention + learning.
- If GPU fights back, fall back to the (reliable) numpy batch path and document the GPU design for later.

## Done so far tonight
- Exp I (uniform cortex), Exp J (vectorized backend, 9× + scale law), Exp K (3/4/5 levels × 10/40/80 MB —
  more data helps a lot, 4th level flat, topic cache modest constant win → motivates attention/boundaries).
