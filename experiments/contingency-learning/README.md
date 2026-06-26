# Contingency-gated learning rate

**Experiment BE · 2026-06-26 · acquisition (G2)**

A reply that *answers* you teaches more than the same words overheard cold (Goldstein & Schwade's
contingent-vs-yoked babbling study). This installs that dial: in `observe()`, weight each count increment
by a soft gain `g = exp(−Δt/τ)` (Δt = steps since the agent last spoke), into two bounded registers —
`tab_hot` (warm replies, loud) + `tab_cold` (background, never silenced); prediction pools both. The agent
lives in a conversation where real-English replies arrive **warm** (right after it speaks) and babble fills
the cold gaps, so the dial up-weights the structured signal and down-weights the noise *from timing alone*.

The load-bearing control is **YOKED**: identical tokens, identical loop, gain drawn from the *scrambled*
timeline — same count mass, wrong alignment. Result: contingency-ON beats YOKED by **+0.45 bpc** and beats
reading-everything-equally by **+0.24 bpc**, at **12/12** dial settings, on both held-out bpc and a
turn-overlap contingency metric. The graded-contingency sweep is the clincher — the gap shrinks to ~0
exactly as timing stops predicting content (r→0.5), the signature of a real dial rather than an artifact.
**WIN; kill did not fire.** Substitutes text8 + frequency-matched babble for CHILDES (not in `data/`).
Online, bounded, no backprop. `lib/contingency.py` + `run.py`. See `RESULTS.md`.
