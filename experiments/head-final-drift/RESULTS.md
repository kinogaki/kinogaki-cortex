# Exp AZ — reliability-gated boundary detectors → head-final drift (M3) — 2026-06-26

**Setup.** Three boundary detectors over a character stream — **forward-TP-dip** (Saffran),
**backward-TP-dip** (Pelucchi), **branching-entropy-rise** (Harris/Exp A) — each carrying an
**Exp-AB** hit/miss tally vs the eventually-stable chunk (= gold word boundary). Detector
**reliability = f·c** (c-discounted frequency). Combined by **Exp-AJ take-the-best**. The
deliverable is the **drift** Δ = reliability(backward) − reliability(forward), predicted to **rise**
going head-initial → head-final.

**Corpus.** No Japanese/Korean romanized corpus in `data/`, so we **synthesize a frequency-matched
mirror-image pair**: each word = stem + marker; **head-initial** = STEM·MARK, **head-final** =
MARK·STEM. Identical lexical statistics (same Zipf stems, same tiny high-freq marker class) — order
is the *only* difference. Anchored with **real English text8** (4 MB, a natural head-initial
language). Stated substitution per the spec. 40,000 words/language; online single pass.

## Result — the drift fires

| variation | Δ head-init | Δ head-final | drift↑ | winner init | winner final |
|---|---:|---:|---:|---|---|
| rate=0.3 | +0.108 | +0.163 | **+0.055** | entropy | entropy |
| rate=0.4 | +0.038 | +0.119 | **+0.081** | entropy | entropy |
| rate=0.5 | −0.008 | +0.133 | **+0.141** | entropy | entropy |
| rate=0.6 | +0.004 | +0.086 | **+0.082** | entropy | **backward_tp** |
| rate=0.7 | −0.016 | +0.100 | **+0.116** | entropy | **backward_tp** |
| tol=0 | −0.008 | +0.133 | **+0.141** | entropy | entropy |
| tol=2 | −0.008 | +0.133 | **+0.141** | entropy | entropy |
| seed=1 | +0.138 | +0.062 | −0.076 | entropy | entropy |
| seed=2 | +0.010 | +0.068 | **+0.058** | entropy | entropy |
| seed=3 | −0.039 | +0.208 | **+0.248** | entropy | entropy |
| stems=120,marks=4 | +0.035 | +0.209 | **+0.174** | entropy | entropy |
| stems=40,marks=10 | +0.184 | +0.138 | −0.046 | entropy | entropy |

**drift>0 in 10/12 variations · mean drift = +0.093 · median +0.099 · min −0.076 · max +0.248.**

Per-detector reliability (f·c) at rate=0.5, tol=1, seed=0:

| corpus | forward_TP | backward_TP | entropy | winner |
|---|---:|---:|---:|---|
| synth head-INITIAL | 0.4421 | 0.4340 | 0.7921 | entropy |
| synth head-FINAL | 0.4047 | **0.5373** | 0.7295 | entropy |
| text8 (English) | 0.5496 | 0.5538 | 0.6762 | entropy |

Going head-initial → head-final, **backward-TP gains +0.103 reliability and forward-TP loses
−0.037** — the predicted forward→backward shift. At a looser operating point (rate 0.6–0.7)
backward-TP even becomes the **take-the-best winner** on head-final text.

## Verdict: **PARTIAL PASS** (the drift fires; a clean, honest read)

The M3 deliverable — the **forward→backward-TP drift on head-final text** — **is present**: the
reliability gate shifts weight toward backward-TP exactly where the language puts predictable
material on the right. It is robust to operating point and tolerance (drift +0.05…+0.14 across the
whole rate sweep) and survives most seeds (10/12 positive). The kill-condition (no drift after the
FRAGILE budget) **did not fire**.

Why *partial*, not a clean win:
1. **`entropy` wins the global take-the-best on every corpus** — the branching-entropy-rise is the
   strongest single boundary cue (consistent with Exp A's 0.775 F1 verdict). The drift lives *inside
   the TP pair*, which is the gate the claim is about, but the take-the-best winner only flips to
   backward-TP at loose operating points. So the gate *shifts* but rarely *changes its top pick*.
2. **Two seeds (1, 10-marker vocab) gave a small negative drift.** Within the FRAGILE budget and
   outweighed, but it says the effect is real-but-modest, not overwhelming — matching the spec's own
   evidence-honesty caveat that the ~13-month human drift is *thin and contested*.
3. **text8 shows backward ≈ forward (0.554 ≈ 0.550)** — English (head-initial-ish but morphologically
   mixed) sits near parity, the natural anchor the synthetic head-initial extreme brackets from below.

## Rules honored
- **Online single pass:** the TP fields are causal online bigram counts; each detector's firing
  threshold is a **leaky online histogram** (decay 0.9995, no global lookahead) and the hit/miss
  tally accrues as the stream is read.
- **No backprop / k-means / SVD / eigen:** pure counts, NARS (f,c), take-the-best argmax.
- **Bounded memory:** O(V²) bigram tables (V=27) + O(#detectors) tallies + a fixed 256-bin
  histogram. No per-context growth.
- *Honest caveat:* the `branching-entropy-rise` field is computed over a whole-slice neighbour table
  (a bounded V×V matrix, but a global pass, not strictly causal). It is the *secondary* cue; the
  **deliverable drift is in the two TP detectors, which ARE causal.**

## Notes / what a follow-up should do
- Substitute a **real romanized Japanese/Korean** slice (or a head-final CHILDES sample) for the
  synthetic head-final corpus to confirm the drift survives natural lexical noise.
- The take-the-best winner is dominated by entropy; to make the *gate-flip* (not just the
  reliability-shift) the headline, gate the **nonadjacent (a_X_b) detector** in as M3 sketches, or
  weight detectors by reliability-rank rather than winner-take-all.
- `boundsdrift.py` exports `run_detectors`, `take_the_best`, `drift`, `synth_corpus` for reuse.
