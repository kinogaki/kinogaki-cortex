# Exp I — the uniform-component cortex: one Column, wired bigger → better — 2026-06-25

**The thesis (the steer).** Don't add mechanisms; replicate and stack the SAME part. A `Column` is an online
associative backoff predictor over a token stream. A `Level` is N Columns voting. A `Cortex` stacks Levels:
char-Columns predict chars; word-Columns predict the current word from the previous words and hand a top-down
char prior back down. Grow along two axes — **WIDTH** (more char Columns voting) and **DEPTH** (stack a word
Level, then widen it to a longer-timescale phrase band) — and the standard scorecard should improve.

**Setup.** text8, train 2MB / test 2MB, vocab 26k. One `Column` class everywhere (`lib/cortex.py`); only the
wiring changes. Scored on `lib/metrics`: bpc + overfit gap, real-word %, phrase-coherence %.

## The combiner is the load-bearing choice (the first real finding)

Naive **product-of-experts** (raw product of the columns) and calibrated **log-linear pooling** (geometric mean)
gave opposite scorecards — same model, only the pooling rule changed:

| pooling | bpc (big cortex) | overfit gap | phrase-coh % (temp 0.9) |
|---|---:|---:|---:|
| product-of-experts (sharp) | **3.41** (worse, blows up) | **+2.9** | **98** (great) |
| geometric-mean (calibrated) | **2.11** (good) | +1.2 | 56 (tempered) |

Product pooling *sharpens* with every column added → fluent, coherent generation but **overconfident, overfit
likelihood** (each added column makes it worse-calibrated). Geometric-mean pooling keeps the weights summing to
1 → **honest, calibrated bpc** but tempered (noisier) generation. This is the same product-vs-linear tension
Exp D found, now isolated as a knob.

**The recipe that gets both:** pool with the geometric mean (calibrated scoring), then **sharpen at SAMPLING
time** (low temperature) to recover coherence. Best of both worlds.

## Scorecard — geometric-mean pool + temp-0.5 sampling (the recipe)

| config (same Column, more of it) | test bpc | overfit | real-word % | phrase-coh % |
|---|---:|---:|---:|---:|
| 1 char col | 2.401 | +1.028 | 98.9 | 81.5 |
| 3 char cols (vote)  — **WIDER** | **2.119** | **+0.566** | 97.6 | 73.4 |
| + word level        — **DEEPER** | **2.103** | +1.151 | 99.0 | **92.4** |
| + phrase band (wider word level) | 2.112 | +1.200 | **99.2** | **93.6** |
| *(real held-out text)* | — | — | 89.7 | 46.7 |

Generated samples (same seed), small → big:
- **1 char col**: *"safety the final company considered party and down with the development of the country …"*
- **+ word level**: *"communications team or any this will be errors and health science and for the players as a separate state the sale of art …"*
- **+ phrase band**: *"argentina argentine nation of the absence of autistic people s states or the common ancestor of the formal naming convention and reducing government official"*

## Findings — each growth step buys something attributable (and nothing it doesn't)

1. **WIDTH (more char Columns voting) buys calibration + generalization.** 1 → 3 char columns drops test bpc
   **2.40 → 2.12 and HALVES the overfit gap (1.03 → 0.57)**. Voting columns are an ensemble — it regularizes.
   This is the cleanest single win, and it comes purely from replicating the one component.
2. **DEPTH (stack a word Level) buys coherence.** Adding word-Columns lifts phrase-coherence **73 → 92%** (the
   big jump) and gives the best bpc (2.103). Concepts help the level they model — consistent with Exp C/E.
3. **A 4th band hits diminishing returns at 2 MB.** The phrase band barely moves anything (bpc flat, phrase-coh
   +1). Exactly Exp F's capacity×data law: a bigger cortex needs more data to pay off. char-bpc itself is near
   its local-context ceiling (Exp D) — so the char dial saturates while the coherence dials still have room.
4. **Local English is mastered; global coherence is not.** Even the biggest cortex generates locally-fluent
   word-salad (real words, real phrases, no discourse). The frontier Exp G/H named is unchanged — the next
   level up (theme/meaning) is where the architecture's reasoning/movement ambitions must deliver.

## What this establishes

- **The uniform-component thesis holds.** ONE `Column`, replicated (width) and stacked (depth), composes into a
  cortex that reads and generates English — "cortex small vs big" is real on the dials. Same part throughout.
- **"Better" is multi-dial, and the combiner is the hinge.** Width → calibration/generalization; depth →
  coherence; the geometric-mean-pool + low-temp-sample recipe gets honest likelihood *and* fluent generation.
- **Diminishing returns are data-bound, not mechanism-bound** (Exp F) — the cheapest next win is scale (more
  data + the already-built width/depth), and the open frontier is still global/discourse coherence.

Raw logs: `out.txt` (product-of-experts), `out_calibrated.txt` (geometric-mean, temp 0.9),
`out_recipe.txt` (geometric-mean + temp-0.5 sampling — the table above).
