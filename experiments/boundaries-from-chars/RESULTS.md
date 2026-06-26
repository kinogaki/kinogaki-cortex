# Experiment A — results (boundary kill-test) — 2026-06-25

**Setup.** Online variable-order char predictor (orders 0–6, belief over orders, add-α smoothing, fully online,
no GPU) on space-stripped *Pride & Prejudice* (563,924 letters, 126,864 true word boundaries). Score every
position by four signals; rank; take top-K (=#true boundaries); F1 at ±1-char tolerance. Pipeline dogfoods
`kinogaki 0.2.0` to write discovered segments as a `.prism` document.

**Result.**

| signal | precision | recall | F1 |
|---|---:|---:|---:|
| random | 0.625 | 0.542 | 0.580 |
| surprisal | 0.772 | 0.642 | 0.701 |
| entropy (forward) | 0.719 | 0.603 | 0.656 |
| bwd_entropy | 0.601 | 0.460 | 0.521 |
| branch_sum (Hf+Hb) | 0.662 | 0.524 | 0.585 |
| **branch_rise (fwd+bwd entropy RISE)** | **0.820** | **0.735** | **0.775** |
| bayes_surprise (KL belief shift) | 0.603 | 0.477 | 0.533 |
| transient bayes_surprise | 0.605 | 0.483 | 0.537 |

**Verdict: QUALIFIED PASS.** Character-level boundaries are genuinely recoverable from prediction error
(branch_rise F1 0.775, lit. range 0.75–0.85) — the low-level half of the "ambiguity = boundary" bet holds.
The winning signal is **forward+backward branching-entropy RISE**, *not* raw surprisal and *not* Bayesian
surprise. The full read→discover→persist loop works end-to-end and dogfoods kinogaki 0.2.0 to write the
discovered words as a `.prism` document.

(The high random baseline is the ±1 tolerance × 0.228 boundary density; the *discrimination* above random is what
matters: surprisal +0.12, bayes −0.04.)

## Findings

1. **The "ambiguity = boundary" bet HOLDS at the char level — but via surprisal/entropy, not Bayesian surprise.**
   Prediction-difficulty recovers word boundaries (surprisal F1 0.70 ≫ 0.58 random). This is the classic
   **branching-entropy** effect (Harris 1955; Elman): the next char is hard to predict exactly when a new word
   starts. Decades of unsupervised word-segmentation rest on it.
2. **Transient Bayesian surprise does NOT transfer down to word boundaries — it scored *below random*.** This
   empirically confirms the open question flagged in the verdict doc. Kumar & Zacks validated it for *semantic
   event* boundaries; at the character level the belief-over-order shift peaks *mid-word* (as a distinctive
   context locks in), not at boundaries. **It is a high-level signal, not a low-level one.**
3. The belief-over-context-order operationalization of Bayesian surprise is the wrong latent for boundaries;
   a higher-level Bayesian surprise would need a belief over *concepts/topics*, not context length.

## Design correction (the cheap course-correct)

**The boundary signal is LEVEL-DEPENDENT** — this updates `ARCHITECTURE_SKETCH.md` §BoundaryDetector:
- **Low levels (chars→words):** branching entropy / surprisal (prediction difficulty). *Validated here.*
- **High levels (sentences→themes/events):** transient Bayesian surprise over a concept/topic belief. *To test
  at the right level once we have higher-level States — Experiment A.2.*

The kill-test did its job: the core bet survives (boundaries are recoverable from prediction error), and we
corrected the boundary mechanism before building the voting field.

## Next

- **A.1 — strengthen the low-level segmenter** (it works; push F1 higher with the literature's actual recipe):
  forward+backward **branching entropy / successor-variety**, score the *rise* (Δ-entropy) not the raw level,
  add the **Bayesian-surprise-at-the-word-level** only once words exist. Target F1 ≳ 0.78 (lit. range).
- **A.2 — test Bayesian surprise at the RIGHT level:** segment into words, predict next *word*, compute
  Bayesian surprise over a topic/concept belief → recover *sentence/discourse* boundaries. This is where the
  paper says it should win.
- Then **B** (the real thesis: online learning across domains without catastrophic forgetting).
