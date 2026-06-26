# Exp AY — two-threshold comprehension/production gate (M17) — 2026-06-26

**The bet (a robust developmental finding: comprehension precedes production by weeks-to-months).**
Split the SAME leaky binding-count, read at two operating points. **Comprehension** is a cheap
recognition read — a word is "understood" once ANY incoming binding (a context→word bigram count)
clears a **LOW** threshold. **Production** is an expensive generation read used by `act()` — a word may
be emitted in a context only if it is that context's **argmax** AND the context's NARS truth `f·c`
(frequency × confidence, AB's hit/miss split) clears a **HIGH** bar. The lag between the two onsets
falls out of the threshold gap, and should **widen** for words with many competitors (high **AO** fan:
many forms fighting to be the argmax of one shared context).

**Setup.** Tokens are **words** (a word = the production unit). A text8 slice of 6,000,000 chars →
1,019,514 words / 53,286 distinct, streamed once. Watch-set = a CDI-like band of 300 mid-frequency
content words (frequency ranks 20–900, skipping the ultra-frequent function words and the hapax tail).
Onsets recorded in 40 stream-time bins; lag = production-bin − comprehension-bin. Baseline =
single-threshold acquisition (produce the moment comprehended → predicts exactly zero lag). LOW=2,
HIGH=0.6 for the main pass; seeds 0,1 (seed shifts the stream offset). Single online pass, no gradients
/ k-means / SVD, bounded memory (word-bigram bindings + a tiny per-watch onset record). ~10 min on CPU
(the Python word-stream build + 15-point operating sweep dominate; one gate pass is seconds).

**Corpus substitution.** The spec names CHILDES-CDS + the 680-word CDI list. CHILDES is not in `data/`.
**Substituted text8** for the input stream (generic, NOT child-directed) and a frequency-band watch-set
as a CDI proxy. This weakens the developmental realism (text8 is adult encyclopedic register) but tests
the mechanism's *signature* — does a C>P lag exist and widen with fan — which is corpus-agnostic.

---

## Q1 — does a C-before-P lag exist? (median lag > 0 in stream-bins)

| condition | seed | #words reaching both onsets | median lag | mean lag | frac lag>0 |
|---|---:|---:|---:|---:|---:|
| **two-threshold** | 0 | 29 | 1.0 | **2.07** | 51.7% |
| single-threshold (baseline) | 0 | 300 | 0.0 | 0.00 | 0.0% |
| **two-threshold** | 1 | 32 | 0.0 | **2.69** | 43.8% |
| single-threshold (baseline) | 1 | 299 | 0.0 | 0.00 | 0.0% |

The two-threshold gate produces a positive lag (mean ~2–2.7 bins; ~44–52% of words production-onset
*after* comprehension-onset). The single-threshold baseline gives **exactly zero** lag for every word,
as predicted — the lag is a genuine product of the dual operating point, not an artifact.

## Q2 — does the lag WIDEN with competitor density? (the kill axis)

| | seed 0 | seed 1 |
|---|---:|---:|
| **Spearman(lag, fan)** | **+0.303** | **+0.585** |

Positive on **both** seeds. Bucketed (seed 0): mean lag rises **1.67** (fan 1–2) → **2.73** (fan 2–9).
High-fan words — those with more forms competing to be the argmax of a shared context — wait longer
between understanding and saying. This is exactly the M17 signature.

## Q3 — robustness across operating points (FRAGILE: many variations before judging)

Sweep over LOW ∈ {1,2,3} × HIGH ∈ {0.4…0.8} (15 points):

- **Spearman(lag, fan) > 0 in ALL 15 operating points** (range +0.086…+0.829).
- median-lag>0 AND rho>0 in **7/15** points (the strict win); the others have rho>0 but median lag
  pinned at 0 because lenient gates (low HIGH) let easy and hard words produce in the same bin — the
  effect lives in the **mean/tail**, not the median, at those settings.
- The lag magnitude grows as the production bar HIGH rises (med lag 0 → 2 as HIGH 0.4 → 0.8): a stricter
  "won't say it yet" gate produces a longer lag, monotonically. The mechanism behaves as designed.

---

## Verdict — WIN (modest)

The kill-condition (**"the C-before-P lag is absent or does NOT widen with competitor density across two
seeds"**) **did NOT fire**:

- A C>P lag **exists** (two-threshold mean 2.07 / 2.69 bins; single-threshold baseline a flat zero).
- It **widens with competitor density** — Spearman(lag, fan) = **+0.30 / +0.59 across two seeds**, and
  stays positive across all 15 operating points.

So the dual-operating-point gate reproduces "understands but won't say it yet," and reproduces that it
is **worse for high-fan (confusable) words** — the developmental signature M17 predicted.

**Honest caveats (why "modest," not a slam-dunk):**
1. **Small n.** Only 29–32 of 300 watch words reach BOTH onsets in this 6 MB slice — most never clear
   the HIGH production bar in the window. The rho is over a few-dozen words; a longer stream would
   tighten it. (A bigger slice is the obvious follow-up — kept small this pass per the brief.)
2. **Median is often 0; the effect is in the mean/tail.** Many words comprehend and produce in the same
   bin; the lag is carried by the high-fan minority. That is consistent with the theory (only confusable
   words show a big lag) but means "median lag > 0" is a fragile headline — report the mean and the rho.
3. **Corpus substitution.** text8, not CHILDES-CDS; a frequency band, not the real 680-word CDI list. The
   *direction* of the result is corpus-agnostic, but the bin-magnitudes are not developmental months.

**Rules honored:** ONLINE single streaming pass (one walk of the word-id sequence, onsets recorded as
they open); NO gradient descent / k-means / SVD / backprop (everything read off integer counts + the
NARS f·c on a running top-1); BOUNDED memory (word-bigram bindings the predictor already needs, plus a
tiny per-watch-word onset record; the production gate only ever *reduces* the emitted vocabulary).

**Follow-up:** (1) scale to a 50–100 MB slice for n in the hundreds and a tighter rho; (2) measure the
lag at the FORM/grammatical level (the spec says it should appear at every level) — e.g. inflectional
slots, not just whole words; (3) swap in a real child-directed register (bible/shakespeare are in
`data/`) to see if the lag-bin magnitudes move toward developmental shape.
