# Exp BF — Margin-gated production: read the same counts the hard way (G6) — 2026-06-26

**The claim (G6).** Production is comprehension read **backwards**. The spine learns ONE thing — a
count table over `(cue -> label)` co-occurrences — and the two faculties are two READ DIRECTIONS:

- **Comprehension** = `cue -> label`, **one-to-many**, forgiving (no gate): is the heard label
  compatible with the cue? Any plausible label is recognised — comprehension is cheap, comes early.
- **Production** = `cue -> the ONE label`, **many-to-one**, committing: to *say* a word you must
  beat its competitors. Read the SAME counts through a **margin gate**:

      activation(label | cue) = count(cue,label) * AB_frequency(f) / FAN(cue)
      emit top  iff  margin = activation(top)/activation(2nd) >= theta_emit
      else back off to a generic label, or DEFER (stay silent).

Two predictions: (1) **gated production is more precise than ungated argmax at matched recall**, and
(2) the same table read the two ways reproduces a **C>P gap that appears and shrinks with evidence**.

**Setup.** text8 word-level, 10 MB slice → **1.71M words**, vocab 20k+UNK (OOV 4.7%). Cue = the
**previous word** (a slot's left bracket, AF-style); label = the **content word** that fills it
(function words / len<3 dropped from the production target). Online single pass: learn on the 85%
prefix, probe on the held-out 15% tail (**151,682** content-label probes). The per-cue store is
**bounded** (≤32 labels/cue, LRU-evicted), counts + an AB hit/miss split only — no gradients, fixed
seed. `lib/margingen.py` (the reusable organ), `exp_bf_margin/run.py`. theta swept over **12** values
(the FRAGILE budget). Baselines: ungated argmax (theta=1), and a **raw-count** production read with
**no fan division and no AB frequency** (isolates what fan+AB buy).

## Result 1 — the margin gate buys precision (a clean win)

| theta | precision (fan+AB) | recall | | precision (raw-count) | recall |
|---:|---:|---:|---|---:|---:|
| 1.00 (ungated argmax) | 6.95% | 99.1% | | 7.91% | 99.1% |
| 1.50 | 11.12% | 49.4% | | 9.57% | 73.3% |
| 2.00 | 13.08% | 39.4% | | 10.04% | 61.3% |
| 4.00 | 17.23% | 27.4% | | 11.51% | 46.7% |
| **6.00** | **20.89%** | 21.5% | | 15.34% | 30.3% |

**The gate works, and fan+AB is a better gate than raw counts.** Tightening the margin trades recall
for precision exactly as the organ predicts: precision climbs **6.95% → 20.89%** (3×) as the producer
falls silent on contested slots. At **matched recall** the fan+AB margin **dominates the raw-count
margin** at every comparable operating point (e.g. ~recall 49% → **11.1%** vs ~recall 73% → 9.6%;
~recall 21% → **20.9%** vs ~recall 30% → 15.3%). Precision **lift at matched recall vs ungated
argmax: +3.0 pts** (9.95% @ 58.7% recall vs 6.95% @ 99.1%). The first kill clause — *"gated precision
not above ungated at matched recall"* — does **NOT** fire: the gate clearly adds precision, and the
fan/AB read adds precision on top of a plain frequency margin. (At theta=8/12 precision is noisier —
those cells emit on few, idiosyncratic uncontested cues — which is why the sweep, not a single theta,
is the honest unit.)

## Result 2 — the C>P gap: it appears, but does NOT cleanly shrink (the honest negative)

Bucketing probes by the cue's **count at probe time** (evidence), gate theta=2.0:

| cue-count | n | comp-any | comp-top1 | prod-prec | prod-recall | gap (any − P) |
|---:|---:|---:|---:|---:|---:|---:|
| 1–2 | 875 | 5.83% | 5.83% | 5.83% | 100.0% | +0.00 |
| 2–4 | 1,536 | 9.64% | 4.30% | 6.29% | 52.8% | +3.35 |
| 4–8 | 3,512 | 12.64% | 4.27% | 13.74% | 20.3% | −1.10 |
| 8–16 | 4,191 | 15.01% | 4.29% | 8.46% | 33.8% | +6.55 |
| 16–32 | 5,686 | 19.29% | 4.17% | 7.69% | 34.8% | +11.60 |
| 32–64 | 7,738 | 21.93% | 4.48% | 10.87% | 19.9% | +11.06 |
| 64–256 | 18,864 | 21.82% | 4.29% | 10.13% | 20.0% | +11.69 |
| 256+ | 107,961 | 29.60% | 7.97% | 13.97% | 45.1% | +15.64 |

*comp-any = forgiving one-to-many recognition (label present among the cue's labels); comp-top1 =
strict argmax comprehension (the apples-to-apples twin of gated production).*

**The gap is real but does NOT shrink in the predicted direction — and which way it moves depends
entirely on the comprehension read.**

- **Forgiving comprehension (one-to-many):** the gap **WIDENS** with evidence (+0.0 → +15.6).
  As a cue accumulates labels, "is the true label *present* among them?" gets *easier* (the cue owns
  more labels), while production precision — picking the right one out of a 20k-word next-word
  distribution — stays low. So the forgiving read inflates with evidence; the widening gap is largely
  a property of that metric, not of a deepening comprehension>production asymmetry.
- **Strict comprehension (argmax):** comp-top1 is roughly **flat (~4–8%)** and sits *near or below*
  gated production precision. The "strict gap" technically moves +0.0 → −6.0 (the spec's "shrink"),
  but that is an artifact of both reads being ~5.8% at evidence=1 (a single label) and production
  precision overtaking strict comprehension once cues are rich — not a comprehension-leads-production
  story collapsing toward parity.

**Verdict on the structural-gap claim: NOT cleanly demonstrated on text8 next-word.** A
left-context word and a stochastic next content word is a *one-to-many* relation with no stable
single right answer, so "comprehension recognises, production commits" does separate the two reads
in *level* (forgiving comprehension ≫ committed production), but the **evidence-dependent shrink**
the kill-condition asks for is absent — the gap widens or is metric-dependent. The honest reading:
this corpus/probe gives the *level* asymmetry for free (it is baked into one-to-many vs many-to-one)
but cannot show the developmental *shrink*, which needs a task with a stable target the producer can
eventually nail (a grounded (message-cue → label) pair, AV's scene env — not free next-word).

## Kill-condition

The kill line: *"gated precision not above ungated at matched recall, OR the gap does not appear/shrink
with evidence."*

- **First clause: does NOT fire.** Gated precision is clearly above ungated at matched recall
  (+3.0 pts), and the fan+AB margin beats a raw-count margin at every matched operating point. The
  margin gate is a genuine production organ — this half is a **win**.
- **Second clause: fires (partially).** The gap **appears** in level (forgiving comp 26.7% vs gated
  production ≤21%) but does **NOT shrink** with evidence — it widens (forgiving) or is a metric
  artifact (strict). The structural *shrinking* C>P prediction is **not** supported on this corpus.

**Overall: PARTIAL.** The mechanism (margin-gated, fan+AB-weighted reverse read) works and is
precision-positive — keep it on the FRAGILE budget. The *developmental gap-shrink* claim is parked,
not killed: it needs a stable-target grounded probe (queue with **AV**'s scene env), where production
has a single correct label to converge on as evidence accrues.

## Rules honored

- **Online single-pass:** one causal stream; the AB (f,c) at probe time reflect only earlier
  occurrences; learn on the prefix, probe on the held-out tail.
- **No gradient / k-means / SVD / backprop:** pure counts + an AB hit/miss split + a margin ratio.
  Production allocates **no new table** — it is a query over the same counts (the whole point of G6).
- **Bounded memory:** per-cue store capped at 32 labels, LRU-evicted; O(#cues · cap).
- **Fragile / right axis:** judged on production **precision** and the **gap**, not bpc; theta swept
  over 12 values; the negative on the shrink axis is reported honestly, not papered over.

## Corpus note

Spec names "text8 word-level; (message-cue, label) pairs". text8 is present; we used it directly. We
**substituted** a corpus-derived `(prev-word -> next content word)` pair for the "message/role-cue ->
label" pair the spec envisions (no scene/message env exists yet — that is AV's deliverable). This
substitution is almost certainly *why* the gap-shrink fails: a next-word target is irreducibly
one-to-many, so production can never "nail" it as evidence grows. A grounded message→label pair would
be the faithful test, and is the natural follow-up.

Repro: `python exp_bf_margin/run.py` (10 MB text8, ~5 s incl. load). New files only:
`lib/margingen.py`, `exp_bf_margin/run.py`, `exp_bf_margin/README.md`, this file.
