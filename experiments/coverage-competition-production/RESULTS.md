# Exp BD — Coverage-competition production (G1) — RESULTS

**Verdict: PARTIAL.** The construction producer is **measurably more well-formed and less
over-generating** than the flat sampler on the constructional battery (BD's primary winning axis) —
**+18.5 pts well-formed, −28.5 % over-generation**, on a NON-circular held-out oracle. The merged Levelt
frame-survival sub-claim falls short (**61 %** vs the 80–95 % target). So BD's own kill-condition
(BUILD_QUEUE: "not measurably more well-formed / less over-generating than flat") **did NOT fire**; the
separately-judgeable Levelt sub-claim is the weak axis.

Corpus: **text8**, 8 MB slice → 1.36 M words; first 80 % is the grammar/train split, last 20 % is the
held-out oracle the producer never built its grammar on. Same AF/AW pipeline (top-10 k words, online
signatures + leader categories, C=400). Fixed seed 0, single streaming pass.

## What ran

- **Producer** (`lib/production.py::CoverageCompetitionProducer`): retrieve frame → coverage×frequency
  (AW-gated) → AJ take-the-best category → fill (frozen verbatim / open-slot productive sample).
- **Articulation** through **AU** `ChunkLexicon.cover_buffer` — every emitted word is spelled as whole
  committed chunks (the integration proof below).
- **Over-generation gate** via **AW** `AssocSlots` (ΔP / PPMI) zeroing base-rate-only slot categories.
- **Oracle** (`HeldoutWellFormedness`): a (frame, word) pair is well-formed if the held-out text contains
  the exact frame→word bigram OR the frame prefers the word's category ≥1.5× its held-out base rate.
  **Non-circular** — it reads held-out empirical frequency only, never the producer's association gate.
  (An earlier train-grammar oracle gave a 100 %-by-construction artefact across all variants; replaced.)
- **Baseline**: `FlatWordSampler` — the gibberish floor at the word grain (samples the next word from the
  global unigram, ignoring the frame → emits any word after any frame). This is G1's "current flat
  geometric-mean sampler" one altitude up, so the constructional battery is apples-to-apples.

## AU integration — emitted words articulated as whole chunks

```
"the ___" -> emit 'san'        articulated as ['sa', 'n']
"and ___" -> emit 'functions'  articulated as ['fu', 'nc', 'ti', 'on', 's']
"to ___"  -> emit 'carry'      articulated as ['ca', 'rr', 'y']
"a ___"   -> emit 'defensive'  articulated as ['de', 'fe', 'ns', 'iv', 'e']
```

AU's chunk lexicon (2 MB slice, 594 minted chunks) supplies the emission vocabulary; the producer covers
each emitted word with the largest committed units. AW's `AssocSlots` supplies the over-generation gate.
Both imported clean — **no fallback needed**.

## FRAGILE sweep — 10 variants (assoc-kind × confidence-bar), held-out oracle

| kind | conf | well-formed % | over-gen % | emit-rate | frame-survival % |
|------|-----:|--------------:|-----------:|----------:|-----------------:|
| ppmi | 0.0  | 47.7 | 52.3 | 100.0 | 52.6 |
| ppmi | 1.0  | 47.7 | 52.3 | 100.0 | 52.6 |
| ppmi | 3.0  | 47.7 | 52.3 | 100.0 | 52.6 |
| dp   | 0.0  | 51.2 | 48.8 | 100.0 | 58.2 |
| dp   | 1.0  | 51.3 | 48.7 |  98.9 | 58.7 |
| **dp** | **3.0** | **53.5** | **46.5** | 93.8 | **61.4** |
| none | 0.0  | 48.2 | 51.8 | 100.0 | 56.7 |
| none | 1.0  | 48.2 | 51.8 | 100.0 | 56.7 |
| none | 3.0  | 48.2 | 51.8 | 100.0 | 56.7 |
| ppmi | 6.0  | 47.7 | 52.3 | 100.0 | 52.6 |
| **FLAT** | — | **34.9** | **65.1** | 100.0 | — |

Best producer (ΔP, conf-bar 3.0): **53.5 % well-formed vs 34.9 % flat = +18.5 pts**; over-generation
**46.5 % vs 65.1 % = −28.5 %**.

## Read

- **Primary axis = WIN.** Every producer variant beats the flat floor on well-formedness by ~13–19 pts and
  cuts over-generation by ~20–29 %. Routing emission through a competing construction (instead of a flat
  vote) is the mechanism that does it.
- **Association (AW) is a near-no-op dial here.** PPMI-gated vs ungated coverage is essentially flat
  (−1.0 % at bar 0, PPMI slightly *worse*); ΔP edges ahead but mostly by being more conservative. The
  win comes from the **construction/coverage structure**, not from the ΔP/PPMI contingency correction —
  consistent with AW's own "raw counts suffice for English text" finding. The real selectivity dial is the
  **confidence bar** (ΔP/3.0 emits 6 % less and is cleanest).
- **Merged Levelt frame-survival = weak (61 %, target 80 %).** The argmax word of the chosen slot category
  often is itself a high-frequency function word whose held-out category profile is flat, so the held-out
  category-lift test refuses it. The construction picks a defensible category but the single most-probable
  word of that category is not always a held-out-confirmed member of *this* frame's slot. This is the
  separately-judgeable sub-claim, and it underperforms its target.

## Rules compliance

- **Online single pass**: the producer has NO learning step — pure scored lookup over tables AF/AW/AU
  already counted in one streaming pass; the chunk lexicon and grammar are each built in a single pass.
- **No gradient / k-means / SVD / eigen / backprop**: categories are online leader-clustering (jepa);
  association is closed-form ΔP/PPMI; competition is AJ argmax; nothing is optimized.
- **Bounded**: reads the bounded constructicon + bounded leader centroids (C=400) + bounded chunk lexicon
  (LFU-capped at 20 k); no new growing store; the producer allocates nothing per emission beyond the slot.

## Kill-condition

BD's BUILD_QUEUE kill ("not measurably more well-formed / less over-generating than the flat sampler on
the constructional battery") **did NOT fire** — the producer wins that axis by +18.5 pts / −28.5 %. Per
the spec, a fired kill would PARK as "needs the situation model (AM frontier)" without killing the
constructicon; that does not apply. The merged Levelt frame-survival sub-claim (a *separate* kill-test)
is unmet (61 % < 80 %), which is why the overall verdict is **PARTIAL** rather than a clean WIN.

## What a follow-up should do

- Frame-survival: score against held-out members of the chosen category *weighted by the frame's own slot
  distribution* (not the single global-argmax word), or sample a few category words and check the held-out
  rate — the 80 % bar is likely reachable without changing the mechanism.
- The situation model (AM frontier) is the named next lever: coverage here is purely lexical/syntactic;
  a message/referent cue (AO content key, the wired-but-defaulted seam) is what would push well-formedness
  past the ~53 % ceiling the construction structure alone reaches.
- Larger slice + 2-gram frames (the `build_frame_counts(order=2)` path) to test whether tighter frames
  raise both well-formedness and survival.
