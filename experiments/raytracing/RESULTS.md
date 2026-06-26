# Exp P — raytracing / proximity columns — 2026-06-25 (autonomous)

Place the top-2500 word-columns in a space (PPMI co-occurrence + eigen-embed — a count factorization, no
backprop), connect by proximity, feed text as a trajectory, predict three ways. text8, 20 MB, 16-D.

## The space is real — proximity = meaning

| word | nearest columns (learned, unsupervised) |
|---|---|
| three | four, five, six, two, one, eight, seven, nine |
| king | son, iv, maria, prince, iii, daughter, afonso, vi |
| war | invasion, soviet, allied, battle, army, iraq, troops |
| water | gas, liquid, heat, energy, copper, oxygen, oil, solid |
| france | spain, italy, sweden, hungary, denmark, portugal, netherlands |
| music | musical, pop, artists, musicians, blues, hip, hop, dance |

"Real columns connected by proximity" **works as a representation** — semantically related columns cluster
cleanly, from counts alone.

## But proximity does NOT predict the next word

| method | next-word top-1 acc (held-out) |
|---|---:|
| bigram baseline | **20.85%** |
| A. spatial gather (euclidean-near columns' followers vote) | 17.07% |
| B. ray extrapolation (current + velocity → nearest column) | **1.79%** |
| C. graph spreading (PMI-graph neighbors' followers — the *endorsed* form) | 20.73% |
| random (1/2500) | 0.04% |

- **Ray extrapolation fails** (1.79%, though 45× random): the "a sentence is a locally-linear trajectory through
  embedding space" hypothesis is **false** — semantic space is not sequentially predictive. The purest form of
  the raytracing idea does not work for next-word prediction.
- **Spatial gather hurts** (17% < 21%): averaging in semantic *neighbors'* followers blurs the exact next word.
- **Graph spreading ties the bigram** (20.73 ≈ 20.85): spreading adds nothing to plain bigram here.

## Honest verdict (matches what the source-mining independently warned)

The HTM forums + senior voices say a Euclidean reference frame for *language* is a dead end (no natural
origin/unit for symbolic space), and **prediction is a sequential/syntactic task that semantic proximity does
not serve.** This experiment confirms it cleanly: the embedding is an excellent **similarity / generalization**
structure but a poor **predictor**.

**Where proximity should pay off (untested here, the real next step):** as a **backoff/smoothing prior for RARE
or unseen contexts** — where the bigram has no counts, a word's semantic neighbors' followers are a principled
fallback. This experiment measured *overall* accuracy, dominated by frequent words where the bigram already wins;
it does not isolate the rare-context regime where gather/spreading should help. That's the fair test to run next.

**Implication for the architecture:** keep the proximity space as a *modulator* (a similarity/backoff prior and
an inspection tool), **not** as the content predictor — exactly the driver-vs-modulator separation the brain
sources recommend. The sequential predictor stays the offset-keyed count model (`IDEAS_FROM_SOURCES_V2.md` #2).
