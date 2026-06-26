# Exp V — Trajectory / change memory & affordances

Operationalizes TBP's **"Deeper Dive into Trajectory Memory for Behavior Models"** (2026/05) on the text
stream. Three claims from the talk, each turned into a measured test:

- **(a) movements are shared, locations are unique** — a behavior is a *sequence of changes*, learned
  *independent of the object* and shared across objects. → **Transfer test.**
- **(b) trajectories are directional** — "I don't think I could write my signature and then halfway
  through start going backwards." Reverse needs its own memory. → **Directionality test.**
- **(c) affordances** — "I see this feature, it suggests the behavior, because that feature only occurred
  when the behavior began or ended." A keyframe feature predicts which trajectory begins. → **Priming test.**

Train 12 MB of text8, eval on a held-out 1 MB tail (Test 1 splits the 12 MB by vocabulary). Single pass,
~16 s end to end.

---

## Test 1 — Change/trajectory generalization across UNSEEN content (the headline)

Split the training vocabulary into **two disjoint halves by word identity** (hash of the word; verified
0 hash-overlap, ~0.89 M vs ~1.16 M words). Train two models on half-A words only, evaluate both on the
**unseen half-B words**:

- **content char-5** = ordinary char backoff (n-gram of order 5) — it memorizes *spellings* (locations).
- **change traj-4** = backoff over the 9-symbol **class-transition** alphabet (vowel/consonant/space →
  vowel/consonant/space), order 4 — it learns object-independent *moves* (the trajectory).

| model | bpsym (seen A) | bpsym (unseen B) | degradation |
|---|---:|---:|---:|
| content char-5 (locations) | 1.731 | 3.707 | **+114.1 %** |
| change traj-4 (movements)  | 1.076 | 1.346 | **+25.1 %** |

**Axis = TRANSFER. WON.** The content n-gram more than **doubles** its loss on unseen vocabulary (it
overfit to specific spellings — locations). The change/trajectory model degrades **~4.5× less** (+25 %),
because vowel/consonant *moves* are genuinely shared across words it never saw. This is "the movements are
shared, the locations are unique," measured. (The two bpsym scales are different alphabets — read the
**% degradation**, the within-model transfer gap, not the absolute columns.)

## Test 2 — Forward directionality ("you can't speak backwards")

One forward char-5 model (predict next | previous). Then: (b) abuse the *same* table to predict the
*previous* char from the *following* context, vs (c) a *separately* counted reverse model.

| direction | bpc | next-acc |
|---|---:|---:|
| (a) forward (next \| prev) | 1.946 | 0.597 |
| (b) forward model used BACKWARD | 5.182 | 0.106 |
| (c) SEPARATE reverse model | 1.944 | 0.600 |

**Axis = DIRECTIONAL ASYMMETRY. WON, strongly.** Running the forward trajectory in reverse is **+3.24 bpc
worse** (accuracy collapses 0.597 → 0.106 — barely above chance). A *dedicated* reverse memory recovers to
1.944 bpc / 0.600 acc — essentially identical to forward, +3.24 bpc better than the abuse. The trajectory is
directional: you replay it forward; reversal is not a free transform, it requires its own learned memory —
exactly the signature claim.

## Test 3 — Affordances (boundary keyframe primes the next trajectory)

At each word boundary, a keyframe feature is present as the next word's trajectory begins. Does it prime the
opening of that trajectory? Trigger strengths: a weak one (just the previous word's last char) and an
**affordance cluster** = (prev word's first char, last char, length-bucket), built by counting.

| predict word-initial char | bits | top-1 | lift |
|---|---:|---:|---:|
| baseline P(first) | 4.129 | 0.150 | |
| primed by boundary char (weak) | 4.058 | 0.162 | +0.071 |
| primed by prev-word cluster (afford) | 3.874 | **0.205** | **+0.255** |

first-char **class** (vowel/consonant register): 0.907 → 0.870 bits (−0.036).

Opening-trajectory P(first *n* chars):

| onset length | base | primed | lift |
|---|---:|---:|---:|
| n=1 | 4.129 | 3.874 | **+0.255** |
| n=2 | 6.591 | 6.258 | **+0.334** |
| n=3 | 8.760 | 9.661 | −0.901 |

**Axis = PRIMING LIFT. WON for n≤2, with an honest sparsity wall at n=3.** The affordance cluster gives a
real top-down prime: **+0.255 bits** on the first char (6.2 % reduction) and **+0.334** on the first two,
lifting first-char top-1 from 0.150 → **0.205** (+37 % relative). The weak boundary-char trigger barely
moves (+0.07) — *which* keyframe feature you use matters, consistent with affordances being specific learned
features, not generic context. The n=3 reversal is honest: the trigger×V³ table is too sparse at 12 MB
(eval onsets land in under-counted cells), so the prime hurts — the effect is real but shallow (primes the
*register/onset*, not the whole word). A bigger corpus or a coarser onset code would extend it; not pursued
here to keep the run single-pass.

---

## Verdict

All three trajectory-memory claims reproduce on text, each on its proper axis:

| idea | axis | result |
|---|---|---|
| change-trajectory generalizes | transfer gap on unseen vocab | **WON** — degrades 4.5× less than content n-gram |
| forward directionality | directional asymmetry | **WON** — +3.24 bpc penalty reversed, fixed by a separate reverse memory |
| affordances | priming lift on trajectory onset | **WON (n≤2)** — +0.255/+0.334 bits, top-1 0.150→0.205; sparsity wall at n=3 |

None of these wins are about beating a bigram on raw bpc on seen content — and they shouldn't be. The
change model has a *worse* job (fewer symbols) but the point is its loss barely moves to new content. The
value of trajectory memory here is exactly what the talk argues: **a representation of change that transfers
across objects, runs in one direction, and lets a keyframe feature prime what comes next.**

## Online-compliance note

Strictly online / count-based. Every model is a single streaming pass of additive counters realized with
`np.unique` (the batch-equivalent of incrementing a count per observed k-gram — order-independent, so the
one-pass batch count equals the streamed count). Predictions are table look-ups with add-α smoothing.
Affordance "clusters" are fixed feature tuples counted online — **no k-means, no SVD, no eigendecomposition,
no gradient descent, no batch optimization.** MLX was not needed (numpy counting saturates the CPU at ~1 M
chars/s and the tables are tiny). Reverse model = forward counter on the reversed stream (still one pass).
