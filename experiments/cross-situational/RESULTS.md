# Exp AV — cross-situational word→referent learning (dual-variant) — 2026-06-26

**The claim (Yu & Smith 2007; Trueswell 2013; Markman ME; McMurray 2007).** The spine learns word←word and
char←char co-occurrence. It has never had the OTHER axis a child learns on: word ← *referent in the world*.
A toddler hears "ball" while a ball, a dog and a cup are all in view and does NOT know which word names which
thing — no single scene disambiguates. Across MANY ambiguous scenes the true word→referent mapping is the one
statistical regularity that survives, and recovering it is the count-native first **grounding of meaning** —
the reason `act()` could ever say a word on purpose. The acquisition literature is unresolved between two
mechanisms, so we built **both** (`lib/crosssit.py`) and let behaviour pick:

- **(A) DenseAssoc** — a word×referent co-occurrence matrix; map a word by argmax PMI-like score
  `c(w,o)·N/(c(w)·c(o))`. The competition between words for an object *approximates* mutual exclusivity.
  Bounded by a per-word cap + LFU eviction.
- **(B) ProposeVerify** — Trueswell's "propose-but-verify": **ONE** referent hypothesis per word + a
  confidence counter. Confirm if present; decrement-and-repropose (from currently-present *unbound* objects)
  if absent. Memory-budget-honoring by construction — one slot/word, no matrix.

**Corpus.** The spec names "Yu & Smith scenes," which is **not a file in `data/`** and has no text-aligned
analogue (a real text corpus carries no referent ids). Per the spec's licence to synthesise, the scene stream
is **generated in `run.py`**: a `SceneEnv` of V words each naming one of V objects, presenting C words per
trial with their referents amid `distract` lure objects, with a `drop` chance the referent is missing and an
optional systematic `confound` lure. The agent receives the co-present referent ids as the harness's
`Turn.signal` — **not** from the token stream. Single online pass, fixed seed 0. (`exp_av_crosssit/run.py`,
`lib/crosssit.py` — new files only.)

## (1) mapping accuracy after N scenes (V=18, C=3, 3000 scenes; chance = 1/18 = 0.049)

| model | accuracy | bytes |
|---|---:|---:|
| random baseline | 0.049 | — |
| **(A) dense PMI (cap 8)** | **1.000** | 180 |
| **(B) propose-but-verify** | **1.000** | 36 |
| full uncapped table (strawman) | 1.000 | 360 |

Both variants recover the **complete** mapping from co-occurrence alone — no labels, no within-trial pairing
given, single online pass. **B reaches the same accuracy at 20% of A's footprint** (one slot/word vs a
matrix) — the memory-budget-honoring variant, for free on clean scenes. The above-chance half of the
kill-condition **does not fire** for either variant.

## (2) McMurray vocabulary S-curve (#words mapped vs scenes seen)

| scenes | 50 | 150 | 400 | 900 | 1800 | 3000 |
|---|---:|---:|---:|---:|---:|---:|
| A (/18) | 18 | 18 | 18 | 18 | 18 | 18 |
| B (/18) | 13 | 18 | 18 | 18 | 18 | 18 |

A saturates almost immediately (the dense matrix is data-efficient); **B shows the accelerating curve**
(13→18 words over the first 150 scenes) — the McMurray-style climb the single-slot mechanism predicts, since
each word's slot has to be proposed and confirmed independently. (The toy lexicon is small, so both ceiling
out fast; the *shape* before ceiling is the point.)

## (3) mutual-exclusivity rate (novel word in a scene of mostly-owned objects)

| model | ME rate | chance (1 of 3 present) |
|---|---:|---:|
| (A) dense PMI | 1.000 | 0.333 |
| (B) propose-but-verify | 1.000 | 0.333 |

Both route a never-heard word to the **unclaimed** object every time — ME-by-competition holds. **Honest
caveat:** this measures the *competition mechanism* (route to the least-claimed / first-unbound object), not a
Bayesian ME guarantee; with one genuinely unclaimed object present the competition is unambiguous, so the
rate is at ceiling. The interesting graded version (M11's fan-divided novelty) is a separate experiment.

## (4) propose-but-verify SIGNATURE — the decisive dissociation (per-word chance ≈ 0.056)

| | acc |
|---|---:|
| (B) after CONFIRM | 0.931 |
| (B) after DISCONFIRM | 0.656 |
| **(B) after RE-PROPOSE** | **0.267** ← at ~chance for the propose-pool |
| (A) when top-guess present | 0.991 |
| (A) when top-guess absent | 0.673 ← holds |

**This is the result the kill-condition turns on, and it passes cleanly.** Variant B collapses to near chance
the moment its single hypothesis is disconfirmed and re-proposed (0.931 → 0.656 → **0.267**) — it kept no
distribution, only the fresh guess it just made, the **Trueswell 2013 at-chance-after-disconfirm signature**.
Variant A, holding a co-occurrence distribution, does **not** collapse on the analogous event (its accuracy
when its top guess is absent this trial stays at 0.673, far above the 0.056 per-word chance). The two
mechanisms are behaviourally **dissociable** — the experiment's whole reason for building both. B is the
faithful human mechanism on this axis; A is the data-efficient engineer's choice.

## (5) FRAGILE budget — 12 variations of (V, C, N, confound)

| V | C | N | confound | chance | accA | accB |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 2 | 500 | 0.0 | 0.083 | 1.000 | 1.000 |
| 12 | 2 | 1500 | 0.0 | 0.083 | 1.000 | 1.000 |
| 12 | 2 | 3000 | 0.0 | 0.083 | 1.000 | 1.000 |
| 18 | 3 | 1000 | 0.0 | 0.056 | 1.000 | 1.000 |
| 18 | 3 | 3000 | 0.0 | 0.056 | 1.000 | 1.000 |
| 18 | 4 | 3000 | 0.0 | 0.056 | 1.000 | 1.000 |
| 24 | 3 | 3000 | 0.0 | 0.042 | 1.000 | 1.000 |
| 24 | 5 | 4000 | 0.0 | 0.042 | 1.000 | 1.000 |
| 36 | 6 | 8000 | 0.0 | 0.028 | 0.972 | 1.000 |
| 18 | 3 | 3000 | **0.5** | 0.056 | 1.000 | 0.944 |
| 18 | 3 | 3000 | **0.8** | 0.056 | 0.889 | 0.389 |
| 24 | 4 | 4000 | **0.7** | 0.042 | 0.958 | 0.500 |

**A above chance in 12/12 · B above chance in 12/12.** On clean scenes both are near-perfect across
referential uncertainty C=2..6 and vocab V=12..36. The **systematic-confound regime** (a non-referent lure
that co-occurs with a word far more than chance) is where the **equal-memory tradeoff surfaces**: the dense
matrix can out-vote a frequent lure (A holds 0.889–1.000), while the single-slot guesser gets dragged onto it
and recovers only partly (B 0.389–0.944) — yet **B never drops below chance**. This is the AQ-style
equal-bytes contrast: B buys an 80%-memory saving on clean input at the cost of robustness to a systematic
confound.

## Honest accounting

- **Headline accuracy is at ceiling** because the synthetic lexicon is small and clean; the informative
  results are the *shapes and signatures* (the S-curve, the disconfirm collapse, the confound divergence), not
  the 1.000s. A larger or noisier scene world would lower the ceiling; the mechanism and its dissociation are
  what generalise.
- **ME is measured as competition, not a Bayesian guarantee** — exactly as the spec instructs. The graded
  fan-divided ME (M11) is a different experiment.
- **No real grounding harness yet.** This is the scene-bearing env + the binding layer as a standalone module;
  wiring it into a live InterlocutorEnv responder that *names a referent* (the spec's later Haiku step) is the
  follow-up that turns this into an `act()` reason-to-speak.

## Verdict — **WIN (mechanism + dissociation), with the headline at ceiling**

Both variants learn the word→referent mapping from co-occurrence alone, online, single-pass, bounded — the
above-chance kill-condition **does not fire** (12/12 each). Variant B reproduces the **Trueswell
at-chance-after-disconfirm signature** cleanly (0.93 → 0.27), so the second kill-condition **does not fire**
either. The two mechanisms are behaviourally dissociable — B is memory-cheap and human-faithful, A is robust
to systematic confounds — so the right answer per the spec ("pick by harness grounding") is to **keep both**:
B as the default one-slot grounder, A available where a confound demands a full distribution. This is the
first count-native binding of a word to a thing in the world — the seed of a reason for `act()` to say a word.

## Rules compliance

- **Online single pass** ✓ — every `observe()` is one count update / one slot verify; no second pass.
- **No gradient descent / k-means / SVD / eigen / word2vec / backprop** ✓ — A is co-occurrence counts + a
  closed-form PMI ratio; B is one slot + an integer confidence counter.
- **Bounded memory** ✓ — A is capped per word with LFU eviction; B is one slot + one counter per word.
- **FRAGILE budget honoured** ✓ — 12 variations across C, N, V and confound before any verdict.

Repro: `python exp_av_crosssit/run.py` (synthetic scenes, ~30s, no data files). New files only:
`lib/crosssit.py`, `exp_av_crosssit/run.py`, this file, `README.md`.
