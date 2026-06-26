# Exp AP — permutation-bound n-grams + FlyHash addressing — 2026-06-26

**The bet (Kanerva's hyperdimensional computing; Joshi/Kanerva HD text; Dasgupta's FlyHash).** The phrase level
suffers from **sparsity**: a literal n-gram counts next-tokens at the *exact* string "a b c", so a phrase seen
once has a near-zero count even when dozens of *similar* phrases ("the b c", "a b d") were common. Each literal
phrase is its own island; the counts never pool. The VSA fix is to give similar phrases **overlapping addresses**
so their counts pool, while still keeping order so "abc" ≠ "cab" (not a bag):

- **Atom + permutation.** Each word gets a fixed random bipolar (±1) **atom** hypervector. Encode the n-gram as
  one **order-preserving address** by binding *shifted* atoms: `addr = ρ²(A) ⊛ ρ(B) ⊛ C`. ρ = a fixed cyclic
  shift (`np.roll`) that tags each slot, so position rides inside the vector; ⊛ = bind = elementwise product.
  Binding is similarity-distributive: phrases that share most (shifted) atoms get *close* dense addresses.
- **FlyHash** (Dasgupta et al. 2017, the fly olfactory circuit). A sparse **expansive** random projection blows
  the dense address up (D = 256 → M = 4000 buckets, each summing s = 12 random input dims) and a top-k = 16
  winner-take-all keeps only the strongest buckets. FlyHash is locality-sensitive: nearby inputs light up
  **overlapping** winner sets. We **count next-token at every active bucket**. Counting at *shared* buckets is
  exactly count-pooling across similar phrases — generalization, with no backprop, no factorization.

**Online rule (enforced).** Atoms + the FlyHash projection are fixed random structure (a random projection,
allowed). Everything learned is additive counts at buckets, one streaming pass, order-independent. NO gradients,
NO SVD/eigen, NO k-means, NO backprop. Bounded memory: M × V bucket counts + N atom rows.

**Setup.** text8, 14 MB → 2.39 M words (86 k types); top-N = 10 000 words get a dense id + an atom. 1.66 M clean
3-word→next-word windows. We held **20 % of the 1.10 M distinct context-phrase strings** out of *every* model's
training (325 910 windows), so the held phrases were **never seen in this exact form** — the literal n-gram must
floor them to the unigram. Three predictors on the same training, one pass each: **literal** phrase-count
baseline, **bag-of-context** (sorted-set key, order-blind control), **perm+FlyHash** (the model). Whole run
**634 s on CPU** (the literal/bag dict builds dominate; the VSA scatter-add is the cheap part). Fixed seed.

---

## Result (a) — GENERALIZATION on held-out phrases (the headline)

60 000 held-out probes, each context phrase **never seen in this exact form** in training (literal count = 0).
Lower perplexity is better; "beats" = fraction of probes where the model gave the *true* next token higher
probability.

| model on held-out phrases | perplexity | beats literal | beats bag |
|---|---:|---:|---:|
| literal n-gram (floors to unigram) | **830.7** | — | — |
| bag-of-context (order-blind) | 1451.8 | — | — |
| **perm + FlyHash (VSA)** | 1206.4 | **67.3 %** | **69.0 %** |

**The mixed truth, stated plainly.** Per probe, the VSA model beats the floored literal **two times out of
three**, and beats the order-blind bag on every axis (lower ppl *and* 69 % of probes). The pooling is real: a
phrase with zero literal count lands on buckets that *common similar phrases* filled, and inherits their
continuation. Examples it recovered while the literal could only floor:

```
'have been aware ___' -> 'of'    permfly 0.0403   literal 0.0370   (unigram floor)
'last true pharaoh ___' -> 'of'  permfly 0.0411   literal 0.0370
'covers the east ___' -> 'of'    permfly 0.0422   literal 0.0370
'advance planning and ___' -> 'a' permfly 0.0198  literal 0.0185
'least two n ___' -> 'two'       permfly 0.0144   literal 0.0130
```

**But on aggregate perplexity the VSA model does NOT beat literal (1206 vs 831) — an honest negative.** The
literal baseline's unigram floor is a *well-tuned* fallback: it never spikes. FlyHash crosstalk is the opposite —
on the ~third of probes where the wrong similar phrases dominate the shared buckets, it puts low mass on the true
token, and those spikes drag the geometric-mean perplexity up. So the model **wins the majority vote but loses
the tail.** The pooling helps where similar phrases agree on the continuation; the hashing hurts where they
disagree and collide.

## Sanity — SEEN phrases (literal should dominate)

| model on seen phrases | perplexity |
|---|---:|
| literal n-gram | **2.2** |
| bag-of-context | 2.5 |
| perm + FlyHash | 423.0 |

FlyHash addressing is a **poor exact-memory**: even for a phrase it trained on, the answer is smeared across
k buckets shared with thousands of other phrases, so the exact continuation is diluted (ppl 423 vs literal's 2.2).
The VSA address is built for *generalization, not recall* — the same crosstalk that pools rare phrases also
blurs common ones. A real system would route exact hits to the literal table and only fall back to FlyHash when
the literal count is zero; that hybrid is the obvious follow-up.

## Result (b) — ORDER-SENSITIVITY (does it stay a sequence, not a bag?)

Scramble the context word order at test (keep the multiset, permute the slots), on seen phrases:

| model | clean ppl | scrambled ppl | factor | order-sensitive? |
|---|---:|---:|---:|---|
| literal n-gram | 2.2 | 352.3 | ×161 | yes (but it just floors the new string) |
| bag-of-context | 2.5 | 2.5 | **×1.00** | **no — invariant by construction** |
| **perm + FlyHash** | 423.0 | 969.2 | **×2.29** | **yes — ρ kept the order** |

This is the clean win. The bag is **exactly** invariant (×1.00) — it threw order away. The perm+FlyHash model
**degrades ×2.29** under scrambling: the ρ-shifted binding genuinely encodes word order, and FlyHash crosstalk
did *not* wash it out. So the VSA address is a *sequence* representation that *also* pools by similarity — it sits
between the brittle literal n-gram and the order-blind bag, which is exactly the niche the construction was
designed for.

---

## Verdict

- **Pooling is real and order survives.** Perm+FlyHash beats the floored literal on two-thirds of never-seen-
  in-form phrases and beats the order-blind bag outright, while staying order-sensitive (×2.29 under scramble vs
  the bag's ×1.00). The fragile idea — *similar phrases should share their counts without becoming a bag* —
  holds on the axis it was built for.
- **The honest negative: crosstalk loses the tail.** On aggregate perplexity it does **not** beat literal
  (1206 vs 831) and is a poor exact-memory on seen phrases (423 vs 2.2). The pooling helps where similar phrases
  agree; the hashing hurts where they collide. FlyHash is a generalizer, not a store.
- **The shape of the fix.** Use perm+FlyHash as a **back-off layer**, not a replacement: literal counts when the
  exact phrase has evidence, fall to the FlyHash address only when the literal count is zero. That keeps the 2.2
  ppl on seen phrases *and* the 67 % generalization on unseen ones — the hybrid is the next experiment.

Corpus text8, single streaming pass, fixed seed, 634 s CPU. `lib/permngram.py`, `exp_ap_permngram/run.py`.
