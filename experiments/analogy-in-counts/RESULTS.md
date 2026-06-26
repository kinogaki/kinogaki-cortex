# Exp AD — is the analogy already in the counts? — 2026-06-26

**The bet.** Two old results say compositional reasoning should fall out of *counting alone*.
(1) The **parallelogram** `a:b::c:d` is already latent in raw co-occurrence counts — PMC11493305 showed
analogy structure is recoverable from the *unfactorized* count matrix, and SVD/word2vec only **smooth** it
to human parity. (2) **NARS** (Wang) derives new links by syllogism with a count-derived truth value
`(f, c)` — induction/abduction from observed evidence, no training. Turney's LRA adds that relations live
in pair co-occurrence patterns. So: can a pure count cortex do analogy and induced-link reasoning?

**The hard online rule (the banned list).** Single streaming pass, learn-while-it-lives. NO gradient
descent, NO batch optimization, and **critically** no SVD / eigendecomposition / PMI-matrix factorization
and no word2vec — those are the smoothing steps that buy the literature its parity, and they are exactly
what we forbid. The only smoothing on the allowed list is **online leader clustering** over context
profiles (`jepa.leader_cluster`), and the headline question is whether *that* count-native pool can
substitute for the banned SVD step.

**Setup.** text8, 16 MB → 2.73 M words (92 k types). Per-word co-occurrence profile over the top **M=6000**
context words (window ±4), for the top **N=30 000** target words. Profile in two **count-native** modes:
`log` = log(1+count); `ppmi` = positive PMI (a per-cell ratio of counts — computing PMI *values* is not
factorizing the matrix, so it is allowed). Relation `r(a→b) = profile[b] − profile[a]`; solve `a:b::c:?` by
**3CosAdd** (cosine to `profile[c] + r(a→b)`), excluding `a,b,c`. Analogy items built from text8 vocab in
four classic families (capital-country, currency, plural, gender): 804 items. Leader smoothing mixes each
word's profile with its online cluster centroid at strength `β` (β=0 = raw counts). Whole run **107 s on
CPU, single pass.** Fixed seed.

---

## (A) Analogy from raw counts — and the SVD-substitution test

Top-1 / top-5 accuracy. **Restricted** = pick `d` among the family's own b-words (the standard
category-restricted protocol; random 14%, frequency 10%). **Open** = pick `d` among all 30 k words (the
honest, hard number). The β columns are the **leader-cluster smoothing sweep** — β=0 is raw counts.

### Restricted candidate set

| profile | family | β=0 (raw) | β=0.3 | β=0.6 | β=0.9 |
|---|---|---:|---:|---:|---:|
| **ppmi** | capital-country | **64 / 92** | 58 / 89 | 45 / 77 | 12 / 44 |
| | currency | 30 / 100 | 33 / 100 | 37 / 100 | 37 / 100 |
| | plural | **89 / 99** | 88 / 97 | 89 / 94 | 40 / 72 |
| | gender | 43 / 87 | 44 / 84 | 42 / 85 | 25 / 69 |
| | **MACRO** | **56 / 94** | 56 / 92 | 53 / 89 | 29 / 71 |
| log | MACRO | 58 / 90 | 58 / 90 | 41 / 85 | 25 / 65 |

(baselines, restricted: random **14%**, frequency **10%** top-1)

### Open vocabulary (all 30 k words are candidates)

| profile | family | β=0 (raw) | β=0.3 | β=0.6 | β=0.9 |
|---|---|---:|---:|---:|---:|
| **ppmi** | capital-country | 6 / 22 | 5 / 22 | 3 / 11 | 0 / 1 |
| | plural | **35 / 66** | 34 / 62 | 27 / 42 | 2 / 5 |
| | gender | 17 / 41 | 19 / 36 | 14 / 27 | 1 / 5 |
| | **MACRO** | **15 / 36** | 15 / 34 | 12 / 23 | 1 / 3 |
| log | MACRO | 17 / 32 | 14 / 23 | 8 / 12 | 3 / 4 |

Worked examples (open vocab, ppmi, raw β=0): `man:woman :: king:?` → **daughter, son, iii**;
`france:paris :: japan:?` → **korea, china, singapore**; `car:cars :: dog:?` → **hound, eat, foxes**.

**Reading it.** Two findings, one positive and one a clean negative — both on the right axis.

- **Yes, the parallelogram is in the raw counts.** Restricted PPMI at β=0 lands **56% macro top-1 / 94%
  top-5** against a 14% random / 10% frequency floor — ~4× the baseline, with plural at 89% top-1 and every
  family well clear of chance. No SVD, no embedding, no gradient: a per-cell ratio of counts and a cosine.
  Open vocab is much harder (15 / 36) but still far above chance — the right answer is usually *near* the
  parallelogram point, just crowded by thousands of distractors when nothing restricts the candidate set.
- **No, online leader-clustering does NOT substitute for SVD smoothing.** This is the headline negative the
  research flagged. Every β > 0 column is **flat-to-worse** than β=0: β=0.3 ties, β=0.6 starts to erode,
  β=0.9 collapses (PPMI macro 56 → 29 top-1). Pooling a word's profile toward its cluster centroid *blurs*
  the very deltas the parallelogram needs — `paris` and `tokyo` share a cluster, so averaging them in
  destroys the `paris − france` direction. SVD smoothing works because it keeps the *axes* while denoising;
  leader-cluster averaging throws the axes away. The count-native pool is the wrong shape of smoothing for
  analogy. (It was the *right* shape for rare-context backoff — Exp Z — so this is a per-task verdict, not a
  blanket one.)

| piece | won on | lost / flat on |
|---|---|---|
| raw-count 3CosAdd (PPMI, β=0) | **analogy: ~4× baseline restricted, strong open top-5** | nothing — this is the win |
| leader-cluster smoothing (β>0) | — | **analogy (flat→worse everywhere); cannot replace SVD** |
| PPMI vs log profile | restricted top-5 (94 vs 90), more robust under β | log slightly higher restricted top-1 |

---

## (B) Induced links (NARS) — transitive count composition on never-co-occurred probes

From `L→A` (gap 1) and `A→B` (gap 2) counts, **induce** `L→B` (gap 3) through every bridge `A`, weighted by
the NARS product `f(L→A)·c(L→A) · f(A→B)·c(A→B)`. Probe = predict the true gap-3 word `B` from `L`, on the
**held-out slice** where `(L,B)` essentially never co-occur directly (direct gap-3 count ≤ 1 — 25.4% of all
gap-3 pairs). 4000 probes, top-6000 vocab. Mean-rank: lower is better (chance = 3000).

| model | top-1 | top-5 | mean-rank |
|---|---:|---:|---:|
| unigram frequency | 0.07% | 1.25% | 1593 |
| direct gap-3 counts | 0.00% | **0.27%** | **345** |
| NARS induced (bridges) | **0.10%** | 1.15% | 1366 |
| *lift: induced − direct* | +0.10% | +0.88% | **−1022** (worse) |
| *lift: induced − unigram* | +0.03% | −0.10% | +227 |

**Reading it honestly — a negative.** Induction does *something* (it beats the direct counter's near-zero
top-5, and edges the unigram on top-1), but it does **not** beat the count baselines where it matters: its
mean-rank (1366) is far worse than the direct counter's (345), and its top-5 (1.15%) merely matches the
order-blind unigram (1.25%). The reason is structural: NARS transitive composition `Σ_A f·c` **spreads mass
broadly** across everything any bridge ever predicted, so it can nudge a rare true target into the top-5 but
it dilutes the sharp local signal the direct counter keeps. And the premise that direct counts are "blind"
on the held-out slice is itself false — even at gap 3 with ≤1 direct co-occurrence, the surrounding
collocational structure gives the direct counter a mean-rank of 345. There is no compositional-prediction
lift here that counting alone didn't already have.

> **The lesson.** Analogy is *already in the counts* — a per-cell PMI ratio and a cosine recover the
> parallelogram at ~4× baseline, no SVD, no gradient, one pass. But the count cortex's own smoothing tool,
> online leader clustering, is the **wrong shape** for it: averaging a word toward its cluster centroid
> blurs the exact relation-deltas the parallelogram rides on, so β>0 only ever hurts. The famous
> SVD/word2vec "smoothing" is not interchangeable with count-pooling — it denoises while *preserving the
> axes*, and we have no count-native operator that does that. NARS transitive induction, likewise, spreads
> probability mass too broadly to beat a direct counter on held-out prediction. Counting gets you the
> analogy for free and the induced link not at all.

**Online-compliance note.** Single streaming pass. Profiles, PMI cells, offset tables, and NARS `(f,c)` are
all closed-form functions of counts; the smoothing is online leader clustering (running-mean prototypes,
assign-or-spawn, no re-assignment). `np.bincount` / `np.unique` builders are batched implementations of
order-independent accumulation, identical to a token-at-a-time online update. **No gradient descent, no
batch optimization, no SVD / eigendecomposition / PMI-matrix factorization, no word2vec.** Computing PMI
*values* per cell is counting; the banned step is *factorizing* that matrix, which we never do.

**Verdict.** (A) is a **keeper**: carry the raw PPMI co-occurrence profile as an analogy / relation-query
organ in the cortex — it reasons by parallelogram with nothing but counts, restricted-set top-5 = 94%.
**Do not** route it through leader-cluster smoothing; that pool is for rare-context backoff (Exp Z), not for
relations — it has the wrong invariance. (B) is an honest **negative, parked**: NARS transitive induction as
built does not lift held-out compositional prediction over direct counts; resurrect only if a *sharpening*
(not mass-spreading) combiner appears — the same "right combiner" thread that limited Z and S.

**Axis.** Headline = analogy top-1/top-5 from pure counts (won, ~4× baseline) + the SVD-substitution
question (answered: leader-clustering does **not** substitute) + induced-link held-out lift (negative).
Repro: `python exp_ad_reasoning/run.py` (~107 s, single pass, fixed seed). New files only:
`lib/reasoning.py`, `exp_ad_reasoning/run.py`, this file.
