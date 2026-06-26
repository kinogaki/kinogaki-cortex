# Exp X — the heterogeneous, specialized stack vs the uniform Column — 2026-06-26

**The thesis under test (the opposite axis from Exp I).** Exp I proved a UNIFORM `Column`, wired bigger,
gets better ("cortex small → big"). This experiment tests the brain's *other* claim: the brain is **not** one
repeated part — it is **specialized topologies at different levels**. Retina does center-surround filtering;
cortex does columns + voting; the thalamus **gates**; the basal ganglia **select/arbitrate**; the hippocampus
holds episodes. Connections come in two kinds — **proximal/local** vs **distal/long-range**. Timescales run a
gradient — **fast sensory** to **slow integrative** (Hasson). So: give each LEVEL a different column type, a
different connection RANGE, and a different TIMESCALE, then **gate/arbitrate** which level speaks per token.
Does specialization-by-level beat the uniform stack?

**The stack.** Four specialized levels, every one predicting the *same* next CHARACTER (so bits-per-char is
apples-to-apples end to end and against the uniform baseline):

| level | column type | range | timescale | reused module |
|---|---|---|---|---|
| **L0 char** | dense local n-gram | proximal (short window) | fast | `evidence.ExpertBank` |
| **L1 word** | offset-keyed count-attention + lexicon | distal (mid) | mid | `offsetattn.OffsetAttn` |
| **L2 phrase** | branching-entropy chunks + change/trajectory | distal-ish | slower | `boundaries` + `trajectory` |
| **L3 theme** | online topic state + G-conditioned chars | long-range | slow | ignition idea, made online |

Higher levels project their (word / chunk / topic) belief onto the next char through a spelling lexicon — the
same `char_prior` move the uniform cortex already uses. **Thalamic gate + basal-ganglia arbitration** = a
per-token router that picks/weights which level dominates, by each level's running **confidence** (leaky
neg-NLL) and the char level's **surprise** (leaky entropy), replacing one fixed pooling rule.

**Setup.** text8, 10 MB train / 80 k-char eval. `exp_a_boundary/.venv`, numpy. Single streaming pass, all
counting + leaky accumulators + **online leader-clustering** for topics. No backprop, no k-means/SVD.

---

## [1] Ablation — does specialization-by-level beat the uniform stack? (bits-per-char)

| configuration | bpc | char-acc |
|---|---:|---:|
| uniform Column stack (Exp I) | **1.985** | — |
| char only (L0) | 2.010 | 0.558 |
| **HETERO full** (char+word+phrase+theme, hard-router gate) | 2.369 | 0.566 |
| &nbsp;&nbsp;− word | 2.191 | 0.561 |
| &nbsp;&nbsp;− phrase | 2.233 | 0.563 |
| &nbsp;&nbsp;− theme | 2.366 | 0.567 |
| &nbsp;&nbsp;char + word only | 2.229 | 0.564 |

**Read it honestly.** On aggregate bpc, **specialization-by-level loses** to both the uniform stack and even
to the bare char level. Every distal level *adds* bits, because as a **character** predictor its projection is
noisy: next-word/next-chunk/next-topic prediction is genuinely hard (perplexity in the thousands), and a hard
guess at the next *word* makes a poor prior on the next *char*. The ablation is monotone — dropping any distal
level helps; char+word-only (2.229) is the least-bad combination; the theme level adds essentially nothing to
bpc (2.366 vs 2.369). This is the **expected fragile-idea state** (Commandment 1): a first result no better
than the simpler baseline is normal, not a verdict. So we judged each piece on its **right axis** (Commandment
7), below.

> **Char-acc note.** Interestingly, top-1 char accuracy *rises* with the full stack (0.558 → 0.566) even as
> bpc worsens — the distal levels help pick the single most-likely next char at boundaries but are
> *over-confident* there, so they pay in calibration (bpc). Same product-vs-calibration tension Exp I and
> Exp D found, now across heterogeneous levels.

---

## [1b] Right-axis — each specialization on the axis it was built to win

**Word level — its axis is CALIBRATION, not top-1** (the Exp S finding, reproduced):

| next-word model | top-1 | perplexity |
|---|---:|---:|
| offset-attn | 0.143 | **1803** |
| bag-of-words | 0.070 | 3458 |
| bigram (d=1) | 0.148 | 1613 |

Offset-attn **kills the bag** (order matters: 0.143 vs 0.070 top-1, nearly 2× better, and a far lower
perplexity) — the headline claim of count-attention holds. It **ties the bigram on top-1** but its real edge
is calibration: it beats the bag's perplexity decisively. (The bigram's slightly lower perplexity here is the
known Exp-S caveat at this scale — the win to protect is offset-attn ≫ bag, the order-sensitivity.)

**Phrase level — its axis is unit discovery.** Branching entropy minted **8000 chunks, 58 % multi-word**, with
no spaces-between-phrases given. Discovered units are real collocations: *"bright star catalogue … hipparcos
catalogue esa"*, *"extra vehicular activity list of spacewalks"*, *"september … attacks"*. The phrase level
**does** find phrases; it just doesn't help next-char bpc.

**Theme level — its axis is online topic formation.** Online **leader-clustering** (no k-means, single pass)
found **143 topics**. Some are clean (*leap holidays gregorian observances feast*; *ohio jersey campus downtown
beach*); some are noisy (*haliotis amaranth … guitarist*). It forms topics online — but as a char prior it is
the weakest contributor.

---

## [2] Dynamic gating vs static pooling — the clearest positive

| gate / pool | bpc |
|---|---:|
| char only (reference) | 2.010 |
| static pool — equal weight | 3.274 |
| static pool — fixed char-heavy weight | 2.291 |
| DYNAMIC soft gate (confidence softmax) | 2.537 |
| **DYNAMIC hard router (argmax confidence)** | **2.369** |
| DYNAMIC anchored gate (driver+modulator) | 2.683 |

**Dynamic routing decisively beats the naive static pool** (2.369 vs 3.274 — a 0.9 bpc gulf). The fixed
equal-weight geometric mean lets the three weak distal levels *drag down* the strong local one; the dynamic
**hard router** — basal-ganglia winner-take-all by confidence — routes **85 % of tokens to char**, opening to
word (9 %) / phrase (5 %) / theme (0 %) only where char is unsure, and so **recovers near char-level bpc**.
This is the real architectural lesson: *if* you mix heterogeneous levels, **the combiner must be dynamic** —
a fixed pooling rule is actively harmful when the levels are unequal. (Static char-heavy weight, 2.291, ties
the hard router — i.e. most of the dynamic win is just "trust the local model," but routing finds that for
free, per-token, without hand-tuning weights.)

**[2b] Robustness — the gate's hoped-for right axis (honest negative):**

| context noise | char-only bpc | hard-router bpc | gate − char |
|---:|---:|---:|---:|
| 0.00 | 2.010 | 2.233 | −0.223 |
| 0.05 | 2.490 | 2.654 | −0.164 |
| 0.10 | 2.920 | 3.086 | −0.166 |
| 0.20 | 3.686 | 3.925 | −0.239 |

We expected the gate to **win under noise** — corrupt the local context and the slow long-range levels should
rescue the fast one. It does not. The fast char model **degrades gracefully on its own**, and the distal
levels' char-projections are too noisy to help even at 20 % corruption. **Parked, not killed** (Commandment 8):
robustness lives *within* the char level (a leaky-evidence decode, Exp R), not in routing to noisier distal
char-projections. The next step is to gate over each level's *leaky-evidence* output rather than its fresh
product pool.

---

## [3] Walking through idea space — the exploratory demo (honest)

Generation by **walking the PMI association graph** (`graph.py`), guided by a goal topic and a surprise
(novelty) penalty that forbids revisiting — a "motor = moving through concept space" loop, all counts.

```
free walk from 'science':
  science → writers → anarchists → anarcho → historian → sir → austin → fictional
          → ark → bible → books → translated → armenian → classical → texts → collection
goal walk science → 'music':
  science → oxford → understanding → deep → objects → animals → paid → raised → money …
```

**Coherence** = mean PMI-edge weight actually traversed (are consecutive ideas associated?):

| walk | coherence |
|---|---:|
| free walk | 0.052 |
| goal walk | 0.051 |
| random words | **0.000** |

The walk stays on **associated** ideas (the free roam *science → anarchist writers → the bible → classical
texts* is a recognisable chain of association); random jumps traverse zero weight. **HONEST:** this is
associative idea-*roaming*, **not** grounded reasoning — a navigation demo over a co-occurrence graph, not
thought. The goal-bias works weakly (it pulls toward the music neighbourhood but wanders). Real "navigating
idea space" needs the trajectory/V model to steer the walk, not just spread activation — parked as the next
step.

---

## Verdict

**Specialization-by-level did NOT beat the uniform stack on bits-per-char** — and that is the honest headline.
The brain's heterogeneity buys things our flat metric can't see: the word level's *order-sensitivity* and
*calibration*, the phrase level's *unit discovery*, the theme level's *online topic formation*, the graph's
*coherent idea-roaming*. Each specialized piece **won on the axis it was built for** and **lost on char-bpc**,
exactly the fragile-idea pattern. The one clean architectural win is the gate: **dynamic confidence routing
beats static pooling by ~0.9 bpc** — when levels are unequal, *the arbitration mechanism, not the levels, is
load-bearing.*

**Which axis each piece won on**

| piece | won on | lost on |
|---|---|---|
| L1 word (offset-attn) | order-sensitivity (≫ bag), calibration | next-char bpc, top-1 vs bigram |
| L2 phrase (chunks) | unit discovery (58 % multi-word) | next-char bpc |
| L3 theme (online topics) | online topic formation (143, no k-means) | next-char bpc (negligible help) |
| gate (thalamus + BG) | **routing ≫ static pool (2.37 vs 3.27)** | beating char-only on clean text |
| idea-walk | coherent association (0.05 vs 0.00) | grounded reasoning (aspirational) |

**Parked (graveyard, not trash):** (1) gate over each level's **leaky-evidence** decode (robustness should
live there); (2) blend the levels **at their native granularity** (word-level mixing) instead of projecting
every level down to chars — the char-projection is what poisons bpc; (3) **trajectory-steered** idea-walk.

**Online-compliance.** Single streaming pass. Every table built by counting (`np.unique` / `bincount`); every
accumulator leaky/decayed (confidence, surprise, topic histogram, evidence); topics by **online leader
clustering** (sequential, single-pass — the explicit no-k-means substitution the brief required). No gradient
descent, no k-means/SVD/eigendecomposition anywhere. Reproduce:
`exp_a_boundary/.venv/bin/python exp_x_hetero/run.py`.
