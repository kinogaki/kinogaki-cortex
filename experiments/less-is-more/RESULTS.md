# Exp AJ — less-is-more: validity-ordered, noncompensatory, early-stopping inference — 2026-06-26

**The bet (Gigerenzer & Goldstein's *take-the-best* + the *recognition heuristic*; Simon's *satisficing*;
the bias–variance theorem).** Our usual combiner pools EVERY cue — all `D` offset-experts, the soft
accumulated channel — with a geometric mean, paying full compute to weigh evidence that mostly does not
matter. Fast-and-frugal heuristics say the opposite: rank cues by their measured **validity**, consult
them one at a time **high-validity-first**, and **STOP at the first cue that discriminates** (clears a
margin). A lower-validity cue never overrides a higher one that already fired — the rule is
**noncompensatory**. And the bias–variance decomposition (`total error = bias² + variance + noise`) says
why this should *win*, not just save compute: a count model is **high-bias / low-variance** by
construction; when data is sparse and single-pass, that is the RIGHT place to be, and a frugal rule that
*ignores* weak channels trades a sliver of bias for a large cut in variance — the **less-is-more effect**.

Everything is read off counts, online. Cues, and how validity is measured (NO gradient, NO fitted knob):

- **Recognition cue** `recog` (`count>0`): the bag-merged table — "the most frequent token I've ever
  seen follow this word." Bare recognition.
- **Per-offset experts** `off1..off8` (reused from Exp S's `offsetattn`): each offset `d` is a cue; its
  bet is the argmax of its offset table for the context word.
- **Soft / accumulated cue** `soft`: the full geometric-mean pool over all firing offsets — the
  "integrate everything" channel, here **demoted to the last-resort fallback**.

  `validity v_j = hits / (hits + misses)` for each cue — Goldstein–Gigerenzer ecological validity,
  counted online (each cue's running argmax scored against the next token).

**Take-the-best inference.** Scan the crisp cues (recognition + offsets) in descending validity; the
first whose **top-1 probability** clears a **satisficing aspiration** bar decides, and we stop. The
aspiration is a leaky accumulator over the deciding cue's discrimination — it RISES when recent
decisions are confident, FALLS when they are scarce (Simon; the bar sets itself, no tuned threshold). If
no crisp cue clears, satisfice with the soft pool. **Less-is-more (α>β):** the soft channel takes over
from the best crisp cue that fired ONLY when its validity exceeds that crisp cue's; when the soft channel
is the *weaker* one it is **ignored** — and the prediction is that on sparse/noisy contexts ignoring it
*improves* accuracy.

**Setup.** Word level on text8, top **40,000** words + UNK (OOV 2.3%), **D=8** offsets, **18 MB** prefix
≈ 3.07M words; train ≈ 2.99M, held-out eval = last **80,000** words. Cue validity + aspiration primed in
one online causal pass over the last **200,000** train words (the ranking and bar are running stats that
stabilize early). Perplexity uses a shared unigram backoff (`λ=0.10`) so every combiner is scored the
same and a top-K-capped tail never starves a token. Fixed seed 0. Whole run ≈ 5 min on CPU.

The primed cue validities (the order take-the-best scans — crisp cues only; `soft` is the fallback):

| cue | validity v | n |
|---|---:|---:|
| **off1** | **0.1748** | 200,000 |
| soft | 0.1229 | 200,000 |
| recog | 0.1161 | 200,000 |
| off2 | 0.1048 | 200,000 |
| off3 | 0.0820 | 200,000 |
| off4 | 0.0764 | 200,000 |
| off5 | 0.0753 | 200,000 |
| off6–off8 | 0.0737 | 200,000 |

The immediately-preceding word (`off1`) is by far the most-valid cue; validity then decays fast and
plateaus — the same shape Exp S found for information gain, now measured as a hit-rate. The satisficing
bar settled at **0.2392**.

---

## Q1 — take-the-best + early-stop vs full integration (accuracy AND compute)

| combiner | top-1 acc | perplexity | cues consulted / step |
|---|---:|---:|---:|
| full integration (all offsets, baseline) | 9.71% | 7,159.6 | 8.00 |
| **TAKE-THE-BEST (validity-ordered, early-stop)** | **15.00%** | **1,917.7** | **4.56** |

**Less is more, and it is not close.** Take-the-best **beats** full integration on accuracy
(15.00% vs 9.71%, **+5.3 pp**), cuts perplexity **3.7×** (1,918 vs 7,160), and does it at **43% less
compute** (4.56 cues/step vs 8.00). This is the headline: the frugal rule does not merely *match*
accuracy at a fraction of the cost — it **wins on every axis at once**. The reason is exactly the
bias–variance story: unweighted full integration pools eight offset-experts of wildly different validity,
and the seven weak far-offsets DILUTE the sharp `off1` signal (variance), dragging the argmax off-target.
Take-the-best lets the single most-valid cue decide and ignores the rest — high bias, low variance, the
right trade for a sparse single-pass count model.

---

## Q2 — the less-is-more effect (α>β): ignore the weak channel

The only difference between the two rows below is the **α>β override**: *less-is-more* refuses to let the
soft channel take over a crisp cue when `v[soft] ≤ v[crisp]` (it IGNORES the weaker channel);
*compensatory* always falls back to the soft pool.

### All contexts

| rule | top-1 acc | perplexity | cues / step |
|---|---:|---:|---:|
| **take-the-best + less-is-more (ignore weak)** | **15.00%** | **1,917.7** | 4.56 |
| take-the-best, compensatory (always integrate) | 13.99% | 3,350.2 | 4.91 |

### Sparse contexts only (the α>β prediction — `t−1` word has <50 train occurrences, 13.5% of eval)

| rule | top-1 acc | perplexity |
|---|---:|---:|
| **less-is-more (ignore weak)** | **11.68%** | **3,443.2** |
| compensatory (always integrate) | 10.96% | 4,623.1 |
| full integration | 7.41% | 7,958.9 |

**The α>β prediction holds, exactly where the theory says it should.** Ignoring the weak soft channel
beats integrating it both overall (15.00% vs 13.99%, perplexity 1,918 vs 3,350) and — the sharp test —
on **sparse contexts** (11.68% vs 10.96%, **+0.72 pp**; perplexity 3,443 vs 4,623). On exactly the
positions where the soft pool is thinnest and least reliable, *not consulting it* is the better call.
Both frugal rules trounce full integration on the sparse slice (7.41%), where over-pooling is most
harmful. The weak channel is not just useless on sparse contexts — using it actively hurts.

---

## Q3 — base-rate guard on clustering (honest negative)

The guard replaces leader-clustering's pure-cosine assignment with `argmax(similarity × clusterCount^γ)`
— a count prior folded in (representativeness → approximate Bayes; γ=0 = the current rule). We rebuild
signatures on a 15%-corrupted stream, re-cluster, and measure assignment **stability** (Rand-style pair
agreement) under that perturbation.

| γ | clusters | words clustered | stability under perturbation |
|---:|---:|---:|---:|
| **0.00** | 1,500 | 28,800 | **0.9817** |
| 0.05 | 1,500 | 28,800 | 0.6890 |
| 0.10 | 1,500 | 28,800 | 0.6093 |
| 0.25 | 1,500 | 28,800 | 0.8232 |
| 0.50 | 1,500 | 28,800 | 0.8996 |

**Honest negative: pure similarity (γ=0) is the most stable; the base-rate guard hurts here.** Any
`γ>0` *lowers* stability (a dip to 0.61 at γ=0.10, partial recovery to 0.90 at γ=0.50) and never reduces
the cluster count. The guard's intent — a count prior consolidating onto large, well-attested prototypes
— does not pay off for this online leader-clusterer: tilting the argmax toward big clusters makes the
assignment MORE sensitive to which prototype happened to grow first, not less. The representativeness →
Bayes intuition is right in spirit but the wrong knob for a single-pass, no-reassignment clusterer.
γ=0 stays the operating point.

---

## Bottom line

- **Frugal beats full, on every axis.** Validity-ordered take-the-best with early-stopping reaches
  **15.00% / ppl 1,918 at 4.56 cues/step** vs full integration's **9.71% / ppl 7,160 at 8.00** — higher
  accuracy, 3.7× lower perplexity, **43% less compute**. Pooling every cue *dilutes* the one that matters.
- **The less-is-more effect is real and predicted by α>β.** Ignoring the weak soft channel beats
  integrating it overall (+1.0 pp) and on the sparse slice where it is thinnest (+0.72 pp), exactly the
  Gigerenzer–Goldstein prediction; both frugal rules crush full integration on sparse contexts.
- **The base-rate guard is an honest negative.** γ=0 (pure similarity) is the most stable; a count prior
  on this single-pass leader-clusterer reduces stability rather than improving it.

**Online note.** Every number is a single causal pass: validity is the online hit/miss of each cue's
running bet, the aspiration is a leaky accumulator, the cue order and bar use only train data, and
nothing iterates to convergence or backprops.

**The axis.** The right question was never "how do we weigh all the evidence?" but "**which one cue do we
trust, and when do we stop looking?**" — and a count model, high-bias by construction, is precisely the
place where stopping early *wins*.

Repro: `python exp_aj_takethebest/run.py` (D=8, vocab=40k, 18 MB text8, ~5 min). New files only:
`lib/takethebest.py`, `exp_aj_takethebest/run.py`, this file.
