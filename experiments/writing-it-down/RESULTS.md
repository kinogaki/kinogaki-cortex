# Experiment AQ — environment-as-memory: writing it down — 2026-06-26

**The third coping route.** The bounded-memory rule has now produced three experiments that all ask the same
thing: under a *fixed* memory budget, what survives eviction? Two routes were already tested — **evict the right
tail** (AE's LTI/ART, AI's ACT-R power law) and **consolidate the head offline** (AA's sleep replay). This one
tests the route humans actually lean on hardest: we don't hold everything in our heads — **we write it down.**
Ericsson & Kintsch's *long-term working memory* says an expert keeps a tiny set of **cues** in a narrow focus and
the **content** in an external store the cues retrieve on demand. The question for our substrate: at **equal total
budget**, does a small internal model + an external store beat one big internal table — especially on the
**rare contexts** the all-internal model evicted?

**Substrate.** Char-level count model, orders 1–5, add-α highest-order-seen backoff (the AE/AI counting core, so
peak quality is comparable). darwin.txt, 200k train / 20k held-out, single streaming pass, fixed seed 0.

- **A AllInternal** — one bounded table per order, cap 4000. On overflow, evict the tail (reservoir-sampled
  lowest leaky-recency use-score). The long tail is forgotten; there is nowhere else for it. (Exp AE's FLAT.)
- **B IntExt** — a *small* internal table (`int_cap` 1000/order) + an **external store** (`ext_cap` 3000/order).
  On overflow the internal table **writes the evicted context's counts down** to the store (merging if already on
  paper), then drops it. At predict time the internal table answers when it is **confident** (its distribution's
  entropy ≤ `conf_h` bits); when **uncertain** it pays a **retrieval** — looks the same context up on paper and
  blends. `int_cap + ext_cap = 4000 = A's cap`, so **A and B hold the same per-order entry budget.**

**Accounting, honestly.** Budget = stored context entries (each a 27-float count vector + scalars — same unit for
both). The external store **counts against the budget** in the strict A-vs-B test (this is the fair test). We also
report a **cheap-external** variant (D), where paper is bigger than skull — the realistic asymmetry, *not* equal
budget. Every external read is counted (consult rate, hit rate), so the store's time cost is visible.

**The rare slice.** A held-out position is *rare* if its order-5 context appeared **1–3×** in training — exactly
the entries a bounded internal table evicts first. 2,141 of 19,995 eval positions (10.7%). This is the slice where
writing-it-down should pay off.

## Result — equal total budget (bpc, lower = better)

| arch | overall | **rare** | common | total entries |
|---|---:|---:|---:|---:|
| **A** all-internal (cap 4000) | **2.2091** | **2.3983** | **2.1864** | 11,839 |
| **B** int+ext (1k+3k, equal budget, conf_h 2.0) | 2.2208 | 2.4113 | 2.1979 | 12,478 |
| C internal-only (cap 1000, **under** budget by 3k) | 2.4708 | 2.7727 | 2.4346 | 3,478 |
| D int + **big** ext (1k+9k, cheap paper, NOT equal) | **1.9791** | **2.2459** | **1.9471** | 24,736 |

**Equal-budget Δ (A − B; + = writing-it-down wins): overall −0.0117, rare −0.0131.**

- **The strict equal-budget verdict is an HONEST NEGATIVE.** At equal budget and selective consultation
  (conf_h 2.0, the external store read on 71% of predictions, 88% hit-rate), the split internal+external model
  does **not** beat the single bounded table — it is **~0.01 bpc worse** overall *and* on the rare slice. Writing
  it down, then re-reading only when unsure, does not buy you anything the one big table didn't already have.
- **Why.** Two losses eat the gain. (1) **The split fragments evidence:** a context's counts now live partly
  internal, partly on paper, and the confident-internal path answers from the *internal fragment alone* — missing
  the written-down half. (2) **At equal budget the big table simply holds more of the live distribution at once**;
  there is no eviction the external store uniquely rescued, because darwin's order-5 tail that a 4000-cap table
  evicts is genuinely low-value (the rare slice is hard for *both* — 2.40 vs 2.39).
- **C proves the budget matters and the store is doing real work:** internal-only at 1000/order is far worse
  (rare 2.77). So B's external store *is* recovering most of the 3k of lost budget (rare 2.41, nearly back to A's
  2.40). It just doesn't **exceed** A. The store earns its keep; it doesn't out-earn the same bytes held internal.

## The one regime where it does win — consult almost always

Sweeping the internal-confidence threshold (how often we bother to re-read paper):

| conf_h | overall | **rare** | consult % | hit % |
|---:|---:|---:|---:|---:|
| 0.5 | 2.2522 | **2.3703** | 96.9% | 92% |
| 1.0 | 2.2369 | 2.4037 | 90.3% | 93% |
| 1.5 | 2.2315 | 2.3833 | 82.1% | 92% |
| 2.0 | 2.2284 | 2.3993 | 71.5% | 92% |
| 3.0 | 2.2907 | 2.5251 | 43.8% | 93% |
| 4.0 | 2.3656 | 2.6520 | 15.5% | 97% |

- At **conf_h 0.5** — distrust the internal model and **re-read the store on 97% of predictions** — B's rare-slice
  bpc (**2.3703**) finally **beats A (2.3983)** by 0.028 bits. The written-down tail *does* help the rare slice —
  **but only when you essentially always go back to the page.** That is the honest shape of the win: externalizing
  pays off exactly when retrieval is cheap enough to do constantly, which is the human reality (paper is cheap) and
  is precisely what the cheap-external variant rewards.
- The selective, "consult only when unsure" version (the *efficient* one) loses, because the internal-confidence
  signal sends you to paper on the wrong subset and lets the fragmented-evidence loss dominate.

## The realistic asymmetric verdict — paper is cheaper than skull (D)

Drop equal budget. Keep internal tiny (1000) and give the store 9000/order — the real human asymmetry, since an
external store *is* cheaper per entry than working memory. **Overall 1.979, rare 2.246, 99% rare hit-rate** — a
**0.23-bit** improvement over the all-internal table at the same tiny internal footprint. This is the unsurprising
part and the practically important one: **if the external store is cheap, externalize aggressively.** The
contribution of this experiment is the *equal-budget* answer, which is the one that isn't obvious.

## Takeaways

1. **At equal byte budget, "write it down" does not beat "hold it all in one table"** on darwin char-prediction —
   an honest negative for the strong form of the hypothesis. The cost is **evidence fragmentation** (the
   confident-internal path answers from the internal fragment, ignoring the written-down half).
2. **It wins on the rare slice only when you re-read the store almost always** (conf_h 0.5: rare 2.37 vs 2.40) —
   i.e. when retrieval is treated as cheap. Selective "consult-when-unsure" is the efficient version and it loses.
3. **The realistic regime is the asymmetric one** (D): paper cheaper than skull → big external store → −0.23 bpc.
   Ericsson & Kintsch's long-term working memory is *cost-asymmetric by construction* (the page is free to keep) —
   so the model that matches the cognition matches its result, and the equal-budget framing is the wrong one for
   the human analogy even though it is the right one for the fair-comparison question.
4. **Fix for a future swing:** never answer from the internal fragment alone — always blend internal + external
   when both hold the context, and make the *consult decision* itself count-based (consult iff the internal
   context is below a learned count threshold, not an entropy one). That removes the fragmentation loss and tests
   whether selective retrieval can win at equal budget.

*Lineage: bounded-memory rule → AE (non-forgetting under shift) → AA (sleep consolidation) → AQ (externalize).
Credit Ericsson & Kintsch (1995), long-term working memory. Single pass, fixed seed 0, ~85 s CPU.*
