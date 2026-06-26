# Exp BG / M24 — Inverse-count / surprise-prioritized spindle replay — 2026-06-26

**The bet (Schapiro 2018; M24).** Sleep doesn't replay uniformly — spindles preferentially reinstate
the **weak, recent, surprising**, protecting *infrequent* words from being forgotten. Exp **AA**'s
sleep pass replays its recent buffer **uniformly**, so at a fixed offline effort almost all of it
lands on the head (high-frequency contexts the count model already nails) and the **rare tail** — the
count model's weakest region — gets nearly nothing. M24: spend the bounded offline budget ∝ **1/count**
(protect the rare tail) or ∝ **surprise** (re-fire worst-predicted) instead. **Kill:** does not beat
uniform on **rare-context bpc at equal budget**, OR harms common more than it helps rare.

**Setup.** text8 (spec asks 60M; this is a see-where-we-are pass) — **8 MB train**, disjoint **2 MB
held-out tail**, char backoff **order 6**, 3 MB recent buffer, `RARE_THRESH=20` (a held-out position
is "rare-context" if its longest matched train context had <20 counts; **23.6 %** of positions). One
online streaming pass builds the substrate; one bounded sleep cycle per policy, with prune+distill
held **identical** across policies so the only variable is the replay distribution. Fixed seed.
**Whole run 132 s on CPU.** The ">=10 variations" M24 asks for = the policy × budget × α sweep below.

The replay mechanism (`lib/replay.py`): one streaming pass over the buffer groups every
`(ctx, next)` event at its longest matched order and reads its current count + surprise; a policy
turns those into per-event weights; a largest-remainder allotment deposits **exactly B** `+1`
increments onto the touched entries (`uniform`: weight = multiplicity; `invcount`: ÷(1+count)^α;
`surprise`: × −log₂P). Online, bounded (total deposited == B), no gradient.

---

## Result 1 — policy × budget (the head of the table is the kill-axis: **rare** bpc)

| policy | budget | bpc | **rare** | common | mem (entries) | Δrare | Δcommon |
|---|---:|---:|---:|---:|---:|---:|---:|
| *base (online, no sleep)* | — | 1.9742 | 2.681 | 1.756 | 2,215,777 | — | — |
| uniform  | 50,000  | 1.9461 | 2.490 | 1.825 | 1,362,657 | −0.191 | +0.070 |
| invcount | 50,000  | 1.9476 | 2.488 | 1.827 | 1,364,582 | −0.193 | +0.071 |
| surprise | 50,000  | 1.9463 | 2.488 | 1.826 | 1,354,858 | −0.193 | +0.070 |
| uniform  | 200,000 | 1.9465 | 2.500 | 1.826 | 1,391,735 | −0.181 | +0.070 |
| **invcount** | 200,000 | 1.9534 | **2.521** | 1.823 | 1,486,981 | −0.160 | +0.067 |
| **surprise** | 200,000 | 1.9474 | **2.483** | 1.830 | 1,378,672 | −0.198 | +0.074 |
| uniform  | 800,000 | 1.9501 | 2.522 | 1.828 | 1,458,351 | −0.158 | +0.072 |
| **invcount** | 800,000 | 1.9670 | **2.563** | 1.829 | 1,681,333 | −0.118 | +0.073 |
| **surprise** | 800,000 | 1.9561 | **2.477** | 1.849 | 1,409,488 | −0.204 | +0.093 |

Read the **rare** column. Every sleep policy improves rare-context bpc over the no-sleep baseline
(2.681 → ~2.5), confirming AA's finding that a sleep pass helps the tail. The M24 claim is narrower:
*prioritized > uniform on rare at equal budget.*

## Result 2 — invcount α sweep (budget 200k) — steepness of the 1/count protection

| α | bpc | rare | common | Δrare | Δcommon |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.9481 | 2.497 | 1.827 | −0.184 | +0.071 |
| 1.0 | 1.9534 | 2.521 | 1.823 | −0.160 | +0.067 |
| 1.5 | 1.9563 | 2.541 | 1.820 | −0.140 | +0.064 |
| 2.0 | 1.9579 | 2.560 | 1.817 | −0.121 | +0.061 |

**Steeper inverse-count protection makes rare bpc *worse*** (2.497 → 2.560 as α 0.5→2.0). The
mechanism's own knob runs the wrong way — strong evidence the 1/count idea hurts the very axis it was
meant to win.

## Result 3 — `protect_floor` ablation (invcount, 200k)

| protect_floor | rare | common | pruned_ctx |
|---|---:|---:|---:|
| True  | 2.521 | 1.823 | 529,955 |
| False | 2.502 | 1.830 | 591,922 |

Lifting reinforced contexts above the prune floor (the "spindle saves the infrequent word" knob)
*slightly worsens* rare bpc — the saved contexts are too sparse to predict well; uniform's pruning of
them and falling back to a denser backoff is mildly better.

## Result 4 — the M24 kill-check (policy vs uniform, equal budget)

| policy | budget | rare help vs uniform | common hurt | verdict (margin 5e-3) |
|---|---:|---:|---:|---|
| invcount | 50,000  | +0.002 | +0.002 | no-beat (tie) |
| invcount | 200,000 | **−0.021** | −0.003 | no-beat (worse) |
| invcount | 800,000 | **−0.040** | +0.001 | no-beat (worse) |
| surprise | 50,000  | +0.002 | +0.001 | no-beat (tie) |
| **surprise** | 200,000 | **+0.017** | +0.004 | **WIN** |
| **surprise** | 800,000 | **+0.046** | +0.021 | **WIN** |

---

## Verdict — PARTIAL (invcount kill **FIRED**; surprise variant **wins**)

The kill-condition is stated on **inverse-count**, and it **fired**: invcount never beats uniform on
rare-context bpc at equal budget — it ties at the smallest budget and is **worse** at the two larger
budgets, and its α knob makes rare bpc monotonically worse. The frequency-stratified half of M24 is a
**clean negative**. *Why:* the rare tail in a char model is rare because it carries little structure;
pouring the budget into 1/count-weighted contexts reinforces noisy two- and three-count contexts
whose sharpened distributions don't generalize to the held-out tail, while *starving* the
medium-frequency contexts that uniform was usefully topping up. "Protect the rare token" is a lexical
intuition that doesn't transfer to char-order-6 backoff, where rare-context ≠ rare-word.

But the **surprise**-driven variant — M24's "uncertain" half — **wins** the same kill-axis: rare help
+0.017 at 200k and +0.046 at 800k over uniform, by re-firing the events the model predicted *worst*
rather than merely the *rarest*. At 800k it does start to overtrade common (+0.021), so the win is
budget-gated, not free. Surprise targets *branching uncertainty*, which (unlike raw inverse-count)
correlates with where extra counts actually move the held-out prediction.

**Takeaway for the angle.** Prioritized replay is real, but the right priority is **surprise, not
inverse-count.** This is a FRAGILE-budget save: do not bury the mechanism on the invcount negative —
promote the `surprise` policy and drop the 1/count rule. A follow-up should (a) test surprise on the
**AE domain-shift** stream and on **word-level** counts (where rare-context = rare-word, the regime
the 1/count intuition was actually built for, and where it may yet win); (b) add the dual
item-vs-regularity (offset/skip-gram) budget M24 sketches, untested here; (c) sweep the
surprise/common overtrade to find the budget knee.

## Rules compliance

- **Online single pass:** the substrate is one streaming pass; replay is one streaming pass over the
  bounded buffer. No epochs.
- **No gradient / k-means / SVD / backprop:** every deposit is a `+1` count; weights are read from
  current counts/surprise; allotment is largest-remainder arithmetic. None of the forbidden
  optimizers.
- **Bounded:** the offline pass deposits **exactly B** increments regardless of buffer size (the
  scarce-offline-budget constraint), and prune+distill keep memory below the pre-sleep size (mem
  drops from 2.22M to ~1.4M entries).
