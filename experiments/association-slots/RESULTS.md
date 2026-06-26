# Exp AW — streaming-ASSOCIATION slot strength (ΔP / PPMI) as the construction substrate — 2026-06-26

**The bet (Ellis cue contingency; Allan's ΔP; collostructional PPMI; Casenhiser-Goldberg skewed input).** AF
(Exp AF) builds open-slot constructions by counting, but it ranks and vetoes a slot's filler-CATEGORIES by the
**raw commitment ratio** `r = c(f,s)/c(f,·)` — pure conditional `P(s|f)`. That share is base-rate-inflated: a
category that follows *everything* earns a high raw share in *every* frame, so raw co-occurrence over-generates
(the 2025 "LLMs learn constructions humans don't know" warning). M5 swaps the substrate for **association**: keep
only four additive marginals per `(frame f, category s)` — `c(f,s), c(f,·), c(·,s), N` — and derive on demand

- `ΔP = P(s|f) − P(s|¬f)` (Allan / Ellis contingency),
- `PPMI = max(0, log(c(f,s)·N / (c(f,·)·c(·,s))))` (positive pointwise MI).

Both discount a category by its global base rate `c(·,s)/N`. Association then drives the skewed-input anchor
(argmax association, not argmax count), the open-slot distribution (count × association), and the preemption veto
(relative *association* between competing frames, not relative commitment ratio).

**Setup (AF's pipeline, equal memory, equal categories).** text8, **8 MB → 1.36 M words (62 k types)**; top-N =
10 000 words get an id + an **online** category (`jepa.py` signatures + leader clustering, single pass), all
10 000 into **C = 400** categories. Held out **30 % of the 364 277 distinct (frame, filler) pairs** (those fillers
*never seen in that frame*). 2 161 open-slot frames induced. **Whole run 19 s on CPU, single streaming pass, fixed
seed.** *Corpus note:* M5 also names a CHILDES/BabyLM CDS subset — **not in `data/`**; substituted text8 (AF's
corpus), the comparison that owns the −39.5 % over-generation bar. CDS purity is M6/M7's axis, not AW's.

---

## DIAL A — over-generation veto (the −39.5 % bar)

Weak-competitor links = a category a frame holds *weakly* (≤ 20 % of its peak commitment) while a rival frame
commits ≥ 2× harder — the "could-occur-but-unobserved → blocked" over-generation forms. We measure the open-slot
head's mean mass on weak-competitor vs strong-attested links, before vs after each veto. **AF's commitment-ratio
veto at this slice: cut −37.6 %, retain 101.4 % of attested mass** (the −39.5 % bar, reproduced at 8 MB).

| veto | floor | weak-competitor cut | strong (attested) retain |
|---|---:|---:|---:|
| **AF raw commitment-ratio** | — | **−37.6 %** | **101.4 %** |
| assoc ΔP | 0.00 | −35.8 % | 81.1 % |
| assoc ΔP | 0.60 | −11.4 % | 95.3 % |
| assoc PPMI | 0.00 | −11.0 % | 80.7 % |
| assoc PPMI | 0.60 | −4.7 % | 94.7 % |

(8 kind × floor variants shown abbreviated; the ΔP/PPMI × {0, .25, .4, .6} grid is monotone between the rows.)

**Association loses Dial A, clearly.** ΔP can match AF's cut (−35.8 %) only by bleeding **19 % of attested mass**;
to retain strong mass it can cut at most −11.4 %. The association-native diagnostic confirms why: PPMI
**hard-vetoes 27.1 % of weak-competitor links but also wrongly zeroes 18.9 % of strong attested links** — almost
the same rate. On English text the contingency score does *not* separate over-generation from real constructions
better than the commitment ratio; it just adds collateral.

## DIAL B — held-out compositional perplexity (raw-count is the bar)

Same compositional head as AF — `P(w|frame) = Σ_c P(c|frame,slot)·P(w|c)` — but `P(c|frame,slot)` comes from the
association-weighted distribution instead of raw category counts. 40 000 held-out pairs. Lower is better.

| open-slot prior | all held-out ppl | **open-slot-only ppl** |
|---|---:|---:|
| n-gram floor (no construction) | 17 638 | — |
| **AF raw-count** | 7 417 | **6 403.6** |
| assoc **ΔP** | 7 378 | **6 357.5** (−0.7 %) |
| assoc PPMI | 7 948 | 7 106.9 (+11.0 %) |

**ΔP gives a razor-thin win (−0.7 %); PPMI is clearly worse (+11 %).** A 0.7 % perplexity move on open-slot pairs
is within noise — not a real improvement. PPMI over-corrects: by punishing base-rate it strips mass from the
ubiquitous-but-correct function-word categories that the held-out fillers actually belong to.

## Memory

Association **prunes** — categories with ΔP ≤ 0 / PPMI = 0 drop. The slot table shrinks to **75 % of AF's raw
links** (27 578 vs 37 014). Bounded ✓.

---

## Verdict — **PARTIAL → PARK** ("raw counts suffice for English text", with a ΔP caveat)

The kill is **conjunctive**: association must beat AF on Dial A **or** Dial B. After the FRAGILE budget (ΔP/PPMI ×
4 floors on Dial A, ΔP/PPMI on Dial B, plus the native hard-veto diagnostic — both dials checked):

- **Dial A — clean negative.** Association does **not** lower over-generation below AF's −39.5 %. Best
  attested-retaining cut is −11.4 % vs AF's −37.6 %; PPMI's hard veto is as likely to fire on a real construction
  as on an over-generation link.
- **Dial B — negligible.** ΔP edges raw counts by 0.7 % (within noise); PPMI loses by 11 %.

Because Dial B is *non-negative* (ΔP ≥ raw, barely), the strict kill-condition **does not fire** — so this is
**partial/park**, not a kill: park as **"raw commitment ratio suffices for the construction substrate on English
text"**, exactly M5's named park outcome. **The honest caveat in association's favor:** ΔP never *hurts*
compositional perplexity and shrinks the table 25 %, so as a memory-bounded, never-worse substrate it is a
defensible swap — but it earns no new accuracy here. The mechanism's likely home is **CDS / morphologically-rich
or free-word-order corpora** (where base rates differ sharply from English text8 and contingency should bite) and
the **generation guardrail** (a hard PPMI = 0 veto before G1/BD emits), not English-text perplexity.

**Rules:** ONLINE single streaming pass (four additive marginals, closed-form on demand) ✓ · BOUNDED (association
prunes, table → 75 % of raw) ✓ · NO gradient descent / k-means / SVD / eigen ✓.
