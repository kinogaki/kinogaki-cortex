#!/usr/bin/env python3
"""Experiment AS — "what survives a budget": re-run the vanished mechanisms UNDER A MEMORY CAP.

The capstone (overnight) found that at scale, with UNBOUNDED memory, the top-down topic prior, word-concept
generalization, and consolidation/sleep all gave ≈0 gain — raw high-order counts subsumed them. The
bounded-memory rule predicts this FLIPS: once you must throw most counts away, the only way to stay good is to
have GENERALIZED first, so each mechanism should EARN ITS KEEP under a fixed entry budget.

This run measures, per mechanism, Δ_unbounded (≈0, the prior finding) vs Δ_bounded (predicted > 0). A FLIP =
flat-or-negative unbounded, positive bounded. Honest if it does not flip.

Single streaming pass to build counts (vectorized np.unique == online counting); heavy-hitter cap (count-min
keep-top-B); sleep is count-based replay. Fixed seed. Reuses lib/consolidate, lib/ignition, lib/constructions,
lib/jepa via lib/budget_dichotomy.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lib import budget_dichotomy as B
from lib.corpus import load_ids

SEED = 0
np.random.seed(SEED)

# data scale: text8 is clean [a-z ]; take a decent slice + disjoint held-out tail.
N_TRAIN = int(os.environ.get("AS_TRAIN", 60_000_000))      # ~60 MB chars
N_EVAL = int(os.environ.get("AS_EVAL", 2_000_000))

# memory budgets — the "you can't keep every count" regime. Mech 1/2 cap stored CONTEXTS on the high orders;
# mech 3 caps stored CELLS (frame→filler entries + concept heads + the shared category lexicon).
BUD_CONSOL = int(os.environ.get("AS_BUD_CONSOL", 20_000))
BUD_IGNIT = int(os.environ.get("AS_BUD_IGNIT", 30_000))
BUD_CONCEPT = int(os.environ.get("AS_BUD_CONCEPT", 150_000))


def banner(s):
    print("\n" + "=" * 80 + f"\n{s}\n" + "=" * 80, flush=True)


def verdict(d_unb, d_bnd, eps=0.002):
    flat_unb = abs(d_unb) <= eps
    pos_bnd = d_bnd > eps
    if flat_unb and pos_bnd:
        return "FLIP (earns keep under budget)"
    if d_unb > eps and pos_bnd:
        return "helps both (no clean flip — already helped unbounded)"
    if pos_bnd and not flat_unb:
        return "helps bounded, mixed unbounded"
    if not pos_bnd:
        return "NO flip (flat/negative under budget too)"
    return "ambiguous"


def show(r):
    print(f"\n  mechanism: {r['name']}   [{r['metric']}]")
    print(f"    UNBOUNDED   off={r['off_unb']:.5f}  on={r['on_unb']:.5f}   Δ={r['d_unb']:+.5f}")
    print(f"    BOUNDED     off={r['off_bnd']:.5f}  on={r['on_bnd']:.5f}   Δ={r['d_bnd']:+.5f}   "
          f"(budget≈{r['budget']:,} ctx; entries off={r['off_E']:,} on={r['on_E']:,})")
    print(f"    VERDICT     {verdict(r['d_unb'], r['d_bnd'])}")
    return r


def main():
    t0 = time.time()
    banner(f"Exp AS — what survives a budget   (text8, train={N_TRAIN:,} eval={N_EVAL:,}, seed={SEED})")
    ids = load_ids("text8", N_TRAIN + N_EVAL)
    train, evl = np.ascontiguousarray(ids[:N_TRAIN]), np.ascontiguousarray(ids[N_TRAIN:N_TRAIN + N_EVAL])
    print(f"  loaded {len(ids):,} chars  ({time.time()-t0:.0f}s)", flush=True)

    results = []

    banner("MECHANISM 1 — consolidation / sleep (prune→distill→promote concepts)")
    t = time.time()
    r1 = B.mech_consolidation(train, evl, order=5, budget=BUD_CONSOL, seed=SEED)
    print(f"  [sleep: distilled={r1['distilled']:,} promoted={r1['promoted']:,} "
          f"concepts={r1['concepts']:,}]  ({time.time()-t:.0f}s)")
    results.append(show(r1))

    banner("MECHANISM 2 — top-down topic prior (ignition: committed global topic G)")
    t = time.time()
    r2 = B.mech_ignition(train, evl, order=5, g_orders=(5, 4, 3), K=64, budget=BUD_IGNIT, seed=SEED)
    print(f"  [topic clusters K={r2['K']}]  ({time.time()-t:.0f}s)")
    results.append(show(r2))

    banner("MECHANISM 3 — word-concept generalization (open-slot constructions)")
    t = time.time()
    r3 = B.mech_concepts(train, evl, budget=BUD_CONCEPT, seed=SEED)
    print(f"  [categories={r3['n_categories']} open-slot frames={r3['n_open_slot']:,}]  ({time.time()-t:.0f}s)")
    results.append(show(r3))

    # ── headline supporting evidence: the consolidation flip is a CURVE in the budget ──
    # Δ_unbounded is fixed (full tables); sweep the cap. Prediction: Δ_bounded grows as the budget tightens
    # and vanishes when the cap stops binding — the bounded-memory rule made quantitative.
    if os.environ.get("AS_SWEEP", "1") == "1":
        banner("SWEEP — consolidation gain vs budget (the flip is a curve, not a point)")
        print(f"  {'ctx budget':>11} | {'Δ_unbounded':>11} | {'Δ_bounded':>10} | entries kept")
        print("  " + "-" * 60)
        for cap in [5_000, 20_000, 60_000, 200_000]:
            rs = B.mech_consolidation(train, evl, order=5, budget=cap, seed=SEED)
            print(f"  {cap:>11,} | {rs['d_unb']:>+11.5f} | {rs['d_bnd']:>+10.5f} | {rs['off_E']:,}")

    banner("SUMMARY — the dichotomy")
    print(f"  {'mechanism':<34} {'Δ_unbounded':>12} {'Δ_bounded':>12}   verdict")
    print("  " + "-" * 92)
    for r in results:
        print(f"  {r['name']:<34} {r['d_unb']:>+12.5f} {r['d_bnd']:>+12.5f}   "
              f"{verdict(r['d_unb'], r['d_bnd'])}")
    n_flip = sum(1 for r in results if verdict(r['d_unb'], r['d_bnd']).startswith("FLIP"))
    print(f"\n  {n_flip}/{len(results)} mechanisms FLIP (flat unbounded → positive bounded).")
    print(f"  total {time.time()-t0:.0f}s", flush=True)

    # machine-readable
    with open(os.path.join(HERE, "results.tsv"), "w") as f:
        f.write("mechanism\tmetric\toff_unb\ton_unb\td_unb\toff_bnd\ton_bnd\td_bnd\toff_E\ton_E\tverdict\n")
        for r in results:
            f.write(f"{r['name']}\t{r['metric']}\t{r['off_unb']:.6f}\t{r['on_unb']:.6f}\t{r['d_unb']:.6f}\t"
                    f"{r['off_bnd']:.6f}\t{r['on_bnd']:.6f}\t{r['d_bnd']:.6f}\t{r['off_E']}\t{r['on_E']}\t"
                    f"{verdict(r['d_unb'], r['d_bnd'])}\n")


if __name__ == "__main__":
    main()
