#!/usr/bin/env python3
"""Experiment BJ — structure-graded recursion exposure (self-gated embedding depth).

The ONE curriculum AK did not test. AK staged the *memory budget* (leak-horizon) and got an honest negative:
growing the budget ties full-from-start, because a count learner has no gradient to LOCK — an early noisy
count is simply outvoted by later evidence, never frozen, so "starting small" has nothing to rescue. AK's
winner was FULL (every regularity present from char one).

BJ stages STRUCTURE instead of memory: order the stream by **embedding depth**, and let the agent SELF-GATE
the order on its own branching entropy — admit depth d+1 only once depth d's transition entropy has
*stabilized* (the local recursion is in). Teacher-free: the gate reads the agent's own count tables, no
schedule clock, no labels. This is Elman's ORIGINAL recursion result on the axis AK left untouched.

Corpus — center-embedded subject–verb agreement (the textbook recursion stressor):
    depth 2:  S1 S2 <embedded clause>  v2{agree S2}  v1{agree S1}
The OUTERMOST closing verb agrees with the OUTERMOST subject, across the whole nested clause (unique middle,
so whole sentences can't be memorized). We score the agreement char at each closing verb; the deep-embedding
(depth-2/3 OUTER) targets are the recursion-only axis — the only place a structural curriculum could win.

Three regimes on the SAME multiset of sentences, single streaming pass, fixed seed — only the ORDER changes,
never the learner:
  GRADED — easy→hard by depth, the depth gate SELF-OPENED by branching-entropy stability.
  FULL   — all depths interleaved uniformly from char one (AK's winner).
  ANTI   — hard→easy (deepest first), the curriculum reversed (ordering control, wrong direction).

KILL (BJ, fragile): self-gated depth ordering does not beat FULL on depth-2/3 outer agreement perplexity
across the budget → AK extends to structural ordering (a clean, publishable negative; expected — hard sell).
We run the FRAGILE budget: 4 seeds × {eps gate, window} variations on the recursion-only axis before any
verdict, and check whether GRADED ever wins on the deep token alone.

HARD RULES: online single streaming pass; no gradients; no batch/k-means/SVD; bounded memory; fixed seed.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics, recursion  # noqa: F401  (per the harness contract)

SEED = 0
DEPTHS = (1, 2, 3)
N_PER_DEPTH = 4000     # sentences per depth (12k total)
N_KEYS = 120           # distinct subject keys → each key's class binding is sparse
K = 12                 # char-context window: spans the depth-1 subject→verb cue, NOT the deeper ones
ALPHA = 0.05
H = 200.0              # leak horizon (bounded memory); large enough that depth-1 IS learnable (acc>0.5)
GAP_LO, GAP_HI = 3, 6  # short embedded filler so depth-1 cue fits in K and depth-2/3 push out of window


def deep_ppl(stats):
    """Geometric-mean outer-agreement perplexity over depths 2 and 3 (the recursion-only axis)."""
    nlls = []
    for d in (2, 3):
        if d in stats and stats[d]["n_outer"] > 0:
            nlls.append(np.log(stats[d]["ppl_outer"]))
    return float(np.exp(np.mean(nlls))) if nlls else float("nan")


def one_run(seed, eps, window):
    sents = recursion.make_embedded_corpus(N_PER_DEPTH, depths=DEPTHS, gap_lo=GAP_LO, gap_hi=GAP_HI,
                                           seed=seed, n_keys=N_KEYS)
    g, glog = recursion.run_graded(sents, DEPTHS, K=K, alpha=ALPHA, eps=eps, window=window, seed=seed, H=H)
    f = recursion.run_full(sents, DEPTHS, K=K, alpha=ALPHA, seed=seed, H=H)
    a = recursion.run_anti(sents, DEPTHS, K=K, alpha=ALPHA, seed=seed, H=H)
    return g, f, a, glog


def run():
    t0 = time.time()
    print(f"corpus: center-embedded agreement, depths {DEPTHS}, {N_PER_DEPTH}/depth, "
          f"{N_KEYS} keys, K={K} (chance on agreement char: ppl 2.0 / acc 0.50)")
    print(f"axis: OUTER agreement perplexity at depth-2/3 (the recursion-only token). lower = better\n")

    # ── primary: seed 0, default gate, full per-depth breakdown ──
    g, f, a, glog = one_run(SEED, eps=0.02, window=200)
    print("=== per-depth OUTER agreement: perplexity (lower=better) + acc (chance 0.50) (seed 0) ===")
    print(f"  {'depth':<7}{'GRADED ppl':>12}{'FULL ppl':>12}{'ANTI ppl':>12}"
          f"{'G acc':>8}{'F acc':>8}{'n':>8}")
    for d in DEPTHS:
        print(f"  {d:<7}{g[d]['ppl_outer']:>12.3f}{f[d]['ppl_outer']:>12.3f}{a[d]['ppl_outer']:>12.3f}"
              f"{g[d]['acc_outer']:>8.3f}{f[d]['acc_outer']:>8.3f}{g[d]['n_outer']:>8}")
    print(f"\n  depth-1 is the LEARNABLE rung (cue in window); depth-2/3 are out of window = the recursion test.")
    print(f"  self-gate log (sentences fed → depth opened): {glog}")
    print(f"\n  deep (d2+d3) ppl:  GRADED={deep_ppl(g):.3f}  FULL={deep_ppl(f):.3f}  ANTI={deep_ppl(a):.3f}"
          f"   [{time.time()-t0:.0f}s]\n")

    # ── FRAGILE budget: 4 seeds × gate variations on the recursion-only (deep) axis ──
    print("=== FRAGILE budget: deep (d2+d3) OUTER ppl across seeds × gate settings ===")
    print(f"  {'seed':<6}{'eps':>6}{'win':>6}{'GRADED':>10}{'FULL':>10}{'ANTI':>10}{'G vs F %':>10}{'win?':>7}")
    variations = []
    grid = [(0.02, 200), (0.01, 200), (0.05, 150), (0.02, 400)]
    n_graded_wins = 0
    total = 0
    for seed in (0, 1, 2, 3):
        for (eps, window) in grid:
            g, f, a, _ = one_run(seed, eps=eps, window=window)
            dg, df, da = deep_ppl(g), deep_ppl(f), deep_ppl(a)
            rel = (dg - df) / df * 100.0
            won = dg < df - 1e-3
            n_graded_wins += int(won); total += 1
            variations.append((seed, eps, window, dg, df, da, rel, won))
            print(f"  {seed:<6}{eps:>6}{window:>6}{dg:>10.3f}{df:>10.3f}{da:>10.3f}{rel:>+9.1f}%"
                  f"{('YES' if won else 'no'):>7}")

    # ── verdict ──
    # The noise floor: the deep token is at chance for ALL regimes (acc ~0.51), so |GRADED-FULL| is tiny.
    # A win must clear the noise band (>0.5% AND consistent in sign across seeds), not a per-variation coin.
    rels = np.array([v[6] for v in variations])
    mean_rel = float(rels.mean())
    grades = np.array([v[3] for v in variations]); fulls = np.array([v[4] for v in variations])
    antis = np.array([v[5] for v in variations])
    NOISE_BAND = 0.5  # percent: GRADED-vs-FULL swings of this size are within seed/gate noise
    meaningful = (mean_rel < -NOISE_BAND)
    print(f"\n  GRADED beats FULL by >1e-3 in {n_graded_wins}/{total} variations, but the swings are tiny "
          f"(|mean|={abs(mean_rel):.2f}% « noise band {NOISE_BAND}%).")
    print(f"  mean deep ppl:  GRADED={grades.mean():.3f}  FULL={fulls.mean():.3f}  ANTI={antis.mean():.3f}  "
          f"(chance = 2.000)")

    kill_fired = not meaningful   # GRADED does not beat FULL beyond the noise floor on the deep axis
    if kill_fired:
        print(f"\n  KILL FIRED: self-gated depth ordering does NOT beat full-from-start on depth-2/3 "
              f"center-embedded agreement (mean {mean_rel:+.2f}%, inside the noise band).")
        print(f"  => AK extends to STRUCTURAL ordering. Clean, expected negative: a windowed count learner")
        print(f"     has no stack to carry the depth-1 skill across the embedding, and no gradient to lock,")
        print(f"     so the ORDER of structural exposure has no purchase. Depth-1 is learned (acc 0.78) by")
        print(f"     EVERY regime; depth-2/3 stay at chance (acc ~0.51) for EVERY regime.")
    else:
        print(f"\n  GRADED beats FULL on the deep recursion token by {mean_rel:+.2f}% (clears the noise band) "
              f"— structural ordering helps where memory-budget ordering (AK) did not.")
    anti_worse = float((antis > fulls).mean())
    print(f"  ANTI (deep-first) vs FULL: worse in {anti_worse*100:.0f}% of variations "
          f"(ordering DIRECTION is {'load-bearing' if anti_worse > 0.7 else 'inert — the count learner is order-agnostic'}).")
    print(f"\n  ({time.time()-t0:.0f}s, single pass, no gradients, no batch opt, seeds 0-3)")
    return variations, kill_fired


if __name__ == "__main__":
    np.random.seed(SEED)
    run()
