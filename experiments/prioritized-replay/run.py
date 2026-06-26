#!/usr/bin/env python3
"""Exp BG / M24 — INVERSE-COUNT, SURPRISE-PRIORITIZED spindle replay.

The cortex doesn't replay its night uniformly. Sharp-wave ripples preferentially reinstate the
weak, the recent, the surprising — Schapiro 2018: spindles protect *infrequent* words from being
forgotten. Exp AA's sleep pass replays its buffer UNIFORMLY: at a fixed offline effort, almost all
of it lands on the head (high-frequency contexts the model already nails), and the rare tail — where
a count model is weakest and where consolidation should matter most — gets nearly nothing.

THE BET. Hold the offline pass fixed (same prune + distill, same recent buffer) and the offline
BUDGET fixed (a hard cap B on reinforcement increments), and change ONLY the replay distribution:
  uniform   (AA)  : spend B frequency-proportionally       -> head soaks it up
  invcount  (M24) : spend B ∝ 1/(1+count)                  -> the rare tail is protected
  surprise  (M24) : spend B ∝ -log2 P(next|ctx)            -> re-fire worst-predicted events
M24's prediction: invcount/surprise beat uniform on RARE-CONTEXT bpc at equal budget, without
harming common-context more than they help rare. KILL if either fails.

Corpus: text8 (spec asks text8 60M; this is a SEE-WHERE-WE-ARE pass so 8 MB train, 2 MB held-out
tail). Char backoff order 6. One online streaming pass builds the substrate; one bounded sleep cycle
per policy. Fixed seed. NO gradient / k-means / SVD: every deposit is a +1 count; total deposited ==
B (bounded). The ">=10 variations" M24 asks for = the policy x budget x alpha sweep below.
"""
import os, sys, time, functools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics, replay
from consolidate import learn_tables, score, memory_size
from replay import replay_sleep

print = functools.partial(print, flush=True)

TRAIN_BYTES = 8_000_000
TEST_BYTES  = 2_000_000
ORDER       = 6
BUFFER      = 3_000_000
RARE_THRESH = 20
SEED        = 0

# fixed prune/distill so the ONLY variable is the replay policy
PRUNE = dict(min_ctx=4, tail_mass=0.999, distill_tau=0.02)


def fmt(s):
    return (f"bpc {s['bpc']:.4f}  acc {s['acc']*100:5.2f}%  | "
            f"rare {s['bpc_rare']:.3f}  common {s['bpc_common']:.3f}")


def main():
    t0 = time.time()
    ids = corpus.load_ids("text8", nbytes=TRAIN_BYTES + TEST_BYTES)
    train = np.ascontiguousarray(ids[:TRAIN_BYTES])
    test = np.ascontiguousarray(ids[TRAIN_BYTES:TRAIN_BYTES + TEST_BYTES])
    buffer = np.ascontiguousarray(train[-BUFFER:])
    print(f"text8: train {len(train):,}  held-out {len(test):,}  buffer {len(buffer):,} "
          f"(load {time.time()-t0:.1f}s)")

    t1 = time.time()
    tab0 = learn_tables(train, ORDER)
    base = score(tab0, test, rare_ctx_thresh=RARE_THRESH)
    m0 = memory_size(tab0)
    print(f"online learn order-{ORDER} in {time.time()-t1:.1f}s | mem {m0['entries']:,} entries")
    print(f"  BEFORE SLEEP : {fmt(base)}   (rare-frac {base['rare_frac']*100:.1f}%)\n")

    BUDGETS = [50_000, 200_000, 800_000]
    POLICIES = ["uniform", "invcount", "surprise"]

    print(f"{'policy':>9} {'budget':>9} {'bpc':>8} {'rare':>7} {'common':>7} "
          f"{'mem':>10}  {'Δrare':>7} {'Δcom':>7}  deposited")
    print(f"{'(base)':>9} {'-':>9} {base['bpc']:8.4f} {base['bpc_rare']:7.3f} "
          f"{base['bpc_common']:7.3f} {m0['entries']:10,}")

    results = {}
    for B in BUDGETS:
        for pol in POLICIES:
            tab, st = replay_sleep(tab0, buffer, ORDER, policy=pol, budget=B,
                                   protect_floor=True, alpha_inv=1.0, seed=SEED, **PRUNE)
            s = score(tab, test, rare_ctx_thresh=RARE_THRESH)
            m = memory_size(tab)
            results[(B, pol)] = (s, m, st)
            dr = s['bpc_rare'] - base['bpc_rare']
            dc = s['bpc_common'] - base['bpc_common']
            print(f"{pol:>9} {B:9,} {s['bpc']:8.4f} {s['bpc_rare']:7.3f} {s['bpc_common']:7.3f} "
                  f"{m['entries']:10,}  {dr:+7.3f} {dc:+7.3f}  {st['deposited']:,}")

    # ── alpha sweep on invcount (steepness of the 1/count protection) at the mid budget ──
    print(f"\n=== invcount alpha sweep (budget 200k) — steepness of rare-tail protection ===")
    print(f"{'alpha':>6} {'bpc':>8} {'rare':>7} {'common':>7}  {'Δrare':>7} {'Δcom':>7}")
    for a in [0.5, 1.0, 1.5, 2.0]:
        tab, st = replay_sleep(tab0, buffer, ORDER, policy="invcount", budget=200_000,
                               protect_floor=True, alpha_inv=a, seed=SEED, **PRUNE)
        s = score(tab, test, rare_ctx_thresh=RARE_THRESH)
        print(f"{a:6.1f} {s['bpc']:8.4f} {s['bpc_rare']:7.3f} {s['bpc_common']:7.3f}  "
              f"{s['bpc_rare']-base['bpc_rare']:+7.3f} {s['bpc_common']-base['bpc_common']:+7.3f}")

    # ── protect_floor ablation (the spindle-saves-the-infrequent-word knob) ──
    print(f"\n=== protect_floor ablation (invcount, budget 200k) ===")
    for pf in [True, False]:
        tab, st = replay_sleep(tab0, buffer, ORDER, policy="invcount", budget=200_000,
                               protect_floor=pf, alpha_inv=1.0, seed=SEED, **PRUNE)
        s = score(tab, test, rare_ctx_thresh=RARE_THRESH)
        print(f"  protect_floor={str(pf):>5}: rare {s['bpc_rare']:.3f} "
              f"(Δ{s['bpc_rare']-base['bpc_rare']:+.3f})  common {s['bpc_common']:.3f}  "
              f"pruned_ctx {st['pruned_ctx']:,}")

    # ── VERDICT: at each budget, does a policy beat uniform on RARE bpc, without harming
    #    common more than it helps rare? (the M24 kill-condition is stated on INVCOUNT.) ──
    MARGIN = 5e-3   # don't count a noise-level tie as a win (fragile-budget honesty)
    def beats_uniform(B, pol):
        su = results[(B, "uniform")][0]; sp = results[(B, pol)][0]
        help_rare = su['bpc_rare'] - sp['bpc_rare']          # +ve = pol's rare bpc is lower (better)
        hurt_common = sp['bpc_common'] - su['bpc_common']    # +ve = pol's common bpc is higher (worse)
        rare_win = help_rare > MARGIN
        overtrade = hurt_common > max(help_rare, 0) + 1e-4
        return rare_win and not overtrade, help_rare, hurt_common

    print(f"\n=== M24 kill-check: does the policy beat UNIFORM on RARE-context bpc at equal budget? ===")
    inv_fired = []
    for pol in ["invcount", "surprise"]:
        print(f"  -- {pol} vs uniform --")
        for B in BUDGETS:
            win, hr, hc = beats_uniform(B, pol)
            if pol == "invcount":
                inv_fired.append(not win)
            print(f"     budget {B:>9,}: rare help {hr:+.3f}  common hurt {hc:+.3f} -> "
                  f"{'WIN' if win else 'no-beat'}")
    kill = all(inv_fired)   # M24's kill is stated on invcount specifically
    print(f"\n  KILL-CONDITION (invcount) {'FIRED (negative)' if kill else 'did NOT fire'}: "
          f"invcount {'never beat' if kill else 'beat'} uniform on rare-context bpc at equal budget "
          f"(margin {MARGIN}).")
    print(f"\ntotal {time.time()-t0:.1f}s")
    return results, base, kill


if __name__ == "__main__":
    main()
