#!/usr/bin/env python3
"""Exp AG — a count-native SYSTEM 2: the deliberate pass.

Everything before this was System 1 — the char-order experts vote in parallel and a leader pops out
(the fast, local, prepotent argmax). The cog-sci convergence (Evans/Stanovich, Botvinick conflict
monitoring, Engle working-memory capacity, Oberauer's focus, Kahneman default-interventionist, Sloman
suppress-not-erase) says System 2 is not a smarter model — it is a tiny, capacity-bounded, SERIAL
workspace, triggered by CONFLICT, that overrides System 1 only when System 1 is wrong. We build it from
counts (no gradient, no batch optimization) and test the Engle signature:

  the override must IMPROVE accuracy where System 1's prepotent (local) answer is WRONG, and MATCH
  System 1 where there is no conflict (no harm on easy cases).

Pieces (all in lib/system2 + lib/confidence):
  System 1 = product-of-experts over SHORT orders {2,3} → vote activations, leader = local argmax.
  Dual trigger = calibrated confidence c (NARS f·c) < theta  OR  Botvinick conflict (top-two product)
    > kappa  ⇒ deploy System 2 (else ship the fast leader).
  Deliberate pass = a capacity-k≈4 FOCUS, a serial race over leaky accumulators seeded by System-1's
    votes + a top-down GOAL bias = the wider LONG-order {5,6} opinion, inhibition-of-return so the loop
    advances, a step BUDGET, suppress-not-erase floor on the default; commit the deliberate winner only
    if it beats the default.

THE PROBE. A CONFLICT subset = positions where the short-order leader and the long-order leader DISAGREE
and both contexts are seen (the prepotent local bet fights the broader context). A NO-CONFLICT subset =
where they agree. We report System-1-only vs the gated System-2 model on each subset separately (Engle:
better on conflict, equal on no-conflict). Plus: gate-fire rate, and graceful fallback at budget 0
(must equal System 1).

Char level on text8. Train on a prefix, eval on a held-out suffix; single causal pass; fixed seed 0.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids
from confidence import CountTruth, unigram_log, order_logdist, _ctx_ids
from system2 import (system1_votes, top_two, conflict_energy, gated_predict,
                     context_confidence, goal_field)

SHORT = (2, 3)          # System 1: the fast, local, prepotent voter
LONG = (5, 6)           # System 2's top-down GOAL: the wider context the fast voter under-weights
DECIDER = 3             # the short order whose NARS f·c is the calibrated confidence c
TRAIN = 12_000_000
EVAL = 300_000
SEED = 0

THETA = 0.35            # confidence floor: below it, deliberate
KAPPA = 0.40            # within-System-1 conflict ceiling: above it, deliberate
KAPPA_X = 0.10          # cross-subsystem conflict ceiling (fast leader vs goal leader disagree, both on)
K_FOCUS = 4             # working-memory capacity (Cowan/Oberauer ~4)
BUDGET = 6              # serial step budget
IOR = 0.7               # inhibition-of-return
GOAL_GAIN = 2.0         # top-down drive strength
DECOUPLE = 0.5          # cognitive-decoupling decay on the prepotent System-1 seed
FLOOR = 0.02            # suppress-not-erase floor


def build_lds(tables, ctx_q, orders, uni, m):
    """Per-order backed-off log-dists aligned to the same m positions (pad the first k-1 with unigram)."""
    lds = []
    for k in orders:
        ld_k, _ = order_logdist(tables[k], ctx_q[k], uni)     # length n-k, aligns to t=k..n-1
        full = np.tile(uni, (m, 1))
        full[k - 1:] = ld_k
        lds.append(full)
    return lds


def acc_on(pred, targets, mask):
    sel = mask
    if sel.sum() == 0:
        return float("nan"), 0
    return float((pred[sel] == targets[sel]).mean()), int(sel.sum())


def main():
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN + EVAL + 1_000_000)
    train = ids[:TRAIN].astype(np.int64)
    ev = ids[TRAIN:TRAIN + EVAL].astype(np.int64)
    targets = ev[1:]
    m = len(targets)
    print(f"loaded {len(ids):,} chars  train {len(train):,}  eval {len(ev):,}  "
          f"short {SHORT} long {LONG}  ({time.time()-t0:.1f}s)")

    all_orders = sorted(set(SHORT) | set(LONG) | {DECIDER})
    tables = {k: CountTruth(k).online_pass(train) for k in all_orders}
    uni = unigram_log(train)
    print(f"online (w+,w-) pass done for orders {all_orders}  ({time.time()-t0:.1f}s)")

    ctx_q = {k: _ctx_ids(ev, k) for k in all_orders}
    short_lds = build_lds(tables, ctx_q, SHORT, uni, m)
    long_lds = build_lds(tables, ctx_q, LONG, uni, m)

    # System 1: fast vote field + prepotent leader; its top-two for conflict.
    a = system1_votes(short_lds)
    i1, a1, i2, a2 = top_two(a)
    conflict = conflict_energy(a1, a2)
    s1_pred = i1                                       # System-1-only = the local argmax

    # System 2 inputs: top-down goal = long-order opinion; calibrated confidence = decider's f·c.
    goal = goal_field(long_lds)
    cq_dec = np.zeros(m, np.int64); cq_dec[:] = -1
    cq_dec[DECIDER - 1:] = ctx_q[DECIDER]
    c_conf = context_confidence(tables[DECIDER], cq_dec)

    # ── the conflict probe: where short-leader vs long-leader disagree, both seen ──
    long_leader = goal.argmax(1)
    seen_short = tables[max(SHORT)].lookup(ctx_q[max(SHORT)])
    seen_long = tables[min(LONG)].lookup(ctx_q[min(LONG)])
    seen_s = np.zeros(m, bool); seen_s[max(SHORT) - 1:] = seen_short >= 0
    seen_l = np.zeros(m, bool); seen_l[min(LONG) - 1:] = seen_long >= 0
    both_seen = seen_s & seen_l
    conflict_set = both_seen & (i1 != long_leader)
    noconf_set = both_seen & (i1 == long_leader)
    print(f"probe: both-seen {both_seen.mean():.3f}  conflict {conflict_set.mean():.3f}  "
          f"no-conflict {noconf_set.mean():.3f}")
    # the prepotent-wrong slice (the Engle target): conflict positions where S1's local argmax is WRONG
    s1_wrong = conflict_set & (s1_pred != targets)
    print(f"  of conflict positions, S1-wrong fraction {float((s1_pred[conflict_set]!=targets[conflict_set]).mean()):.3f}")

    # ── the gated System-2 model (the deliberate race) ──
    pred, fired, override = gated_predict(a, goal, c_conf, THETA, KAPPA,
                                          k=K_FOCUS, budget=BUDGET, ior=IOR,
                                          goal_gain=GOAL_GAIN, floor=FLOOR,
                                          decouple=DECOUPLE, kappa_x=KAPPA_X)
    # the gated DEFERRAL: same trigger, but the operator is trivial — on every fired position, defer to
    # the reflective (goal) answer. The minimal count-native System 2: think = consult the wider context.
    defer = s1_pred.copy()
    defer[fired] = long_leader[fired]
    print(f"gate fired on {fired.mean():.3f} of positions; "
          f"the deliberate race overrode the default on {override.mean():.3f}  ({time.time()-t0:.1f}s)")

    # ── the Engle signature: accuracy by subset, System-1 vs the two System-2 variants ──
    print("\n=== Engle signature: accuracy by subset ===")
    print(f"{'subset':>14} | {'n':>8} | {'Sys-1':>7} | {'S2 defer':>8} | {'Δdefer':>7} | {'S2 race':>7} | {'Δrace':>7}")
    rows = []
    for name, mask in [("ALL", np.ones(m, bool)),
                       ("CONFLICT", conflict_set),
                       ("NO-CONFLICT", noconf_set)]:
        s1a, n = acc_on(s1_pred, targets, mask)
        dfa, _ = acc_on(defer, targets, mask)
        s2a, _ = acc_on(pred, targets, mask)
        rows.append((name, n, s1a, dfa, dfa - s1a, s2a, s2a - s1a))
        print(f"{name:>14} | {n:>8,} | {s1a:>7.4f} | {dfa:>8.4f} | {dfa-s1a:>+7.4f} | "
              f"{s2a:>7.4f} | {s2a-s1a:>+7.4f}")

    # ── does the elaborate deliberate RACE earn its keep over the trivial DEFERRAL? (same gate) ──
    agree_s2 = float((defer[fired] == pred[fired]).mean()) if fired.any() else float("nan")
    print(f"\ndeliberate race vs trivial deferral (SAME gate): race agrees with deferral on "
          f"{agree_s2:.3f} of fired positions.")
    diff = fired & (pred != defer)
    if diff.sum():
        print(f"  where they DIFFER ({int(diff.sum()):,} positions): "
              f"race {float((pred[diff]==targets[diff]).mean()):.4f}  "
              f"vs deferral {float((defer[diff]==targets[diff]).mean()):.4f}  "
              f"→ {'race earns it' if (pred[diff]==targets[diff]).mean() > (defer[diff]==targets[diff]).mean() else 'deferral wins; the workspace machinery does NOT help'}")

    # ── graceful fallback: budget 0 must equal System 1 ──
    pred0, fired0, override0 = gated_predict(a, goal, c_conf, THETA, KAPPA,
                                             k=K_FOCUS, budget=0, ior=IOR,
                                             goal_gain=GOAL_GAIN, floor=FLOOR,
                                             decouple=DECOUPLE, kappa_x=KAPPA_X)
    same = bool((pred0 == s1_pred).all())
    print(f"\nbudget=0 fallback equals System-1 exactly: {same}  "
          f"(overrides at budget 0: {int(override0.sum())})")

    # ── where the override acted: did it fix S1-wrong conflict cases? ──
    fixed = int(((s1_pred != targets) & (pred == targets) & conflict_set).sum())
    broke = int(((s1_pred == targets) & (pred != targets) & conflict_set).sum())
    print(f"on conflict subset: overrides FIXED {fixed}, BROKE {broke}  "
          f"(net {fixed - broke})")
    nc_fixed = int(((s1_pred != targets) & (pred == targets) & noconf_set).sum())
    nc_broke = int(((s1_pred == targets) & (pred != targets) & noconf_set).sum())
    print(f"on no-conflict subset: FIXED {nc_fixed}, BROKE {nc_broke}  (net {nc_fixed - nc_broke})")

    print(f"\ntotal {time.time()-t0:.1f}s")
    return rows


if __name__ == "__main__":
    main()
