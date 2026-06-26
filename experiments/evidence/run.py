#!/usr/bin/env python3
"""Exp R — evidence accumulation with decay (robust voting + a free boundary signal).

From Thousand Brains: don't recompute the belief fresh every step. Keep a LEAKY log-evidence
accumulator E_t = gamma*E_{t-1} + sum_k logp_k(.) so one noisy step can't tank the prediction, and
read its DROP as a boundary signal.

Two measured results:
  1. ROBUST VOTING under noise. Experts = char orders {2,3,4,5}. Baseline = fresh product-of-experts
     each step. Evidence = same experts via the leaky accumulator. Corrupt 0/5/10/20/30% of CONTEXT
     chars; measure next-char accuracy + bpc of both. Hypothesis: the accumulator degrades gracefully.
  2. BOUNDARY SIGNAL. Running confidence conf_t = gamma*conf_{t-1} + logP(observed char). Boundary =
     a sharp DROP / local min in conf. Score F1 vs true word boundaries (spaces); compare head-to-head
     with the forward branching-entropy RISE signal (Exp A's winner). Tolerance ±1, ±2 chars.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids
from evidence import (ExpertBank, fresh_pool_logp, evidence_logp, decode_metrics,
                      running_confidence, forward_entropy, drop_signal, rise_signal,
                      f1_at_rate, corrupt_context, V)

ORDERS = (2, 3, 4, 5)
GAMMA = 0.8
TRAIN = 10_000_000
EVAL = 200_000
NOISE = [0.0, 0.05, 0.10, 0.20, 0.30]


def main():
    rng = np.random.default_rng(0)
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN + EVAL + 1_000_000)
    train = ids[:TRAIN]
    eval_clean = ids[TRAIN:TRAIN + EVAL].copy()
    print(f"loaded {len(ids):,} chars  ({time.time()-t0:.1f}s)")
    print(f"train {len(train):,}   eval {len(eval_clean):,}   orders {ORDERS}   gamma {GAMMA}\n")

    bank = ExpertBank(ORDERS).learn(train)
    print(f"learned experts ({time.time()-t0:.1f}s)\n")

    # targets = the TRUE next-char at each predicted position (clean truth), positions t = 1..n-1
    targets = eval_clean[1:].astype(np.int64)

    # ── RESULT 1: robust voting under context noise ──
    print("=== RESULT 1: next-char accuracy / bpc under context corruption ===")
    print(f"{'noise':>7} | {'fresh acc':>9} {'fresh bpc':>9} | {'evid acc':>9} {'evid bpc':>9} | "
          f"{'acc drop f':>10} {'acc drop e':>10}")
    rows1 = []
    base_fresh_acc = base_evid_acc = None
    for noise in NOISE:
        noisy = corrupt_context(eval_clean, noise, np.random.default_rng(100 + int(noise * 1000)))
        orders_lp, _ = bank.logp_orders(noisy)          # per-order logdist over noisy context
        fresh = fresh_pool_logp(orders_lp)
        evid = evidence_logp(orders_lp, gamma=GAMMA)
        fa, fb = decode_metrics(fresh, targets)
        ea, eb = decode_metrics(evid, targets)
        if noise == 0.0:
            base_fresh_acc, base_evid_acc = fa, ea
        df = base_fresh_acc - fa
        de = base_evid_acc - ea
        rows1.append((noise, fa, fb, ea, eb, df, de))
        print(f"{noise:>7.0%} | {fa:>9.4f} {fb:>9.4f} | {ea:>9.4f} {eb:>9.4f} | "
              f"{df:>10.4f} {de:>10.4f}")
    print()

    # ── RESULT 1b: gamma sweep at a fixed noise level (the leak/robustness tradeoff) ──
    print("=== RESULT 1b: gamma sweep — clean vs 20%-noise bpc (fresh has no gamma) ===")
    noisy20 = corrupt_context(eval_clean, 0.20, np.random.default_rng(777))
    olp_clean, _ = bank.logp_orders(eval_clean)
    olp_noisy, _ = bank.logp_orders(noisy20)
    fa_c, fb_c = decode_metrics(fresh_pool_logp(olp_clean), targets)
    fa_n, fb_n = decode_metrics(fresh_pool_logp(olp_noisy), targets)
    print(f"  fresh pool          : clean bpc {fb_c:.3f} acc {fa_c:.3f} | "
          f"20%-noise bpc {fb_n:.3f} acc {fa_n:.3f}")
    print(f"  {'gamma':>6} | {'clean bpc':>9} {'clean acc':>9} | {'noisy bpc':>9} {'noisy acc':>9}")
    rows1b = []
    for g in (0.5, 0.6, 0.7, 0.8, 0.9):
        ec = evidence_logp(olp_clean, gamma=g); en = evidence_logp(olp_noisy, gamma=g)
        eac, ebc = decode_metrics(ec, targets); ean, ebn = decode_metrics(en, targets)
        rows1b.append((g, ebc, eac, ebn, ean))
        print(f"  {g:>6.1f} | {ebc:>9.3f} {eac:>9.3f} | {ebn:>9.3f} {ean:>9.3f}")
    print()

    # ── RESULT 2: boundary detection (clean eval stream) ──
    print("=== RESULT 2: boundary detection vs true word boundaries (spaces) ===")
    orders_lp, _ = bank.logp_orders(eval_clean)
    fresh = fresh_pool_logp(orders_lp)                   # use fresh pool for entropy/confidence base
    # true boundaries: a space at output position t marks a boundary. predicted positions are t=1..n-1,
    # so boundary index in the m-length signal arrays = (t-1) where eval_clean[t]==space.
    space_at = np.nonzero(eval_clean[1:] == (V - 1))[0]  # indices into the m-length signal arrays
    m = len(targets)

    conf = running_confidence(fresh, targets, gamma=GAMMA)
    drop = drop_signal(conf)
    H = forward_entropy(fresh)
    rise = rise_signal(H)
    # raw surprisal of observed char (no accumulator) as a sanity reference
    surpr = -fresh[np.arange(m), targets]

    # truth: a boundary's prediction-difficulty lands on the NEW-WORD-START char (right after the
    # space), not on the space itself (spaces are very predictable). score against new-word starts.
    truth = space_at + 1
    truth = truth[truth < m]
    rng_rate = len(truth) / m
    print(f"{m:,} positions, {len(truth):,} true word-start boundaries (density {rng_rate:.3f})\n")
    rows2 = []
    for tol in (1, 2):
        print(f"  -- tolerance ±{tol} char --")
        print(f"  {'signal':>28} | {'prec':>6} {'rec':>6} {'F1':>6}")
        for name, score in [("random", rng.random(m)),
                            ("instantaneous surprisal", surpr),
                            ("entropy RISE (Exp A winner)", rise),
                            ("confidence DROP (evidence)", drop)]:
            p, r, f = f1_at_rate(score, truth, m, tol=tol)
            rows2.append((tol, name, p, r, f))
            print(f"  {name:>28} | {p:>6.3f} {r:>6.3f} {f:>6.3f}")
        print()

    print(f"done ({time.time()-t0:.1f}s)")
    return rows1, rows2, len(space_at), m, rng_rate


if __name__ == "__main__":
    main()
