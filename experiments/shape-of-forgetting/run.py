#!/usr/bin/env python3
"""Experiment AI — POWER-LAW memory and budgeted eviction (Anderson & Schooler / ACT-R).

The substrate weights memory by RAW COUNT (frequency only) or an EXPONENTIAL EMA (geometric recency). ACT-R's
rational analysis says the right curve is a POWER LAW: base-level activation B = ln(Σ_k age_k^(−d)), d≈0.5 —
frequency adds terms, recency weights them, decay d is the leak; need-odds ∝ exp(B). We build a count model whose
per-context activation is the incremental ACT-R approximation of B and use exp(B) for BOTH prediction weighting
AND eviction, then test on the axes a memory budget makes visible.

  1. SPACING EFFECT — retention of a motif seen SPACED vs MASSED at equal total count. Power-law B should keep
     spaced more accessible; an exponential EMA cannot (it only sees the last use).
  2. EVICTION UNDER A FIXED BUDGET — cap each order's table; compare evict-lowest-B (power-law) vs LRU vs LFU vs
     EMA on held-out bpc at several cap sizes. Hypothesis: lowest-B preserves the long, sparse, repeated tail
     best → best bpc-under-budget.
  3. DOWNSTREAM BPC — power-law-WEIGHTED prediction vs plain highest-order counts, UNBOUNDED (sanity check).

HARD RULES: single streaming pass; no gradient descent; no batch optimization. Activation is a per-entry O(1)
recurrence; eviction is reservoir-sampled lowest-score. Fixed seed.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lib.powerlaw import CountModel, spacing_probe, load_ids, shift_eviction

DATA = os.path.join(HERE, "..", "..", "data")
SEED = 0
np.random.seed(SEED)

K = 5
D = 0.5                     # ACT-R decay
N_TRAIN = 200_000           # chars (single streaming pass)
N_EVAL = 20_000             # disjoint held-out tail
CAPS = [500, 1500, 4000]    # per-order table caps (the budget)
POLICIES = ["powerlaw", "lru", "lfu", "ema"]


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def main():
    t0 = time.time()

    # ── Test 1: spacing effect (pure accumulator, no eviction) ──────────────────────────────────
    banner("TEST 1 — spacing effect: same count, MASSED vs SPACED (final retrieval weight)")
    sp = spacing_probe(d=D, count=20, stream_len=20000, seed=SEED)
    print(f"  power-law exp(B):  massed={sp['powerlaw_massed']:.5f}  spaced={sp['powerlaw_spaced']:.5f}  "
          f"spaced/massed = {sp['powerlaw_ratio']:.2f}×")
    print(f"  exponential EMA :  massed={sp['ema_massed']:.5e}  spaced={sp['ema_spaced']:.5e}  "
          f"spaced/massed = {sp['ema_ratio']:.2f}×")
    print(f"  → spacing benefit (spaced more accessible than massed) requires ratio > 1.")

    # ── load corpus for the budget + downstream tests ───────────────────────────────────────────
    ids = load_ids(os.path.join(DATA, "darwin.txt"), N_TRAIN + N_EVAL)
    train, ev = ids[:N_TRAIN], ids[N_TRAIN:N_TRAIN + N_EVAL]
    print(f"\n  corpus darwin.txt: {len(train):,} train / {len(ev):,} held-out chars, orders 1–{K}, d={D}")

    # ── Test 2: eviction quality under a fixed memory budget ────────────────────────────────────
    banner("TEST 2 — eviction quality under a fixed memory budget (held-out bpc, lower=better)")
    print(f"  {'cap/order':>10} | " + " | ".join(f"{p:>9}" for p in POLICIES) + " |   best")
    print("  " + "-" * 64)
    budget_rows = []
    for cap in CAPS:
        row = {}
        for pol in POLICIES:
            m = CountModel(K=K, cap=cap, d=D, policy=pol, weighted=False)
            m.train_stream(train)
            row[pol] = m.eval_bpc(ev)
        best = min(row, key=row.get)
        budget_rows.append((cap, row, best))
        print(f"  {cap:>10,} | " + " | ".join(f"{row[p]:9.4f}" for p in POLICIES) +
              f" |  {best} ({row[best]:.4f})")

    # ── Test 2b: eviction under DOMAIN SHIFT (A then B flood; bpc back on A) ─────────────────────
    banner("TEST 2b — eviction under DOMAIN SHIFT: stream A→B under a cap, bpc back on A (lower=better)")
    a = load_ids(os.path.join(DATA, "darwin.txt"), N_TRAIN + N_EVAL)
    b = load_ids(os.path.join(DATA, "shakespeare.txt"), N_TRAIN)
    a_tr, a_ev = a[:N_TRAIN], a[N_TRAIN:N_TRAIN + N_EVAL]
    print(f"  stream darwin({len(a_tr):,}) → shakespeare({len(b):,}) flood, eval on darwin held-out")
    shift_rows = []
    for cap in CAPS:
        row = shift_eviction(a_tr, b, a_ev, K=K, cap=cap, d=D)
        best = min(row, key=row.get)
        shift_rows.append((cap, row, best))
        print(f"  {cap:>10,} | " + " | ".join(f"{row[p]:9.4f}" for p in POLICIES) +
              f" |  {best} ({row[best]:.4f})")

    # ── Test 3: downstream bpc — power-law-weighted vs plain counts, UNBOUNDED ───────────────────
    banner("TEST 3 — downstream bpc, UNBOUNDED memory: power-law-WEIGHTED vs plain highest-order counts")
    m_plain = CountModel(K=K, cap=0, d=D, policy="none", weighted=False); m_plain.train_stream(train)
    bpc_plain = m_plain.eval_bpc(ev)
    m_wpl = CountModel(K=K, cap=0, d=D, policy="none", weighted=True); m_wpl.train_stream(train)
    bpc_wpl = m_wpl.eval_bpc(ev)
    print(f"  plain highest-order backoff : {bpc_plain:.4f} bpc   (table {m_plain.size():,})")
    print(f"  power-law-weighted blend    : {bpc_wpl:.4f} bpc   (table {m_wpl.size():,})")
    print(f"  Δ = {bpc_wpl - bpc_plain:+.4f} bpc  ({'weighting helps' if bpc_wpl < bpc_plain else 'no gain'})")

    print(f"\n  total wall {time.time()-t0:.0f}s")

    # machine-readable dump for RESULTS.md
    print("\n[DUMP]")
    print("spacing", sp)
    for cap, row, best in budget_rows:
        print("budget", cap, {p: round(row[p], 4) for p in POLICIES}, "best", best)
    for cap, row, best in shift_rows:
        print("shift", cap, {p: round(float(row[p]), 4) for p in POLICIES}, "best", best)
    print("downstream", {"plain": round(bpc_plain, 4), "weighted": round(bpc_wpl, 4)})


if __name__ == "__main__":
    main()
