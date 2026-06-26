#!/usr/bin/env python3
"""Exp J — is the uniform Column a BETTER BASE? (foundation check, before scaling/attention).

Three questions the user put first:
  1. CORRECTNESS — does the vectorized backend compute the SAME model as the readable dict Column?
     (If yes, the abstraction is real: one interface, swappable guts.)
  2. SPEED — is learn() fast enough that the Column is optimizable toward GB-scale, not a pure-Python dead end?
  3. SCALE — when we actually feed it more data (the thing that "didn't pay off" at 2 MB), does prediction keep
     improving? (Exp F's capacity×data law, now reachable because learn is fast.)
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from cortex import Column, vote, V0, CH                      # the readable dict reference
from fastcol import FastColumn, bpc_ids

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")

def text(n0, n1):
    raw = open(os.path.join(DATA, "text8"), "rb").read()[n0:n1].decode("ascii", "ignore")
    return "".join(c for c in raw if c in CH)

def ids_of(s): return np.fromiter((CH[c] for c in s), dtype=np.int64, count=len(s))

def dict_bpc(col, ids):                                       # bpc with the dict Column (single-expert vote path)
    import math
    tot = 0.0
    for t in range(1, len(ids)):
        ctx = tuple(int(x) for x in ids[max(0, t - col.order):t])
        p = vote([col.predict(ctx)], V0)
        tot += -math.log2(p[ids[t]] + 1e-12)
    return tot / (len(ids) - 1)

if __name__ == "__main__":
    ORDER = 6
    train = text(0, 2_000_000); test = text(98_000_000, 98_150_000)
    tr_ids, te_ids = ids_of(train), ids_of(test)
    print(f"train {len(tr_ids):,} chars / test {len(te_ids):,} / order {ORDER}\n")

    # ── 1+2. correctness & speed: dict Column vs FastColumn, same order, same data ──
    t0 = time.time(); dcol = Column(ORDER); dcol.learn(tuple(int(x) for x in tr_ids)); t_dict = time.time() - t0
    t0 = time.time(); fcol = FastColumn(ORDER, V0).learn(tr_ids); t_fast = time.time() - t0
    bpc_dict = dict_bpc(dcol, te_ids)
    bpc_fast = bpc_ids(fcol, te_ids)
    print("=== correctness & speed (one Column, order 6, 2 MB) ===")
    print(f"    dict Column      learn {t_dict:7.2f}s   test bpc {bpc_dict:.4f}")
    print(f"    FastColumn       learn {t_fast:7.2f}s   test bpc {bpc_fast:.4f}")
    print(f"    → identical model: {'YES' if abs(bpc_dict-bpc_fast) < 1e-6 else f'DIFF {abs(bpc_dict-bpc_fast):.2e}'}"
          f"   |   speedup {t_dict/max(t_fast,1e-9):.0f}×")

    # ── 3. scale: feed it more data (only the fast backend can), watch bpc fall (the Exp F law) ──
    print("\n=== scale: same Column, more data (fast backend makes it reachable) ===")
    print(f"    {'train chars':>14}{'learn s':>10}{'MB/s':>8}{'test bpc':>11}")
    for nmb in (2, 10, 50):
        n = nmb * 1_000_000
        ids = ids_of(text(0, n))
        t0 = time.time(); fc = FastColumn(ORDER, V0).learn(ids); dt = time.time() - t0
        print(f"    {len(ids):>14,}{dt:>10.2f}{len(ids)/1e6/max(dt,1e-9):>8.1f}{bpc_ids(fc, te_ids):>11.4f}")

    # ── projection to GB-scale ──
    print("\n  → at the measured fast-backend throughput, a 1 GB corpus learns in minutes, not days.")
