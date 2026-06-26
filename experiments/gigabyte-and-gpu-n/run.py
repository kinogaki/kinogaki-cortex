#!/usr/bin/env python3
"""Exp N — gigabyte char scaling with fast columns + batch eval.

Does prediction keep improving from megabytes to a GIGABYTE, and can we learn that fast? Pure char Column
(order 5), id-space (no Python strings), batch bpc (no per-position loop). Corpus: enwik9 (1 GB Wikipedia).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids
from fastchar import FastChar
from fastcol import FastColumn, bpc_ids   # known-good per-position reference

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("loading enwik9 (1 GB) → id-space ...", flush=True)
    t0 = time.time(); ids = load_ids("enwik9"); print(f"  {len(ids):,} chars in {time.time()-t0:.1f}s")
    test = ids[-5_000_000:]; test_eval = test[:1_000_000]

    # ── correctness: batch FastChar == per-position FastColumn (same model) ──
    sm = ids[:1_000_000]
    fc = FastChar(5).learn(sm); ref = FastColumn(5, 27).learn(sm)
    b_batch, b_ref = fc.batch_logloss(test_eval[:100_000]), bpc_ids(ref, test_eval[:100_000])
    print(f"\ncorrectness: batch {b_batch:.4f} vs per-position {b_ref:.4f} → "
          f"{'MATCH' if abs(b_batch-b_ref) < 1e-6 else f'DIFF {abs(b_batch-b_ref):.2e}'}")

    # ── scale: same Column (order 5), more data, to a gigabyte ──
    print("\n=== char order-5 scaling to 1 GB (fixed 1M-char held-out) ===")
    print(f"  {'train chars':>15}{'learn s':>9}{'MB/s':>8}{'test bpc':>10}")
    for n in (10_000_000, 100_000_000, 300_000_000, len(ids) - 5_000_000):
        tr = ids[:n]
        t0 = time.time(); m = FastChar(5).learn(tr); dt = time.time() - t0
        bpc = m.batch_logloss(test_eval)
        print(f"  {n:>15,}{dt:>9.1f}{n/1e6/max(dt,1e-9):>8.1f}{bpc:>10.4f}", flush=True)
        del m
