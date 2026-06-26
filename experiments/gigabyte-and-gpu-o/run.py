#!/usr/bin/env python3
"""Exp O — how fast can a Column really go? np.unique (sort) vs dense bincount (CPU) vs MLX dense (GPU/Metal).

Same char Column (order 5), same backoff model, same bpc — three backends, timed (learn + batch eval) at
100 MB and 300 MB on enwik9. Answers "really fast columns + GPU" and, honestly, WHERE the GPU actually helps.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids
from fastchar import FastChar
from densechar import DenseChar
from gpuchar import GPUChar

def timed(fn):
    t0 = time.time(); r = fn(); return r, time.time() - t0

if __name__ == "__main__":
    print("loading enwik9 ...", flush=True); ids = load_ids("enwik9")
    test = ids[-1_000_000:]
    print(f"  {len(ids):,} chars\n")
    print(f"  {'size':>7} {'backend':>16} {'learn s':>9} {'eval s':>8} {'test bpc':>10}")
    for nmb in (100, 300):
        tr = ids[:nmb * 1_000_000]
        for name, ctor in [("np.unique (sort)", FastChar), ("bincount (CPU dense)", DenseChar),
                            ("MLX (GPU/Metal)", GPUChar)]:
            m, lt = timed(lambda: ctor(5).learn(tr))
            bpc, et = timed(lambda: m.batch_logloss(test))
            print(f"  {nmb:>5}MB {name:>16} {lt:>9.2f} {et:>8.2f} {bpc:>10.4f}", flush=True)
            del m
        print(flush=True)
