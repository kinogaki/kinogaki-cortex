#!/usr/bin/env python3
"""Experiment F — the DYNAMICS: how does prediction scale with (a) training data and (b) cortex capacity?

Before extending the architecture we characterize it. Two axes on text8 (100MB clean English, a-z+space):
  DATA:     train on 1 / 3 / 10 / 30 / 90 MB  (1x → ~90x)
  CAPACITY: char n-gram ORDER K = 2..5  (how much context the cortex holds = its "size")
Metric: bits-per-char on a FIXED held-out slice. The grid shows where we're data-limited (a column keeps
dropping with more data) vs capacity-limited (a column flattens — the cortex is too small to use more data),
and where higher capacity needs more data to pay off (high order is worse when data-starved → sparsity).

Fully vectorized count model (np.bincount), add-α smoothing, fixed-order (no backoff) so capacity = K cleanly.
"""
import os, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")
ALPHA = 0.05; B = 27

def load_syms():
    raw = np.frombuffer(open(os.path.join(DATA, "text8"), "rb").read(), dtype=np.uint8)
    m = np.full(256, 26, np.int64)                      # space/other -> 26
    for i in range(26): m[ord('a') + i] = i             # a-z -> 0..25
    return m[raw]

def ctx_pair_ids(s, K):
    """context id (base-27, K chars) and (context*27+next) pair id, for positions t in [K, len)."""
    n = len(s)
    cid = np.zeros(n - K, dtype=np.int64)
    for i in range(K):                                  # context chars s[t-K+i]
        cid = cid * B + s[i:n - K + i]
    nxt = s[K:n]
    return cid, cid * B + nxt, nxt

def train_counts(s, K):
    _, pair, _ = ctx_pair_ids(s, K)
    counts = np.bincount(pair, minlength=B ** (K + 1)).reshape(B ** K, B).astype(np.float64)
    return counts, counts.sum(1)

def eval_bpc(counts, rowsum, s_test, K):
    cid, _, nxt = ctx_pair_ids(s_test, K)
    num = counts[cid, nxt] + ALPHA
    den = rowsum[cid] + ALPHA * B
    return float(np.mean(-np.log2(num / den)))

if __name__ == "__main__":
    s = load_syms()
    print(f"text8 loaded: {len(s):,} chars")
    test = s[98_000_000:100_000_000]                    # fixed 2M-char held-out (not in any train slice)
    MB = 1_000_000
    data_sizes = [1, 3, 10, 30, 90]
    orders = [2, 3, 4, 5]
    print(f"held-out: {len(test):,} chars | data sizes (MB): {data_sizes} | orders K: {orders}\n")

    grid = {}
    t0 = time.time()
    for K in orders:
        for mb in data_sizes:
            try:
                counts, rowsum = train_counts(s[:mb * MB], K)
                bpc = eval_bpc(counts, rowsum, test, K)
                grid[(K, mb)] = bpc
                del counts, rowsum
            except MemoryError:
                grid[(K, mb)] = None
        print(f"  order K={K} done ({time.time()-t0:.0f}s)")

    print("\n=== bits-per-char  (rows = order/capacity, cols = train MB) — lower is better ===")
    print(f"  {'K\\MB':<6}" + "".join(f"{mb:>9}" for mb in data_sizes))
    for K in orders:
        print(f"  {K:<6}" + "".join((f"{grid[(K,mb)]:>9.3f}" if grid[(K,mb)] is not None else f"{'OOM':>9}") for mb in data_sizes))

    # diagnostics: data-limited vs capacity-limited
    print("\n=== dynamics ===")
    for K in orders:
        row = [grid[(K, mb)] for mb in data_sizes if grid[(K, mb)] is not None]
        if len(row) >= 2:
            improved = row[0] - row[-1]          # positive = bpc went down with more data
            last_step = row[-2] - row[-1]        # positive = still improving on the last 3x-data step
            print(f"  K={K}: bpc {row[0]:.3f}→{row[-1]:.3f} (improved {improved:.3f} over {data_sizes[0]}→{data_sizes[-1]}MB); "
                  f"last 3x step −{last_step:.3f}  "
                  f"{'[STILL data-hungry]' if last_step > 0.01 else '[saturated — capacity-limited]'}")
    for mb in data_sizes:
        col = [(K, grid[(K, mb)]) for K in orders if grid[(K, mb)] is not None]
        if len(col) >= 2:
            best = min(col, key=lambda x: x[1])
            print(f"  {mb}MB data: best order K={best[0]} ({best[1]:.3f}); "
                  f"higher K {'helps' if best[0]==orders[-1] else 'stops helping past K='+str(best[0])+' (data-starved sparsity)'}")
