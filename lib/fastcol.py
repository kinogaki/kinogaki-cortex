"""Vectorized backend for the Column — SAME interface, SAME numbers, ~100× faster learn.

The dict-based `Column` (lib/cortex.py) is the readable reference. It is also slow: learn() is a Python loop
of tuple-keyed dict increments, O(n·order) with per-op hashing — minutes for 2 MB, hopeless for GBs. The
question for scaling: is the Column *design* optimizable, or fundamentally slow? Answer: optimizable. The count
table is just "for each context, a histogram over next tokens" — which numpy builds in one `np.unique` over the
whole stream. Encode every (context, next-token) as one int64, unique-with-counts, store sorted; predict() is a
`searchsorted` slice. The Column ABSTRACTION is unchanged (learn(seq) / predict→counts); only the backend is.

This is the foundation claim: the uniform `Column` is not just clean, it scales — same wiring, fast guts.
"""
import math
import numpy as np

ALPHA = 0.05

def _encode(ids, k, V):
    """For every position t in [k, n): the order-k context id and the next token. Vectorized, no Python loop."""
    n = len(ids)
    if k == 0:
        return np.zeros(n - 0, np.int64)[:n], ids            # empty context → id 0 for all
    w = np.lib.stride_tricks.sliding_window_view(ids, k)[:n - k]   # window i = ids[i:i+k] = ctx for t=i+k
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w.astype(np.int64) @ powers, ids[k:]

class FastColumn:
    """Same role as Column: an order-`order` backoff associative predictor. Vectorized count tables."""
    def __init__(self, order, V):
        self.order = order; self.V = V; self.tab = []         # tab[k] = (vals_sorted, counts), val = ctx_id*V + token
    def learn(self, ids):
        ids = np.ascontiguousarray(ids, dtype=np.int64); self.tab = []
        for k in range(self.order + 1):
            if k == 0:
                self.tab.append((None, np.bincount(ids, minlength=self.V).astype(np.float64))); continue
            ctx_ids, toks = _encode(ids, k, self.V)
            vals, counts = np.unique(ctx_ids * self.V + toks, return_counts=True)
            self.tab.append((vals, counts.astype(np.float64)))
        return self
    def _smooth(self, hist):
        s = hist.sum()
        return None if s <= 0 else (hist + ALPHA) / (s + ALPHA * self.V)
    def predict_dense(self, ctx):
        """ctx = tuple/list of recent token ids. Backoff to the highest order whose context was seen → smoothed
        next-token distribution. Identical to dict-Column + single-expert vote()."""
        for k in range(min(self.order, len(ctx)), -1, -1):
            if k == 0:
                p = self._smooth(self.tab[0][1])
                if p is not None: return p
                continue
            cid = 0
            for x in ctx[-k:]: cid = cid * self.V + int(x)
            vals, counts = self.tab[k]
            lo = np.searchsorted(vals, cid * self.V); hi = np.searchsorted(vals, cid * self.V + self.V)
            if hi > lo:
                hist = np.zeros(self.V); hist[vals[lo:hi] % self.V] = counts[lo:hi]
                return (hist + ALPHA) / (hist.sum() + ALPHA * self.V)
        return np.full(self.V, 1.0 / self.V)

def bpc_ids(model, ids):
    tot = 0.0
    for t in range(1, len(ids)):
        p = model.predict_dense(ids[max(0, t - model.order):t])
        tot += -math.log2(p[ids[t]] + 1e-12)
    return tot / (len(ids) - 1)
