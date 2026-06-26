"""DenseChar — the char Column as DENSE histograms (np.bincount), not sorted-unique. For bounded order (≤5,
so 27^(k+1) bins fit) this is O(n) learn (no sort) + O(1) gather eval — much faster than the np.unique path,
and the exact representation the GPU wants (scatter to learn, gather to predict). Same backoff model, same bpc.
"""
import numpy as np

V = 27; ALPHA = 0.05

def ctx_tok(ids, k):
    n = len(ids)
    if k == 0:
        return np.zeros(n, np.int64), ids.astype(np.int64)
    w = np.lib.stride_tricks.sliding_window_view(ids, k)[:n - k].astype(np.int64)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w @ powers, ids[k:].astype(np.int64)

class DenseChar:
    def __init__(self, order=5):
        self.order = order; self.tab = []; self.rs = []
    def learn(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); self.tab = []; self.rs = []
        for k in range(self.order + 1):
            ctx, tok = ctx_tok(ids, k)
            t = np.bincount(ctx * V + tok, minlength=V ** (k + 1)).astype(np.float32)
            self.tab.append(t); self.rs.append(t.reshape(-1, V).sum(1))
        return self
    def batch_logloss(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); m = n - 1
        logp = np.full(m, np.log2(1.0 / V)); resolved = np.zeros(m, bool)
        for k in range(self.order, -1, -1):
            if k == 0:
                idx = ~resolved
                p = (self.tab[0][ids[1:][idx]] + ALPHA) / (self.tab[0].sum() + ALPHA * V)
                logp[idx] = np.log2(p); break
            ctx, tok = ctx_tok(ids, k); off = k - 1; sl = slice(off, off + len(ctx))
            rs = self.rs[k][ctx]; cnt = self.tab[k][ctx * V + tok]
            p = (cnt + ALPHA) / (rs + ALPHA * V)
            newly = (rs > 0) & ~resolved[sl]
            tgt = np.zeros(m, bool); tgt[sl] = newly
            logp[tgt] = np.log2(p[newly]); resolved[tgt] = True
            if resolved.all(): break
        return float(-logp.mean())
