"""GPUChar — the char Column on the Apple GPU via MLX (Metal). Dense histograms: scatter-add to LEARN, gather
to PREDICT. Order ≤ 5 so the combined id ctx·27+tok fits int32 (27^6 = 3.87e8 < 2.1e9). Same backoff model.

MLX compiles these to Metal kernels on the GPU; unified memory means no host↔device copies of the big tables.
"""
import numpy as np
import mlx.core as mx

V = 27; ALPHA = 0.05

def _comb(ids, k):
    """int32 (context, token) combined ids for every position t in [k, n). ids: mx int32 array."""
    n = ids.size
    if k == 0:
        return mx.zeros((n,), mx.int32), ids
    ctx = mx.zeros((n - k,), mx.int32)
    for j in range(k):                                       # ctx = Σ ids[j .. ] * 27^(k-1-j)
        ctx = ctx * V + ids[j:n - k + j]
    return ctx, ids[k:]

class GPUChar:
    def __init__(self, order=5):
        self.order = order; self.tab = []; self.rs = []
    def learn(self, ids_np):
        ids = mx.array(np.ascontiguousarray(ids_np, np.int32)); self.tab = []; self.rs = []
        for k in range(self.order + 1):
            ctx, tok = _comb(ids, k)
            comb = ctx * V + tok
            h = mx.zeros((V ** (k + 1),), mx.int32)
            h = h.at[comb].add(mx.ones(comb.shape, mx.int32))
            h = h.astype(mx.float32); mx.eval(h)
            self.tab.append(h); self.rs.append(h.reshape(-1, V).sum(axis=1)); mx.eval(self.rs[-1])
        return self
    def batch_logloss(self, ids_np):
        ids = mx.array(np.ascontiguousarray(ids_np, np.int32)); n = ids.size; m = n - 1
        logp = mx.full((m,), float(np.log2(1.0 / V))); chosen = mx.zeros((m,), mx.bool_)
        for k in range(self.order, -1, -1):
            if k == 0:
                tok0 = ids[1:]
                p = (self.tab[0][tok0] + ALPHA) / (mx.sum(self.tab[0]) + ALPHA * V)
                logp = mx.where(chosen, logp, mx.log2(p)); break
            ctx, tok = _comb(ids, k); off = k - 1
            rs = self.rs[k][ctx]; cnt = self.tab[k][ctx * V + tok]
            p = (cnt + ALPHA) / (rs + ALPHA * V)
            lp_k = mx.log2(p)                                 # length n-k, aligns to logp[off : off+len]
            pad_l = mx.full((off,), -1e9); pad_r = mx.full((m - off - lp_k.size,), -1e9)
            lp_full = mx.concatenate([pad_l, lp_k, pad_r])
            seen_k = mx.concatenate([mx.zeros((off,), mx.bool_), rs > 0,
                                     mx.zeros((m - off - lp_k.size,), mx.bool_)])
            take = seen_k & (~chosen)
            logp = mx.where(take, lp_full, logp); chosen = chosen | take
        mx.eval(logp)
        return float(-mx.mean(logp).item())
