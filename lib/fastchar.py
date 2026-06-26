"""FastChar — vectorized char backoff with BATCH evaluation. The Column, built for scale.

learn(): per order k, count k-gram→next via np.unique (same as FastColumn). Also keep per-context totals.
batch_bpc(): evaluate ALL positions at once (no Python per-position loop — that was Exp K's bottleneck).
For each order high→low: encode every position's context, searchsorted into the table, fill the not-yet-resolved
positions with their smoothed true-token prob. Fully vectorized → millions of positions per second.

Char order ≤ 6 (combined id ctx·27+tok stays in int64). For the GPU/int32 path use order ≤ 5 (fits int32).
"""
import numpy as np

V = 27; ALPHA = 0.05

def _ctx_tok(ids, k):
    """For every t in [k, n): order-k context id and next token (vectorized)."""
    n = len(ids)
    if k == 0:
        return np.zeros(n, np.int64), ids.astype(np.int64)
    w = np.lib.stride_tricks.sliding_window_view(ids, k)[:n - k].astype(np.int64)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w @ powers, ids[k:].astype(np.int64)

class FastChar:
    def __init__(self, order=5):
        self.order = order; self.tab = []                    # tab[k] = (keys, counts, gctx, gtot)
    def learn(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); self.tab = []
        for k in range(self.order + 1):
            if k == 0:
                cnt = np.bincount(ids, minlength=V).astype(np.float64)
                self.tab.append((None, None, None, cnt.sum(), cnt)); continue
            ctx, tok = _ctx_tok(ids, k)
            keys, counts = np.unique(ctx * V + tok, return_counts=True)   # sorted by (ctx,tok)
            counts = counts.astype(np.float64)
            gctx = keys // V                                  # context per key (sorted)
            change = np.concatenate([[0], np.nonzero(np.diff(gctx))[0] + 1])
            gtot = np.add.reduceat(counts, change)            # total count per context
            self.tab.append((keys, counts, gctx[change], gtot, None))
        return self
    def batch_logloss(self, ids):
        """Mean bits-per-char over `ids`, batched. Backoff to highest seen order, add-α smoothing."""
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids)
        m = n - 1                                             # predict positions 1..n-1
        logp = np.full(m, np.log2(1.0 / V)); resolved = np.zeros(m, bool)
        for k in range(self.order, -1, -1):
            if k == 0:
                _, _, _, tot0, cnt0 = self.tab[0]
                idx = ~resolved
                p = (cnt0[ids[1:][idx]] + ALPHA) / (tot0 + ALPHA * V)
                logp[idx] = np.log2(p); resolved[:] = True; break
            keys, counts, gctx, gtot, _ = self.tab[k]
            if keys is None or len(keys) == 0: continue
            # context for predicting position t = ids[t-k:t]; t in [k, n)
            ctx, tok = _ctx_tok(ids, k)                       # length n-k, aligns to positions t=k..n-1
            off = k - 1                                       # position t maps to logp index t-1; ctx index t-k
            sl = slice(off, off + len(ctx))                  # logp[off:] corresponds to these positions
            pos = np.searchsorted(gctx, ctx)
            seen = (pos < len(gctx)) & (gctx[np.minimum(pos, len(gctx) - 1)] == ctx)
            newly_local = seen & ~resolved[sl]
            if not newly_local.any(): continue
            comb = ctx * V + tok
            kp = np.searchsorted(keys, comb)
            hit = (kp < len(keys)) & (keys[np.minimum(kp, len(keys) - 1)] == comb)
            cnt = np.where(hit, counts[np.minimum(kp, len(keys) - 1)], 0.0)
            tot = gtot[np.minimum(pos, len(gctx) - 1)]
            p = (cnt + ALPHA) / (tot + ALPHA * V)
            tgt = np.zeros(m, bool); tgt[sl] = newly_local
            logp[tgt] = np.log2(p[newly_local]); resolved[tgt] = True
        return float(-logp.mean())
