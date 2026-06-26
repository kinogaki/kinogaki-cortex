"""Top-down prior / ignition broadcast — the lever on GLOBAL COHERENCE (Exp T).

A higher level commits to ONE global context state G (the topic) and the lower-level char predictor
conditions its counts on G:  P(next | local-ctx, G), backing off to P(next | local-ctx) when (ctx,G) is
unseen. "Ignition" (global-workspace): G only CHANGES when one topic cluster decisively dominates the recent
window (margin threshold) — otherwise the previous G is HELD (hysteresis). This file builds G online and a
G-conditioned backoff char model, reusing the FastChar count machinery so WITH-G vs WITHOUT-G is apples-to-apples.

  TopicCoder   : content-word vocab → K topic clusters (k-means on a PPMI co-occurrence signature, vectorized).
                 .topic_of[word_id] -> cluster id.
  commit_G     : online recency-weighted topic histogram + ignition/hysteresis → committed G at every char.
  GCondChar    : G-conditioned char backoff. learn(ids, Gchar); batch_logloss(ids, Gchar). Highest order is
                 keyed by (G, ctx); lower orders fall back to the plain (ctx)-only table (== FastChar) so the
                 WITHOUT-G numbers are recovered exactly by passing G≡0 and one cluster.
"""
import numpy as np

V = 27; SPACE = 26; ALPHA = 0.05


# ───────────────────────────── topic clustering ─────────────────────────────

class TopicCoder:
    """Cluster content words into K topics by co-occurrence. Each word gets a PPMI signature over the top-D
    context words (the words it appears near); k-means on those signatures → a topic id per word. Stopwords
    (the ~`n_stop` most frequent words) and rare words map to topic -1 (ignored when committing G)."""
    def __init__(self, K=128, n_stop=100, top_context=400, min_count=5, iters=12, seed=0):
        self.K = K; self.n_stop = n_stop; self.top_context = top_context
        self.min_count = min_count; self.iters = iters; self.seed = seed
        self.topic_of = None                                  # word_id -> cluster (or -1)

    def fit(self, wids, vocab):
        rng = np.random.default_rng(self.seed)
        freq = np.bincount(wids, minlength=vocab).astype(np.float64)
        order = np.argsort(freq)[::-1]
        stop = set(order[:self.n_stop].tolist())              # drop most-frequent = stopwords
        # candidate content words: frequent enough, not a stopword
        content_mask = (freq >= self.min_count)
        for s in stop: content_mask[s] = False
        content_words = np.nonzero(content_mask)[0]
        # context vocabulary = top-D frequent NON-stop words (the columns of the co-occurrence matrix)
        ctx_pool = [w for w in order if w not in stop]
        ctx_words = np.array(ctx_pool[:self.top_context], dtype=np.int64)
        ctx_index = -np.ones(vocab, np.int64); ctx_index[ctx_words] = np.arange(len(ctx_words))
        cw_index = -np.ones(vocab, np.int64); cw_index[content_words] = np.arange(len(content_words))

        # co-occurrence counts within a ±W word window (vectorized over offsets)
        W = 8; n = len(wids)
        cooc = np.zeros((len(content_words), len(ctx_words)), np.float64)
        ci = cw_index[wids]                                   # content-row per position (-1 if not content)
        for d in range(1, W + 1):
            for a, b in ((wids[:-d], wids[d:]), (wids[d:], wids[:-d])):
                ra = ci[: len(a)] if a is wids[:-d] else ci[d:]
                cb = ctx_index[b]
                ok = (ra >= 0) & (cb >= 0)
                if ok.any():
                    np.add.at(cooc, (ra[ok], cb[ok]), 1.0)

        # PPMI signature, L2-normalized → cosine-friendly k-means
        row = cooc.sum(1, keepdims=True); col = cooc.sum(0, keepdims=True); tot = cooc.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((cooc * tot) / (row * col + 1e-9) + 1e-9)
        sig = np.maximum(pmi, 0.0)
        nrm = np.linalg.norm(sig, axis=1, keepdims=True); nrm[nrm == 0] = 1.0; sig /= nrm

        # spherical k-means (cosine == dot on unit vectors)
        keep = sig.any(1)                                     # drop all-zero-signature words from clustering
        X = sig[keep]; kept_words = content_words[keep]
        K = min(self.K, len(X))
        cent = X[rng.choice(len(X), size=K, replace=False)].copy()
        for _ in range(self.iters):
            assign = np.argmax(X @ cent.T, axis=1)
            for k in range(K):
                m = assign == k
                if m.any():
                    c = X[m].sum(0); nc = np.linalg.norm(c)
                    if nc > 0: cent[k] = c / nc
        assign = np.argmax(X @ cent.T, axis=1)
        self.topic_of = -np.ones(vocab, np.int64)
        self.topic_of[kept_words] = assign
        self.K = K
        return self


# ───────────────────────────── online ignition ─────────────────────────────

def commit_G(topic_seq, K, halflife=40.0, margin=0.18):
    """Online committed global topic G over a sequence of per-content-word topic ids (-1 = stopword/unknown).
    Recency-weighted histogram (exponential decay) over recent topics; IGNITION = switch G only when the
    leading cluster's mass-fraction beats the current G's by > `margin` (else HOLD = hysteresis).
    Returns G per element of topic_seq (the committed topic AFTER seeing that word)."""
    decay = 0.5 ** (1.0 / halflife)
    hist = np.zeros(K, np.float64)
    G = 0; out = np.empty(len(topic_seq), np.int64)
    for i, t in enumerate(topic_seq):
        hist *= decay
        if t >= 0:
            hist[t] += 1.0
        s = hist.sum()
        if s > 0:
            lead = int(np.argmax(hist))
            if lead != G and (hist[lead] - hist[G]) / s > margin:
                G = lead                                       # ignition: decisive winner commits
        out[i] = G
    return out


# ───────────────────────── G-conditioned char model ─────────────────────────

def _ctx_tok(ids, k):
    n = len(ids)
    if k == 0:
        return np.zeros(n, np.int64), ids.astype(np.int64)
    w = np.lib.stride_tricks.sliding_window_view(ids, k)[:n - k].astype(np.int64)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w @ powers, ids[k:].astype(np.int64)


class GCondChar:
    """G-conditioned char backoff. The top `g_orders` orders are keyed by (G, ctx); below that it backs off to
    the plain (ctx)-only tables — identical to FastChar. So `use_g=False` (or one cluster) gives the exact
    no-top-down baseline, and any improvement is purely the committed topic G."""
    def __init__(self, order=6, g_orders=(6, 5, 4, 3), K=1):
        self.order = order; self.g_orders = set(g_orders); self.K = K
        self.plain = []; self.gtab = {}                       # gtab[k] = (keys, counts, gctx, gtot) keyed (G*V^k+ctx)

    def learn(self, ids, Gchar=None):
        ids = np.ascontiguousarray(ids, np.int64); self.plain = []; self.gtab = {}
        for k in range(self.order + 1):
            if k == 0:
                cnt = np.bincount(ids, minlength=V).astype(np.float64)
                self.plain.append((None, None, None, cnt.sum(), cnt)); continue
            ctx, tok = _ctx_tok(ids, k)
            keys, counts = np.unique(ctx * V + tok, return_counts=True)
            counts = counts.astype(np.float64); gctx = keys // V
            change = np.concatenate([[0], np.nonzero(np.diff(gctx))[0] + 1])
            gtot = np.add.reduceat(counts, change)
            self.plain.append((keys, counts, gctx[change], gtot, None))
            # G-conditioned table for this order
            if Gchar is not None and k in self.g_orders:
                g = Gchar[k:].astype(np.int64)                # G committed at the predicted position t=k..n-1
                gkey = (g * (V ** k) + ctx) * V + tok          # fold G into the context
                keys2, counts2 = np.unique(gkey, return_counts=True)
                counts2 = counts2.astype(np.float64); gctx2 = keys2 // V
                ch2 = np.concatenate([[0], np.nonzero(np.diff(gctx2))[0] + 1])
                gtot2 = np.add.reduceat(counts2, ch2)
                self.gtab[k] = (keys2, counts2, gctx2[ch2], gtot2)
        return self

    def batch_logloss(self, ids, Gchar=None, use_g=True):
        """Mean bpc, batched. With use_g: try (G,ctx) at g_orders first, then fall back to plain (ctx) backoff.
        Returns (overall_bpc, per_position_bits) so callers can slice post-boundary spikes."""
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); m = n - 1
        logp = np.full(m, np.log2(1.0 / V)); resolved = np.zeros(m, bool)

        def fill(keys, counts, gctx, gtot, ctx, tok, comb, sl, mask):
            pos = np.searchsorted(gctx, ctx)
            seen = (pos < len(gctx)) & (gctx[np.minimum(pos, len(gctx) - 1)] == ctx)
            newly = seen & mask
            if not newly.any(): return
            kp = np.searchsorted(keys, comb)
            hit = (kp < len(keys)) & (keys[np.minimum(kp, len(keys) - 1)] == comb)
            cnt = np.where(hit, counts[np.minimum(kp, len(keys) - 1)], 0.0)
            tot = gtot[np.minimum(pos, len(gctx) - 1)]
            p = (cnt + ALPHA) / (tot + ALPHA * V)
            tgt = np.zeros(m, bool); tgt[sl] = newly
            logp[tgt] = np.log2(p[newly]); resolved[tgt] = True

        for k in range(self.order, -1, -1):
            if k == 0:
                _, _, _, tot0, cnt0 = self.plain[0]; idx = ~resolved
                p = (cnt0[ids[1:][idx]] + ALPHA) / (tot0 + ALPHA * V)
                logp[idx] = np.log2(p); resolved[:] = True; break
            ctx, tok = _ctx_tok(ids, k); off = k - 1
            sl = slice(off, off + len(ctx)); comb = ctx * V + tok
            # 1) G-conditioned attempt at this order
            if use_g and Gchar is not None and k in self.gtab:
                g = Gchar[k:].astype(np.int64)
                gctx_full = g * (V ** k) + ctx; gcomb = gctx_full * V + tok
                keys2, counts2, gc2, gt2 = self.gtab[k]
                fill(keys2, counts2, gc2, gt2, gctx_full, tok, gcomb, sl, ~resolved[sl])
            # 2) plain (ctx-only) backoff at this order
            keys, counts, gctx, gtot, _ = self.plain[k]
            if keys is None or len(keys) == 0: continue
            fill(keys, counts, gctx, gtot, ctx, tok, comb, sl, ~resolved[sl])
        return float(-logp.mean()), -logp
