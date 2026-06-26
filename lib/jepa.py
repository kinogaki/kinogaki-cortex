"""JEPA-style prediction in REPRESENTATION space, for the count world — Exp U. ONLINE, NO backprop.

LeCun's JEPA thesis: don't reconstruct the raw input; mask part of it and predict the masked part's abstract
REPRESENTATION from the visible context; add sparsity/structure so the representation doesn't collapse. We
translate that into the count-cortex's vocabulary with a HARD online-only rule — single streaming pass,
"learn while it lives", NO gradient descent, NO k-means, NO SVD/eigendecomposition, NO matrix factorization.
Everything below is order-independent accumulation (counting) plus an online leader-clustering pass. The
vectorized builders (np.unique / bincount) are just batched implementations of order-independent accumulation —
the counts are identical to a token-at-a-time online update; nothing here iterates-to-convergence or backprops.

  - REPRESENTATION (the latent) = a word's concept-CLUSTER id, built by ONLINE LEADER CLUSTERING:
      * SIGNATURE: each word keeps a running count vector over a small FIXED feature space — its co-occurring
        words HASHED into D dims (a fixed random sign-hash = a random projection done by accumulation, no
        factorization). Accumulating these counts is order-independent → online.
      * CLUSTERS: stream the words once in first-appearance (stream) order; the first time a word is "ripe"
        (enough accumulated evidence), assign it to the nearest existing PROTOTYPE by cosine (a prototype is
        the running MEAN of its members' signatures). If best cosine < a threshold, SPAWN a new prototype
        (capped at Cmax). Update the winner's running mean incrementally. One pass, no re-assignment.
    Because the latent is GROUNDED in fixed counts (the target cluster is computed by counting, never co-trained
    against the predictor), it CANNOT collapse the way a gradient JEPA encoder can — rep-space prediction with
    NO collapse-prevention machinery (no stop-grad, no VICReg, no EMA teacher), and online.

  - MASKED PREDICTION (JEPA core) = hide a target word; build BIDIRECTIONAL context from words within ±W
    (offset-keyed counts so left/right and distance carry signal); predict either the masked WORD (input space)
    or its CLUSTER (rep space / JEPA). Two heads share the SAME counted context evidence.

  - INTER-LAYER SPARSITY = represent the context as a SPARSE top-k code over CLUSTERS (only the k most-active
    context clusters carry signal) and predict the masked cluster from it. Sweep k.
"""
import numpy as np


# ── ONLINE SIGNATURES: hashed running context vectors (a random projection by accumulation) ──

def _hash_table(N, D, seed=0):
    """Fixed random map: context-word dense-id -> (D-bucket, +-1 sign). A random sign-projection applied by
    accumulation (no SVD/factorization). Deterministic given seed; order-independent."""
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, D, size=N)
    sign = rng.integers(0, 2, size=N).astype(np.float64) * 2 - 1
    return bucket, sign


def online_signatures(seq, N, D=64, window=5, seed=0):
    """ONE pass over the stream -> sig[word]=(N,D) running hashed-signed context-count vector; cnt[word]=evidence.

    For every position and every neighbour within ±window, credit sign[nb]*idf[nb] into sig[word, bucket[nb]] —
    the per-word "running count vector over a small fixed feature space" the spec asks for, with each neighbour
    DOWN-WEIGHTED by its inverse log frequency (idf = 1/log(2+running_count)). The IDF is itself an online
    running count, so this stays a single streaming accumulation (order-independent) = the canonical online
    learner. Without it, ubiquitous function words ('the','of') dominate every signature and the leader
    clusterer collapses to one mega-cluster; IDF lets the DISCRIMINATIVE neighbours shape the latent.
    Vectorized per offset (identical to a token-at-a-time update). seq is dense top-ids (-1=OOV, skipped)."""
    bucket, sign = _hash_table(N, D, seed)
    freq = np.bincount(seq[seq >= 0], minlength=N).astype(np.float64)   # running per-word counts (online)
    idf = sign / np.log(2.0 + freq)                                     # signed, frequency-discounted weight
    sig = np.zeros(N * D, np.float64)
    cnt = np.zeros(N, np.int64)
    for g in range(1, window + 1):
        for a, b in ((seq[:-g], seq[g:]), (seq[g:], seq[:-g])):   # both directions = bidirectional
            m = (a >= 0) & (b >= 0)
            wa, wb = a[m], b[m]
            np.add.at(sig, wa * D + bucket[wb], idf[wb])
            np.add.at(cnt, wa, 1)
    return sig.reshape(N, D), cnt


# ── ONLINE LEADER CLUSTERING: single pass, nearest-prototype-or-spawn, running-mean prototypes ──

def leader_cluster(sig, cnt, order, min_evidence=40, thresh=0.55, Cmax=400):
    """Assign each word a cluster by ONLINE LEADER CLUSTERING, single pass in `order` (stream order).

    A word with cnt<min_evidence stays unclustered (-1) — not ripe yet (the online "decide when ripe" stance).
    For a ripe word: cosine-match its L2-normalized signature to existing prototypes (running means); if best
    cosine>=thresh join+fold into that running mean, else SPAWN (up to Cmax; past the cap, force the argmax).
    No iteration, no re-assignment — a true single pass. Returns clu:(N,) in [0,C) or -1, and C."""
    N, D = sig.shape
    unit = sig / np.maximum(np.linalg.norm(sig, axis=1, keepdims=True), 1e-9)
    proto = np.zeros((Cmax, D), np.float64)          # running SUM of member unit-sigs (mean ∝ sum)
    C = 0
    clu = -np.ones(N, np.int64)
    for w in order:
        if cnt[w] < min_evidence:
            continue
        u = unit[w]
        if C == 0:
            proto[0] = u; C = 1; clu[w] = 0
            continue
        dots = proto[:C] @ u
        cos = dots / np.maximum(np.linalg.norm(proto[:C], axis=1), 1e-9)   # cosine(u, mean_c)
        best = int(cos.argmax()); bcos = float(cos[best])
        if bcos >= thresh or C >= Cmax:
            proto[best] += u; clu[w] = best
        else:
            proto[C] = u; clu[w] = C; C += 1
    used = np.unique(clu[clu >= 0])
    relabel = -np.ones(Cmax, np.int64); relabel[used] = np.arange(len(used))
    clu = np.where(clu >= 0, relabel[clu], -1)
    return clu, len(used)


# ── the offset-keyed BIDIRECTIONAL context model — VECTORIZED counting, SPARSE pooling ──

def _build_offset_counts(seq, target, W, T):
    """For each offset o in [-W..W]\\{0}: counts[o] is a dict ctxword -> (np.array tgt_ids, np.array counts).
    `target` gives the target id for each position (the word id, or its cluster id). T = #target classes.
    Built with np.unique over packed (ctxword, target) keys per offset — vectorized, order-independent."""
    n = len(seq)
    out = {}
    for o in range(-W, W + 1):
        if o == 0:
            continue
        if o > 0:
            ctx = seq[:-o]; tg = target[o:]
        else:
            ctx = seq[-o:]; tg = target[:o]
        m = (ctx >= 0) & (tg >= 0)
        ctx, tg = ctx[m], tg[m]
        if ctx.size == 0:
            out[o] = {}
            continue
        key = ctx.astype(np.int64) * T + tg.astype(np.int64)
        uk, uc = np.unique(key, return_counts=True)
        ucw = uk // T; utg = (uk % T).astype(np.int64)
        d = {}
        # split into per-ctxword runs (uk sorted -> ucw sorted)
        edges = np.nonzero(np.diff(ucw))[0] + 1
        starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(uk)]])
        for s, e in zip(starts, ends):
            d[int(ucw[s])] = (utg[s:e], uc[s:e].astype(np.float64))
        out[o] = d
    return out


class JepaCounts:
    """P(target | one context word at a signed offset), BIDIRECTIONAL over ±W. Two heads share the SAME context
    evidence: tok (target=WORD, input space) and clu (target=CLUSTER, rep space / JEPA). Built vectorized;
    pooled SPARSELY (log-linear over the union of the active experts' supports, like cortex.vote_sparse)."""
    def __init__(self, W, N, C, alpha=0.1):
        self.W, self.N, self.C, self.alpha = W, N, C, alpha
        self.tok = None; self.clu = None

    def fit(self, seq, clu_of):
        tgt_clu = np.where(seq >= 0, clu_of[np.clip(seq, 0, self.N - 1)], -1)   # cluster of each position
        self.tok = _build_offset_counts(seq, seq, self.W, self.N)
        self.clu = _build_offset_counts(seq, tgt_clu, self.W, self.C)
        return self

    @staticmethod
    def _pool_sparse(experts, size, alpha):
        """experts: list of (tgt_ids, counts). Log-linear pool over the UNION of supports; smoothing denom is
        per-expert-constant across candidates so it cancels -> only supports matter. Returns dense prob (size)."""
        if not experts:
            return None
        # union of candidate ids
        cand = np.unique(np.concatenate([e[0] for e in experts]))
        logp = np.zeros(len(cand))
        for ids, cs in experts:
            tot = cs.sum() + alpha * size
            base = np.log(alpha / tot)
            row = np.full(len(cand), base)
            # add counts where this expert has them
            pos = np.searchsorted(cand, ids)
            row[pos] = np.log((cs + alpha) / tot)
            logp += row
        logp -= logp.max()
        e = np.exp(logp)
        p = np.zeros(size); p[cand] = e / e.sum()
        return p

    def _experts(self, store, ctx):
        out = []
        for o, cw in ctx:
            d = store.get(o)
            if d:
                e = d.get(cw)
                if e is not None:
                    out.append(e)
        return out

    def predict_tok(self, ctx):
        return self._pool_sparse(self._experts(self.tok, ctx), self.N, self.alpha)

    def predict_clu(self, ctx):
        return self._pool_sparse(self._experts(self.clu, ctx), self.C, self.alpha)


def context_at(seq, t, W):
    """Visible bidirectional context at t: list of (signed_offset, ctxword), OOV dropped."""
    out = []
    for j in range(max(0, t - W), min(len(seq), t + W + 1)):
        if j == t:
            continue
        cw = seq[j]
        if cw >= 0:
            out.append((j - t, cw))
    return out


# ── inter-layer sparsity: a sparse top-k code over CLUSTERS for the context ──

class SparseCluCounts:
    """Predict the masked cluster from a SPARSE top-k cluster code of the context. The context window's visible
    words map to clusters; keep only the k most-frequent context clusters (the active SDR-like set). Evidence is
    counted per ACTIVE-CLUSTER -> {target_cluster}. Sweeping k (1..dense) tests 'sparsity reduces interference /
    improves generalization' (watch the train-test gap). Built vectorized over precomputed context codes."""
    def __init__(self, C, alpha=0.1):
        self.C, self.alpha = C, alpha
        self.tab = None     # active_cluster -> (tgt_ids, counts)

    def fit_codes(self, active_flat, tgt_flat):
        """active_flat, tgt_flat: 1-D arrays — one (active_cluster, target_cluster) pair per context-code entry,
        already expanded by the caller (vectorized). Counted by np.unique."""
        if len(active_flat) == 0:
            self.tab = {}; return self
        key = active_flat.astype(np.int64) * self.C + tgt_flat.astype(np.int64)
        uk, uc = np.unique(key, return_counts=True)
        ua = uk // self.C; ut = (uk % self.C).astype(np.int64)
        edges = np.nonzero(np.diff(ua))[0] + 1
        starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(uk)]])
        tab = {}
        for s, e in zip(starts, ends):
            tab[int(ua[s])] = (ut[s:e], uc[s:e].astype(np.float64))
        self.tab = tab
        return self

    def predict(self, active_clusters):
        experts = [self.tab[a] for a in active_clusters if a in self.tab]
        return JepaCounts._pool_sparse(experts, self.C, self.alpha)


def topk_code(ctx_clusters, C, k):
    """The sparse active set: the k most-frequent context clusters (or all if k is None / k>=#distinct)."""
    if len(ctx_clusters) == 0:
        return np.empty(0, np.int64)
    cnt = np.bincount(ctx_clusters, minlength=C)
    nz = int((cnt > 0).sum())
    if k is None or k >= nz:
        return np.nonzero(cnt)[0]
    return np.argsort(cnt)[::-1][:k]


# ── corruption: replace a fraction of context words with random in-vocab words ──

def corrupt(seq, frac, N, seed=0):
    """Copy of seq with `frac` of non-OOV positions replaced by random top-words — perturbs CONTEXT to measure
    robustness of input- vs rep-space prediction (JEPA claims rep-space degrades more gracefully)."""
    rng = np.random.default_rng(seed)
    out = seq.copy()
    valid = np.nonzero(seq >= 0)[0]
    k = int(len(valid) * frac)
    if k == 0:
        return out
    pick = rng.choice(valid, size=k, replace=False)
    out[pick] = rng.integers(0, N, size=k)
    return out
