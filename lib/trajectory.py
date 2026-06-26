"""Trajectory / change-memory primitives for Exp V (online, count-based, no backprop).

Grounded in TBP's "Trajectory Memory for Behavior Models": a behavior is a SEQUENCE OF CHANGES with no
reference frame, learned INDEPENDENT of the object and SHARED across objects ("the movements are shared,
the locations are unique"). Trajectories are DIRECTIONAL ("you can't write your signature backwards").
A keyframe feature is an AFFORDANCE — it predicts which behavior is beginning.

Everything here is a streaming counter / leaky accumulator. We build count TABLES via np.unique (one
pass, no optimisation) — the same online-equivalent counting FastChar uses — and predict by table lookup.
NO gradient descent, NO k-means/SVD/eigendecomposition.

Alphabet (from corpus.py): a..z = 0..25, space = 26 ; V = 27.
"""
import numpy as np

V = 27; SPACE = 26; ALPHA = 0.5

# ── change/delta alphabets ────────────────────────────────────────────────────
# A "change" is a category of TRANSITION between adjacent chars, not the char itself. This is the object-
# independent movement: the same vowel→consonant "move" happens in thousands of different words.

VOWELS = np.zeros(V, bool)
for c in "aeiou": VOWELS[ord(c) - 97] = True

def char_class(ids):
    """Per-char class: 0=space, 1=vowel, 2=consonant. (No digits in [a-z ] corpus.)"""
    cls = np.full(len(ids), 2, np.int64)            # default consonant
    cls[ids == SPACE] = 0
    cls[VOWELS[np.clip(ids, 0, 25)] & (ids != SPACE)] = 1
    return cls

NCLS = 3
def class_change(ids):
    """The CHANGE stream: id = from_class*NCLS + to_class for adjacent chars. Length n-1.
    This is the directional 'move' alphabet, fully content-agnostic (9 symbols)."""
    cls = char_class(ids)
    return cls[:-1] * NCLS + cls[1:]
NCHG = NCLS * NCLS

# ── generic online k-gram count table over an arbitrary symbol stream ─────────
def _ctx_tok(seq, k, base):
    n = len(seq)
    if k == 0:
        return np.zeros(n, np.int64), seq.astype(np.int64)
    w = np.lib.stride_tricks.sliding_window_view(seq, k)[:n - k].astype(np.int64)
    powers = (base ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w @ powers, seq[k:].astype(np.int64)

class CountColumn:
    """Online backoff k-gram over a symbol stream of cardinality `base`. predict P(next | recent k).
    learn() = pure counting (np.unique); equivalent to a single streaming pass of leaky-free counters."""
    def __init__(self, base, order=4):
        self.base = base; self.order = order; self.tab = []
    def learn(self, seq):
        seq = np.ascontiguousarray(seq, np.int64); self.tab = []; b = self.base
        for k in range(self.order + 1):
            if k == 0:
                cnt = np.bincount(seq, minlength=b).astype(np.float64)
                self.tab.append((None, None, None, cnt.sum(), cnt)); continue
            ctx, tok = _ctx_tok(seq, k, b)
            keys, counts = np.unique(ctx * b + tok, return_counts=True)
            counts = counts.astype(np.float64); gctx = keys // b
            change = np.concatenate([[0], np.nonzero(np.diff(gctx))[0] + 1])
            gtot = np.add.reduceat(counts, change)
            self.tab.append((keys, counts, gctx[change], gtot, None))
        return self
    def batch_logloss(self, seq):
        """Mean bits/symbol predicting seq[k:] from preceding context, backoff high→low order."""
        seq = np.ascontiguousarray(seq, np.int64); n = len(seq); b = self.base
        m = n - 1
        if m <= 0: return float("nan")
        logp = np.full(m, np.log2(1.0 / b)); resolved = np.zeros(m, bool)
        for k in range(self.order, -1, -1):
            if k == 0:
                _, _, _, tot0, cnt0 = self.tab[0]; idx = ~resolved
                if idx.any():
                    p = (cnt0[seq[1:][idx]] + ALPHA) / (tot0 + ALPHA * b)
                    logp[idx] = np.log2(p)
                resolved[:] = True; break
            if k >= len(self.tab): continue
            keys, counts, gctx, gtot, _ = self.tab[k]
            if keys is None or len(keys) == 0: continue
            ctx, tok = _ctx_tok(seq, k, b)
            off = k - 1; sl = slice(off, off + len(ctx))
            pos = np.searchsorted(gctx, ctx)
            seen = (pos < len(gctx)) & (gctx[np.minimum(pos, len(gctx) - 1)] == ctx)
            newly = seen & ~resolved[sl]
            if not newly.any(): continue
            comb = ctx * b + tok; kp = np.searchsorted(keys, comb)
            hit = (kp < len(keys)) & (keys[np.minimum(kp, len(keys) - 1)] == comb)
            cnt = np.where(hit, counts[np.minimum(kp, len(keys) - 1)], 0.0)
            tot = gtot[np.minimum(pos, len(gctx) - 1)]
            p = (cnt + ALPHA) / (tot + ALPHA * b)
            tgt = np.zeros(m, bool); tgt[sl] = newly
            logp[tgt] = np.log2(p[newly]); resolved[tgt] = True
        return float(-logp.mean())
    def acc(self, seq):
        """Top-1 next-symbol accuracy over seq (argmax of the resolved-order distribution)."""
        seq = np.ascontiguousarray(seq, np.int64); n = len(seq); b = self.base
        m = n - 1
        if m <= 0: return float("nan")
        pred = np.zeros(m, np.int64); resolved = np.zeros(m, bool)
        for k in range(self.order, -1, -1):
            if k == 0:
                _, _, _, _, cnt0 = self.tab[0]; idx = ~resolved
                pred[idx] = int(np.argmax(cnt0)); resolved[:] = True; break
            if k >= len(self.tab): continue
            keys, counts, gctx, gtot, _ = self.tab[k]
            if keys is None or len(keys) == 0: continue
            ctx, _ = _ctx_tok(seq, k, b)
            off = k - 1; sl = slice(off, off + len(ctx))
            pos = np.searchsorted(gctx, ctx)
            seen = (pos < len(gctx)) & (gctx[np.minimum(pos, len(gctx) - 1)] == ctx)
            newly = seen & ~resolved[sl]
            if not newly.any(): continue
            # argmax next symbol per context: scan the table for the best tok in each seen ctx
            base_ctx = gctx[np.minimum(pos, len(gctx) - 1)]
            best = self._argmax_for_ctx(k, ctx)
            tgt = np.zeros(m, np.int64); slots = np.arange(m)[sl]
            sel = slots[newly]; pred[sel] = best[newly]; resolved[sl] = resolved[sl] | newly
        truth = seq[1:]
        return float((pred == truth).mean())
    def _argmax_for_ctx(self, k, query_ctx):
        """Best next symbol for each queried context id (table lookup, vectorized)."""
        keys, counts, gctx, gtot, _ = self.tab[k]; b = self.base
        # group keys by ctx, take argmax count within group → map ctx -> best tok
        kc = keys // b; kt = keys % b
        change = np.concatenate([[0], np.nonzero(np.diff(kc))[0] + 1, [len(keys)]])
        best_tok = np.empty(len(change) - 1, np.int64); ctx_ids = kc[change[:-1]]
        for i in range(len(change) - 1):
            a, z = change[i], change[i + 1]
            best_tok[i] = kt[a + np.argmax(counts[a:z])]
        pos = np.searchsorted(ctx_ids, query_ctx)
        pos = np.clip(pos, 0, len(ctx_ids) - 1)
        out = np.where(ctx_ids[pos] == query_ctx, best_tok[pos], 0)
        return out

# ── word-level helpers (for transfer split + affordances) ─────────────────────
def word_arrays(ids, spans):
    """Compact per-word features from (start,end) spans: first char, last char, length-bucket."""
    starts = np.array([s for s, e in spans]); ends = np.array([e for s, e in spans])
    first = ids[starts].astype(np.int64)
    last = ids[ends - 1].astype(np.int64)
    length = (ends - starts).astype(np.int64)
    lbucket = np.clip(length, 1, 8) - 1          # 0..7
    return first, last, length, lbucket

def first_class(first):
    """Class of a word's first char: 0=vowel, 1=consonant (the affordance trigger / keyframe feature)."""
    return np.where(VOWELS[np.clip(first, 0, 25)], 0, 1).astype(np.int64)
