"""chunkvote.py — the chunk lexicon as ONE EXPERT in the calibrated pool (close AU's bpc gap).

AU's ChunkLexicon WON the splice axis (Isbilen sub-unit interference, count-native) but LOST held-out
bpc by +0.20: its ChunkAgent REPLACED the calibrated backoff n-gram with raw chunk-completion, and a
raw completion-by-whole-chunk throws away the backoff's per-sub-transition calibration. That is a
read-out bug, not a lexicon bug.

The cognitive frame: the parser (chunk lexicon) and the sequence-predictor (char Columns) are NOT
rivals — they are two experts the cortex consults at once. The lexicon should HELP where it is
confident (mid-committed-chunk: "I'm inside 'thedog', the next char is almost surely the one that
finishes it") and SHUT UP elsewhere, letting the backoff carry the rest. That is exactly what a
log-linear / geometric-mean pool (cortex.vote) does: each expert contributes its log-distribution,
abstainers contribute nothing, and the consensus is the calibrated intersection of views.

So this module does ONE thing: at each position it gathers
  (a) the char-Column predict-dicts (verbatim, as harness.CortexAgent does), and
  (b) the chunk-completion next-char COUNT-dict from AU's ChunkLexicon (lib/chunklex, READ not
      modified) — the longest trailing partial-chunk's continuations, scaled to a chunk-expert weight,
and pools ALL of them through the SAME calibrated geometric-mean math as cortex.vote — never replacing
the backoff, only adding the chunk vote as one more expert.

The FRAGILE dial: `chunk_w` = the weight of the chunk expert in the pool (0 = pure n-gram; >0 = blend).
Weighting in a geometric-mean pool = scaling that expert's log-contribution; we generalize cortex.vote
to accept per-expert weights and reuse its calibration (the same ALPHA smoothing, the same /Σw mean).

ONLINE single pass (counts only). NO gradient / k-means / SVD / backprop. BOUNDED (the lexicon caps +
LFU-evicts itself; the char Columns are a fixed bounded band). Reuses cortex.Column / cortex.vote and
the AU ChunkLexicon verbatim.
"""
import numpy as np

from cortex import Column, vote, V0, CH, ALPHA
from chunklex import ChunkLexicon


def vote_weighted(dicts, weights, vocab):
    """cortex.vote, generalized to a WEIGHTED geometric mean (log-linear pool). Identical calibration:
    each expert -> ALPHA-smoothed distribution -> its log added with weight w_i; the pool is divided by
    Σw_i (not n) so it stays a calibrated mean. weights[i]==0 or dicts[i] falsy => the expert abstains
    (contributes nothing), exactly like cortex.vote's None handling. With all weights 1 this IS vote."""
    logp = np.zeros(vocab)
    wsum = 0.0
    for d, w in zip(dicts, weights):
        if not d or w <= 0.0:
            continue
        wsum += w
        p = np.full(vocab, ALPHA)
        tot = ALPHA * vocab
        for tok, c in d.items():
            if tok < vocab:
                p[tok] += c
                tot += c
        logp += w * np.log(p / tot)
    if wsum == 0.0:
        return np.full(vocab, 1.0 / vocab)
    logp /= wsum
    z = logp - logp.max()
    e = np.exp(z)
    return e / e.sum()


class NgramAgent:
    """(1) The number to beat: a plain backoff n-gram char predictor — the same band of replicated
    Columns harness.CortexAgent uses, pooled by cortex.vote verbatim. Online single pass. `.K`/`.dist`
    adapt to lib/metrics unchanged. This is the calibrated backoff AU's ChunkAgent threw away."""

    def __init__(self, orders=(0, 1, 2, 3, 4, 5, 6)):
        self.cols = [Column(o) for o in orders]
        self.maxord = max(orders)
        self.V = V0
        self.K = 64
        self.buf = []

    def observe(self, ids):
        ids = list(int(x) for x in ids)
        if not ids:
            return
        start = len(self.buf)
        self.buf.extend(ids)
        for col in self.cols:
            tab = col.tab
            for t in range(start, len(self.buf)):
                nx = self.buf[t]
                for k in range(min(col.order, t) + 1):
                    d = tab[k].setdefault(tuple(self.buf[t - k:t]), {})
                    d[nx] = d.get(nx, 0) + 1
        if len(self.buf) > self.K + self.maxord:
            self.buf = self.buf[-(self.K + self.maxord):]

    def _col_dicts(self, ctx):
        return [c.predict(tuple(ctx)) for c in self.cols]

    def _dist_ids(self, ctx):
        return vote(self._col_dicts(ctx), self.V)

    def dist(self, suffix):
        ids = [CH[c] for c in suffix if c in CH][-self.K:]
        return self._dist_ids(ids)


def chunk_completion_dict(lex, by_prefix, ids, max_len):
    """AU's chunk-completion read-out, returned as a COUNT-dict (NOT yet a distribution) so the pool
    smooths/calibrates it like any other expert. Over the trailing PARTIAL chunk (longest suffix that
    is a known chunk PREFIX), look up which chars extend it, weighted by committed chunk weight. This
    is the lexicon's confidence: peaky mid-committed-chunk, empty (abstain) when no chunk extends."""
    for L in range(min(max_len - 1, len(ids)), -1, -1):
        cand = tuple(ids[len(ids) - L:]) if L else ()
        ext = by_prefix.get(cand)
        if ext:
            return {tok: w for tok, w in ext.items() if tok < V0 and w > 0}
    return None


class ChunkOnlyAgent:
    """(2) Reproduce AU's +0.20 loser: chunk-completion blended ONLY with a low-order char backoff
    floor (lam mix), the lexicon REPLACING the calibrated high-order n-gram. This is AU's ChunkAgent
    shape (lib/chunklex.ChunkAgent), rebuilt here over the same lexicon so all three agents share one
    trained lexicon and one held-out slice. Lower bpc is better; AU found this +0.20 worse than (1)."""

    def __init__(self, lex, order=2, alpha=0.05, lam=0.6):
        self.lex = lex
        self.V = V0
        self.K = 64
        self.order = order
        self.alpha = alpha
        self.lam = lam
        self.ctx = [dict() for _ in range(order + 1)]
        self.by_prefix = build_prefix_index(lex)

    def observe(self, ids):
        s = list(int(x) for x in ids)
        for t in range(len(s)):
            nx = s[t]
            for k in range(min(self.order, t) + 1):
                d = self.ctx[k].setdefault(tuple(s[t - k:t]), {})
                d[nx] = d.get(nx, 0) + 1

    def _backoff(self, ids):
        for k in range(min(self.order, len(ids)), -1, -1):
            d = self.ctx[k].get(tuple(ids[len(ids) - k:]) if k else ())
            if d:
                p = np.full(self.V, self.alpha)
                tot = self.alpha * self.V
                for tok, c in d.items():
                    p[tok] += c
                    tot += c
                return p / tot
        return np.full(self.V, 1.0 / self.V)

    def _dist_ids(self, ids):
        back = self._backoff(ids)
        comp = chunk_completion_dict(self.lex, self.by_prefix, ids, self.lex.max_len)
        if comp is None:
            return back
        cp = np.full(self.V, 0.0)
        tot = 0.0
        for tok, w in comp.items():
            cp[tok] += w
            tot += w
        cp = cp / tot
        p = self.lam * cp + (1 - self.lam) * back
        return p / p.sum()

    def dist(self, suffix):
        ids = [CH[c] for c in suffix if c in CH][-self.K:]
        return self._dist_ids(ids)


class ChunkVoteAgent:
    """(3) THE FIX: the chunk-completion dist as ONE EXPERT in the calibrated geometric-mean pool
    ALONGSIDE the char Columns — never replacing them. At each position: gather the char-Column
    predict-dicts (the calibrated backoff, kept whole) AND the chunk-completion count-dict, then pool
    ALL of them through vote_weighted (cortex.vote's math, with a weight on the chunk expert).

    chunk_w is the FRAGILE dial: 0 == the plain n-gram (1); large == the lexicon dominates. The bet:
    a modest weight lets the lexicon sharpen mid-chunk while the backoff carries the rest -> bpc <=
    the n-gram's. Char columns each carry weight 1 (matching cortex.vote); the chunk expert carries
    chunk_w. Online single pass; the lexicon is trained separately and frozen for held-out scoring."""

    def __init__(self, lex, orders=(0, 1, 2, 3, 4, 5, 6), chunk_w=1.0):
        self.lex = lex
        self.cols = [Column(o) for o in orders]
        self.maxord = max(orders)
        self.V = V0
        self.K = 64
        self.chunk_w = chunk_w
        self.buf = []
        self.by_prefix = build_prefix_index(lex)

    def observe(self, ids):
        ids = list(int(x) for x in ids)
        if not ids:
            return
        start = len(self.buf)
        self.buf.extend(ids)
        for col in self.cols:
            tab = col.tab
            for t in range(start, len(self.buf)):
                nx = self.buf[t]
                for k in range(min(col.order, t) + 1):
                    d = tab[k].setdefault(tuple(self.buf[t - k:t]), {})
                    d[nx] = d.get(nx, 0) + 1
        if len(self.buf) > self.K + self.maxord:
            self.buf = self.buf[-(self.K + self.maxord):]

    def _dist_ids(self, ctx):
        col_dicts = [c.predict(tuple(ctx)) for c in self.cols]
        comp = chunk_completion_dict(self.lex, self.by_prefix, list(ctx), self.lex.max_len)
        dicts = col_dicts + [comp]
        weights = [1.0] * len(col_dicts) + [self.chunk_w]
        return vote_weighted(dicts, weights, self.V)

    def dist(self, suffix):
        ids = [CH[c] for c in suffix if c in CH][-self.K:]
        return self._dist_ids(ids)


def build_prefix_index(lex):
    """Index a ChunkLexicon's chunks by prefix -> {next_char: summed committed weight}, for fast
    completion look-up (AU's ChunkAgent._index, factored out so all agents share one index)."""
    by_prefix = {}
    for ch, w in lex.w.items():
        if w <= 0:
            continue
        for L in range(len(ch)):
            d = by_prefix.setdefault(ch[:L], {})
            d[ch[L]] = d.get(ch[L], 0.0) + w
    return by_prefix
