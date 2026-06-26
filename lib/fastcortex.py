"""FastCortex — the uniform-component cortex (Exp I) on the vectorized backend (Exp J), so it scales to big data.

Same wiring as cortex.Cortex: a char Column predicts chars; word Levels predict the current word from previous
words and hand a top-down char prior down; experts pooled by geometric mean (calibrated). Only the guts are
vectorized — char counts via FastColumn (np.unique), word n-grams via FastNgram (packed-context np.unique).

LEVELS (the thing we scale): level 0 = chars; each added word level extends the word-context span by one;
an optional top level = a recency/topic cache (needs a long eval window to matter). "K levels" = char + (K-1)
higher bands. The question this enables: does DEPTH pay off once the data is no longer starved (Exp J showed it
was, at 2 MB)?
"""
import numpy as np
from fastcol import FastColumn
from cortex import vote_sparse, char_prior, V0, CH

class FastNgram:
    """One word Level of a single order — vectorized. Context (k word-ids) packed into one int64 key; the
    (key, next-word) histogram built by np.unique; predict = searchsorted slice → sparse {wid: count}."""
    def __init__(self, order, bits):
        self.order = order; self.bits = bits; self.ckeys = None; self.toks = None; self.counts = None
    def learn(self, wids):
        n = len(wids)
        if n <= self.order: self.ckeys = np.zeros(0, np.int64); return self
        w = np.lib.stride_tricks.sliding_window_view(wids, self.order + 1)   # rows [ctx..., token]
        key = np.zeros(len(w), np.int64)
        for i in range(self.order): key = (key << self.bits) | w[:, i]
        stack = np.stack([key, w[:, self.order]], axis=1)
        uniq, counts = np.unique(stack, axis=0, return_counts=True)          # lexsorted by (key, token)
        self.ckeys = np.ascontiguousarray(uniq[:, 0]); self.toks = np.ascontiguousarray(uniq[:, 1])
        self.counts = counts.astype(np.float64)
        return self
    def predict(self, ctx):
        if self.ckeys is None or len(self.ckeys) == 0 or len(ctx) < self.order: return None
        key = 0
        for x in ctx[-self.order:]: key = (key << self.bits) | int(x)
        lo = np.searchsorted(self.ckeys, key, "left"); hi = np.searchsorted(self.ckeys, key, "right")
        if hi <= lo: return None
        return {int(self.toks[j]): float(self.counts[j]) for j in range(lo, hi)}

class FastCortex:
    def __init__(self, char_order=6, word_orders=(1, 2), use_cache=False, lam=0.4, vocab_cap=250_000):
        self.char_order = char_order; self.word_orders = list(word_orders)
        self.use_cache = use_cache; self.lam = lam; self.vocab_cap = vocab_cap
        self.K = 400                                          # eval window (chars) — long enough for a topic level
        self.charcol = None; self.ngrams = []; self.w2id = {}; self.id2spell = {0: None}
    def fit(self, text):
        self.charcol = FastColumn(self.char_order, V0).learn(np.fromiter((CH[c] for c in text), np.int64, len(text)))
        if not self.word_orders and not self.use_cache: return self
        words = [w for w in text.split(" ") if w]
        freq = {}
        for w in words: freq[w] = freq.get(w, 0) + 1
        keep = [w for w, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= 2][:self.vocab_cap]
        self.w2id = {w: i + 1 for i, w in enumerate(keep)}
        self.id2spell = {i: w for w, i in self.w2id.items()}; self.id2spell[0] = None
        bits = max(1, len(self.w2id).bit_length())            # keep order*bits <= 62 (orders <= 3 here)
        wids = np.fromiter((self.w2id.get(w, 0) for w in words), np.int64, len(words))
        self.ngrams = [FastNgram(o, bits).learn(wids) for o in self.word_orders]
        return self
    def dist(self, suffix):
        cids = [CH[c] for c in suffix[-8:] if c in CH]
        pc = self.charcol.predict_dense(cids)
        if not self.ngrams and not self.use_cache: return pc
        parts = suffix.split(" "); prefix = parts[-1]
        words = [p for p in parts[:-1] if p]
        ctx = [self.w2id.get(w, 0) for w in words[-3:]]
        experts = [ng.predict(ctx) for ng in self.ngrams]
        if self.use_cache and len(words) > 1:
            cache = {}
            for w in words[-70:-1]:
                wid = self.w2id.get(w, 0)
                if wid: cache[wid] = cache.get(wid, 0.0) + 1.0
            if cache: experts.append(cache)
        pw = vote_sparse(experts)
        if not pw: return pc
        cp = char_prior(pw, prefix, self.id2spell)
        if cp is None: return pc
        return self.lam * pc + (1 - self.lam) * (cp / cp.sum())
