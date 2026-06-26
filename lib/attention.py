"""Associative attention — "attention but in our sense": learn WHICH earlier words and HOW FAR to attend, by
ONLINE COUNTING, not gradient-trained Q/K/V.

Fixed n-grams assume fixed distance + equal weight per slot. This drops both:
  - assoc[A] = the distribution of words that FOLLOW A within a wide window, learned by counting (skip-grams).
    Content links ("united"→"states", "not"→negation) emerge regardless of exact distance; non-forgetting.
  - attention weight of a context word A = how PEAKED assoc[A] is (top-1 mass). Content words (sharp) dominate;
    function words (flat — predict everything) auto-suppress. "Which words to attend to", LEARNED from counts.
  - any word in a long window contributes, weighted by association strength → effective SPAN is data-learned
    per word, not a fixed n. A strong long-range link survives distance; weak ones fade.
  - surprise gate (optional): a word that was surprising when it arrived carries new info → weight it more.
Pooled by the same calibrated geometric-mean vote, blended to the char prior. All online, associative, inspectable.
"""
import math
import numpy as np
from cortex import char_prior, V0, CH, ALPHA
from fastcol import FastColumn

def attn_vote(dicts, weights):
    """Weighted log-linear pool (calibrated): logp(k) = Σ wᵢ·log(dᵢ[k]) / Σ wᵢ, over the union of supports."""
    items = [(d, w) for d, w in zip(dicts, weights) if d and w > 0]
    if not items: return None
    keys = set().union(*[set(d) for d, _ in items]); wt = sum(w for _, w in items)
    out = {}
    for k in keys:
        lp = 0.0
        for d, w in items: lp += w * math.log(d.get(k, 0) + ALPHA)
        out[k] = lp / wt
    m = max(out.values()); z = sum(math.exp(v - m) for v in out.values())
    return {k: math.exp(v - m) / z for k, v in out.items()}

class AssocAttention:
    def __init__(self, char_order=6, assoc_window=8, ctx_window=60, beta=3.0, lam=0.4, vocab_cap=200_000):
        self.char_order = char_order; self.assoc_window = assoc_window; self.ctx_window = ctx_window
        self.beta = beta; self.lam = lam; self.vocab_cap = vocab_cap
        self.K = 600                                          # eval window (chars) — attention needs long context
        self.charcol = None; self.w2id = {}; self.id2spell = {0: None}
        self.A_keys = None; self.B = None; self.cnt = None; self.weight = None
    def fit(self, text):
        self.charcol = FastColumn(self.char_order, V0).learn(np.fromiter((CH[c] for c in text), np.int64, len(text)))
        words = [w for w in text.split(" ") if w]
        freq = {}
        for w in words: freq[w] = freq.get(w, 0) + 1
        keep = [w for w, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= 2][:self.vocab_cap]
        self.w2id = {w: i + 1 for i, w in enumerate(keep)}
        self.id2spell = {i: w for w, i in self.w2id.items()}; self.id2spell[0] = None
        wids = np.fromiter((self.w2id.get(w, 0) for w in words), np.int64, len(words))
        # skip-gram associations: (A, B) for every B within assoc_window positions after A, all gaps stacked
        pairs = [np.stack([wids[:-g], wids[g:]], axis=1) for g in range(1, self.assoc_window + 1)]
        allp = np.concatenate(pairs, axis=0)
        allp = allp[(allp[:, 0] > 0) & (allp[:, 1] > 0)]      # ignore unk on either side
        uniq, counts = np.unique(allp, axis=0, return_counts=True)   # lexsorted by (A, B)
        self.A_keys = np.ascontiguousarray(uniq[:, 0]); self.B = np.ascontiguousarray(uniq[:, 1])
        self.cnt = counts.astype(np.float64)
        # per-A informativeness = top-1 mass of assoc[A] (peakedness) — the LEARNED attention weight
        self.weight = np.zeros(len(self.w2id) + 1)
        edges = np.searchsorted(self.A_keys, np.arange(len(self.w2id) + 2))   # block boundaries per A id
        for a in range(1, len(self.w2id) + 1):
            lo, hi = edges[a], edges[a + 1]
            if hi > lo:
                c = self.cnt[lo:hi]; self.weight[a] = float(c.max() / c.sum())
        self._edges = edges
        return self
    def _assoc(self, a):
        lo, hi = self._edges[a], self._edges[a + 1]
        return None if hi <= lo else {int(self.B[j]): float(self.cnt[j]) for j in range(lo, hi)}
    def dist(self, suffix):
        cids = [CH[c] for c in suffix[-8:] if c in CH]
        pc = self.charcol.predict_dense(cids)
        parts = suffix.split(" "); prefix = parts[-1]
        words = [p for p in parts[:-1] if p][-self.ctx_window:]
        seen, dicts, wts = set(), [], []
        for w in words:                                       # each distinct context word attends by its weight
            a = self.w2id.get(w, 0)
            if a == 0 or a in seen: continue
            seen.add(a); d = self._assoc(a)
            if d: dicts.append(d); wts.append(self.weight[a] ** self.beta)
        pw = attn_vote(dicts, wts)
        if not pw: return pc
        cp = char_prior(pw, prefix, self.id2spell)
        if cp is None: return pc
        return self.lam * pc + (1 - self.lam) * (cp / cp.sum())
    def attends_to(self, word, topn=8):
        """Inspectability: what does `word` predict, and how strong is its attention weight?"""
        a = self.w2id.get(word, 0); d = self._assoc(a)
        if not d: return self.weight[a] if a else 0.0, []
        top = sorted(d.items(), key=lambda x: -x[1])[:topn]
        return self.weight[a], [(self.id2spell[b], int(c)) for b, c in top]
