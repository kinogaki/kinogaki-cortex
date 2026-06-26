#!/usr/bin/env python3
"""Exp H — score the CONCEPT-hierarchy model (char + word lexicon, Exp C) on the generalization suite vs the
plain char baseline. Question: do concepts improve *coherence* (real-word generation, prediction horizon),
not just bits-per-char?
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import metrics as M
from metrics import A, V, CH, WIN

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")
ALPHA = 0.05; K = 6

def text(n0, n1):
    raw = open(os.path.join(DATA, "text8"), "rb").read()[n0:n1].decode("ascii", "ignore")
    return "".join(c for c in raw if c in CH)

class CharBackoff:
    def __init__(self, K): self.K = K; self.c = [dict() for _ in range(K + 1)]
    def fit(self, s):
        for t in range(len(s)):
            nx = CH[s[t]]
            for k in range(self.K + 1):
                if k > t: break
                ctx = s[t-k:t]; d = self.c[k].get(ctx)
                if d is None: d = np.zeros(V); self.c[k][ctx] = d
                d[nx] += 1
    def cdist(self, suffix):
        for k in range(min(self.K, len(suffix)), -1, -1):
            d = self.c[k].get(suffix[len(suffix)-k:] if k else "")
            if d is not None and d.sum() > 0: return (d + ALPHA) / (d.sum() + ALPHA*V)
        return np.full(V, 1.0/V)
    def dist(self, suffix): return self.cdist(suffix)

class Trie:
    def __init__(self): self.kids = {}; self.end = 0.0; self.tot = 0.0
    def add(self, w, c):
        n = self
        for ch in w: n.tot += c; n = n.kids.setdefault(ch, Trie())
        n.tot += c; n.end += c
    def prior(self, prefix):
        n = self
        for ch in prefix:
            n = n.kids.get(ch)
            if n is None or n.tot <= 0: return None
        p = np.zeros(V); p[CH[" "]] = n.end
        for ch, kid in n.kids.items(): p[CH[ch]] = kid.tot
        s = p.sum(); return p / s if s > 0 else None

class HierLex:
    """char backoff + word lexicon prior (Exp C), as a generative model: P = λ·char + (1-λ)·lex(prefix)."""
    def __init__(self, char, lex, lam): self.char = char; self.lex = lex; self.lam = lam; self.K = max(char.K, WIN)
    def dist(self, suffix):
        pc = self.char.cdist(suffix[-self.char.K:] if len(suffix) > self.char.K else suffix)
        prefix = suffix.split(" ")[-1]            # chars since last space = current word prefix
        pl = self.lex.prior(prefix)
        return pc if pl is None else self.lam * pc + (1 - self.lam) * pl

if __name__ == "__main__":
    train = text(0, 10_000_000); test = text(98_000_000, 100_000_000)
    vocab = set(w for w in train.split(" ") if w)
    tr_s, te_s = train[:400_000], test[:400_000]; seed = test[400_000:600_000]
    print(f"train {len(train):,} / test {len(test):,} / vocab {len(vocab):,} words; char order K={K}")

    char = CharBackoff(K); char.fit(train)
    lex = Trie()
    freq = {}
    for w in train.split(" "):
        if w: freq[w] = freq.get(w, 0) + 1
    for w, f in freq.items(): lex.add(w, f)
    hier = HierLex(char, lex, lam=0.4)

    rc = M.report("char-only (baseline)", char, tr_s, te_s, vocab, seed)
    rh = M.report("char + word concepts (Exp C)", hier, tr_s, te_s, vocab, seed)

    # ── Exp E: word-level generator (unigram·bigram·cache product) → does the PHRASE level add coherence? ──
    toks = [w for w in train.split(" ") if w]
    big = {}
    for a_, b_ in zip(toks, toks[1:]): big.setdefault(a_, {}); big[a_][b_] = big[a_].get(b_, 0) + 1
    bigram_set = set((a_, b_) for a_ in big for b_ in big[a_])
    top = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:8000]]
    def gen_words(n, prev="the"):
        out = [prev]; cache = {}
        for _ in range(n):
            d = dict(big.get(prev, {}))
            for w, v in cache.items(): d[w] = d.get(w, 0) + 0.5 * v       # topic/recency
            if not d: d = {w: freq[w] for w in top[:200]}
            ws = list(d); wt = np.array([d[w] for w in ws], float); wt /= wt.sum()
            prev = ws[np.random.choice(len(ws), p=wt)]
            out.append(prev)
            for w in list(cache): cache[w] *= 0.99
            cache[prev] = cache.get(prev, 0.0) + 1.0
        return " ".join(out)

    print("\n=== PHRASE COHERENCE — % of generated word-bigrams that are REAL (seen in training) ===")
    real_pc = M.phrase_coherence(test[400_000:700_000], bigram_set)
    cgen = M.generate(char, seed[:48], 3000, 0.9); hgen = M.generate(hier, seed[:48], 3000, 0.9); wgen = gen_words(700)
    print(f"    REAL held-out text          {real_pc*100:5.1f}%   (ceiling)")
    print(f"    char-only generation        {M.phrase_coherence(cgen, bigram_set)*100:5.1f}%")
    print(f"    char + word concepts (C)     {M.phrase_coherence(hgen, bigram_set)*100:5.1f}%")
    print(f"    WORD-LEVEL phrases (E)       {M.phrase_coherence(wgen, bigram_set)*100:5.1f}%   <-- the phrase level")

    print("\n=== summary: what each level buys (vs char-only) ===")
    print(f"    {'level':<24}{'test bpc':>10}{'overfit':>9}{'real-word%(t1)':>16}{'phrase-coh%':>13}")
    print(f"    {'char-only':<24}{rc['test_bpc']:>10.3f}{rc['test_bpc']-rc['train_bpc']:>+9.3f}{rc['val10']*100:>16.1f}{M.phrase_coherence(cgen,bigram_set)*100:>13.1f}")
    print(f"    {'+ word concepts (C)':<24}{rh['test_bpc']:>10.3f}{rh['test_bpc']-rh['train_bpc']:>+9.3f}{rh['val10']*100:>16.1f}{M.phrase_coherence(hgen,bigram_set)*100:>13.1f}")
    print(f"    {'+ phrases word-level (E)':<24}{'n/a':>10}{'n/a':>9}{100.0:>16.1f}{M.phrase_coherence(wgen,bigram_set)*100:>13.1f}")
    print("\n  word-level (Exp E) sample: " + " ".join(wgen.split()[:40]))
