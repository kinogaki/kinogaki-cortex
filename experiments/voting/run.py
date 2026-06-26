#!/usr/bin/env python3
"""Experiment D — the integrated cortex: multi-level concepts + many-views VOTING (English, associative).

Brings together everything: several "views" (experts), each predicting the next char from a different level —
local chars, the word lexicon, word-context PHRASES (reach beyond the char window), and a TOPIC cache (whole
document) — combined by GATED ADAPTIVE VOTING (each view's weight tracks where it's actually good, by position
regime; multiplicative-weights / Hedge). This is cortical voting + attention gating, online, no backprop.

Experts (all output a 27-dim next-char distribution):
  E.char8  : order-8 char n-gram        (local)
  E.char3  : order-3 char n-gram        (a shorter, diverse view)
  E.lex    : word lexicon trie prior    (within-word, level 1 — the Exp C winner)
  E.phrase : predict next WORD from prev 1-2 words (tri/bi-gram), marginalize to next char  (level 2, > char window)
  E.topic  : decaying word cache → recently-active words, marginalize to next char           (level 3, document-range)

Tests: (1) does voting beat the best single view? (2) does adding phrase+topic COMPOUND below Exp C's 1.65?
"""
import os, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
VOCAB = " abcdefghijklmnopqrstuvwxyz"; CH2I = {c: i for i, c in enumerate(VOCAB)}; V = len(VOCAB); SP = 0
ALPHA = 0.02; TOPN = 300

def load(name="english.txt"):
    raw = open(os.path.join(DATA, name), encoding="utf-8", errors="ignore").read()
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S); m2 = re.search(r"\*\*\* END OF", raw, re.S)
    body = raw[m1.end():m2.start()] if (m1 and m2) else raw
    return re.sub(r" +", " ", re.sub(r"[^a-z]+", " ", body.lower()).strip())

class CharNGram:
    def __init__(self, K): self.K = K; self.c = [dict() for _ in range(K + 1)]
    def fit(self, s):
        for t in range(len(s)):
            for k in range(self.K + 1):
                if k > t: continue
                ctx = s[t-k:t]; d = self.c[k].get(ctx)
                if d is None: d = np.zeros(V); self.c[k][ctx] = d
                d[CH2I[s[t]]] += 1
    def dist(self, ctx):
        for k in range(min(self.K, len(ctx)), -1, -1):
            d = self.c[k].get(ctx[len(ctx)-k:] if k else "")
            if d is not None and d.sum() > 0: return (d + ALPHA) / (d.sum() + ALPHA*V)
        return np.full(V, 1.0/V)

class Trie:
    def __init__(self): self.kids = {}; self.end = 0.0; self.tot = 0.0
    def add(self, w, c):
        n = self
        for ch in w: n.tot += c; n = n.kids.setdefault(ch, Trie())
        n.tot += c; n.end += c
    def node(self, prefix):
        n = self
        for ch in prefix:
            n = n.kids.get(ch)
            if n is None: return None
        return n

def marginalize(words, wt, prefix):
    """next-char dist by summing candidate words' char at position len(prefix) (or space if the word ends)."""
    p = np.zeros(V); pos = len(prefix)
    for w, q in zip(words, wt):
        p[CH2I[w[pos]] if pos < len(w) else SP] += q
    s = p.sum(); return (p / s) if s > 0 else None

def filt(words, wt, prefix):
    keep = [(w, q) for w, q in zip(words, wt) if len(w) >= len(prefix) and w[:len(prefix)] == prefix]
    return ([w for w, _ in keep], np.array([q for _, q in keep])) if keep else ([], np.array([]))

if __name__ == "__main__":
    s = load(); cut = int(len(s)*0.85); train, test = s[:cut], s[cut:cut+100_000]
    words = train.split(" ")
    print(f"english train {len(train):,} / test {len(test):,} chars; {len(words):,} words")

    cm8 = CharNGram(8); cm8.fit(train)
    cm3 = CharNGram(3); cm3.fit(train)
    freq = {};
    for w in words: freq[w] = freq.get(w, 0) + 1
    lex = Trie()
    for w, f in freq.items(): lex.add(w, f)
    uni = sorted(freq.items(), key=lambda x: -x[1])[:TOPN]; uni_w = np.array([f for _, f in uni], float); uni_w /= uni_w.sum()
    big = {}; tri = {}
    for a, b in zip(words, words[1:]):
        big.setdefault(a, {}); big[a][b] = big[a].get(b, 0)+1
    for a, b, c in zip(words, words[1:], words[2:]):
        tri.setdefault((a, b), {}); tri[(a, b)][c] = tri[(a, b)].get(c, 0)+1

    def phrase_cands(p1, p2):
        d = tri.get((p1, p2)) or big.get(p2) or None
        Q = {}
        if d:
            tot = sum(d.values())
            for w, c in d.items(): Q[w] = 0.7*c/tot
        for (w, _), pw in zip(uni, uni_w): Q[w] = Q.get(w, 0.0) + 0.3*pw
        it = sorted(Q.items(), key=lambda x: -x[1])[:TOPN]
        return [w for w, _ in it], np.array([v for _, v in it])

    EXPERTS = ["char8", "char3", "lex", "phrase", "topic"]
    NE = len(EXPERTS)
    NR = 3                              # gating regimes by prefix length: 0=start, 1=len1-2, 2=len>=3
    Wg = np.full((NR, NE), 1.0/NE)      # LINEAR voting weights (for comparison)
    eta, leak = 0.3, 0.001
    Ag = np.full((NR, NE), 0.5)         # PRODUCT-of-experts log-linear weights, learned per regime
    lr_a = 0.02
    def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()

    # accumulators
    solo = np.zeros(NE); voted = 0.0; unif = 0.0; prod = 0.0
    cache = {}; prefix = ""; p1 = p2 = ""
    pw_words, pw_wt = phrase_cands("", "")          # phrase candidates
    tp_words, tp_wt = [], np.array([])              # topic candidates

    for t in range(1, len(test)):
        ctx = test[max(0, t-8):t]
        ds = np.empty((NE, V))
        ds[0] = cm8.dist(ctx)
        ds[1] = cm3.dist(ctx[-3:] if len(ctx) >= 3 else ctx)
        n = lex.node(prefix)
        if n is None or n.tot <= 0: ds[2] = np.full(V, 1.0/V)
        else:
            p = np.zeros(V); p[SP] = n.end
            for ch, kid in n.kids.items(): p[CH2I[ch]] = kid.tot
            ds[2] = p / p.sum()
        d3 = marginalize(pw_words, pw_wt, prefix) if len(pw_words) else None
        ds[3] = d3 if d3 is not None else np.full(V, 1.0/V)
        d4 = marginalize(tp_words, tp_wt, prefix) if len(tp_words) else None
        ds[4] = d4 if d4 is not None else np.full(V, 1.0/V)
        ds = np.clip(ds, 1e-9, None)

        g = 0 if prefix == "" else (1 if len(prefix) <= 2 else 2)
        c = CH2I[test[t]]
        # LINEAR voting (mixture) — for comparison
        w = Wg[g]; Pv = w @ ds; Pv /= Pv.sum()
        Pu = ds.mean(0); Pu /= Pu.sum()
        # PRODUCT-of-experts (log-linear pooling) with learned per-regime weights — constraints multiply,
        # uniform experts auto-abstain. This is the real "consensus = intersection of views".
        z = np.log(ds)                         # (NE, V)
        a = Ag[g]
        Pp = softmax(a @ z)
        voted += -math.log2(Pv[c]); unif += -math.log2(Pu[c]); prod += -math.log2(Pp[c] + 1e-12)
        solo += -np.log2(ds[:, c])
        # update linear weights (Hedge)
        losses = -np.log(ds[:, c]); w = w * np.exp(-eta*(losses - losses.min())); Wg[g] = (1-leak)*w/w.sum() + leak/NE
        # update product weights (online log-linear gradient): raise a_v if view v scored the true char above its own mean
        Ag[g] = np.clip(a + lr_a * (z[:, c] - z @ Pp), 0.0, 5.0)
        # advance state
        if test[t] == " ":
            for k in list(cache): cache[k] *= 0.997
            if prefix: cache[prefix] = cache.get(prefix, 0.0)+1.0
            p1, p2 = p2, prefix; prefix = ""
            pw_words, pw_wt = phrase_cands(p1, p2)
            if cache:
                it = sorted(cache.items(), key=lambda x: -x[1])[:TOPN]
                tp_words, tp_wt = [k for k, _ in it], np.array([v for _, v in it])
        else:
            prefix += test[t]
            pw_words, pw_wt = filt(pw_words, pw_wt, prefix)
            tp_words, tp_wt = filt(tp_words, tp_wt, prefix)

    N = len(test) - 1
    print("\n  === solo views (bits-per-char) ===")
    for j, name in enumerate(EXPERTS): print(f"    {name:<8} {solo[j]/N:.4f}")
    print("\n  === combinations (bits-per-char) ===")
    print(f"    uniform linear mix     {unif/N:.4f}")
    print(f"    gated LINEAR voting    {voted/N:.4f}")
    print(f"    gated PRODUCT-of-experts {prod/N:.4f}")
    best_solo = solo.min()/N
    print(f"\n  best single view {best_solo:.4f}  |  Exp C char+lex 1.653  |  PRODUCT {prod/N:.4f}  "
          f"({(1.653-prod/N)/1.653*100:+.1f}% vs Exp C, {(best_solo-prod/N)/best_solo*100:+.1f}% vs best view)")
    print("\n  learned PRODUCT weights per regime (start / len1-2 / len3+):")
    for r in range(NR): print(f"    regime{r}: " + "  ".join(f"{EXPERTS[j]}={Ag[r][j]:.2f}" for j in range(NE)))
