#!/usr/bin/env python3
"""Exp K — does DEPTH pay off at SCALE? (3/4/5 levels × big data)

Exp J showed the data axis was starved at 2 MB. So now: take the uniform cortex up to 80 MB and ask whether
stacking more levels (3 → 4 → 5) keeps paying off as the data grows — the question that "didn't pay off at 2 MB"
was really asking. Levels (same Column unit): char + word-bands, the 5th a recency/topic cache (long window).

  3 levels: char + word{1,2}
  4 levels: char + word{1,2,3}
  5 levels: char + word{1,2,3} + topic cache

Scorecard (lean, window 400 chars so the topic level has room): bpc + overfit gap, real-word %, phrase-coh %.
"""
import os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from fastcortex import FastCortex
from cortex import CH

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")
WIN = 400

def text(n0, n1):
    raw = open(os.path.join(DATA, "text8"), "rb").read()[n0:n1].decode("ascii", "ignore")
    return "".join(c for c in raw if c in CH)

def bpc(model, s):
    return float(np.mean([-math.log2(model.dist(s[max(0, t - WIN):t])[CH[s[t]]] + 1e-12) for t in range(1, len(s))]))

def generate(model, seed, n, temp=0.5):
    A = "abcdefghijklmnopqrstuvwxyz "; s = seed
    for _ in range(n):
        p = model.dist(s[-WIN:]) ** (1.0 / temp); p /= p.sum()
        s += A[np.random.choice(27, p=p)]
    return s[len(seed):]

def real_word(s, vocab):
    ws = [w for w in s.split(" ") if w]; return np.mean([w in vocab for w in ws]) if ws else 0.0

def phrase_coh(s, bigrams):
    ws = [w for w in s.split(" ") if w]; pr = list(zip(ws, ws[1:]))
    return np.mean([p in bigrams for p in pr]) if pr else 0.0

CONFIGS = [("3 levels", dict(word_orders=[1, 2])),
           ("4 levels", dict(word_orders=[1, 2, 3])),
           ("5 levels", dict(word_orders=[1, 2, 3], use_cache=True))]

if __name__ == "__main__":
    np.random.seed(0)
    test = text(98_000_000, 98_200_000); te_s = test[:60_000]; seed = test[80_000:80_400]
    print(f"{'data':>7} {'levels':>9} {'fit s':>7} {'train bpc':>10} {'test bpc':>9} {'overfit':>8} {'real-wd%':>9} {'phrase%':>8}")
    vocab = bigrams = None
    for nmb in (10, 40, 80):
        train = text(0, nmb * 1_000_000)
        vocab = set(w for w in train.split(" ") if w)
        toks = [w for w in train.split(" ") if w]; bigrams = set(zip(toks, toks[1:]))
        tr_s = train[:60_000]
        for name, kw in CONFIGS:
            t0 = time.time(); cx = FastCortex(**kw).fit(train); ft = time.time() - t0
            tb, eb = bpc(cx, tr_s), bpc(cx, te_s)
            gen = generate(cx, seed[:48], 2500, 0.5)
            print(f"{nmb:>5}MB {name:>9} {ft:>7.1f} {tb:>10.3f} {eb:>9.3f} {eb-tb:>+8.3f}"
                  f" {real_word(gen, vocab)*100:>9.1f} {phrase_coh(gen, bigrams)*100:>8.1f}")
        sys.stdout.flush()
    print(f"\n  (real held-out text vs 80MB vocab):  real-word% {real_word(test[80_000:160_000], vocab)*100:.1f}"
          f"   phrase-coh% {phrase_coh(test[80_000:160_000], bigrams)*100:.1f}")
