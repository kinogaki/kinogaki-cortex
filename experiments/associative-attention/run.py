#!/usr/bin/env python3
"""Exp L — associative attention vs fixed n-grams: learn WHICH words and HOW FAR to attend, by online counting.

Baseline = FastCortex 3-level (char + fixed word bigram/trigram). Treatment = AssocAttention (skip-gram
associations + learned per-word attention weight + long context window). Same calibrated pooling, same char
bridge. Question: does content-based, variable-distance, count-learned attention beat fixed n-grams on coherence
— and is what it learns inspectable/sensible?
"""
import os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from fastcortex import FastCortex
from attention import AssocAttention
from cortex import CH

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")
WIN = 400

def text(n0, n1):
    raw = open(os.path.join(DATA, "text8"), "rb").read()[n0:n1].decode("ascii", "ignore")
    return "".join(c for c in raw if c in CH)

def bpc(m, s):
    return float(np.mean([-math.log2(m.dist(s[max(0, t-WIN):t])[CH[s[t]]] + 1e-12) for t in range(1, len(s))]))

def generate(m, seed, n, temp=0.5):
    A = "abcdefghijklmnopqrstuvwxyz "; s = seed
    for _ in range(n):
        p = m.dist(s[-WIN:]) ** (1.0/temp); p /= p.sum(); s += A[np.random.choice(27, p=p)]
    return s[len(seed):]

def real_word(s, vocab):
    ws = [w for w in s.split(" ") if w]; return np.mean([w in vocab for w in ws]) if ws else 0.0
def phrase_coh(s, bg):
    ws = [w for w in s.split(" ") if w]; pr = list(zip(ws, ws[1:])); return np.mean([p in bg for p in pr]) if pr else 0.0

if __name__ == "__main__":
    np.random.seed(0)
    train = text(0, 10_000_000); test = text(98_000_000, 98_200_000)
    vocab = set(w for w in train.split(" ") if w); toks = [w for w in train.split(" ") if w]
    bg = set(zip(toks, toks[1:])); tr_s, te_s = train[:30_000], test[:30_000]; seed = test[80_000:80_600]
    print(f"train {len(train):,} chars / vocab {len(vocab):,}\n")

    models = [("fixed n-grams (Exp K 3-lvl)", FastCortex(word_orders=[1, 2])),
              ("associative attention", AssocAttention())]
    rows = []
    for name, m in models:
        t0 = time.time(); m.fit(train); ft = time.time() - t0
        tb, eb = bpc(m, tr_s), bpc(m, te_s)
        gen = generate(m, seed[:48], 1500, 0.5)
        rows.append((name, ft, tb, eb, real_word(gen, vocab), phrase_coh(gen, bg), gen))

    print(f"  {'model':<30}{'fit s':>7}{'test bpc':>10}{'overfit':>9}{'real-wd%':>10}{'phrase%':>9}")
    for name, ft, tb, eb, rw, ph, _ in rows:
        print(f"  {name:<30}{ft:>7.1f}{eb:>10.3f}{eb-tb:>+9.3f}{rw*100:>10.1f}{ph*100:>9.1f}")
    print(f"  {'(real held-out text)':<30}{'':>7}{'':>10}{'':>9}{real_word(test[80_000:160_000], vocab)*100:>10.1f}"
          f"{phrase_coh(test[80_000:160_000], bg)*100:>9.1f}")
    for name, *_ , gen in rows:
        print(f"\n  [{name}] sample:\n    " + " ".join(gen.split()[1:34]))

    # ── inspectability: what did attention LEARN to attend to (and what does it ignore)? ──
    attn = models[1][1]
    print("\n  what attention learned (word → top followers, with LEARNED attention weight):")
    for w in ["united", "new", "the", "of", "world", "north", "you", "first"]:
        wt, top = attn.attends_to(w, 5)
        print(f"    {w:<9} weight {wt:.2f}  → " + ", ".join(f"{b}({c})" for b, c in top[:5]))
