#!/usr/bin/env python3
"""Experiment G — generalization metrics: is it actually learning English, and how far can it see?

Three metrics on text8, with a generative backoff char model (orders 0..K), trained on a train slice and
measured on a disjoint held-out slice:

  1. GENERALIZATION / OVERFITTING — train-bpc vs test-bpc (gap = overfitting), swept over capacity K.
  2. PREDICTION HORIZON — from a held-out context, greedily predict forward; how many chars (and words) match
     the true continuation before the first error? (how far can it 'see').
  3. TEXT vs GIBBERISH — sample free generations; what % of generated words are REAL words (in the training
     vocab), vs real held-out text (ceiling) and random chars (floor)? Plus eyeball samples.
"""
import os, math, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")
A = "abcdefghijklmnopqrstuvwxyz "; V = len(A); CH = {c: i for i, c in enumerate(A)}
ALPHA = 0.05; random.seed(0)

def text(n0, n1):
    raw = open(os.path.join(DATA, "text8"), "rb").read()[n0:n1].decode("ascii", "ignore")
    return "".join(c for c in raw if c in CH)

class Backoff:
    def __init__(self, K): self.K = K; self.c = [dict() for _ in range(K + 1)]
    def fit(self, s):
        c = self.c
        for t in range(len(s)):
            nxt = CH[s[t]]
            for k in range(self.K + 1):
                if k > t: break
                ctx = s[t - k:t]; d = c[k].get(ctx)
                if d is None: d = np.zeros(V); c[k][ctx] = d
                d[nxt] += 1
    def dist(self, ctx):
        for k in range(min(self.K, len(ctx)), -1, -1):
            d = self.c[k].get(ctx[len(ctx) - k:] if k else "")
            if d is not None and d.sum() > 0:
                return (d + ALPHA) / (d.sum() + ALPHA * V)
        return np.full(V, 1.0 / V)
    def bpc(self, s):
        return float(np.mean([-math.log2(self.dist(s[max(0, t - self.K):t])[CH[s[t]]]) for t in range(1, len(s))]))
    def generate(self, seed, n, temp=1.0):
        s = seed
        for _ in range(n):
            p = self.dist(s[-self.K:]) ** (1.0 / temp); p /= p.sum()
            s += A[np.random.choice(V, p=p)]
        return s[len(seed):]

def prediction_horizon(model, test, n_probes=2000, ctx_len=40, max_pred=50):
    runs = []; acc_at = {d: [] for d in (1, 2, 4, 8, 16, 32)}
    for _ in range(n_probes):
        t = random.randint(ctx_len, len(test) - max_pred - 1)
        ctx = test[t - ctx_len:t]; truth = test[t:t + max_pred]
        run = 0
        gen = ctx
        for d in range(max_pred):
            pred = A[int(np.argmax(model.dist(gen[-model.K:])))]
            if d < max_pred:
                if pred == truth[d]:
                    if run == d: run += 1
                for dd in acc_at:
                    if d == dd - 1: acc_at[dd].append(1 if pred == truth[d] else 0)
            gen += truth[d]   # teacher-forcing for accuracy@d; run already counts the matched prefix
        runs.append(run)
    return runs, {d: float(np.mean(v)) for d, v in acc_at.items()}

def word_validity(s, vocab):
    ws = [w for w in s.split(" ") if w]
    if not ws: return 0.0
    return np.mean([1 if w in vocab else 0 for w in ws])

if __name__ == "__main__":
    train = text(0, 10_000_000); test = text(98_000_000, 100_000_000)
    vocab = set(train.split(" "))
    print(f"train {len(train):,} chars / test {len(test):,} / training vocab {len(vocab):,} words\n")

    # ---- 1. GENERALIZATION / OVERFITTING vs capacity ----
    print("=== 1. generalization / overfitting (bits-per-char) ===")
    print(f"  {'K':<4}{'train bpc':>11}{'test bpc':>11}{'gap (overfit)':>16}")
    models = {}
    tr_s, te_s = train[:500_000], test[:500_000]
    for K in (2, 3, 4, 6, 8):
        m = Backoff(K); m.fit(train); models[K] = m
        tb, eb = m.bpc(tr_s), m.bpc(te_s)
        print(f"  {K:<4}{tb:>11.3f}{eb:>11.3f}{eb - tb:>+16.3f}")

    m = models[8]
    # ---- 2. PREDICTION HORIZON ----
    print("\n=== 2. prediction horizon (greedy, from 40-char held-out contexts) ===")
    runs, acc = prediction_horizon(m, test)
    runs = np.array(runs)
    print(f"  exact-match run length: mean {runs.mean():.1f} chars, median {int(np.median(runs))}, "
          f"90th pct {int(np.percentile(runs,90))}, max {runs.max()}")
    print("  next-char accuracy @ distance:  " + "  ".join(f"d{d}={acc[d]:.2f}" for d in sorted(acc)))

    # ---- 3. TEXT vs GIBBERISH ----
    print("\n=== 3. text vs gibberish (% generated words that are real / in training vocab) ===")
    real_rate = word_validity(test[:200_000], vocab)
    rnd = "".join(random.choice(A) for _ in range(200_000))
    rand_rate = word_validity(rnd, vocab)
    print(f"  REAL held-out text      {real_rate*100:5.1f}%   (ceiling)")
    print(f"  random chars            {rand_rate*100:5.1f}%   (floor)")
    for temp in (0.7, 1.0):
        gens = [m.generate(test[i:i+40], 400, temp=temp) for i in range(0, 200000, 20000)]
        rate = np.mean([word_validity(g, vocab) for g in gens])
        print(f"  GENERATED (temp={temp})    {rate*100:5.1f}%")
    print("\n  sample generation (K=8, temp=0.8), seed='the history of the ':")
    print("   " + m.generate("the history of the ", 300, temp=0.8).replace("\n", " "))
