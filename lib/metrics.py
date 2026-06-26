"""Reusable generalization-metric suite for kinogaki-cortex models.

A model is any object with `.K` (context window it needs) and `.dist(suffix) -> np.array(V)` giving the
next-char distribution from the recent text suffix (stateless — current word-prefix is derivable from the
suffix). This lets char-only and concept-hierarchy models be scored identically.

Metrics: bpc (+ train/test gap = overfitting), prediction_horizon (how far it predicts the true text),
text_validity (% of generated words that are real). See Exp G for definitions.
"""
import math, random
import numpy as np

A = "abcdefghijklmnopqrstuvwxyz "; V = len(A); CH = {c: i for i, c in enumerate(A)}
WIN = 64

def bpc(model, s):
    return float(np.mean([-math.log2(model.dist(s[max(0, t - WIN):t])[CH[s[t]]] + 1e-12) for t in range(1, len(s))]))

def generate(model, seed, n, temp=1.0):
    s = seed
    for _ in range(n):
        p = model.dist(s[-WIN:]) ** (1.0 / temp); p /= p.sum()
        s += A[np.random.choice(V, p=p)]
    return s[len(seed):]

def prediction_horizon(model, test, n_probes=2000, ctx_len=48, max_pred=50, seed=0):
    rnd = random.Random(seed); runs = []; acc = {d: [] for d in (1, 2, 4, 8, 16)}
    for _ in range(n_probes):
        t = rnd.randint(ctx_len, len(test) - max_pred - 1)
        ctx = test[t - ctx_len:t]; truth = test[t:t + max_pred]
        run = 0; matched = True
        for d in range(max_pred):
            pred = A[int(np.argmax(model.dist((ctx + truth[:d])[-WIN:])))]
            hit = (pred == truth[d])
            for dd in acc:
                if d == dd - 1: acc[dd].append(1 if hit else 0)
            if matched and hit: run += 1
            elif matched: matched = False
        runs.append(run)
    return np.array(runs), {d: float(np.mean(v)) for d, v in acc.items()}

def text_validity(s, vocab):
    ws = [w for w in s.split(" ") if w]
    return float(np.mean([1 if w in vocab else 0 for w in ws])) if ws else 0.0

def phrase_coherence(s, bigrams):
    """% of generated word-bigrams that are REAL (seen in training) — measures local phrase/grammar coherence,
    beyond just 'are the words real'."""
    ws = [w for w in s.split(" ") if w]
    pairs = list(zip(ws, ws[1:]))
    return float(np.mean([1 if p in bigrams else 0 for p in pairs])) if pairs else 0.0

def report(name, model, train_s, test_s, vocab, samples_seed):
    tb, eb = bpc(model, train_s), bpc(model, test_s)
    runs, acc = prediction_horizon(model, test_s)
    val = {temp: np.mean([text_validity(generate(model, samples_seed[i:i+48], 400, temp), vocab)
                          for i in range(0, len(samples_seed) - 500, (len(samples_seed)-500)//8)])
           for temp in (0.7, 1.0)}
    print(f"\n  ── {name} ──")
    print(f"    bpc:    train {tb:.3f}  test {eb:.3f}  gap(overfit) {eb-tb:+.3f}")
    print(f"    horizon: greedy run mean {runs.mean():.1f}  median {int(np.median(runs))}  90pct {int(np.percentile(runs,90))}  "
          f"| next-char acc d1={acc[1]:.2f} d4={acc[4]:.2f} d16={acc[16]:.2f}")
    print(f"    real-word %: temp0.7 {val[0.7]*100:.1f}   temp1.0 {val[1.0]*100:.1f}")
    return dict(train_bpc=tb, test_bpc=eb, run=runs.mean(), val07=val[0.7], val10=val[1.0])
