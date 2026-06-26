#!/usr/bin/env python3
"""Experiment B — catastrophic forgetting, the core differentiator (English-only).

Stream a char predictor through 4 English registers in sequence (austen → shakespeare → darwin → bible),
one pass, NO replay. After each register, evaluate on held-out from ALL registers → retention matrix.

  DENSE  (the forgetting baseline): char-embedding → tanh hidden → softmax. SGD updates ALL weights each step.
  SPARSE (our design):              fixed random projection → k-WTA sparse code → softmax readout. The code is
                                    sparse, so each step's gradient touches only the k ACTIVE readout columns —
                                    a new register barely perturbs the columns carrying an old one. Localized
                                    learning = low interference = (the hypothesis) no catastrophic forgetting.
  COUNT  (non-forgetting reference): online backoff n-gram; accumulates, never overwrites.

Metric: FORGETTING = how much austen (task 1) bits-per-char degrades after learning the later registers.
Hypothesis: DENSE forgets a lot; SPARSE retains; both at comparable peak. Honest if it fails.
"""
import os, re, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
REGS = [("austen", "english.txt"), ("shakespeare", "shakespeare.txt"),
        ("darwin", "darwin.txt"), ("bible", "bible.txt")]
VOCAB = " abcdefghijklmnopqrstuvwxyz.,'"          # 30 symbols
CH2I = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)
W = 12                  # context window
T_TRAIN = 120_000       # chars trained per register (one pass, no replay)
T_EVAL = 12_000         # held-out chars per register
BATCH = 256
rng = np.random.default_rng(0)

def clean(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S); m2 = re.search(r"\*\*\* END OF", raw, re.S)
    body = raw[m1.end():m2.start()] if (m1 and m2) else raw
    body = body.lower().replace("\n", " ")
    body = re.sub(r"[^a-z.,' ]+", " ", body)
    body = re.sub(r" +", " ", body)
    return np.array([CH2I[c] for c in body if c in CH2I], dtype=np.int64)

def examples(ids):
    n = len(ids) - W
    X = np.stack([ids[i:i + W] for i in range(n)])
    y = ids[W:W + n]
    return X, y

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

# ---------- DENSE online MLP ----------
class Dense:
    def __init__(self, d=24, H=300, lr=0.1):
        self.lr = lr
        self.E = rng.normal(0, 0.1, (V, d))
        self.W1 = rng.normal(0, 1/np.sqrt(W*d), (W*d, H)); self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, 1/np.sqrt(H), (H, V));     self.b2 = np.zeros(V)
        self.params = V*d + W*d*H + H*V
    def fwd(self, X):
        emb = self.E[X].reshape(len(X), -1)        # (B, W*d)
        h = np.tanh(emb @ self.W1 + self.b1)
        p = softmax(h @ self.W2 + self.b2)
        return emb, h, p
    def train(self, X, y):
        emb, h, p = self.fwd(X); B = len(X)
        dz = p; dz[np.arange(B), y] -= 1; dz /= B
        dW2 = h.T @ dz; db2 = dz.sum(0)
        dh = (dz @ self.W2.T) * (1 - h*h)
        dW1 = emb.T @ dh; db1 = dh.sum(0)
        demb = (dh @ self.W1.T).reshape(B, W, -1)
        self.W2 -= self.lr*dW2; self.b2 -= self.lr*db2
        self.W1 -= self.lr*dW1; self.b1 -= self.lr*db1
        np.add.at(self.E, X, -self.lr*demb)
    def eval(self, X, y):
        _, _, p = self.fwd(X)
        bpc = -np.log2(p[np.arange(len(y)), y] + 1e-12).mean()
        acc = (p.argmax(1) == y).mean()
        return bpc, acc

# ---------- SPARSE online (our design) ----------
class Sparse:
    def __init__(self, M=2048, k=40, lr=0.3):
        self.M, self.k, self.lr = M, k, lr
        self.R = rng.normal(0, 1, (W*V, M))        # FIXED random projection
        self.Wr = np.zeros((M, V)); self.br = np.zeros(V)
        self.params = M*V                           # only the readout is learned
    def code(self, X):
        B = len(X)
        oh = np.zeros((B, W*V))
        cols = X + (np.arange(W)*V)                 # one-hot of each context position
        oh[np.arange(B)[:, None], cols] = 1.0
        proj = oh @ self.R                          # (B, M)
        idx = np.argpartition(proj, -self.k, 1)[:, -self.k:]
        h = np.zeros((B, self.M)); h[np.arange(B)[:, None], idx] = 1.0
        return h
    def train(self, X, y):
        h = self.code(X); B = len(X)
        p = softmax(h @ self.Wr + self.br)
        dz = p; dz[np.arange(B), y] -= 1; dz /= B
        self.Wr -= self.lr * (h.T @ dz)             # only active columns (h=0 elsewhere) get updated
        self.br -= self.lr * dz.sum(0)
    def eval(self, X, y):
        p = softmax(self.code(X) @ self.Wr + self.br)
        bpc = -np.log2(p[np.arange(len(y)), y] + 1e-12).mean()
        acc = (p.argmax(1) == y).mean()
        return bpc, acc

# ---------- COUNT backoff n-gram (non-forgetting reference) ----------
class Count:
    def __init__(self, K=6, alpha=0.1): self.K=K; self.alpha=alpha; self.c=[dict() for _ in range(K+1)]
    def train(self, X, y):
        for ctx_row, t in zip(X, y):
            for k in range(self.K+1):
                key = ctx_row[W-k:].tobytes() if k else b""
                d = self.c[k].get(key)
                if d is None: d = np.zeros(V); self.c[k][key] = d
                d[t] += 1
    def eval(self, X, y):
        bpc=0.0; acc=0
        for ctx_row, t in zip(X, y):
            P = np.full(V, 1.0/V); w = 1.0
            for k in range(self.K, -1, -1):
                key = ctx_row[W-k:].tobytes() if k else b""
                d = self.c[k].get(key)
                if d is not None and d.sum()>0:
                    P = (d+self.alpha)/(d.sum()+self.alpha*V); break
            bpc += -math.log2(P[t]+1e-12); acc += (P.argmax()==t)
        return bpc/len(y), acc/len(y)

def stream_train(model, X, y):
    for i in range(0, min(len(X), T_TRAIN), BATCH):
        model.train(X[i:i+BATCH], y[i:i+BATCH])

if __name__ == "__main__":
    print("loading english registers (char-level, vocab=%d, W=%d)..." % (V, W))
    data = {}
    for name, fn in REGS:
        ids = clean(os.path.join(DATA, fn))
        Xtr, ytr = examples(ids[:T_TRAIN+W])
        Xev, yev = examples(ids[T_TRAIN+W:T_TRAIN+W+T_EVAL+W])
        data[name] = (Xtr, ytr, Xev, yev)
        print(f"  {name:<12} {len(ids):>9,} chars")
    names = [n for n, _ in REGS]
    models = {"dense": Dense(), "sparse": Sparse(), "count": Count()}
    print("learned params:  dense=%d  sparse=%d (readout only)" % (models['dense'].params, models['sparse'].params))

    # retention[model][after_i][eval_j] = bpc; also track acc
    bpc_hist = {m: [] for m in models}; acc_hist = {m: [] for m in models}
    peak = {m: {} for m in models}
    t0 = time.time()
    for i, name in enumerate(names):
        Xtr, ytr, _, _ = data[name]
        for m in models: stream_train(models[m], Xtr, ytr)
        row_b = {m: {} for m in models}; row_a = {m: {} for m in models}
        for m in models:
            for j, ev in enumerate(names):
                _, _, Xev, yev = data[ev]
                b, a = models[m].eval(Xev[:T_EVAL], yev[:T_EVAL])
                row_b[m][ev] = b; row_a[m][ev] = a
            bpc_hist[m].append(row_b[m]); acc_hist[m].append(row_a[m])
        # peak = performance on a register right after training it
        for m in models: peak[m][name] = (row_b[m][name], row_a[m][name])
        print(f"  trained through [{name}]  ({time.time()-t0:.0f}s)")

    print("\n=== RETENTION: bits-per-char on each register after each training phase (lower=better) ===")
    for m in models:
        print(f"\n  -- {m} --")
        print("  after \\ eval " + "".join(f"{n[:5]:>9}" for n in names))
        for i, name in enumerate(names):
            print(f"  {name[:10]:<12}" + "".join(f"{bpc_hist[m][i][n]:>9.3f}" for n in names))

    print("\n=== FORGETTING (Δ bits-per-char on austen, after-all minus right-after-training-it) ===")
    print(f"  {'model':<8}{'austen peak bpc':>16}{'austen final bpc':>18}{'forgetting Δ':>14}")
    for m in models:
        peak_b = peak[m]["austen"][0]
        final_b = bpc_hist[m][-1]["austen"]
        print(f"  {m:<8}{peak_b:>16.3f}{final_b:>18.3f}{final_b-peak_b:>+14.3f}")
    # also report mean peak acc (did each model actually LEARN each register?)
    print("\n=== peak top-1 acc per register (did it learn?) ===")
    print(f"  {'model':<8}" + "".join(f"{n[:7]:>9}" for n in names))
    for m in models:
        print(f"  {m:<8}" + "".join(f"{peak[m][n][1]:>9.3f}" for n in names))
