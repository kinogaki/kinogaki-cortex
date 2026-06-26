"""kinogaki-cortex — the uniform-component architecture.

ONE repeated unit, the Column: an online associative predictor over a stream of tokens at one timescale.
The whole system is Columns wired together — scale by REPLICATION (more columns voting per Level) and by
STACKING (a Level's chunks become the tokens for the Level above). "cortex small vs big", same part.

  - Column  : the unit. backoff associative n-gram over int tokens. learn() + predict() (sparse count-dict).
  - Level   : N replicated Columns over one token stream at different orders (views). They VOTE
              (product-of-experts) — dense for the small char vocab, sparse for the huge word vocab.
  - Cortex  : a stack of Levels. char-Columns predict chars; word-Columns predict the CURRENT word from the
              previous words and (via the spelling lexicon) hand a top-down char prior back down to the char
              Level. Adding columns = wider; adding a higher-order word band = a longer-timescale level.

Char-columns generate characters; word-columns generate words BY USING the char-columns' chunks; the next-char
distribution is the char vote blended with every higher level's top-down prior. Alphabet matches lib/metrics.
"""
import math
import numpy as np

A = "abcdefghijklmnopqrstuvwxyz "; V0 = len(A); CH = {c: i for i, c in enumerate(A)}   # space-LAST, == metrics
ALPHA = 0.05

class Column:
    """The one repeated unit: an online backoff associative predictor over a stream of int tokens."""
    def __init__(self, order):
        self.order = order
        self.tab = [dict() for _ in range(order + 1)]       # tab[k][ctx_tuple] -> {token: count}
    def learn(self, seq):                                    # seq: tuple of ints
        for t in range(len(seq)):
            nx = seq[t]
            for k in range(self.order + 1):
                if k > t: break
                d = self.tab[k].setdefault(seq[t - k:t], {})
                d[nx] = d.get(nx, 0) + 1
    def predict(self, ctx):                                  # ctx: tuple of ints -> {token: count} or None
        for k in range(min(self.order, len(ctx)), -1, -1):
            d = self.tab[k].get(ctx[len(ctx) - k:] if k else ())
            if d: return d
        return None

def vote(dicts, vocab):
    """Voting across views → DENSE distribution (small vocab). LOG-LINEAR POOLING (geometric mean of the
    experts) — calibrated 'consensus = intersection of views' without the product-of-experts overconfidence
    (raw product sharpens with each added column → overfits; the geometric mean keeps the weights summing to 1).
    Abstainers (None) contribute nothing."""
    logp = np.zeros(vocab); n = 0
    for d in dicts:
        if not d: continue
        n += 1
        p = np.full(vocab, ALPHA); tot = ALPHA * vocab
        for tok, c in d.items():
            if tok < vocab: p[tok] += c; tot += c
        logp += np.log(p / tot)
    if n == 0: return np.full(vocab, 1.0 / vocab)
    logp /= n                                            # geometric mean = calibrated pool
    z = logp - logp.max(); e = np.exp(z); return e / e.sum()

def vote_sparse(dicts):
    """Log-linear pooling over a HUGE vocab, kept sparse: restrict to the union of the experts' supports
    (each expert's smoothing denominator is constant across candidates → cancels under normalization)."""
    active = [d for d in dicts if d]
    if not active: return None
    keys = set().union(*[set(d) for d in active]); n = len(active)
    out = {}
    for k in keys:
        lp = 0.0
        for d in active: lp += math.log(d.get(k, 0) + ALPHA)
        out[k] = lp / n                                  # geometric mean = calibrated pool
    m = max(out.values()); z = sum(math.exp(v - m) for v in out.values())
    return {k: math.exp(v - m) / z for k, v in out.items()}

class Level:
    """N replicated Columns over the SAME token stream at different orders, voting. Dense vote if `vocab`
    is given (chars), sparse otherwise (words)."""
    def __init__(self, orders, vocab=None):
        self.cols = [Column(o) for o in orders]; self.vocab = vocab
    def learn(self, seq):
        for c in self.cols: c.learn(seq)
    def dist(self, ctx):                                     # dense distribution over `vocab`
        return vote([c.predict(ctx) for c in self.cols], self.vocab)
    def predict_sparse(self, ctx):                           # sparse {token: prob}
        return vote_sparse([c.predict(ctx) for c in self.cols])

def char_prior(pw, prefix, id2spell):
    """Top-down: turn a predicted-next-WORD distribution (pw) + the current word prefix into a next-CHAR prior
    — the load-bearing 'higher level constrains the lower' step, expressed once and reused."""
    cp = np.zeros(V0); any_ = False
    for wid, prob in pw.items():
        sp = id2spell.get(wid)
        if sp is None or not sp.startswith(prefix): continue
        nc = sp[len(prefix)] if len(sp) > len(prefix) else " "      # finish the word, else a space
        if nc in CH: cp[CH[nc]] += prob; any_ = True
    return cp if any_ and cp.sum() > 0 else None

class Cortex:
    """A stack of Levels built from the ONE Column. Same component, wired bigger → better (the thesis).
       char_orders : the char Level's columns (wider vote = bigger).
       word_orders : the word Level's columns; [] = no word level. Higher orders = longer-timescale band.
    """
    def __init__(self, char_orders, word_orders=(), lam=0.4):
        self.charL = Level(char_orders, vocab=V0)
        self.wordL = Level(word_orders) if word_orders else None
        self.lam = lam
        self.K = 64                                          # context window the metrics suite should pass
        self.w2id = {}; self.id2spell = {0: None}
    def fit(self, text):
        self.charL.learn(tuple(CH[c] for c in text if c in CH))
        if self.wordL is None: return self
        words = [w for w in text.split(" ") if w]
        freq = {}
        for w in words: freq[w] = freq.get(w, 0) + 1
        self.w2id = {w: i + 1 for i, w in enumerate(w for w, c in freq.items() if c >= 2)}
        self.id2spell = {i: w for w, i in self.w2id.items()}; self.id2spell[0] = None
        self.wordL.learn(tuple(self.w2id.get(w, 0) for w in words))
        return self
    def dist(self, suffix):
        ids = tuple(CH[c] for c in suffix[-8:] if c in CH)
        pc = self.charL.dist(ids)
        if self.wordL is None: return pc
        parts = suffix.split(" ")
        prefix = parts[-1]
        completed = [p for p in parts[:-1] if p][-3:]
        pw = self.wordL.predict_sparse(tuple(self.w2id.get(w, 0) for w in completed))
        if not pw: return pc
        cp = char_prior(pw, prefix, self.id2spell)
        if cp is None: return pc
        return self.lam * pc + (1 - self.lam) * (cp / cp.sum())

def branch_chunk(stream, cols_fwd, cols_bwd, vocab, target_rate):
    """Unsupervised boundaries where forward+backward predictive entropy RISES (Exp A) — the same Columns,
    reused to discover the next level's tokens instead of relying on spaces. (Available; exp_i uses spaces
    for tractability — Exp A/C already proved the unsupervised version.)"""
    n = len(stream)
    def ent(cols, s):
        H = np.zeros(len(s))
        for t in range(len(s)):
            p = vote([c.predict(s[max(0, t - c.order):t]) for c in cols], vocab)
            H[t] = float(-(p * np.log2(p + 1e-12)).sum())
        return H
    Hf = ent(cols_fwd, stream); Hb = ent(cols_bwd, stream[::-1])[::-1]
    rise = lambda H: np.concatenate([[0.0], np.maximum(0.0, H[1:] - H[:-1])])
    score = rise(Hf) + rise(Hb[::-1])[::-1]
    thr = np.quantile(score, 1 - target_rate)
    chunks, cur = [], []
    for t in range(n):
        if t > 0 and score[t] >= thr and cur: chunks.append(tuple(cur)); cur = []
        cur.append(stream[t])
    if cur: chunks.append(tuple(cur))
    return chunks
