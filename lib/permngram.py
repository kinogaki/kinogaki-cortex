"""Exp AP — permutation-bound n-grams + FlyHash addressing. ONLINE, count-based, NO backprop.

The phrase-level sparsity problem. A literal n-gram counts next-tokens at the EXACT string "a b c". Seen once,
its continuation has a near-zero count even if dozens of SIMILAR phrases ("the b c", "a b d") were common: each
literal phrase is its own island, and the counts never pool. The bag-of-context fix ("count next-token given the
SET {a,b,c}") pools too much — it throws away ORDER, so "abc" and "cab" collapse to the same key.

The VSA fix (Kanerva's hyperdimensional computing; Joshi/Kanerva HD text; Dasgupta's FlyHash). Give every word a
fixed random ATOM hypervector. Encode an n-gram as a single ORDER-PRESERVING address by BINDING shifted atoms:

    addr = rho^2(A) (x) rho^1(B) (x) rho^0(C)      for the 3-gram "A B C"

  - rho = a fixed cyclic PERMUTATION (np.roll). rho^k tags the k-th-from-last slot, so POSITION is carried in the
    vector itself: "abc" and "cab" bind different shifted atoms -> different addresses. Order is preserved, unlike
    a bag.
  - (x) = BIND = elementwise product of bipolar (+-1) vectors. Binding is similarity-DISTRIBUTIVE: if two phrases
    share most of their (shifted) atoms, their bound addresses stay CLOSE in Hamming/cosine space. Similar phrases
    -> similar dense addresses. That is the pooling we want.

Then FLYHASH the dense address into a SPARSE one (Dasgupta et al. 2017, the fly olfactory circuit): a sparse
EXPANSIVE random projection (dense D -> wide M, M >> D) followed by top-k winner-take-all. FlyHash is
locality-sensitive: NEARBY inputs light up OVERLAPPING winner sets. So similar phrases get overlapping sparse
addresses (a small set of active buckets), and we COUNT the next token at every active bucket. Counting at shared
buckets is exactly count POOLING across similar phrases — generalization, with no backprop, no factorization.

At predict time we FlyHash the query phrase the same way, gather the next-token count vectors from its active
buckets, sum them, and normalize. A phrase never seen in this exact form still lands on buckets that COMMON
similar phrases populated, so it inherits their continuation statistics.

HARD ONLINE RULE (enforced). Atom vectors + the FlyHash projection are FIXED random structure (a random
projection, allowed). Everything learned is additive COUNTS at buckets, accumulated in one streaming pass —
order-independent, identical to a token-at-a-time online update. NO gradients, NO SVD/eigen, NO k-means, NO
backprop. Bounded memory: M buckets x V next-token counts, plus N atom rows.

Three predictors, same data, same single pass:
  LiteralNgram   count next-token at the EXACT phrase string (the sparsity baseline).
  BagContext     count next-token at the unordered SET of context words (order-blind control).
  PermFlyNgram   permutation-bind -> FlyHash -> count at sparse buckets (the VSA model under test).
"""
import numpy as np


# ── fixed random structure: atom hypervectors + a cyclic permutation ──────────────────────────────

def atom_vectors(N, D, seed=0):
    """Fixed bipolar (+-1) atom hypervector per word id (N x D). A random projection's basis — not learned."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(N, D)).astype(np.int8) * 2 - 1)          # +-1


def bind_ngram(atoms, ctx_ids, D):
    """Order-preserving bind of an n-gram's atoms: rho^(n-1-k) applied to the k-th context atom, then elementwise
    product across the window. ctx_ids: (M, n) int array of dense word ids (oldest..newest, left to right).
    rho = cyclic shift by 1 (np.roll along the feature axis). Returns (M, D) bipolar address per row.

    rho^j(atom) = np.roll(atom, j) shifts features by j; binding (product) of differently-shifted atoms keeps the
    SLOT identity of each word, so order matters. Vectorized over all M rows (a batched online encode)."""
    M, n = ctx_ids.shape
    addr = np.ones((M, D), np.int8)
    for k in range(n):
        shift = n - 1 - k                                                     # newest word gets rho^0
        a = atoms[ctx_ids[:, k]]                                              # (M, D) bipolar
        if shift:
            a = np.roll(a, shift, axis=1)
        addr = addr * a                                                       # bind = elementwise product
    return addr                                                              # (M, D) in +-1


# ── FlyHash: sparse expansive random projection + top-k winner-take-all (Dasgupta et al. 2017) ────

class FlyHash:
    """Locality-sensitive sparse code. A SPARSE random projection blows D up to M (M >> D); each output bucket
    sums a few random input dims; top-k WTA keeps only the k strongest buckets. Nearby inputs -> overlapping
    winners. Fixed given seed (random structure, not learned). Bounded: M buckets, s connections each."""

    def __init__(self, D, M, k, s=8, seed=0):
        self.D, self.M, self.k = D, M, k
        rng = np.random.default_rng(seed)
        # each output bucket samples s input dims (sparse expansive projection, the fly's PN->KC connectivity)
        self.proj = rng.integers(0, D, size=(M, s)).astype(np.int64)         # (M, s) input indices per bucket

    def encode(self, addr):
        """addr: (B, D) bipolar. Return (B, k) int bucket ids = the k strongest buckets (the sparse code).
        Bucket activity = sum of its s sampled input dims; top-k by activity (winner-take-all)."""
        B = addr.shape[0]
        act = addr[:, self.proj].sum(axis=2)                                 # (B, M) bucket activities
        # top-k buckets per row (unordered partition is enough — we only use the SET of winners)
        return np.argpartition(act, -self.k, axis=1)[:, -self.k:]            # (B, k) winning bucket ids


# ── the three predictors ──────────────────────────────────────────────────────────────────────────

class LiteralNgram:
    """Count next-token at the EXACT n-gram string. Dict keyed by packed context tuple -> next-token counts.
    The sparsity baseline: an unseen-in-this-exact-form phrase backs off to the unigram (floors it)."""

    def __init__(self, V, n, alpha=0.1):
        self.V, self.n, self.alpha = V, n, alpha
        self.table = {}                                                      # ctx_key -> dict(next -> count)
        self.uni = np.zeros(V, np.float64)

    def fit(self, ctx_ids, nxt):
        # vectorized pack of the context tuple into one int key (online additive counts, order-independent)
        key = _pack(ctx_ids, self.V)
        order = np.argsort(key, kind="stable")
        key, nx = key[order], nxt[order]
        edges = np.nonzero(np.diff(key))[0] + 1
        starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(key)]])
        for s, e in zip(starts, ends):
            c = np.bincount(nx[s:e], minlength=self.V).astype(np.float64)
            self.table[int(key[s])] = c
        self.uni = np.bincount(nxt, minlength=self.V).astype(np.float64)
        self.uni /= self.uni.sum()
        return self

    def prob(self, ctx_row):
        c = self.table.get(int(_pack(ctx_row[None, :], self.V)[0]))
        if c is None:
            return (self.uni + self.alpha / self.V) / (1 + self.alpha)       # backoff: floored unigram
        return (c + self.alpha * self.uni) / (c.sum() + self.alpha)


class BagContext:
    """ORDER-BLIND control: count next-token at the SORTED set of the context word ids. 'abc' and 'cab' collapse
    to the same key here — so it pools across permutations (and should NOT degrade under scrambling)."""

    def __init__(self, V, n, alpha=0.1):
        self.V, self.n, self.alpha = V, n, alpha
        self.table = {}
        self.uni = np.zeros(V, np.float64)

    def fit(self, ctx_ids, nxt):
        bag = np.sort(ctx_ids, axis=1)                                       # order-blind key
        key = _pack(bag, self.V)
        order = np.argsort(key, kind="stable")
        key, nx = key[order], nxt[order]
        edges = np.nonzero(np.diff(key))[0] + 1
        starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(key)]])
        for s, e in zip(starts, ends):
            self.table[int(key[s])] = np.bincount(nx[s:e], minlength=self.V).astype(np.float64)
        self.uni = np.bincount(nxt, minlength=self.V).astype(np.float64); self.uni /= self.uni.sum()
        return self

    def prob(self, ctx_row):
        c = self.table.get(int(_pack(np.sort(ctx_row)[None, :], self.V)[0]))
        if c is None:
            return (self.uni + self.alpha / self.V) / (1 + self.alpha)
        return (c + self.alpha * self.uni) / (c.sum() + self.alpha)


class PermFlyNgram:
    """The VSA model. permutation-bind the n-gram -> FlyHash to k sparse buckets -> COUNT next-token at every
    active bucket. Similar phrases share buckets, so their next-token counts POOL (generalization). Prediction
    sums the next-token count vectors over the query's active buckets. Bounded memory: M x V counts.

    Memory note: M*V dense floats can be large; we keep per-bucket counts in a dict-of-sparse only if needed,
    but for the experiment sizes here a dense (M, V) float32 table is fine and fast (vectorized scatter-add)."""

    def __init__(self, atoms, fly, V, n, alpha=0.1):
        self.atoms, self.fly, self.V, self.n, self.alpha = atoms, fly, V, n, alpha
        self.D = atoms.shape[1]
        self.bucket = np.zeros((fly.M, V), np.float32)                       # bounded count table
        self.uni = np.zeros(V, np.float64)

    def _addr_buckets(self, ctx_ids):
        addr = bind_ngram(self.atoms, ctx_ids, self.D)                       # (B, D) bipolar
        return self.fly.encode(addr)                                         # (B, k) bucket ids

    def fit(self, ctx_ids, nxt):
        buckets = self._addr_buckets(ctx_ids)                                # (B, k)
        # scatter every (bucket, next) pair: counting at shared buckets = pooling across similar phrases
        flat_b = buckets.reshape(-1)
        flat_n = np.repeat(nxt, self.fly.k)
        np.add.at(self.bucket, (flat_b, flat_n), 1.0)                        # online additive counts
        self.uni = np.bincount(nxt, minlength=self.V).astype(np.float64); self.uni /= self.uni.sum()
        return self

    def prob_batch(self, ctx_ids):
        """Vectorized predict for many rows: sum next-token counts over each row's k active buckets, smooth."""
        buckets = self._addr_buckets(ctx_ids)                                # (B, k)
        agg = self.bucket[buckets].sum(axis=1).astype(np.float64)            # (B, V) pooled counts
        tot = agg.sum(axis=1, keepdims=True)
        return (agg + self.alpha * self.uni) / (tot + self.alpha)

    def prob(self, ctx_row):
        return self.prob_batch(ctx_row[None, :])[0]


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────

def _pack(ctx_ids, V):
    """Pack an (M, n) id array into one int64 key per row (mixed-radix V). Deterministic, order-as-given."""
    M, n = ctx_ids.shape
    key = np.zeros(M, np.int64)
    for k in range(n):
        key = key * V + ctx_ids[:, k]
    return key


def make_windows(seq, n):
    """seq: dense top-word id stream (-1 = OOV). Return (ctx_ids:(M,n), nxt:(M,)) over windows with NO OOV in
    the context or target. ctx[k] is oldest..newest; nxt is the word after the window. Single pass, vectorized."""
    cols = [seq[i:len(seq) - n + i] for i in range(n)]                       # context columns
    nxt = seq[n:]
    ctx = np.stack(cols, axis=1)                                             # (M, n)
    m = (ctx >= 0).all(axis=1) & (nxt >= 0)
    return ctx[m].astype(np.int64), nxt[m].astype(np.int64)


def perplexity(p):
    return float(np.exp(-np.mean(np.log(np.clip(p, 1e-12, 1.0)))))
