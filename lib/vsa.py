"""Vector Symbolic Architecture — gradient-free COMPOSITION and DECODE over hypervectors.

The lineage (research/IDEAS, PROVENANCE): Plate's Holographic Reduced Representations (HRR), Kanerva's
Hyperdimensional Computing (HDC / "spatter codes"), and Frady/Kent/Sommer's RESONATOR NETWORKS. A VSA
gives three algebraic operators on fixed random ±1 hypervectors, all of them pure elementwise arithmetic:

    BIND    a ⊛ b   = elementwise product   (its OWN inverse for ±1: a⊛a = 1)
    BUNDLE  a ⊕ b   = sum, then sign         (a superposition that stays SIMILAR to its parts)
    PERMUTE rho(a)  = cyclic shift           (encode order / protect a factor from commuting away)

No gradients, no batch optimization, no SVD/eigen/word2vec — assigning a fixed random sign vector per atom
is a hash, not a learned embedding; everything after is multiply/add/shift. This is exactly the count cortex's
allowed kit (random projection + accumulation), now wearing the VSA algebra.

WHY HERE. Exp AD found the analogy parallelogram already lives in raw co-occurrence counts, but the cortex
lacks a count-native combiner that SHARPENS a composition without BLURRING it (leader-cluster averaging
destroyed the relation deltas). VSA is a candidate sharpening-combiner with a bonus the count side never
had: it can be READ BACK OUT. A bundled triple role_s⊛S + role_v⊛V + role_o⊛O can be DECODED — unbind a
role, clean up the noisy result against the codebook — recovering structure from a sum. That is composition
you can invert. Its known failure mode is CAPACITY: a bundle of too many noisy terms drowns the signal, and
the resonator's iterative factoring has a hard combinatorial ceiling. We measure exactly where it breaks.
"""
import numpy as np


# ── codebook: one fixed random ±1 hypervector per atom (a hash, not a learned embedding) ──────────────

class Codebook:
    """A fixed bank of N random ±1 hypervectors of dimension D — the atoms (words, roles). Assigned once
    from a seeded RNG; never updated. This is the VSA's only 'memory' of an atom's identity, and it is a
    RANDOM PROJECTION (allowed), not a trained vector. Cleanup = nearest-neighbor against this bank, which
    is exactly the leader-cluster cleanup the cortex already does, run over the codebook instead of clusters."""

    def __init__(self, n, D, seed=0):
        rng = np.random.default_rng(seed)
        self.D = D
        self.V = rng.choice(np.array([-1.0, 1.0], np.float32), size=(n, D))   # (n,D) ±1 atoms
        self._unit = None

    def __len__(self):
        return self.V.shape[0]

    @property
    def unit(self):
        """L2-normalized atoms, cached — for cosine cleanup (rows are ±1 so norm = sqrt(D), but keep general)."""
        if self._unit is None:
            self._unit = self.V / np.maximum(np.linalg.norm(self.V, axis=1, keepdims=True), 1e-9)
        return self._unit

    def cleanup(self, x, topk=1, restrict=None):
        """Nearest codebook atom(s) to a noisy vector x (best first). restrict: optional id subset to score
        only those atoms — the category-restricted cleanup. Cosine similarity, argpartition for speed."""
        xn = x / max(np.linalg.norm(x), 1e-9)
        U = self.unit if restrict is None else self.unit[restrict]
        sims = U @ xn
        if topk >= sims.shape[0]:
            order = np.argsort(sims)[::-1]
        else:
            part = np.argpartition(sims, -topk)[-topk:]
            order = part[np.argsort(sims[part])[::-1]]
        ids = order if restrict is None else restrict[order]
        return [int(i) for i in ids[:topk]]

    def cleanup_sims(self, x, restrict=None):
        """Raw cosine of x to every (restricted) atom — for resonator updates that want the full similarity."""
        xn = x / max(np.linalg.norm(x), 1e-9)
        U = self.unit if restrict is None else self.unit[restrict]
        return U @ xn


# ── the three VSA operators (pure elementwise arithmetic, no learning) ─────────────────────────────────

def bind(a, b):
    """BIND = elementwise product. For ±1 vectors this is its own inverse (a⊛a = all-ones), so unbind = bind."""
    return a * b

def unbind(a, b):
    """Recover the other factor of a bind: (a⊛b) ⊛ a = b (for ±1 atoms). Same op as bind — multiply it back."""
    return a * b

def bundle(vs, binarize=True):
    """BUNDLE = superposition: sum the vectors, optionally sign() back to ±1. The sum stays SIMILAR to each
    summand (that is the whole trick) but similarity DEGRADES with the number of terms — the capacity limit.
    Keeping the real-valued sum (binarize=False) preserves more signal for decode; sign() is the HDC default."""
    s = np.sum(vs, axis=0)
    if binarize:
        s = np.sign(s)
        s[s == 0] = 1.0          # break ties deterministically (no RNG in the hot path)
    return s.astype(np.float32)

def permute(a, k=1):
    """PERMUTE = cyclic shift by k — encodes order / protects a factor so bind stays non-commutative."""
    return np.roll(a, k)

def unpermute(a, k=1):
    return np.roll(a, -k)


# ── compositional structure: role-filler triples ──────────────────────────────────────────────────────

def encode_triple(roles, fillers, binarize=True):
    """Encode a structured record as a bundle of role⊛filler binds:  T = Σ_i role_i ⊛ filler_i.
    roles, fillers: lists of equal length of ±1 hypervectors. Returns one hypervector (the whole record).
    Decode a slot by unbinding its role and cleaning up:  cleanup(T ⊛ role_i) ≈ filler_i."""
    return bundle([bind(r, f) for r, f in zip(roles, fillers)], binarize=binarize)

def decode_slot(T, role):
    """Unbind a role from a bundled record, returning the NOISY filler estimate (clean it up against a codebook).
    Works because role_i ⊛ T = filler_i + Σ_{j!=i} role_i⊛role_j⊛filler_j — signal + zero-mean crosstalk."""
    return unbind(role, T)


# ── resonator network: factor a bundle of PRODUCTS into its atoms (Frady/Kent/Sommer) ──────────────────

class Resonator:
    """A RESONATOR NETWORK factors a hypervector known to be a BIND of one atom from each of F codebooks:
    s = x1 ⊛ x2 ⊛ ... ⊛ xF, where x_f is some unknown atom of codebook f. It cannot be unbound directly
    (we don't know the other factors), so the resonator holds an ESTIMATE x̂_f per factor and iterates:

        x̂_f  <-  cleanup_f( s ⊛ Π_{g!=f} x̂_g )

    i.e. unbind everyone else's current guess, clean the residual up to the nearest atom of codebook f, in
    parallel for all f, until the estimates stop moving. The cleanup is the SAME nearest-neighbor / leader
    step the cortex already runs. This searches a product space of size Π|codebook_f| WITHOUT enumerating it
    — the gradient-free 'reading structure back out of a product'. Its ceiling is combinatorial: too many
    factors or too-small D and it never locks (spurious fixed points). We report where that ceiling is."""

    def __init__(self, codebooks, max_iter=100, seed=0):
        self.cbs = codebooks                    # list of Codebook, one per factor
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    def factor(self, s):
        """Return (ids, iters, locked): the recovered atom id per factor, #iterations, whether it converged
        (estimates stable two steps running). Soft cleanup: keep a real-valued estimate = Σ sim_a * atom_a so
        the network can hold superposed guesses early and sharpen — the standard resonator update."""
        F = len(self.cbs)
        # init each estimate as the superposition of its whole codebook (mean atom) — unbiased start
        est = [cb.V.mean(0).astype(np.float32) for cb in self.cbs]
        last_ids = None
        for it in range(self.max_iter):
            new = []
            for f in range(F):
                resid = s.copy()
                for g in range(F):
                    if g != f:
                        resid = resid * est[g]          # unbind everyone else's current estimate
                sims = self.cbs[f].cleanup_sims(resid)  # cosine to every atom of codebook f
                w = np.maximum(sims, 0.0)               # rectify — keep only supporting atoms
                if w.sum() <= 0:
                    w = np.ones_like(sims)
                e = (w[:, None] * self.cbs[f].V).sum(0) # soft estimate = similarity-weighted atom sum
                new.append(np.sign(e).astype(np.float32))   # binarize back to the code manifold
            est = new
            ids = tuple(int(np.argmax(cb.cleanup_sims(est[f]))) for f, cb in enumerate(self.cbs))
            if ids == last_ids:
                return ids, it + 1, True
            last_ids = ids
        return last_ids, self.max_iter, False


# ── mapping vector: gradient-free analogy by a single transform (the AD comparison) ────────────────────

def mapping_vector(a, b):
    """The transform that carries a -> b under BIND:  T = a ⊛ b  (since a is its own inverse, a⊛T = b).
    This is the VSA analogue of AD's relation delta r(a->b) = profile[b] - profile[a], but multiplicative.
    Apply to c:  c ⊛ T should land near the d that completes a:b::c:d  — IF the relation is bind-stable."""
    return bind(a, b)

def apply_mapping(c, T):
    """Carry c across the analogy: c ⊛ T ≈ d for a:b::c:d, when T = a⊛b. Clean up against a codebook."""
    return bind(c, T)
