"""Count-only compositional reasoning — analogy + induced links, ONLINE, NO backprop — Exp AD.

Two old results meet here. (1) The parallelogram a:b::c:d is *already latent in raw co-occurrence
counts* (PMC11493305: word-analogy structure is recoverable from a PPMI matrix BEFORE any
factorization — SVD/word2vec only *smooth* it). (2) NARS (Wang) derives new links by syllogism with a
truth value (frequency f, confidence c) — induction/abduction from observed evidence, no training.
We translate both into the count cortex's hard online rule: a single streaming pass of order-independent
counting + leaky accumulators + online leader clustering. NOTHING here factorizes a matrix.

HARD ONLINE RULE (enforced, the banned list): single pass, learn-while-it-lives. NO gradient descent,
NO batch optimization, NO SVD / eigendecomposition / PMI-matrix factorization, NO word2vec. The ONLY
"smoothing" allowed is online LEADER CLUSTERING over context profiles (jepa.leader_cluster) — the
research question is precisely whether that count-native pooling can substitute for the banned SVD
smoothing that gives word2vec/LRA their human-parity analogy accuracy.

  CountProfiles   per-word bidirectional co-occurrence counts over a top-M context vocabulary, plus the
                  log-count profile logcount(.|w) and its sparse leader-cluster-smoothed twin.
  AnalogyCounts   a:b::c:? by 3CosAdd in LOG-COUNT space: find d ~ logcount(.|c) + (logcount(.|b) -
                  logcount(.|a)). Two profile sources: raw vs leader-smoothed (the SVD-substitution test).
  induce_links    NARS: from A->B (f1,c1) and A->C (f2,c2), INDUCE B<->C with a derived (f,c); from
                  A->B and observed B, ABDUCE A. Used to answer probes the pair never co-occurred on.

Everything below is `np.bincount` / `np.unique` accumulation = a batched form of a token-at-a-time
online update (the counts are identical to streaming one token at a time); reuses jepa.leader_cluster.
"""
import numpy as np
from jepa import online_signatures, leader_cluster


# ── co-occurrence count profiles: per-word context counts over a fixed top-M context vocab ──────────

class CountProfiles:
    """For each of N target words, a sparse count vector over the top-M CONTEXT words: how often context
    word j occurred within ±window of target word i. This IS the (unfactorized) co-occurrence matrix the
    analogy is supposed to live inside (PMC11493305). The log-count profile logcount(.|w) = log(1+count)
    is the per-word vector the parallelogram operates on — a SPARSE log-count delta, no SVD.

    Built vectorized per offset (np.bincount), identical to a token-at-a-time online accumulation."""

    def __init__(self, N, M, window=4, mode="ppmi"):
        self.N, self.M, self.window = N, M, window     # N target words, M context words (M<=N)
        self.mode = mode                               # "log" = log(1+count); "ppmi" = positive PMI weighting
        self.cooc = None                               # (N,M) float co-occurrence counts

    def fit(self, seq):
        """seq: dense top-word id stream (-1 = OOV). Accumulate ±window co-occurrence counts into (N,M).
        Context = the SAME id space, capped to the first M ids (the M most-frequent words, since ids are
        frequency-ranked by the caller). A target's context vector is its raw co-occurrence row. We then
        build the per-cell PROFILE matrix once, in the chosen mode."""
        N, M = self.N, self.M
        cooc = np.zeros(N * M, np.float64)
        for g in range(1, self.window + 1):
            for a, b in ((seq[:-g], seq[g:]), (seq[g:], seq[:-g])):   # both directions = symmetric context
                m = (a >= 0) & (b >= 0) & (b < M)                     # target a (any of N), context b (top-M)
                np.add.at(cooc, a[m] * M + b[m], 1.0)
        self.cooc = cooc.reshape(N, M)
        self.row_tot = self.cooc.sum(1)                              # total context mass per target word
        self.P = self._profile_matrix()                             # (N,M) the analogy operates on this
        return self

    def _profile_matrix(self):
        """The per-word profile the parallelogram operates on. Two count-native modes:
          'log'  : log(1+count) — compress the heavy count tail.
          'ppmi' : positive pointwise mutual information PMI(i,j)=log( p(i,j)/(p(i)p(j)) ), clipped at 0.
        PPMI is the representation the analogy literature says the parallelogram lives in (PMC11493305) — and
        it is PURE COUNTING: every cell is a closed-form ratio of counts. The BANNED step is *factorizing*
        this matrix (SVD/word2vec); computing the PMI values themselves is not factorization. No iteration."""
        C = self.cooc
        if self.mode == "log":
            return np.log1p(C)
        tot = C.sum()
        if tot <= 0:
            return np.log1p(C)
        pij = C / tot
        pi = C.sum(1, keepdims=True) / tot               # target marginal
        pj = C.sum(0, keepdims=True) / tot               # context marginal
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(pij / (pi * pj + 1e-12) + 1e-12)
        return np.maximum(pmi, 0.0)                       # positive PMI = clip negatives (count-native, no SVD)

    def logcount(self, w):
        """The profile row for word w (in the chosen mode) — the sparse vector the parallelogram operates on."""
        return self.P[w]

    def logcount_matrix(self, idx=None):
        """The profile matrix (or a subset) — (k,M). Vectorized; used to score candidate d's."""
        return self.P if idx is None else self.P[idx]


# ── online leader-cluster smoothing of the profiles (the SVD-substitution candidate) ────────────────

class LeaderSmoothing:
    """Replace the banned SVD smoothing with ONLINE LEADER CLUSTERING over context profiles. SVD/word2vec
    earn analogy parity by smoothing the sparse co-occurrence matrix into a dense low-rank space. We test
    whether a count-native pool does the same job: cluster the N words by their online co-occurrence
    SIGNATURE (jepa.online_signatures, a hashed/IDF random-sign projection by accumulation), then SMOOTH
    each word's log-count profile by mixing in its cluster CENTROID's mean profile:

        smoothed(w) = (1-beta) * logcount(.|w) + beta * mean_{v in cluster(w)} logcount(.|v)

    The centroid is the running mean of its members' log-count profiles — a leaky accumulator, not a
    factorization. beta=0 recovers raw counts; beta=1 is the pure centroid. This is the ONLY smoothing on
    the allowed list, and the headline question is whether it recovers the analogy signal SVD would."""

    def __init__(self, N, sig_D=96, sig_window=4, min_evidence=40, cos_thresh=0.8, cmax=600, seed=0):
        self.N = N
        self.sig_D, self.sig_window = sig_D, sig_window
        self.min_evidence, self.cos_thresh, self.cmax, self.seed = min_evidence, cos_thresh, cmax, seed
        self.clu = None; self.C = 0

    def fit(self, seq, profiles):
        """Build online signatures, online-leader-cluster the words (single pass, first-appearance order),
        then form each cluster's CENTROID log-count profile (running mean of members). profiles: CountProfiles."""
        sig, ev = online_signatures(seq, N=self.N, D=self.sig_D, window=self.sig_window, seed=self.seed)
        first_seen = np.full(self.N, len(seq), np.int64)
        valid = np.nonzero(seq >= 0)[0]
        np.minimum.at(first_seen, seq[valid], valid)
        order = np.argsort(first_seen)
        order = order[ev[order] >= self.min_evidence]
        clu, C = leader_cluster(sig, ev, order, min_evidence=self.min_evidence,
                                thresh=self.cos_thresh, Cmax=self.cmax)
        self.clu, self.C, self.ev = clu, C, ev
        # cluster centroid in LOG-COUNT space (mean of members' log-count profiles) — leaky-accumulator mean
        L = profiles.logcount_matrix()                          # (N,M)
        cent = np.zeros((max(C, 1), profiles.M), np.float64)
        cnt = np.zeros(max(C, 1), np.float64)
        m = clu >= 0
        np.add.at(cent, clu[m], L[m])
        np.add.at(cnt, clu[m], 1.0)
        cent /= np.maximum(cnt[:, None], 1.0)
        self.centroid = cent
        return self

    def smooth_matrix(self, profiles, beta):
        """All-rows smoothed log-count profiles (N,M): mix each word's profile with its cluster centroid.
        Unclustered words (-1) keep their raw profile (no centroid to borrow)."""
        L = profiles.logcount_matrix()
        if beta <= 0 or self.C == 0:
            return L
        out = L.copy()
        m = self.clu >= 0
        out[m] = (1 - beta) * L[m] + beta * self.centroid[self.clu[m]]
        return out


# ── analogy: 3CosAdd in log-count space, raw vs leader-smoothed ─────────────────────────────────────

class AnalogyCounts:
    """Solve a:b::c:? from a log-count profile matrix P (N,M). The relation vector r(a->b) = P[b] - P[a]
    is the sparse LOG-COUNT DELTA (NARS-style "what changes from a to b"). 3CosAdd ranks candidates d by
    cosine(P[d], P[c] + r(a->b)) = cosine(P[d], P[c] + P[b] - P[a]) — the parallelogram, read straight off
    counts. We mask out a,b,c from the candidate set (the standard, honest protocol). NO SVD."""

    def __init__(self, P):
        self.P = P                                              # (N,M) log-count (raw or smoothed) profiles
        n = np.linalg.norm(P, axis=1, keepdims=True)
        self.unit = P / np.maximum(n, 1e-9)                     # L2-normalized rows for cosine

    def solve(self, a, b, c, topk=10, restrict=None):
        """Return the top-k candidate word ids for d in a:b::c:d (best first). restrict: optional id array to
        score only those candidates (e.g. country names) — sharpens a fair, category-restricted evaluation."""
        target = self.P[c] + self.P[b] - self.P[a]              # parallelogram target in log-count space
        tn = target / max(np.linalg.norm(target), 1e-9)
        if restrict is None:
            scores = self.unit @ tn                             # cosine to every word
            cand = np.arange(self.P.shape[0])
        else:
            cand = np.asarray(restrict)
            scores = self.unit[cand] @ tn
        order = cand[np.argsort(scores)[::-1]]
        # exclude the three given words (standard analogy protocol)
        ban = {int(a), int(b), int(c)}
        out = [int(w) for w in order if int(w) not in ban]
        return out[:topk]


# ── NARS-style induced links: induction + abduction over count-derived (f,c) ───────────────────────

def truth_value(hits, total, k=1.0):
    """NARS truth value from counts: frequency f = hits/total (the conditional probability from counts),
    confidence c = total/(total+k) (more observations -> more confidence). Pure counting; k is the
    evidential horizon. Returns (f, c)."""
    if total <= 0:
        return 0.0, 0.0
    f = hits / total
    c = total / (total + k)
    return f, c


def induce(f1, c1, f2, c2):
    """NARS INDUCTION: from A->B (f1,c1) and A->C (f2,c2), induce B<->C. The induced frequency is f1 (the
    shared-premise term per NARS induction), the confidence is the product discounted by the standard
    w = c1*c2*f2 / (c1*c2*f2 + 1) form (more co-evidence from A -> more confidence in the B–C link)."""
    w = c1 * c2 * f2
    return f1, w / (w + 1.0)


def abduce(f1, c1, f2, c2):
    """NARS ABDUCTION: from A->B (f1,c1) and observed C->B (f2,c2), abduce A<-C (explain B by A). Symmetric
    confidence form to induction with the roles of the explained term swapped. Frequency = f2."""
    w = c1 * c2 * f1
    return f2, w / (w + 1.0)


class InducedLinks:
    """Held-out compositional prediction by NARS induction over forward count links. We split the window:
    LEFT context word L predicts a MIDDLE word A (link L->A), and A predicts a RIGHT target B (link A->B).
    For a probe (L, B) that NEVER co-occurred directly, we INDUCE L->B through every shared bridge A:

        P_induced(B | L) ∝ Σ_A  f(L->A)*c(L->A) * f(A->B)*c(A->B)

    i.e. transitive count composition — the model answers a relational probe it never saw co-occur. The
    direct count P(B|L) is the baseline; the lift is induced − direct on the held-out (never-co-occurred)
    slice. All counts + the NARS (f,c) combiner; single pass; no backprop."""

    def __init__(self, N, M, near=1, far=2, k=1.0):
        self.N, self.M = N, M
        self.near, self.far = near, far    # L is `near` steps left of A; B is `far` steps right of A
        self.k = k

    def fit(self, seq):
        """Build two forward count tables over offsets: LEFT->MID (gap=near) and MID->RIGHT (gap=far)."""
        self.la = _offset_table(seq, self.near, self.N, self.M)     # L (top-M) -> A (top-M)  counts
        self.ab = _offset_table(seq, self.far, self.N, self.M)      # A (top-M) -> B (top-M)  counts
        self.la_tot = self.la.sum(1)
        self.ab_tot = self.ab.sum(1)
        # direct L->B at the BRIDGED gap (near+far): the baseline + the never-co-occurred test
        self.lb = _offset_table(seq, self.near + self.far, self.N, self.M)
        self.lb_tot = self.lb.sum(1)
        return self

    def direct(self, L):
        """Direct count distribution P(B|L) at the bridged gap — the baseline (and the co-occurrence test)."""
        tot = self.lb_tot[L]
        return (self.lb[L] / tot) if tot > 0 else np.zeros(self.M)

    def induced(self, L, top_bridges=40):
        """Induced P(B|L) = Σ_A f(L->A)c(L->A) · f(A->B)c(A->B), over the strongest bridges A from L."""
        la_row = self.la[L]
        if self.la_tot[L] <= 0:
            return np.zeros(self.M)
        bridges = np.argsort(la_row)[::-1][:top_bridges]
        acc = np.zeros(self.M)
        for A in bridges:
            n_la = la_row[A]
            if n_la <= 0:
                continue
            f1, c1 = truth_value(n_la, self.la_tot[L], self.k)
            ab_row = self.ab[A]; tA = self.ab_tot[A]
            if tA <= 0:
                continue
            # vectorized over all B: f2 = ab_row/tA, c2 = tA/(tA+k); contribution = f1*c1 * f2*c2
            f2 = ab_row / tA
            c2 = tA / (tA + self.k)
            acc += (f1 * c1) * (f2 * c2)
        s = acc.sum()
        return acc / s if s > 0 else acc


def _offset_table(seq, gap, N, M):
    """Dense (N,M) forward co-occurrence counts at a fixed positive offset `gap`: how often context word b
    (top-M) appeared exactly `gap` steps after target word a (top-N). Vectorized = online accumulation."""
    a = seq[:-gap]; b = seq[gap:]
    m = (a >= 0) & (b >= 0) & (b < M)
    tab = np.zeros(N * M, np.float64)
    np.add.at(tab, a[m] * M + b[m], 1.0)
    return tab.reshape(N, M)
