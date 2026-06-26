"""Exp Z — similarity reps as a REPRESENTATION FACTORY, projected INTO the count predictors. ONLINE, NO backprop.

The thesis (from Exp P / W). Proximity / "raytracing" is a bad PREDICTOR but a good SIMILARITY tool: nearness
in co-occurrence space is meaning, cleanly (numbers cluster, countries cluster). So stop asking it to predict.
Use it as a representation factory — build similarity reps HIERARCHICALLY (words, then phrases) — and PROJECT
those reps INTO the count-based columnar predictors. The rare-context play: when a context word/phrase has few
or zero direct next-word counts, substitute/augment with its similarity CLUSTER's aggregated counts, so a rare
item inherits its neighbours' statistics. Hybrids are the point.

HARD ONLINE RULE (enforced): single streaming pass, learn-while-it-lives. NO gradient descent, NO k-means, NO
SVD/eigendecomposition, NO PPMI factorization. The reps are built by reusing jepa.py's ONLINE machinery:
  - online_signatures: per-word running hashed co-occurrence count vector (random sign-projection by
    accumulation, IDF-discounted) — order-independent counting.
  - leader_cluster: single-pass nearest-running-mean-prototype-or-spawn online leader clustering.
Everything here is counting + leaky/aggregate accumulation. The np.unique / bincount builders are batched
implementations of order-independent accumulation (identical to a token-at-a-time online update); nothing
iterates-to-convergence or backprops.

Pieces:
  WordReps      L1: word -> online co-occurrence signature -> online leader-cluster id.
  PhraseReps    L2: branching-entropy phrase units -> bag of member word-reps -> online leader-cluster id.
  BigramCounts  the count predictor to project INTO: P(next_word | prev_word), plus per-cluster aggregated
                next-word counts so a rare/zero context word can BACK OFF onto its similarity cluster.
  SimBackoffLM  the hybrid: bigram with similarity-cluster backoff (word level, and +phrase level).
"""
import numpy as np
from jepa import online_signatures, leader_cluster


# ── L1 word representations: online signature -> online leader-cluster id ─────────────────────────

class WordReps:
    """Each word -> an online co-occurrence signature (jepa.online_signatures) -> an online leader-cluster id
    (jepa.leader_cluster). Single streaming pass; clusters are running-mean prototypes, assign-or-spawn."""

    def __init__(self, N, D=64, sig_window=5, min_evidence=40, cos_thresh=0.75, cmax=400, seed=0):
        self.N, self.D, self.sig_window = N, D, sig_window
        self.min_evidence, self.cos_thresh, self.cmax, self.seed = min_evidence, cos_thresh, cmax, seed
        self.clu = None; self.C = 0; self.ev = None

    def fit(self, seq):
        """seq: dense top-word id stream (-1 = OOV). Build signatures, then cluster in first-appearance order."""
        sig, ev = online_signatures(seq, N=self.N, D=self.D, window=self.sig_window, seed=self.seed)
        first_seen = np.full(self.N, len(seq), np.int64)
        valid = np.nonzero(seq >= 0)[0]
        np.minimum.at(first_seen, seq[valid], valid)            # earliest stream position per top-word
        order = np.argsort(first_seen)
        order = order[ev[order] >= self.min_evidence]
        clu, C = leader_cluster(sig, ev, order, min_evidence=self.min_evidence,
                                thresh=self.cos_thresh, Cmax=self.cmax)
        self.clu, self.C, self.ev, self.sig = clu, C, ev, sig
        return self


# ── L2 phrase representations: branching-entropy units -> bag of word-reps -> leader-cluster ──────

def _follower_entropy(seq, vocab):
    """Forward branching entropy per token id (reused logic from boundaries.py; kept local to avoid editing it)."""
    nxt = {}
    for a, b in zip(seq[:-1], seq[1:]):
        d = nxt.setdefault(int(a), {}); d[int(b)] = d.get(int(b), 0) + 1
    H = np.zeros(vocab)
    for a, d in nxt.items():
        c = np.array(list(d.values()), float); p = c / c.sum()
        H[a] = float(-(p * np.log2(p + 1e-12)).sum())
    return H


def phrase_cuts(wids, vocab, target_rate=0.5):
    """Cut the word stream into phrases where forward+backward branching entropy rises. List of (start,end)
    spans over wids. Same signal as boundaries.phrase_cuts, replicated here so we don't edit the shared file."""
    wids = np.asarray(wids)
    Hf = _follower_entropy(wids, vocab)
    Hb = _follower_entropy(wids[::-1], vocab)
    score = Hf[wids[:-1]] + Hb[wids[1:][::-1]][::-1]
    thr = np.quantile(score, 1 - target_rate)
    cut = np.concatenate([[True], score >= thr, [True]])
    edges = np.nonzero(cut)[0]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


class PhraseReps:
    """L2: discover phrase units by branching entropy, represent each phrase by the MEAN of its member words'
    online signatures (a bag of word-reps), then ONLINE-LEADER-CLUSTER the phrase signatures -> phrase concept
    ids. Higher-unit similarity, built the same online way.

    A 'phrase type' is keyed by the tuple of its member dense-word-ids (so the same multi-word unit recurs).
    Its signature is the count-weighted running sum of member word signatures (mean ∝ sum) — accumulation, no
    factorization. Clustering reuses jepa.leader_cluster (single pass, running-mean prototypes)."""

    def __init__(self, word_reps, min_evidence=8, cos_thresh=0.75, cmax=300, target_rate=0.5, seed=0):
        self.wr = word_reps
        self.min_evidence, self.cos_thresh, self.cmax = min_evidence, cos_thresh, cmax
        self.target_rate, self.seed = target_rate, seed
        self.phrase_clu = None; self.PC = 0
        self.key2pid = {}; self.pid_clu = None

    def fit(self, seq, vocab):
        """seq: dense top-word stream (-1=OOV). Returns spans + per-span phrase-type id; builds phrase clusters.
        Phrase types with <2 words are skipped (single words already have an L1 rep)."""
        spans = phrase_cuts(np.where(seq < 0, vocab, seq), vocab + 1, target_rate=self.target_rate)
        wsig = self.wr.sig                                  # (N,D) word signatures (online-accumulated)
        D = wsig.shape[1]
        key2pid = {}
        psig_sum = []                                       # running sum of member-word signatures per phrase type
        pev = []                                            # evidence (#occurrences) per phrase type
        members = []                                        # member word reps per phrase type (for display)
        span_pid = np.full(len(spans), -1, np.int64)
        first_seen = []
        for i, (s, e) in enumerate(spans):
            ws = seq[s:e]
            ws = ws[ws >= 0]
            if len(ws) < 2:                                 # need a multi-word unit
                continue
            key = ws.tobytes()
            pid = key2pid.get(key)
            if pid is None:
                pid = len(key2pid); key2pid[key] = pid
                psig_sum.append(wsig[ws].mean(axis=0))      # bag-of-word-reps = mean of member signatures
                pev.append(0); members.append(ws); first_seen.append(i)
            pev[pid] += 1
            span_pid[i] = pid
        PT = len(key2pid)
        if PT == 0:
            self.spans, self.span_pid, self.key2pid = spans, span_pid, key2pid
            self.pid_clu = np.zeros(0, np.int64); self.PC = 0
            return self
        psig = np.array(psig_sum)                           # (PT, D) one signature per phrase TYPE
        pev = np.array(pev, np.int64)
        order = np.argsort(np.array(first_seen))            # first-appearance order = stream order
        order = order[pev[order] >= self.min_evidence]
        pid_clu, PC = leader_cluster(psig, pev, order, min_evidence=self.min_evidence,
                                     thresh=self.cos_thresh, Cmax=self.cmax)
        self.spans, self.span_pid, self.key2pid = spans, span_pid, key2pid
        self.pid_clu, self.PC, self.pev, self.members = pid_clu, PC, pev, members
        return self

    def phrase_cluster_at(self, seq, span):
        """The phrase-cluster id covering a span key, or -1 (unseen / sub-threshold phrase)."""
        s, e = span
        ws = seq[s:e]; ws = ws[ws >= 0]
        pid = self.key2pid.get(ws.tobytes())
        if pid is None or self.PC == 0:
            return -1
        return int(self.pid_clu[pid])


# ── the count predictor to project INTO: bigram + per-cluster aggregated next-word counts ─────────

class BigramCounts:
    """P(next_word | prev_word) by counting (vectorized = order-independent online accumulation). Also
    aggregates, for each word CLUSTER, the next-word counts pooled over all member words — so a rare or unseen
    prev-word can BACK OFF onto its similarity cluster's statistics (projecting the rep onto the count keys).

    Stored as CSR-like flat arrays for a fast, vectorized eval over a probe set."""

    def __init__(self, N, clu, C, alpha=0.1):
        self.N, self.clu, self.C, self.alpha = N, clu, C, alpha

    def fit(self, seq):
        """seq: dense top-word stream (-1=OOV). Count bigrams; build per-cluster aggregated next-word dists."""
        a, b = seq[:-1], seq[1:]
        m = (a >= 0) & (b >= 0)
        a, b = a[m].astype(np.int64), b[m].astype(np.int64)
        self.prev_count = np.bincount(a, minlength=self.N).astype(np.float64)   # how often each word is a context
        self.uni = np.bincount(b, minlength=self.N).astype(np.float64)
        self.uni_total = float(self.uni.sum())

        # per prev-WORD next dist (CSR over prev word)
        self._wkey = a * self.N + b
        uk, uc = np.unique(self._wkey, return_counts=True)
        self.w_prev = (uk // self.N).astype(np.int64)
        self.w_next = (uk % self.N).astype(np.int64)
        self.w_cnt = uc.astype(np.float64)

        # per prev-CLUSTER next dist (project the rep onto the count keys: pool members' next-word counts)
        ca = np.where(self.clu[a] >= 0, self.clu[a], -1)
        mc = ca >= 0
        ckey = ca[mc].astype(np.int64) * self.N + b[mc]
        if ckey.size:
            uk, uc = np.unique(ckey, return_counts=True)
            self.c_prev = (uk // self.N).astype(np.int64)
            self.c_next = (uk % self.N).astype(np.int64)
            self.c_cnt = uc.astype(np.float64)
            self.c_tot = np.bincount(self.c_prev, weights=self.c_cnt, minlength=self.C)
        else:
            self.c_prev = np.zeros(0, np.int64); self.c_next = np.zeros(0, np.int64)
            self.c_cnt = np.zeros(0); self.c_tot = np.zeros(self.C)
        # index next-dist rows by prev word / prev cluster for fast lookup
        self._w_index = _row_index(self.w_prev, self.N)
        self._c_index = _row_index(self.c_prev, self.C)
        return self

    def word_row(self, w):
        """(next_ids, counts) for prev-word w, or (empty, empty)."""
        s, e = self._w_index[w], self._w_index[w + 1]
        return self.w_next[s:e], self.w_cnt[s:e]

    def cluster_row(self, c):
        if c < 0:
            return np.zeros(0, np.int64), np.zeros(0)
        s, e = self._c_index[c], self._c_index[c + 1]
        return self.c_next[s:e], self.c_cnt[s:e]


def _row_index(prev_sorted, n_rows):
    """Build a CSR offset array: index[r]=start of row r in the (sorted-by-prev) arrays. prev_sorted is the
    unique-sorted prev key column (np.unique returns sorted keys, so prev cols are non-decreasing)."""
    idx = np.zeros(n_rows + 1, np.int64)
    counts = np.bincount(prev_sorted, minlength=n_rows)
    idx[1:] = np.cumsum(counts)
    return idx


# ── the hybrid language models ────────────────────────────────────────────────────────────────────

def _dist_from_row(next_ids, cnt, N, alpha, uni, uni_total):
    """Turn a sparse (next_ids,counts) row into a dense add-alpha-smoothed distribution backed by the unigram
    prior (so unseen next-words get the corpus unigram floor, not a flat floor)."""
    prior = (uni + alpha) / (uni_total + alpha * N)          # unigram prior (a count-based backoff)
    p = prior.copy() * alpha                                 # smoothing mass spread by the prior
    tot = cnt.sum() + alpha
    p[next_ids] += cnt
    return p / (tot)


class SimBackoffLM:
    """The hybrid. Score P(next | prev) three ways, sharing the SAME count tables (BigramCounts):

      mode 'bigram'   : count-only baseline. P(next|prev_word) add-alpha over the unigram prior.
      mode 'wordrep'  : similarity backoff at the WORD level. Mix the prev word's own next dist with its
                        cluster's aggregated next dist; the cluster's weight RISES as the prev word's direct
                        evidence FALLS (rare items inherit neighbours' counts). The rare-context play.
      mode 'hier'     : 'wordrep' PLUS the PHRASE-cluster of the recent phrase as a third, coarser expert,
                        again weighted up only when the word-level evidence is thin.

    Weighting is by count mass, not learned: lam_c = kappa / (kappa + prev_count). No gradients."""

    def __init__(self, bc, word_reps, phrase_reps=None, kappa=20.0, kappa_p=40.0):
        self.bc, self.wr, self.pr = bc, word_reps, phrase_reps
        self.kappa, self.kappa_p = kappa, kappa_p
        self.N, self.C, self.alpha = bc.N, bc.C, bc.alpha

    def _word_dist(self, prev):
        ni, cn = self.bc.word_row(prev)
        return _dist_from_row(ni, cn, self.N, self.alpha, self.bc.uni, self.bc.uni_total)

    def _cluster_dist(self, prev):
        c = int(self.wr.clu[prev]) if prev >= 0 else -1
        ni, cn = self.bc.cluster_row(c)
        if ni.size == 0:
            return None
        return _dist_from_row(ni, cn, self.N, self.alpha, self.bc.uni, self.bc.uni_total)

    def _phrase_dist(self, phrase_cluster):
        """Aggregated next-word dist for words that follow any member of this phrase cluster's words — we reuse
        the WORD-cluster aggregate of the phrase's last word's cluster as the coarse expert. (Cheap projection:
        the phrase rep routes to a word-cluster aggregate.) Returns None if unavailable."""
        if phrase_cluster < 0:
            return None
        return None  # phrase contribution is supplied via _hier_phrase_word_cluster below

    def prob_bigram(self, prev):
        return self._word_dist(prev)

    def prob_wordrep(self, prev):
        pw = self._word_dist(prev)
        pc = self._cluster_dist(prev)
        if pc is None:
            return pw
        pcount = self.bc.prev_count[prev] if prev >= 0 else 0.0
        lam = self.kappa / (self.kappa + pcount)             # cluster weight rises as direct evidence falls
        return (1 - lam) * pw + lam * pc

    def prob_hier(self, prev, phrase_word_cluster):
        """phrase_word_cluster: an extra word-CLUSTER id derived from the recent phrase (e.g. the phrase's
        bag-of-words majority cluster), used as a coarse expert. -1 to disable."""
        base = self.prob_wordrep(prev)
        if phrase_word_cluster is None or phrase_word_cluster < 0:
            return base
        ni, cn = self.bc.cluster_row(int(phrase_word_cluster))
        if ni.size == 0:
            return base
        pp = _dist_from_row(ni, cn, self.N, self.alpha, self.bc.uni, self.bc.uni_total)
        pcount = self.bc.prev_count[prev] if prev >= 0 else 0.0
        lamp = self.kappa_p / (self.kappa_p + pcount)        # phrase expert weight also rises when ctx is thin
        return (1 - 0.5 * lamp) * base + 0.5 * lamp * pp
