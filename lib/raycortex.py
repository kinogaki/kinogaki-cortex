"""raycortex.py — Exp W: the integrated "ray-cortex" word predictor.

Resurrects the PARKED raytracing/proximity idea (Exp P) — NOT naive Euclidean, and NOT alone. The bet
(fragile-ideas commandment 9): proximity-gather earns its keep only in COMBINATION, judged on the RIGHT
axis (rare-context backoff + robustness), never as a standalone next-word predictor.

Everything here is ONLINE / COUNT-BASED / NO-BACKPROP:
  - offset-keyed count tables           (offsetattn.OffsetAttn — the sequential core)
  - PMI association graph + spreading    (graph.build_graph/spread — PMI READ from co-occurrence counts;
                                          spreading = a sparse mat-vec; this is the ONLINE "raytracing")
  - leaky log-evidence pooling           (evidence-style accumulator, here over the word experts)
  - online leader-clustered topic prior  (jepa.online_signatures + leader_cluster — running-mean prototypes,
                                          spawn-on-distance; NO k-means, NO SVD) committed by ignition-style
                                          hysteresis, injected as a SOFT shrink-blend at backoff only.

No gradient descent, no batch optimization (k-means / SVD / eigendecomposition are banned and absent).
Counting, decayed accumulators, leader-clustering, and PMI-from-counts are all single streaming pass.

THE EXPERTS (each emits a sparse {word_id: log-score}; pooled by weighted geometric mean = calibrated):
  A. OffsetAttn         — per-relative-offset successor counts over the recent D context words.
  B. ProximityGather    — the raytracing backoff. Activate the recent context words on the PMI graph,
                          spread 1-2 hops, and let every reached node vote its SUCCESSOR distribution
                          (what follows that word), weighted by spread activation. For RARE contexts the
                          direct counts are empty/thin and the neighbours' followers fill in.
  T. TopicPrior         — committed online topic G → its word histogram, a smooth fallback distribution.

POOLING (RayCortex.predict): the offset expert is the backbone; proximity and topic are gated IN only as
their backbone weight thins (a continuous backoff weight = how sparse the direct evidence is). Evidence
accumulation runs a leaky carry of the pooled log-scores across the window so one thin position can't tank
the belief. Final distribution = softmax over the union candidate set.
"""
import math
import numpy as np

import offsetattn
import graph as graphlib

ALPHA = 0.05


# ────────────────────────────── B. proximity gather (the raytracing) ──────────────────────────────

class ProximityGather:
    """Spreading-activation successor vote over the PMI association graph — the online "raytracing".

    Built from counts only: build_graph gives top-N nodes + a row-normalized top-k PMI adjacency A; we
    also hold, per node, its SUCCESSOR distribution succ[node] = {next_word: count} (what immediately
    follows that word in the stream). A query = activate the context words, spread `hops`, then pool the
    successor distributions of the activated nodes weighted by activation. This is proximity-as-PREDICTOR
    in the graph form the sources endorse (NOT Euclidean): "near in association" → "votes for my followers".
    """

    def __init__(self, N=3000, window=5, k_edges=20, hops=1, spread_alpha=0.5, topn_nodes=24, topk_vote=48):
        self.N = N; self.window = window; self.k_edges = k_edges
        self.hops = hops; self.spread_alpha = spread_alpha
        self.topn_nodes = topn_nodes      # cap activated nodes per query (sparsity)
        self.topk_vote = topk_vote        # cap successor candidates per node

    def fit(self, stream):
        self.top, self.remap, self.A = graphlib.build_graph(
            stream, N=self.N, window=self.window, k_edges=self.k_edges)
        self.AT = np.ascontiguousarray(self.A.T)      # precompute transpose for the spread matvec
        n = len(self.top)
        # successor counts in NODE space: for adjacent positions both in the top-N, count node_a -> word_b.
        # Stored as a dense small per-node top-k list (node -> (word_ids, probs)) for fast pooling.
        a = self.remap[stream[:-1]]; b = stream[1:]
        m = a >= 0
        a, b = a[m], b[m]
        succ = [dict() for _ in range(n)]
        # vectorized accumulate via unique on packed (node_a, word_b)
        Vb = int(stream.max()) + 1
        key = a.astype(np.int64) * Vb + b.astype(np.int64)
        uk, uc = np.unique(key, return_counts=True)
        ua = uk // Vb; uw = uk % Vb
        for node, w, c in zip(ua.tolist(), uw.tolist(), uc.tolist()):
            succ[node][int(w)] = int(c)
        # freeze each node's top-k successors as (ids, log-weights) for the geometric pool
        self.succ_ids = []; self.succ_logc = []
        for d in succ:
            if not d:
                self.succ_ids.append(np.empty(0, np.int64)); self.succ_logc.append(np.empty(0)); continue
            items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[: self.topk_vote]
            ids = np.array([k for k, _ in items], np.int64)
            cs = np.array([v for _, v in items], np.float64)
            self.succ_ids.append(ids); self.succ_logc.append(np.log(cs + ALPHA))
        return self

    def gather(self, ctx_words):
        """ctx_words: recent word-ids (any order). Returns {word_id: log-score} from spread-activated
        neighbours' successor distributions, or None if nothing activates. The vote is a weighted
        geometric mean (sum of activation-weighted log-counts) over the union of candidates.

        SPARSE spread: the seed activation has only the few context nodes nonzero, so the matvec
        A.T @ a0 = sum over seed nodes s of a0[s]·A[s,:] (a few adjacency-row reads), not a dense matvec."""
        seeds = []
        for w in ctx_words:
            nd = self.remap[w] if 0 <= w < len(self.remap) else -1
            if nd >= 0:
                seeds.append(int(nd))
        if not seeds:
            return None
        seeds = np.array(seeds, np.int64)
        sc = np.bincount(seeds, minlength=len(self.top)).astype(np.float64)   # a0 (multiplicity = weight)
        # 1-hop: act = a0 + alpha * (sum of seed rows of A), weighted by seed multiplicity
        act = sc.copy()
        sn = np.unique(seeds)
        hop = self.spread_alpha * (sc[sn] @ self.A[sn])
        act += hop
        if self.hops >= 2:
            nz2 = np.nonzero(hop)[0]
            act += self.spread_alpha * (hop[nz2] @ self.A[nz2])
        nz = np.nonzero(act)[0]
        if nz.size == 0:
            return None
        if nz.size > self.topn_nodes:
            nz = nz[np.argsort(act[nz])[::-1][: self.topn_nodes]]
        # pool successor dists weighted by activation: out[word] += act_node * logcount_node(word)
        out = {}
        wsum = 0.0
        for nd in nz.tolist():
            ids = self.succ_ids[nd]
            if ids.size == 0:
                continue
            w = float(act[nd]); wsum += w
            lc = self.succ_logc[nd]
            idl = ids.tolist(); lcl = lc.tolist()
            for i in range(len(idl)):
                wid = idl[i]
                out[wid] = out.get(wid, 0.0) + w * lcl[i]
        if not out or wsum <= 0:
            return None
        # keep the strongest topk_vote candidates (sparse, pre-capped so the pool needn't re-sort)
        if len(out) > self.topk_vote:
            keep = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[: self.topk_vote]
            out = dict(keep)
        for k in out:
            out[k] /= wsum
        return out


# ────────────────────────────── T. online topic prior ──────────────────────────────

class TopicPrior:
    """Online leader-clustered topic state + a soft per-topic word distribution.

    Words → online signatures (hashed signed co-occurrence, IDF-online) → leader-clustering (running-mean
    prototypes, spawn-on-distance) gives topic_of[word]. We stream the topic ids and commit a global G by
    ignition/hysteresis (recency histogram; switch only on a decisive lead). Per topic we hold a word
    histogram (counts of words seen while that G was committed) as the SOFT fallback distribution. NO
    k-means / NO SVD anywhere — pure online counting + leader clustering."""

    def __init__(self, N, halflife=50.0, margin=0.16, min_evidence=40, thresh=0.55, Cmax=300):
        self.N = N; self.halflife = halflife; self.margin = margin
        self.min_evidence = min_evidence; self.thresh = thresh; self.Cmax = Cmax

    def fit(self, stream):
        from jepa import online_signatures, leader_cluster
        seq = np.where(stream < self.N, stream, -1).astype(np.int64)   # only top-N words carry topic signal
        sig, cnt = online_signatures(seq, self.N, D=64, window=5, seed=0)
        order = np.argsort(cnt)[::-1]                                   # ripen high-evidence words first
        self.topic_of, self.K = leader_cluster(sig, cnt, order,
                                                min_evidence=self.min_evidence,
                                                thresh=self.thresh, Cmax=self.Cmax)
        # committed G per position (ignition + hysteresis), then per-topic word histograms
        topic_seq = np.where(seq >= 0, self.topic_of[np.clip(seq, 0, self.N - 1)], -1)
        self.Gseq = _commit_G(topic_seq, max(self.K, 1), self.halflife, self.margin)
        # word histogram per committed topic (soft fallback dist); store as log-prob rows over a capped vocab
        self.Vtop = self.N
        self.topic_logp = {}
        g = self.Gseq
        wv = np.where(stream < self.Vtop, stream, -1)
        for k in range(self.K):
            mask = (g == k) & (wv >= 0)
            if mask.sum() < 20:
                continue
            h = np.bincount(wv[mask].astype(np.int64), minlength=self.Vtop).astype(np.float64)
            p = (h + ALPHA) / (h.sum() + ALPHA * self.Vtop)
            self.topic_logp[k] = np.log(p)
        return self

    def commit_over(self, stream):
        """Commit G online over an arbitrary (e.g. held-out) stream using the learned topic_of map.
        Returns a G per position — the same ignition/hysteresis rule, run forward on new text."""
        seq = np.where(stream < self.N, stream, -1).astype(np.int64)
        topic_seq = np.where(seq >= 0, self.topic_of[np.clip(seq, 0, self.N - 1)], -1)
        return _commit_G(topic_seq, max(self.K, 1), self.halflife, self.margin)

    def dist_at(self, g, topk=64):
        """Soft topic fallback as {word_id: log-score} (top-k of the committed topic's word histogram)."""
        lp = self.topic_logp.get(int(g))
        if lp is None:
            return None
        idx = np.argpartition(lp, -topk)[-topk:]
        return {int(i): float(lp[i]) for i in idx}


def _commit_G(topic_seq, K, halflife, margin):
    """Ignition/hysteresis commit of a global topic over a topic-id stream (-1 = stop/unknown). Same rule as
    ignition.commit_G, inlined to avoid importing the k-means-bearing module."""
    decay = 0.5 ** (1.0 / halflife)
    hist = np.zeros(K, np.float64)
    G = 0; out = np.empty(len(topic_seq), np.int64)
    for i, t in enumerate(topic_seq):
        hist *= decay
        if t >= 0:
            hist[t] += 1.0
        s = hist.sum()
        if s > 0:
            lead = int(np.argmax(hist))
            if lead != G and (hist[lead] - hist[G]) / s > margin:
                G = lead
        out[i] = G
    return out


# ────────────────────────────── the integrated predictor ──────────────────────────────

class RayCortex:
    """Integrated hierarchical word predictor. The offset-attention expert is the backbone; proximity and
    topic are gated in as the backbone's direct evidence thins (rare-context backoff). All experts pool by
    weighted geometric mean.

    EVIDENCE ACCUMULATION (across the experts, WITHIN a position — the robust-voting form of Exp R / TBP).
    The plain geometric-mean pool lets ONE expert (e.g. a corrupted context word at one offset that votes a
    confident wrong successor) swing the log-score arbitrarily far. Evidence-pooling WINSORIZES each expert's
    per-candidate deviation from that expert's own median vote to ±ev_clip before summing: no single
    hypothesis can be eliminated (or asserted) by one bad observation — exactly the leaky-evidence robustness
    property, expressed as a per-step robust combine instead of a cross-step carry. ev_clip<∞ turns it on.

    Knobs (each ablatable):
      w_off    backbone weight (offset attention)
      w_prox   proximity-gather weight at FULL backoff (scaled by the sparsity gate)
      w_topic  topic-prior weight at FULL backoff (scaled by the sparsity gate)
      ev_clip  evidence winsorization radius in nats (np.inf = off = plain geometric mean)
    """

    def __init__(self, D=6, off_gamma=8.0, N_graph=3000, hops=1, w_off=1.0, w_prox=1.0,
                 w_topic=0.6, ev_clip=np.inf, use_prox=True, use_topic=True):
        self.D = D
        self.off = offsetattn.OffsetAttn(D=D, gamma=off_gamma)
        self.prox = ProximityGather(N=N_graph, hops=hops) if use_prox else None
        self.topic = None
        self.use_topic = use_topic
        self.w_off = w_off; self.w_prox = w_prox; self.w_topic = w_topic
        self.ev_clip = ev_clip

    def fit(self, stream, topic_N=None):
        self.off.fit(stream)
        if self.prox is not None:
            self.prox.fit(stream)
        if self.use_topic:
            tN = topic_N if topic_N is not None else self.prox.N if self.prox else 3000
            self.topic = TopicPrior(N=tN).fit(stream)
        return self

    # --- per-position scoring ---------------------------------------------------------------------

    def _direct_count(self, ctx):
        """Sparsity gate = how thin the IMMEDIATE predictive context is. We use the offset-1 successor mass
        (how many times the directly-preceding word has been seen with a known follower) — the standard
        n-gram "context count". When the preceding word is RARE this is small, the backbone is starved, and
        the proximity/topic backoff should earn its keep. (Summing all offsets washed this out: a common word
        somewhere back in the window always inflated the count, so nothing ever looked rare.)"""
        a = int(ctx[-1])
        dd = self.off.tab[1].get(a)
        return sum(dd.values()) if dd else 0

    def _experts(self, ctx, g, dc):
        """Assemble (weight, dict, is_log) experts for this position, given precomputed direct-count `dc`."""
        gate = 1.0 / (1.0 + dc / 8.0)                 # backoff gate: dc=0 → 1, dc=8 → 0.5, dc large → ~0

        experts = []
        # A. offset backbone — emit as count-dict experts (geometric pool handles the +ALPHA floor)
        for d in range(1, self.D + 1):
            a = int(ctx[-d])
            dd = self.off.tab[d].get(a)
            if dd:
                experts.append((self.w_off * float(self.off.w[d]), dd, False))

        # B. proximity gather — a log-score dict; weight scaled UP by the backoff gate (rare-context role)
        if self.prox is not None and self.w_prox > 0:
            pg = self.prox.gather(ctx)
            if pg:
                experts.append((self.w_prox * gate, pg, True))

        # T. topic prior — a log-prob dict; soft, scaled by the backoff gate (only fills in at backoff)
        if self.topic is not None and self.w_topic > 0 and g is not None:
            td = self.topic.dist_at(g)
            if td:
                experts.append((self.w_topic * gate, td, True))
        return experts

    def _pool(self, experts, topk=64):
        """Weighted (robust) pool over count-dicts (is_log=False) and log-score dicts (is_log=True).
        Returns (cand_ids ndarray, logscore ndarray) UNnormalized (caller softmaxes).

        Each expert contributes a per-candidate log-vector (count-dict words it never saw get log(ALPHA)).
        With ev_clip=∞ this is the plain weighted geometric mean (sum of w·logp / ΣW). With ev_clip finite
        (EVIDENCE ON) each expert's per-candidate value is WINSORIZED to its own median ± ev_clip before the
        weighted sum — so a single corrupted/over-confident expert can neither zero nor assert a hypothesis.
        That is the leaky-evidence robustness property as a per-step robust combine."""
        if not experts:
            return None, None
        W = sum(w for w, _, _ in experts)
        if W <= 0:
            return None, None
        keys = set()
        for _, d, _ in experts:
            if len(d) <= topk:
                keys.update(d)
            else:
                top = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:topk]
                keys.update(k for k, _ in top)
        keys = np.fromiter(keys, np.int64, len(keys))
        kidx = {int(k): i for i, k in enumerate(keys)}
        n = len(keys)
        lA = math.log(ALPHA)
        clip = self.ev_clip
        robust = np.isfinite(clip)
        score = np.zeros(n)
        for w, d, is_log in experts:
            vec = np.full(n, lA if not is_log else 0.0)      # baseline: unseen-count floor / log-score 0
            if is_log:
                vec[:] = -50.0                               # log-score experts: absent candidate ≈ very low
                for k, v in d.items():
                    i = kidx.get(int(k))
                    if i is not None:
                        vec[i] = v
            else:
                for k, c in d.items():
                    i = kidx.get(int(k))
                    if i is not None:
                        vec[i] = math.log(c + ALPHA)
            if robust:
                med = np.median(vec)
                vec = med + np.clip(vec - med, -clip, clip)   # winsorize this expert's deviations
            score += w * vec
        score /= W
        return keys, score

    def predict_stream(self, stream, positions, gseq=None, corrupt_ctx=None):
        """Score the next-word log-prob at each position in `positions` (each t predicts stream[t]).
        ctx for position t = stream[t-D:t] (from `corrupt_ctx` if given, else clean stream).
        Returns (logp_target, direct_counts) arrays aligned to positions. Robust evidence-pooling (when
        ev_clip is finite) is applied per position inside _pool — no cross-position state."""
        ctx_src = corrupt_ctx if corrupt_ctx is not None else stream
        logp = np.full(len(positions), np.log(1e-9))
        dcs = np.zeros(len(positions), np.int64)
        for j, t in enumerate(positions):
            if t < self.D:
                continue
            ctx = ctx_src[t - self.D: t]
            g = int(gseq[t]) if gseq is not None else None
            dc = self._direct_count(ctx)
            dcs[j] = dc
            experts = self._experts(ctx, g, dc)
            ids, sc = self._pool(experts)
            if ids is None:
                continue
            m = sc.max(); z = np.exp(sc - m).sum()
            tgt = int(stream[t])
            hit = np.nonzero(ids == tgt)[0]
            if hit.size:
                logp[j] = (sc[hit[0]] - m) - math.log(z)
            else:
                logp[j] = math.log(ALPHA) - m - math.log(z + 1.0)   # target outside candidate set → tail mass
        return logp, dcs


# ────────────────────────────── baselines (online count models) ──────────────────────────────

def bigram_logp(stream, positions, vocab, corrupt_ctx=None):
    """Plain bigram P(w_t | w_{t-1}) with add-ALPHA backoff to unigram. Online counts, vectorized."""
    ctx_src = corrupt_ctx if corrupt_ctx is not None else stream
    big = {}
    a = stream[:-1].astype(np.int64); b = stream[1:].astype(np.int64)
    V = int(stream.max()) + 1
    uk, uc = np.unique(a * V + b, return_counts=True)
    ua = uk // V; ub = uk % V
    from collections import defaultdict
    tot = defaultdict(int)
    for x, y, c in zip(ua.tolist(), ub.tolist(), uc.tolist()):
        big.setdefault(x, {})[y] = c; tot[x] += c
    uni = np.bincount(stream, minlength=V).astype(np.float64); uni_lp = np.log((uni + ALPHA) / (uni.sum() + ALPHA * V))
    out = np.empty(len(positions))
    for j, t in enumerate(positions):
        prev = int(ctx_src[t - 1])
        d = big.get(prev)
        tgt = int(stream[t])
        if d and tgt in d:
            out[j] = math.log((d[tgt] + ALPHA) / (tot[prev] + ALPHA * V))
        elif d:
            out[j] = math.log(ALPHA / (tot[prev] + ALPHA * V))
        else:
            out[j] = uni_lp[tgt]
    return out
