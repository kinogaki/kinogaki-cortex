"""situation.py — Exp AM: a persistent SITUATION MODEL that PREDICTS over long spans. ONLINE, NO backprop.

Lineage. Exp AC built a Bayesian-surprise event model: KL of the one-step belief update DETECTS topic
boundaries beautifully, but the single persistent event-slot helped PREDICTION only on the ~1% backoff slice
(where the local n-gram had run out). Exp T (ignition) found the same altitude law for a global topic G. Exp U
gave us online concept clusters. The open question on the global-coherence thread: can a persistent situation
state lower perplexity GENERALLY — not just where local context fails — and keep generated text on-topic over
long spans?

This module adds two mechanisms that AC's single bag-of-clusters slot lacked, both built ONLINE (single
streaming pass of leaky accumulators + online counting, NO gradients/k-means/SVD):

  (a) NARRATIVE-SCHEMA EVENT CHAINS (Chambers-Jurafsky 2008). C-J learn "narrative event chains" by counting
      verb pairs that share a coreferring entity (protagonist), then PMI-scoring them so the current event
      predicts the EXPECTED NEXT event. We have no parser and no coreference, so we approximate at the cortex's
      altitude: an EVENT is the concept-CLUSTER of a content word; "coreference" is approximated by RECENCY of
      a shared entity-ish cluster — two events that fire while the same who/where slot is active are treated as
      a coreferring pair. We count ordered cluster-pair co-occurrences within a short event window, PMI-score
      them online, and read off P(next event-cluster | current event-cluster). That distribution is a
      top-down prior on which CLUSTER comes next — i.e. the schema predicts the next event, then we spread the
      event-cluster mass over its member words.

  (b) MULTI-DIMENSIONAL TYPED SITUATION SLOTS (Zwaan's event-indexing situation model — who/where/when/what).
      Instead of one undifferentiated slot, a few TYPED persistent leaky accumulators, each over a different
      slice of the cluster space, each contributing its own word prior that is mixed in:
        who   — recent entity-LIKE clusters (clusters whose words are capitalized-ish / proper-noun-shaped;
                here, high-IDF rare-ish content clusters that recur — the protagonists).
        where — recent location-LIKE clusters (clusters that co-occur with prepositional/locative anchors).
        topic — the slow background topic (the AC-style bag of active clusters, long half-life).
        when  — a coarse positional / recency phase (kept minimal; mostly a control dimension).
      Each typed slot is a leaky cluster histogram with its OWN half-life (who/where drift faster than topic),
      and each maps to a word prior the same way (cluster mass spread over member words by unigram).

The SITUATION PRIOR is the (weighted) blend of the schema prediction + the typed-slot priors, used as a soft
TOP-DOWN PRIOR on the word distribution AT EVERY STEP (not gated to the backoff slice) — that is the whole
point: AC restricted the prior to backoff; AM asks whether a richer, typed, schema-driven situation can earn
its keep generally. We measure both ways (general vs backoff-only) to stay honest.

Everything is online: typed slots are leaky accumulators; the schema is PMI over running co-occurrence counts;
cluster→word spreading is a precomputed sparse mass matrix; nothing iterates to convergence or backprops.

  EventChain      — Chambers-Jurafsky-style ordered cluster-pair PMI: P(next event-cluster | current).
  TypedSlots      — who/where/topic/when leaky cluster histograms, each its own half-life + word prior.
  SituationModel  — drives both over a token stream; emits a per-step situation word-prior to blend in.
  cluster_word_mass — P(word | cluster), the spreader used to turn cluster priors into word priors.
  entityness / locationness — cheap ONLINE per-cluster typings (no parser): which clusters look who/where.
"""
import numpy as np
from collections import defaultdict

EPS = 1e-9


# ───────────────────────── cluster → word spreading (shared by every prior) ─────────────────────────

def cluster_word_mass(clu_of, uni, C, V):
    """P(word | cluster): for each cluster, its member words' unigram mass, row-normalized. The single
    spreader that turns ANY prior over clusters into a prior over words (P(w) ∝ Σ_c prior[c]·P(w|c)). Built
    once from counts — order-independent."""
    M = np.zeros((C, V))
    for wid in range(min(V, len(clu_of))):
        c = clu_of[wid]
        if c >= 0:
            M[c, wid] = uni[wid]
    M /= np.maximum(M.sum(1, keepdims=True), EPS)
    return M


# ───────────────────────── cheap ONLINE per-cluster typings (no parser) ─────────────────────────

def cluster_typings(clu_of, uni, word_idf, V, C, stream, locative_words):
    """Assign each cluster a soft who-ness and where-ness from cheap online statistics — we have no parser, so
    we type clusters by surface co-occurrence, the cortex-altitude stand-in for NER.

      entityness[c] (who)  — average IDF of the cluster's member words, weighted by unigram mass. Proper
        nouns / protagonists are rarer and higher-IDF than function/topic words; high-IDF content clusters are
        the entity-like ones. (C-J's "protagonist" is a recurring entity → rare, content-bearing.)
      locationness[c] (where) — fraction of the cluster's occurrences that immediately FOLLOW a locative anchor
        word ('in','at','on','near','from','to' …). A cheap "is this argued about as a place?" signal.

    Both are online: IDF is a running count; the locative-follow count is a single streaming pass. Returned
    L2-style normalized to [0,1] so they can gate the typed slots."""
    # who: IDF-weighted mass per cluster
    ent = np.zeros(C)
    massc = np.zeros(C)
    for wid in range(min(V, len(clu_of))):
        c = clu_of[wid]
        if c >= 0:
            ent[c] += uni[wid] * word_idf[wid]
            massc[c] += uni[wid]
    ent = ent / np.maximum(massc, EPS)
    ent = (ent - ent.min()) / max(ent.max() - ent.min(), EPS)

    # where: fraction of cluster-token occurrences that follow a locative anchor word
    loc_after = np.zeros(C)
    occ = np.zeros(C)
    is_loc = np.zeros(V, bool)
    is_loc[list(locative_words)] = True
    prev_loc = False
    cl = clu_of
    n = len(stream)
    for t in range(n):
        w = int(stream[t])
        c = cl[w] if w < len(cl) else -1
        if c >= 0:
            occ[c] += 1.0
            if prev_loc:
                loc_after[c] += 1.0
        prev_loc = (w < V) and is_loc[w]
    locn = loc_after / np.maximum(occ, EPS)
    locn = (locn - locn.min()) / max(locn.max() - locn.min(), EPS)
    return ent, locn


# ───────────────────────── (a) narrative-schema event chains (Chambers-Jurafsky) ─────────────────────────

class EventChain:
    """Chambers-Jurafsky narrative event chains, at the cluster altitude. An EVENT = the concept-cluster of a
    content word. We count ORDERED cluster-pair co-occurrences inside a short EVENT WINDOW (the events near
    each other in the discourse), restricted to pairs that share an active WHO/WHERE context (the cortex-level
    stand-in for "coreferring protagonist" — two events count as a chain link if the same entity-ish slot was
    hot for both). Then PMI-score them and read off P(next event-cluster | current event-cluster). Online:
    pure co-occurrence counting; PMI is a closed-form transform of the running counts."""

    def __init__(self, C, window=8, alpha=0.1):
        self.C = C
        self.window = window
        self.alpha = alpha
        self.pair = defaultdict(lambda: defaultdict(float))   # cur cluster -> {next cluster: weighted count}
        self.row = np.zeros(C)                                # marginal weight of each cluster as 'current'
        self.col = np.zeros(C)                                # marginal weight of each cluster as 'next'
        self.tot = 0.0
        self._pmi_cache = {}

    def fit(self, event_seq, share_weight):
        """event_seq: 1-D array of event-cluster ids (content-word clusters, -1 for non-events / OOV) in
        discourse order. share_weight[t] in [0,1] = how strongly event t shares an active entity context with
        its recent neighbours (the coreference proxy) — high weight = a real chain link. We add, for each
        ordered pair (e_i, e_j) with 0<j-i<=window, a co-occurrence weighted by the shared-context strength of
        the LATER event (so pairs bound by a persistent protagonist dominate, exactly C-J's selection)."""
        ev = event_seq
        n = len(ev)
        for g in range(1, self.window + 1):
            cur = ev[:-g]; nxt = ev[g:]; w = share_weight[g:]
            m = (cur >= 0) & (nxt >= 0) & (cur != nxt)
            cur, nxt, w = cur[m], nxt[m], w[m]
            for a, b, ww in zip(cur, nxt, w):
                self.pair[int(a)][int(b)] += ww
                self.row[int(a)] += ww
                self.col[int(b)] += ww
                self.tot += ww
        self._pmi_cache.clear()
        return self

    def next_prior(self, cur_cluster):
        """P(next event-cluster | cur) reweighted by positive PMI (the C-J score): score(a,b) =
        p(b|a) · max(PMI(a,b), 0). Returns a dense (C,) prior over clusters (sums to 1) or None. Memoized per
        current-cluster (the counts are frozen post-fit)."""
        if cur_cluster < 0 or self.tot <= 0:
            return None
        cached = self._pmi_cache.get(cur_cluster)
        if cached is not None:
            return cached
        d = self.pair.get(cur_cluster)
        if not d:
            return None
        rowsum = self.row[cur_cluster] + EPS
        out = np.zeros(self.C)
        for b, cnt in d.items():
            p_b_given_a = cnt / rowsum
            p_a = self.row[cur_cluster] / self.tot
            p_b = self.col[b] / self.tot
            p_ab = cnt / self.tot
            pmi = np.log2((p_ab + EPS) / (p_a * p_b + EPS))
            out[b] = p_b_given_a * max(pmi, 0.0)
        s = out.sum()
        prior = out / s if s > EPS else None
        self._pmi_cache[cur_cluster] = prior
        return prior


# ───────────────────────── (b) multi-dimensional TYPED situation slots (Zwaan) ─────────────────────────

class TypedSlots:
    """Zwaan event-indexing: maintain a few TYPED persistent leaky accumulators over clusters — who / where /
    topic — each with its OWN half-life and its OWN gating mask, each emitting a cluster prior. A token's
    cluster is folded into a slot with a weight = that slot's typing of the cluster (who-ness / where-ness /
    1.0 for the slow topic), so the who-slot accumulates entity-ish clusters, the where-slot location-ish ones,
    and the topic-slot everything slowly. Each slot's prior is its normalized leaky histogram. Online: leaky
    counters, one fold per token."""

    def __init__(self, C, entityness, locationness,
                 hl_who=120.0, hl_where=200.0, hl_topic=2000.0):
        self.C = C
        self.ent = entityness
        self.loc = locationness
        self.d_who = 0.5 ** (1.0 / hl_who)
        self.d_where = 0.5 ** (1.0 / hl_where)
        self.d_topic = 0.5 ** (1.0 / hl_topic)
        self.who = np.zeros(C)
        self.where = np.zeros(C)
        self.topic = np.zeros(C)

    def observe(self, cluster):
        self.who *= self.d_who
        self.where *= self.d_where
        self.topic *= self.d_topic
        if cluster >= 0:
            self.who[cluster] += self.ent[cluster]          # entity-ish clusters load the who slot
            self.where[cluster] += self.loc[cluster]        # location-ish clusters load the where slot
            self.topic[cluster] += 1.0                      # everything slowly loads topic

    def who_hotness(self):
        """How concentrated / active the who-slot is, in [0,1) — used as the 'shared protagonist' strength for
        the event chain (a hot, peaked who-slot = a coreferring protagonist is on stage)."""
        s = self.who.sum()
        if s <= EPS:
            return 0.0
        p = self.who / s
        # peakedness = 1 - normalized entropy; a single dominant entity → near 1
        H = -(p[p > 0] * np.log(p[p > 0])).sum()
        return float(1.0 - H / np.log(self.C))

    @staticmethod
    def _norm(v):
        s = v.sum()
        return v / s if s > EPS else None

    def priors(self):
        return dict(who=self._norm(self.who), where=self._norm(self.where), topic=self._norm(self.topic))


# ───────────────────────── the driver: combine schema + typed slots into one word prior ─────────────────────────

class SituationModel:
    """Drives the event chain + typed slots over a token stream and emits, at each step, a single SITUATION
    PRIOR over WORDS = weighted blend of {schema next-event prior, who prior, where prior, topic prior},
    each spread cluster→word. This is the soft top-down prior the predictor blends in.

    The blend weights are fixed (no training): the schema and who/where slots are the 'foreground' situation,
    topic the slow background. The caller decides how strongly to mix the situation prior into the word
    distribution, and whether to do so everywhere or only on the backoff slice (we test both)."""

    def __init__(self, C, V, clu_of, cw_mass, chain, entityness, locationness,
                 w_schema=0.40, w_who=0.20, w_where=0.15, w_topic=0.25,
                 hl_who=120.0, hl_where=200.0, hl_topic=2000.0, refresh=256):
        self.C, self.V = C, V
        self.clu_of = clu_of
        self.cw = cw_mass                       # (C, V) P(word|cluster)
        self.chain = chain
        self.slots = TypedSlots(C, entityness, locationness, hl_who, hl_where, hl_topic)
        self.w = dict(schema=w_schema, who=w_who, where=w_where, topic=w_topic)
        self.refresh = refresh
        self.last_event = -1                    # most recent content-word cluster (the 'current event')
        self._step = 0
        self._cache_word_prior = None
        self._cache_cluster_prior = None

    def observe(self, word_id):
        """Fold one observed word into the situation: update typed slots, and if it is a content (clustered)
        word, advance the 'current event' for the schema."""
        c = self.clu_of[word_id] if word_id < len(self.clu_of) else -1
        self.slots.observe(c)
        if c >= 0:
            self.last_event = c
        self._step += 1
        if self._step % self.refresh == 0:
            self._cache_word_prior = None       # invalidate; recompute lazily

    def cluster_prior(self):
        """The combined SITUATION prior over CLUSTERS (before spreading to words). Weighted blend of the schema
        next-event prediction and the typed-slot histograms. Returns dense (C,) or None."""
        parts = []
        ps = self.slots.priors()
        sch = self.chain.next_prior(self.last_event)
        if sch is not None:
            parts.append((self.w["schema"], sch))
        for name in ("who", "where", "topic"):
            p = ps[name]
            if p is not None:
                parts.append((self.w[name], p))
        if not parts:
            return None
        tot_w = sum(w for w, _ in parts)
        out = np.zeros(self.C)
        for w, p in parts:
            out += (w / tot_w) * p
        return out

    def word_prior(self):
        """The combined situation prior over WORDS — cluster prior spread by P(word|cluster). Cached between
        refreshes (it drifts slowly; recomputing the C×V matvec every token is wasteful and changes nothing
        material — the AC refresh trick)."""
        if self._cache_word_prior is not None:
            return self._cache_word_prior
        cp = self.cluster_prior()
        wp = (cp @ self.cw) if cp is not None else None
        self._cache_word_prior = wp
        self._cache_cluster_prior = cp
        return wp
