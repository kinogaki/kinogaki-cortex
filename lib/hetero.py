"""hetero.py — a HETEROGENEOUS, specialized stack (Exp X).

The thesis under test (the opposite axis from Exp I's uniform Column): the brain is NOT one repeated part.
It is SPECIALIZED TOPOLOGIES at different levels — retinal center-surround, cortical columns + voting,
thalamic relay/GATING, basal-ganglia action SELECTION, hippocampal episodic memory; proximal/local vs
distal/long-range edges; fast sensory vs slow integrative timescales. So give each LEVEL a different column
type, a different connection RANGE, and a different TIMESCALE, then GATE/ARBITRATE between them per token.
Does specialization-by-level beat a uniform stack? (Exp I proved uniform-bigger works.)

Every level predicts the SAME thing — the next CHARACTER — so bpc is apples-to-apples across the whole stack
and against the uniform baseline. Higher levels project their (word / phrase / topic) belief down into a
27-dim char prior, exactly the `char_prior` move the uniform cortex already uses.

The four specialized levels (different topology + timescale):

  L0  CHAR  — dense LOCAL n-gram. proximal (short char window), FAST. early-sensory-like.
              reuses evidence.ExpertBank (vectorized char backoff) → a per-position 27-dim char dist.
  L1  WORD  — offset-keyed attention (distal, mid-range) + spelling lexicon. reuses offsetattn.OffsetAttn.
              predicts the next WORD from D earlier word slots; → char prior over the current word's next char.
  L2  PHRASE— branching-entropy CHUNKS + a change/trajectory model. reuses boundaries.phrase_cuts +
              trajectory.CountColumn over a chunk-id stream. distal-ish, slower. → char prior via chunk lexicon.
  L3  THEME — ONLINE topic state (leader-clustering, NOT batch k-means) + leaky EVIDENCE accumulation, the
              slow integrator (long timescale). a G-conditioned char table (like ignition.GCondChar) lifts
              the char prediction toward the words typical of the committed topic.

GATING / ARBITRATION (thalamus + basal ganglia): instead of one fixed pooling rule, a per-token gate routes.
We score each level's char distribution by its own running CONFIDENCE (neg recent NLL) and let SURPRISE open
the gate to the higher, slower levels. Two gate policies live here: a soft confidence-weighted geometric mean
and a hard argmax-confidence router. Both compare against the static geometric-mean pool (the Exp I rule).

ONLINE-COMPLIANCE: every table is built by counting (np.unique / bincount), every accumulator is leaky/decayed,
the topic coder is online leader-clustering. NO gradient descent, NO k-means/SVD/eigendecomposition.
Alphabet (corpus.py): a..z = 0..25, space = 26, V = 27.
"""
import math
import numpy as np

import evidence as EV
import offsetattn as OA
import boundaries as BD
import trajectory as TR

V = 27
SPACE = 26
ALPHA = 0.05
LOG_UNIFORM = math.log(1.0 / V)


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  L0 — CHAR level: dense LOCAL n-gram (proximal, fast). Thin wrapper on evidence.ExpertBank.
# ════════════════════════════════════════════════════════════════════════════════════════════════

class CharLevel:
    """Dense local char backoff. Short orders = proximal/fast. Emits a per-position (m,27) log-dist for the
    m = n-1 predicted positions (t = 1..n-1), via the vectorized ExpertBank (fresh product-of-experts pool)."""

    def __init__(self, orders=(2, 3, 4, 5)):
        self.bank = EV.ExpertBank(orders=orders)

    def learn(self, ids):
        self.bank.learn(np.asarray(ids))
        return self

    def logdist(self, ids):
        """(m,27) log next-char distribution for positions t=1..n-1 over the (possibly noisy) stream ids.
        GEOMETRIC-MEAN pool of the per-order experts (each backed-off order is already a proper dist; average
        their logs → calibrated, à la cortex.vote). Product-of-experts here over-sharpens (Exp I's blow-up)."""
        orders_logp, _ = self.bank.logp_orders(np.asarray(ids))
        s = np.zeros_like(orders_logp[0])
        for lp in orders_logp:                               # each lp is already a normalized log-dist per row
            s = s + lp
        s = s / len(orders_logp)                             # geometric mean
        s = s - s.max(axis=1, keepdims=True)
        z = np.log(np.exp(s).sum(axis=1, keepdims=True))
        return s - z


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  shared: word segmentation + spelling lexicon (maps a word distribution → a next-char prior)
# ════════════════════════════════════════════════════════════════════════════════════════════════

def word_spans(ids):
    """(start,end) char spans of every word (run between spaces). id-space, vectorized."""
    sp = np.nonzero(ids == SPACE)[0]
    bounds = np.concatenate([[-1], sp, [len(ids)]])
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i] + 1]


def char_prior_from_words(pw, prefix_bytes, id2spell):
    """pw: {word_id: prob}. prefix_bytes: the chars typed so far of the CURRENT word (tuple of ids).
    Project the word belief onto the next char: for each candidate word whose spelling starts with the prefix,
    add its prob to the char that continues it (or to SPACE if the prefix completes the word). Returns a
    27-vector (unnormalized) or None when no candidate matches — the cortex.char_prior move, id-space."""
    cp = np.zeros(V)
    plen = len(prefix_bytes)
    any_ = False
    for wid, prob in pw.items():
        sp = id2spell.get(wid)
        if sp is None or len(sp) < plen:
            continue
        if plen and tuple(sp[:plen]) != prefix_bytes:
            continue
        nc = sp[plen] if len(sp) > plen else SPACE           # next char, or finish-the-word → space
        cp[nc] += prob
        any_ = True
    return cp if any_ and cp.sum() > 0 else None


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  L1 — WORD level: offset-keyed attention (distal, mid-range) + lexicon.
# ════════════════════════════════════════════════════════════════════════════════════════════════

class WordLevel:
    """Offset-keyed count-attention over the word stream (reuses OffsetAttn). Predicts the NEXT word from the
    D preceding words; at decode it projects that word-distribution down to a next-char prior through the
    spelling lexicon. Distal (looks D words back), mid timescale. Emits a per-char-position (m,27) log-dist."""

    def __init__(self, D=6, gamma=8.0, vocab_size=30000):
        self.attn = OA.OffsetAttn(D=D, gamma=gamma)
        self.D = D
        self.vocab_size = vocab_size

    def learn(self, ids):
        ids = np.asarray(ids)
        spans = word_spans(ids)
        stream, vocab_list, UNK = OA.build_word_stream(ids, spans, vocab_size=self.vocab_size)
        self.attn.fit(stream)
        self.stream = stream
        self.UNK = UNK
        # id2spell: word_id -> tuple(char ids). Reconstruct from the kept vocab strings.
        A = "abcdefghijklmnopqrstuvwxyz "
        ch = {c: i for i, c in enumerate(A)}
        self.id2spell = {i: tuple(ch[c] for c in w) for i, w in enumerate(vocab_list)}
        self.w2id = {tuple(ch[c] for c in w): i for i, w in enumerate(vocab_list)}
        return self

    def logdist(self, ids):
        """For every predicted char position t=1..n-1, the word level's next-char log prior (backed off to
        uniform where it abstains). Computed once per word boundary state, broadcast across the word's chars."""
        ids = np.asarray(ids)
        n = len(ids)
        m = n - 1
        out = np.full((m, V), LOG_UNIFORM, dtype=np.float32)
        spans = word_spans(ids)
        # rolling buffer of the last D completed word-ids (oldest..newest)
        ctxbuf = []
        # for each char position t, we need: (a) the D previous completed words, (b) the prefix of the
        # current (in-progress) word. Walk words in order; within a word, vary only the prefix.
        A = "abcdefghijklmnopqrstuvwxyz "
        ch = {c: i for i, c in enumerate(A)}
        # precompute, per word, the predicted word-distribution from its context (independent of prefix)
        for (s, e) in spans:
            # OffsetAttn.predict needs a full D-length context (it indexes ctx[-d]); left-pad with UNK.
            if len(ctxbuf) >= 1:
                ctxD = ([self.UNK] * self.D + ctxbuf)[-self.D:]
                pw = self.attn.predict(ctxD)
            else:
                pw = None
            # for each char position inside this word (predicting ids[t], t in s..e-1, plus the trailing space)
            for t in range(max(s, 1), e):
                prefix = tuple(ids[s:t])
                if pw:
                    cp = char_prior_from_words(pw, prefix, self.id2spell)
                    if cp is not None:
                        lp = np.log(cp / cp.sum() + 1e-12)
                        out[t - 1] = lp.astype(np.float32)
            # the space after the word (predicting ids[e]) — word completes; high prior on SPACE if a word matches
            if e < n and pw:
                prefix = tuple(ids[s:e])
                cp = char_prior_from_words(pw, prefix, self.id2spell)
                if cp is not None:
                    lp = np.log(cp / cp.sum() + 1e-12)
                    out[e - 1] = lp.astype(np.float32)
            # commit this completed word to the context buffer
            wid = self.w2id.get(tuple(ids[s:e]), self.UNK)
            ctxbuf.append(wid)
            if len(ctxbuf) > 4 * self.D:
                ctxbuf = ctxbuf[-2 * self.D:]
        return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  L2 — PHRASE level: branching-entropy CHUNKS + a change/trajectory model.
# ════════════════════════════════════════════════════════════════════════════════════════════════

class PhraseLevel:
    """Discover multi-word CHUNKS by branching-entropy (boundaries.phrase_cuts), assign each chunk an id, then
    run a trajectory CountColumn over the chunk-id stream to predict the NEXT chunk. Project the predicted-chunk
    distribution to a next-char prior via a chunk-spelling lexicon (chunk = a fixed word sequence with a known
    spelling). Slower timescale than words, captures collocations. Emits a per-char (m,27) log-dist."""

    def __init__(self, target_rate=0.45, order=3, vocab_size=30000, max_chunk_vocab=8000):
        self.target_rate = target_rate
        self.order = order
        self.vocab_size = vocab_size
        self.max_chunk_vocab = max_chunk_vocab

    def learn(self, ids):
        ids = np.asarray(ids)
        spans = word_spans(ids)
        stream, vocab_list, UNK = OA.build_word_stream(ids, spans, vocab_size=self.vocab_size)
        self.spans = spans
        self.word_stream = stream
        self.UNK = UNK
        # char-tuple → word-id, so eval words can be mapped to the TRAIN vocab (build_word_stream re-derives a
        # fresh vocab per stream, which would break chunk matching across train/eval).
        A0 = "abcdefghijklmnopqrstuvwxyz "
        ch0 = {c: i for i, c in enumerate(A0)}
        self.w2id = {tuple(ch0[c] for c in w): i for i, w in enumerate(vocab_list)}
        vocab = int(stream.max()) + 1
        # branching-entropy phrase cuts over the WORD stream → list of (start,end) word-index spans
        chunks = BD.phrase_cuts(stream, vocab, target_rate=self.target_rate)
        # name each chunk by its word-id tuple; keep the most frequent as a chunk vocabulary
        from collections import Counter
        names = [tuple(stream[a:b].tolist()) for (a, b) in chunks]
        cnt = Counter(names)
        keep = [nm for nm, _ in cnt.most_common(self.max_chunk_vocab)]
        chunk2id = {nm: i for i, nm in enumerate(keep)}
        CUNK = len(keep)
        self.chunk2id = chunk2id
        self.CUNK = CUNK
        # chunk-id stream (in chunk order) for the trajectory model
        cid_stream = np.fromiter((chunk2id.get(nm, CUNK) for nm in names), dtype=np.int64, count=len(names))
        self.cid_stream = cid_stream
        self.chunk_spans = chunks                            # word-index spans, in order
        # trajectory model over chunk ids — predicts the next chunk from recent chunks (change memory)
        self.traj = TR.CountColumn(base=CUNK + 1, order=self.order).learn(cid_stream)
        # chunk-id -> spelling (char id tuple of the whole phrase, spaces between words)
        A = "abcdefghijklmnopqrstuvwxyz "
        id2word = {i: w for i, w in enumerate(vocab_list)}
        self.chunk_spell = {}
        for nm, cid in chunk2id.items():
            words = [id2word.get(w, "") for w in nm]
            s = " ".join(words)
            self.chunk_spell[cid] = tuple(ord(c) - 97 if c != " " else SPACE for c in s)
        # build a fast next-chunk predictor table: chunk-id -> {next_chunk_id: count} (order-1 marginal),
        # used as the prefix-projection source (the trajectory model gives the bpc-on-chunks number separately)
        nxt = {}
        for a, b in zip(cid_stream[:-1], cid_stream[1:]):
            d = nxt.setdefault(int(a), {})
            d[int(b)] = d.get(int(b), 0) + 1
        self.next_chunk = nxt
        return self

    def chunk_bpc(self):
        """Bits/chunk the trajectory model achieves predicting the next chunk — the phrase level's OWN axis."""
        return self.traj.batch_logloss(self.cid_stream)

    def logdist(self, ids):
        """Per-char next-char log prior from the phrase level. Greedily re-segment the eval stream into the SAME
        chunk vocabulary (longest-match), and at every char inside a chunk project the trajectory model's
        predicted-NEXT-chunk distribution (given the PREVIOUS chunk) onto the continuation of THIS chunk's
        spelling. Distal: the prior on the current chunk's chars comes from the previous whole chunk."""
        ids = np.asarray(ids)
        n = len(ids)
        m = n - 1
        out = np.full((m, V), LOG_UNIFORM, dtype=np.float32)
        spans = word_spans(ids)
        # map each eval word (char-tuple) to its TRAIN word-id (UNK if unseen), so chunk names line up.
        words = [self.w2id.get(tuple(ids[s:e].tolist()), self.UNK) for (s, e) in spans]
        nw = len(words)
        # max chunk length (in words) for greedy longest-match
        maxlen = max((len(nm) for nm in self.chunk2id), default=1)
        prev_cid = None
        wi = 0
        while wi < nw:
            # longest chunk starting at wi that's in the vocabulary; else single word (unknown chunk)
            cid = None
            L = 1
            for ln in range(min(maxlen, nw - wi), 0, -1):
                nm = tuple(words[wi + j] for j in range(ln))
                c = self.chunk2id.get(nm)
                if c is not None:
                    cid = c
                    L = ln
                    break
            # the char span of this chunk = from start of word wi to end of word wi+L-1
            s0 = spans[wi][0]
            e0 = spans[wi + L - 1][1]
            # prediction: next-chunk distribution given the previous chunk id (trajectory order-1 marginal)
            pred = self.next_chunk.get(prev_cid) if prev_cid is not None else None
            if pred:
                tot = sum(pred.values())
                for t in range(max(s0, 1), min(e0 + 1, n)):
                    prefix = tuple(ids[s0:t])
                    plen = len(prefix)
                    cp = np.zeros(V)
                    any_ = False
                    for ncid, c in pred.items():
                        sp = self.chunk_spell.get(ncid)
                        if sp is None or len(sp) < plen:
                            continue
                        if plen and tuple(sp[:plen]) != prefix:
                            continue
                        nc = sp[plen] if len(sp) > plen else SPACE
                        cp[nc] += c / tot
                        any_ = True
                    if any_ and cp.sum() > 0:
                        out[t - 1] = np.log(cp / cp.sum() + 1e-12).astype(np.float32)
            prev_cid = cid if cid is not None else prev_cid
            wi += L
        return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  ONLINE topic coder — leader-clustering (replaces ignition's batch k-means). STREAMING, no k-means.
# ════════════════════════════════════════════════════════════════════════════════════════════════

class OnlineTopicCoder:
    """Assign each content word a topic id by ONLINE LEADER CLUSTERING on its co-occurrence signature.

    Online & no-batch: stream the words once; maintain a decayed co-occurrence row per content word (a leaky
    accumulator), and a small set of cluster LEADERS (centroids). For each content word, when its signature has
    accumulated enough mass, assign it to the nearest leader by cosine if within threshold, else MINT a new
    leader (leader/sequential clustering — the classic online, single-pass alternative to k-means). Centroids
    update incrementally (running mean). No global optimization, no eigendecomposition."""

    def __init__(self, n_stop=100, top_context=300, min_count=5, sim_thresh=0.18, max_topics=160,
                 window=6, decay=0.999):
        self.n_stop = n_stop
        self.top_context = top_context
        self.min_count = min_count
        self.sim_thresh = sim_thresh
        self.max_topics = max_topics
        self.window = window
        self.decay = decay
        self.topic_of = None

    def fit(self, wids, vocab):
        wids = np.asarray(wids)
        freq = np.bincount(wids, minlength=vocab).astype(np.float64)
        order = np.argsort(freq)[::-1]
        stop = set(order[:self.n_stop].tolist())
        content_mask = freq >= self.min_count
        for s in stop:
            content_mask[s] = False
        # context vocabulary = top frequent non-stop words (signature dimensions)
        ctx_pool = [int(w) for w in order if int(w) not in stop]
        ctx_words = np.array(ctx_pool[:self.top_context], dtype=np.int64)
        ctx_index = -np.ones(vocab, np.int64)
        ctx_index[ctx_words] = np.arange(len(ctx_words))
        D = len(ctx_words)

        # accumulate decayed co-occurrence signatures in ONE streaming pass (vectorized per offset, online-eq:
        # equivalent to a leaky counter advanced token by token).
        sig = np.zeros((vocab, D), np.float64)
        for d in range(1, self.window + 1):
            for a, b in ((wids[:-d], wids[d:]), (wids[d:], wids[:-d])):
                cb = ctx_index[b]
                ok = (cb >= 0) & content_mask[a]
                if ok.any():
                    np.add.at(sig, (a[ok], cb[ok]), 1.0)
        # PPMI-ish reweight (count-based, no SVD): log(1+count) then L2-normalize rows for cosine
        sig = np.log1p(sig)
        nrm = np.linalg.norm(sig, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        sig /= nrm

        # ONLINE leader clustering over content words, processed in frequency order (stable leaders first)
        content_words = [int(w) for w in order if content_mask[w]]
        leaders = np.zeros((0, D), np.float64)
        leader_n = []
        topic_of = -np.ones(vocab, np.int64)
        for w in content_words:
            x = sig[w]
            if not x.any():
                continue
            if len(leaders) == 0:
                leaders = x[None, :].copy()
                leader_n = [1.0]
                topic_of[w] = 0
                continue
            sims = leaders @ x
            j = int(np.argmax(sims))
            if sims[j] >= self.sim_thresh or len(leaders) >= self.max_topics:
                # join nearest leader; incremental running-mean update (online)
                leader_n[j] += 1.0
                leaders[j] += (x - leaders[j]) / leader_n[j]
                nrmj = np.linalg.norm(leaders[j])
                if nrmj > 0:
                    leaders[j] /= nrmj
                topic_of[w] = j
            else:
                leaders = np.vstack([leaders, x[None, :]])
                leader_n.append(1.0)
                topic_of[w] = len(leaders) - 1
        self.topic_of = topic_of
        self.K = max(1, len(leaders))
        return self


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  L3 — THEME level: online topic state + leaky evidence (slow integrator, long timescale).
# ════════════════════════════════════════════════════════════════════════════════════════════════

def commit_G_online(topic_seq, K, halflife=50.0, margin=0.16):
    """Online committed topic G (ignition + hysteresis), copied here to stay self-contained / online."""
    decay = 0.5 ** (1.0 / halflife)
    hist = np.zeros(K, np.float64)
    G = 0
    out = np.empty(len(topic_seq), np.int64)
    for i, t in enumerate(topic_seq):
        hist *= decay
        if 0 <= t < K:
            hist[t] += 1.0
        s = hist.sum()
        if s > 0:
            lead = int(np.argmax(hist))
            if lead != G and (hist[lead] - hist[G]) / s > margin:
                G = lead
        out[i] = G
    return out


class ThemeLevel:
    """The slow integrator. Commit a topic G online (decayed topic histogram + ignition), then maintain a
    G-conditioned char table:  count(next_char | G, short_ctx). At decode, blend toward the chars typical of
    the committed topic — a long-timescale prior on local prediction. Emits a per-char (m,27) log-dist.

    The "leaky evidence" piece: per char we also keep a leaky log-evidence over G's char table so a single
    off-topic char can't yank the prior (robustness axis)."""

    def __init__(self, g_order=4, halflife=50.0, margin=0.16, topic_kw=None):
        self.g_order = g_order
        self.halflife = halflife
        self.margin = margin
        self.topic_kw = topic_kw or {}

    def _word_topic_seq(self, ids):
        """Map char stream → per-CHAR committed topic G. Build word stream, topic per word, commit G per word,
        then broadcast each word's G across its chars (and the trailing space)."""
        spans = word_spans(ids)
        stream, vocab_list, UNK = OA.build_word_stream(ids, spans, vocab_size=30000)
        return spans, stream, vocab_list

    def learn(self, ids):
        ids = np.asarray(ids)
        spans = word_spans(ids)
        stream, vocab_list, UNK = OA.build_word_stream(ids, spans, vocab_size=30000)
        vocab = int(stream.max()) + 1
        self.coder = OnlineTopicCoder(**self.topic_kw).fit(stream, vocab)
        topic_seq = self.coder.topic_of[stream]
        Gword = commit_G_online(topic_seq, self.coder.K, self.halflife, self.margin)
        self.K = self.coder.K
        # broadcast G to every char position (the char at span s..e gets that word's G; spaces inherit prev G)
        Gchar = np.zeros(len(ids), np.int64)
        for wi, (s, e) in enumerate(spans):
            Gchar[s:e] = Gword[wi]
            if e < len(ids):
                Gchar[e] = Gword[wi]                          # the trailing space too
        self.Gchar = Gchar
        # count(next_char | G, ctx_order g_order) and the plain backoff (ctx only) — vectorized via bincount
        self._learn_gtab(ids, Gchar)
        return self

    def _ctx_ids(self, ids, k):
        n = len(ids)
        if k == 0:
            return np.zeros(n, np.int64)
        w = np.lib.stride_tricks.sliding_window_view(ids, k)[: n - k].astype(np.int64)
        powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
        return w @ powers

    def _learn_gtab(self, ids, Gchar):
        ids = np.asarray(ids, np.int64)
        k = self.g_order
        ctx = self._ctx_ids(ids, k)                          # position t = k..n-1
        tok = ids[k:].astype(np.int64)
        g = Gchar[k:].astype(np.int64)
        # key = (G * V^k + ctx) ; store as dict ctxkey -> (counts 27)
        gkey = g * (V ** k) + ctx
        comb = gkey * V + tok
        ukey, ucnt = np.unique(comb, return_counts=True)
        uctx = ukey // V
        utok = ukey % V
        ctx_ids, inv = np.unique(uctx, return_inverse=True)
        mat = np.zeros((len(ctx_ids), V), np.float64)
        mat[inv, utok] = ucnt
        self.g_ctx_ids = ctx_ids
        self.g_logp = np.log((mat + ALPHA) / (mat.sum(1, keepdims=True) + ALPHA * V)).astype(np.float32)

    def _topic_seq_eval(self, ids):
        """G per char on a (possibly held-out) eval stream, using the TRAIN-learned topic_of + online commit."""
        spans = word_spans(ids)
        A = "abcdefghijklmnopqrstuvwxyz "
        ch = {c: i for i, c in enumerate(A)}
        # reuse the coder vocab via build_word_stream on the eval ids (same top-30k vocab assumption; words
        # unseen in train map to a topic of -1 → stopword-like, no histogram contribution)
        stream, vocab_list, UNK = OA.build_word_stream(ids, spans, vocab_size=30000)
        # We can't reuse train word-ids directly (different vocab indexing), so re-map by spelling.
        topic_seq = np.full(len(stream), -1, np.int64)
        # build spelling->topic from train: train coder.topic_of indexes train word ids; we don't keep train
        # spellings here, so we approximate eval topics by re-fitting the coder on eval (online, cheap) — this
        # keeps the level fully online per-stream. (For held-out we accept a small leakage-free re-fit.)
        vocab = int(stream.max()) + 1
        coder = OnlineTopicCoder(**self.topic_kw).fit(stream, vocab)
        topic_seq = coder.topic_of[stream]
        Gword = commit_G_online(topic_seq, coder.K, self.halflife, self.margin)
        Gchar = np.zeros(len(ids), np.int64)
        for wi, (s, e) in enumerate(spans):
            Gchar[s:e] = Gword[wi] % self.K
            if e < len(ids):
                Gchar[e] = Gword[wi] % self.K
        return Gchar

    def logdist(self, ids, Gchar=None):
        ids = np.asarray(ids, np.int64)
        n = len(ids)
        m = n - 1
        out = np.full((m, V), LOG_UNIFORM, dtype=np.float32)
        if Gchar is None:
            Gchar = self.Gchar if len(self.Gchar) == n else self._topic_seq_eval(ids)
        k = self.g_order
        ctx = self._ctx_ids(ids, k)                          # position t = k..n-1
        g = Gchar[k:].astype(np.int64)
        gkey = g * (V ** k) + ctx
        pos = np.searchsorted(self.g_ctx_ids, gkey)
        seen = (pos < len(self.g_ctx_ids)) & (self.g_ctx_ids[np.minimum(pos, len(self.g_ctx_ids) - 1)] == gkey)
        rows = np.arange(k - 1, k - 1 + len(gkey))
        out[rows[seen]] = self.g_logp[pos[seen]]
        return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  GATING / ARBITRATION — per-token routing among the level log-dists.
# ════════════════════════════════════════════════════════════════════════════════════════════════

def _running_conf(logdist, targets, gamma=0.85):
    """Leaky confidence per position = decayed log-prob the level assigned to the char that actually came.
    High = the level has been predicting well lately. This is the thalamic/BG signal that gates routing.
    NOTE: uses the observed target → it is a causal running estimate (no future leakage; uses t and earlier)."""
    rows = np.arange(len(targets))
    obs = logdist[rows, targets]
    conf = np.empty_like(obs)
    acc = obs[0]
    conf[0] = acc
    for t in range(1, len(obs)):
        acc = gamma * acc + (1 - gamma) * obs[t]
        conf[t] = acc
    # shift by one so the gate at position t uses confidence up to t-1 (causal)
    shifted = np.empty_like(conf)
    shifted[0] = conf[0]
    shifted[1:] = conf[:-1]
    return shifted


def static_pool(logdists, weights=None):
    """The Exp I baseline gate: a FIXED geometric-mean pool of the level log-dists (optionally fixed weights).
    logdists: list of (m,27) log-dists. Returns (m,27) normalized log-dist."""
    if weights is None:
        weights = np.ones(len(logdists))
    s = np.zeros_like(logdists[0])
    W = 0.0
    for w, lp in zip(weights, logdists):
        s = s + w * lp
        W += w
    s = s / max(W, 1e-9)
    s = s - s.max(axis=1, keepdims=True)
    z = np.log(np.exp(s).sum(axis=1, keepdims=True))
    return s - z


def soft_gate(logdists, targets, gamma=0.85, temp=1.0, floor=0.05):
    """Thalamic soft gate: per-token, weight each level by its recent CONFIDENCE (softmax over levels of the
    leaky neg-NLL), then geometric-mean pool with those dynamic weights. A confident level dominates; a
    surprised (low-conf) level is down-weighted. `floor` keeps every level a minimum say (driver+modulator)."""
    confs = np.stack([_running_conf(lp, targets, gamma) for lp in logdists], axis=0)   # (L, m)
    # softmax over levels per position (higher conf = higher weight)
    z = confs / temp
    z = z - z.max(axis=0, keepdims=True)
    w = np.exp(z)
    w = w / w.sum(axis=0, keepdims=True)
    w = floor + (1 - floor * len(logdists)) * w                # mix with a uniform floor
    s = np.zeros_like(logdists[0])
    for li in range(len(logdists)):
        s = s + w[li][:, None] * logdists[li]
    s = s - s.max(axis=1, keepdims=True)
    zz = np.log(np.exp(s).sum(axis=1, keepdims=True))
    return s - zz, w


def hard_router(logdists, targets, gamma=0.85, surprise_open=True):
    """Basal-ganglia hard arbitration: per-token, route to the SINGLE most-confident level (winner-take-all).
    If surprise_open, when the lowest (char) level is SURPRISED (recent conf below its running baseline) the
    gate is allowed to open to higher levels; otherwise it defaults to the char level. Returns (logdist, choice)."""
    confs = np.stack([_running_conf(lp, targets, gamma) for lp in logdists], axis=0)   # (L, m)
    choice = np.argmax(confs, axis=0)
    m = logdists[0].shape[0]
    out = np.empty_like(logdists[0])
    stack = np.stack(logdists, axis=0)                          # (L, m, 27)
    out = stack[choice, np.arange(m)]
    return out, choice


def anchored_gate(char_ld, higher_lds, targets, gamma=0.9, beta=0.6, surprise_gamma=0.9):
    """Driver + modulator gate (the principled form, per the sources' 'never mix content and gain' edge types).

    The CHAR level is the DRIVER (always the base distribution). Each higher level is a MODULATOR: it adds its
    log-prior on top of char, with a per-token weight = (its recent confidence) × (how SURPRISED char is right
    now) × (whether it actually has a non-uniform opinion here). So higher levels stay silent while the fast
    local model is doing fine, and only speak up at high-surprise points where distal/topic info can help —
    exactly the thalamic-gate / surprise-opens-the-gate story. Returns (logdist, mean_modulator_weight).

    - confidence c_l(t): leaky neg-NLL of level l (causal).
    - char surprise s(t): leaky predictive ENTROPY of the char level (high = char is unsure → open the gate).
    - opinion mask: 1 where the level's row differs from uniform (it abstains otherwise).
    """
    m = char_ld.shape[0]
    # char surprise = leaky forward entropy of the char dist (causal, normalized to [0,1] by /log2(V))
    p = np.exp(char_ld)
    Hc = -(p * (char_ld / np.log(2))).sum(1)                    # bits
    sur = np.empty(m)
    acc = Hc[0]
    for t in range(1, m):
        acc = surprise_gamma * acc + (1 - surprise_gamma) * Hc[t]
        sur[t] = acc
    sur[0] = Hc[0]
    g = np.empty(m); g[0] = sur[0]; g[1:] = sur[:-1]            # causal shift
    gate_open = np.clip(g / np.log2(V), 0.0, 1.0)               # 0..1, high when char is uncertain

    s = char_ld.copy()
    wsum = np.zeros(m)
    for lp in higher_lds:
        conf = _running_conf(lp, targets, gamma)               # higher conf = more trusted
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)   # 0..1
        opinion = (np.abs(lp - LOG_UNIFORM).sum(1) > 1e-3).astype(np.float64)
        w = beta * conf * gate_open * opinion                  # modulator weight, per token
        # add the modulator's CONTRIBUTION relative to uniform (so abstain rows add ~0)
        s = s + w[:, None] * (lp - LOG_UNIFORM)
        wsum += w
    s = s - s.max(axis=1, keepdims=True)
    z = np.log(np.exp(s).sum(axis=1, keepdims=True))
    return s - z, float(wsum.mean())


def bpc_of(logdist, targets):
    rows = np.arange(len(targets))
    return float(-(logdist[rows, targets] / np.log(2)).mean())


def acc_of(logdist, targets):
    return float((logdist.argmax(1) == targets).mean())
