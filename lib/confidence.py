"""confidence.py — Exp AB: NARS-style calibrated truth values on count-based associations.

The thesis (Pei Wang's NARS; predictive-coding precision; ACT-R). A raw count `w` says how OFTEN a
context fired but nothing about whether it was RIGHT. Split it. Every time a context is used to
predict, score its top prediction against what actually came: increment HITS `w+` on a correct top-1,
MISSES `w-` on a wrong one. Then attach a NARS TRUTH VALUE to the association:

    frequency  f = w+ / w           (w = w+ + w-)   — how often this context's bet pays off
    confidence c = w / (w + k)      (k = 1)         — how much evidence stands behind that frequency

Both are read straight off counts — no gradient, no threshold fitting. Two things fall out:

  REVISION (evidence-additive pooling). To combine several order-experts for one position, SUM their
  w+ and w- (independent evidence adds), then recompute (f,c) on the totals. Each expert's vote is
  naturally weighted by its own c (a low-evidence expert barely moves the pooled w). This is the
  NARS revision rule, and it replaces the hand-weighted geometric mean.

  PRECISION (predictive-coding gain). pi_k = 1 / running-variance of order-k's signed error, a leaky
  accumulator. A level that has been reliably right has low error-variance → high precision → its
  vote is amplified. We offer precision as an ALTERNATIVE voter weight to c, and try both.

  A PRINCIPLED GATE. The classic gate opens on a hand-tuned confidence threshold. Here the gate opens
  when the high-order expert's NARS confidence c exceeds the point where its expected accuracy (f·c,
  the c-discounted frequency) beats the backoff's — a threshold the data sets, not the operator.

Alphabet matches lib/cortex / lib/fastchar: a..z = 0..25, space = 26, V = 27. Everything here is a
SINGLE causal pass: a context's (w+,w-) at position t reflect only positions < t, so train and eval
are the same online stream. Vectorized where the online semantics permit (the prediction a context
makes is fixed by its accumulated counts, so we can resolve a position's top-1 from the counts as of
its own arrival — see CountTruth.online_pass).
"""
import numpy as np

V = 27
K = 1.0          # NARS evidential horizon (c = w/(w+k)); k=1 is the canonical default


def _ctx_ids(ids, k):
    """For every t in [k, n): the order-k context id (base-V encode of ids[t-k:t]). Vectorized."""
    n = len(ids)
    if k == 0:
        return np.zeros(n, np.int64)
    w = np.lib.stride_tricks.sliding_window_view(ids, k)[: n - k].astype(np.int64)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w @ powers


def _seg_cummax(x, grp_start):
    """Segmented inclusive running maximum of int64 x, reset at each group start (Hillis–Steele scan,
    blocked at group boundaries so no group's max leaks into the next). O(m log m) work, fully
    vectorized — no python loop over groups."""
    m = len(x)
    if m == 0:
        return x.copy()
    NEG = np.iinfo(np.int64).min // 2
    grp_id = np.zeros(m, np.int64); grp_id[grp_start] = 1; grp_id = np.cumsum(grp_id)
    out = x.astype(np.int64).copy()
    step = 1
    while step < m:
        prev = np.full(m, NEG, np.int64)
        prev[step:] = out[:-step]
        same = np.zeros(m, bool)
        same[step:] = grp_id[step:] == grp_id[:-step]
        out = np.where(same, np.maximum(out, prev), out)
        step *= 2
    return out


# ── core: online (w+, w-) per (context, predicted-token) ─────────────────────────────────────────

class CountTruth:
    """One char-order's count table that, in a single online pass, tracks for each context BOTH the
    raw next-token counts AND the NARS hit/miss split of its TOP-1 bet.

    Online semantics. Walk the stream once. At position t the context `ctx` has accumulated next-token
    counts from all earlier occurrences; its current top-1 prediction is the argmax of those counts.
    We score that bet against ids[t]: HIT if argmax == ids[t], else MISS. Then we fold ids[t] into the
    counts (so position t+1 sees it). The result is, per context: total raw counts `cnt[ctx]` (27-dim),
    a hit total w+[ctx] and a miss total w-[ctx]. f = w+/(w+ + w-), c = (w+ + w-)/(w+ + w- + k).

    This is exactly an online single-pass count of "was this context's leading guess right?" — no
    second pass, no labels beyond the next char itself.
    """

    def __init__(self, order):
        self.order = order
        self.ctx_ids = None      # sorted unique context ids seen (int64)
        self.cnt = None          # (n_ctx, 27) final raw counts
        self.wp = None           # (n_ctx,) hit total of the context's running top-1 bet
        self.wm = None           # (n_ctx,) miss total

    def online_pass(self, ids):
        """Single causal pass, fully vectorized. Returns self with cnt / wp / wm filled.

        For each predicted position t we need: did the context's RUNNING top-1 (the argmax over its
        next-token counts accumulated from its EARLIER occurrences only) equal ids[t]? We compute this
        without any per-context python loop.

        Sort positions by (context, stream-order) so each context's occurrences are a contiguous run.
        Within a run, let pre_b(j) = how many times token b has appeared BEFORE position j. The running
        argmax before j is the token maximizing pre with the canonical "first to reach the max" tie
        break. We need only test one token per position — the token nxt_s[j] that actually arrives —
        against the leader. Encode each occurrence's strength as a composite key that increases with its
        running count and breaks ties by EARLIER first-arrival:
            key(j) = post_b(j) * BIG - first_arrival_rank_of_b
        where post_b(j) = pre_b(j)+1 is the count INCLUDING j. A segmented inclusive running-max of this
        key over each group gives, at j, the strongest (count, earliest) claim seen so far INCLUDING j.
        The leader strictly BEFORE j is that running-max shifted one step within the group. nxt_s[j] is
        the running top-1 (a HIT) iff its own pre-strength key — post_b(j-occurrence) with j excluded —
        is >= the leader-before key. Because we only ever fold one token per step, the test reduces to:
        the leader-before key equals the strength key this token had at its PREVIOUS occurrence (or, if
        this is the token's first occurrence, it cannot be the prior leader). All segmented ops are
        vectorized cumulative maxima — no python over contexts."""
        ids = np.ascontiguousarray(ids, np.int64)
        k = self.order
        ctx = _ctx_ids(ids, k)               # context at each predicted position t = k..n-1
        nxt = ids[k:].astype(np.int64)        # the token that actually came at t
        order = np.argsort(ctx, kind="stable")   # group by context, preserve stream order within group
        ctx_s = ctx[order]
        nxt_s = nxt[order]
        m = len(ctx_s)
        uniq, inv = np.unique(ctx_s, return_inverse=True)
        n_ctx = len(uniq)
        if m == 0:
            self.ctx_ids = uniq
            self.cnt = np.zeros((n_ctx, V)); self.wp = np.zeros(n_ctx); self.wm = np.zeros(n_ctx)
            return self
        grp_start = np.concatenate([[0], np.nonzero(np.diff(inv))[0] + 1])   # first index of each group
        # exclusive within-(group,token) prefix count = pre_b(j)
        gt = inv * V + nxt_s
        s2 = np.argsort(gt, kind="stable")
        gt_s = gt[s2]
        tok_start = np.concatenate([[0], np.nonzero(np.diff(gt_s))[0] + 1])  # first index of each (g,tok) run
        run_of = np.searchsorted(tok_start, np.arange(m), side="right") - 1
        pre_s = np.arange(m) - tok_start[run_of]            # 0,1,2,... within each (g,tok) run
        pre = np.empty(m, np.int64); pre[s2] = pre_s        # pre_b(j) back in (group,stream) order
        # Composite strength key: higher running COUNT wins; ties broken by LOWER token id (the
        # canonical np.argmax convention the naive reference uses). key = post*V - tok.
        BIG = np.int64(V)
        post = pre + 1
        strength = post * BIG - nxt_s                        # key INCLUDING this occurrence
        # segmented inclusive running max of strength within each group (stream order)
        cum = _seg_cummax(strength, grp_start)
        # leader-before-j key = running max over strictly-earlier same-group positions = cum shifted 1
        leader_before = np.empty(m, np.int64); leader_before[:] = -1
        nonfirst = np.ones(m, bool); nonfirst[grp_start] = False
        leader_before[nonfirst] = cum[np.nonzero(nonfirst)[0] - 1]
        # nxt_s[j] is the running top-1 before j iff its OWN strength BEFORE j (pre*V - tok) is the
        # leader-before key. (If pre==0 the token hasn't been seen, its pre-strength = -tok < any real
        # leader, so it's not the leader — correct: an unseen token can't be the argmax.)
        my_pre_strength = pre * BIG - nxt_s
        has_bet = nonfirst                                   # a bet exists once the group has ≥1 prior token
        hit = has_bet & (pre > 0) & (my_pre_strength == leader_before)
        cnt = np.zeros((n_ctx, V), np.float64)
        np.add.at(cnt, (inv, nxt_s), 1.0)
        wp = np.zeros(n_ctx, np.float64); wm = np.zeros(n_ctx, np.float64)
        np.add.at(wp, inv[hit], 1.0)
        miss = has_bet & ~hit
        np.add.at(wm, inv[miss], 1.0)
        self.ctx_ids = uniq
        self.cnt = cnt
        self.wp = wp
        self.wm = wm
        return self

    def lookup(self, ctx_query):
        """Map a query context-id array → row indices (or -1 if unseen)."""
        pos = np.searchsorted(self.ctx_ids, ctx_query)
        pos_c = np.minimum(pos, len(self.ctx_ids) - 1)
        seen = (pos < len(self.ctx_ids)) & (self.ctx_ids[pos_c] == ctx_query)
        return np.where(seen, pos_c, -1)


def truth_of(wp, wm, k=K):
    """NARS (frequency, confidence) from hit/miss totals. w = wp+wm; f = wp/w; c = w/(w+k).
    With no evidence (w=0): f=0.5 (max ignorance), c=0."""
    w = wp + wm
    f = np.where(w > 0, wp / np.maximum(w, 1e-12), 0.5)
    c = w / (w + k)
    return f, c


# ── distributions per order (for bpc / perplexity) ──────────────────────────────────────────────

ALPHA = 0.05

def order_logdist(table, ctx_query, uni_log):
    """Smoothed log next-token distribution for each query position from one CountTruth table.
    Unseen contexts get the unigram fallback `uni_log`. Returns (m,27) and a `seen` mask."""
    m = len(ctx_query)
    rows = table.lookup(ctx_query)
    out = np.tile(uni_log, (m, 1))
    seen = rows >= 0
    if seen.any():
        c = table.cnt[rows[seen]]
        p = (c + ALPHA) / (c.sum(1, keepdims=True) + ALPHA * V)
        out[seen] = np.log(p)
    return out, seen


def unigram_log(ids):
    cnt = np.bincount(np.asarray(ids, np.int64), minlength=V).astype(np.float64)
    p = (cnt + ALPHA) / (cnt.sum() + ALPHA * V)
    return np.log(p)


# ── three combiners over a stack of orders ───────────────────────────────────────────────────────

def bare_count_pool(order_lds):
    """BASELINE: product-of-experts (sum of per-order log-dists), unweighted. Each order votes equally,
    by its raw smoothed distribution — no truth value used. Renormalize per row."""
    s = np.zeros_like(order_lds[0])
    for ld in order_lds:
        s = s + ld
    s = s - s.max(1, keepdims=True)
    z = np.log(np.exp(s).sum(1, keepdims=True))
    return s - z


def weighted_pool(order_lds, weights):
    """Confidence/precision-weighted product-of-experts: each order's log-dist scaled by its per-position
    weight (m,) before summing. weights[k] is a length-m array. Renormalize per row."""
    s = np.zeros_like(order_lds[0])
    for ld, w in zip(order_lds, weights):
        s = s + ld * w[:, None]
    s = s - s.max(1, keepdims=True)
    z = np.log(np.exp(s).sum(1, keepdims=True))
    return s - z


def revision_truth(tables, ctx_queries, uni_log):
    """NARS REVISION pool — the evidence-additive combiner. For each position, SUM the matched orders'
    per-(context,token) evidence. We approximate per-token (w+,w-) by apportioning a context's hit/miss
    mass across the tokens it predicts, in proportion to that token's count share — i.e. token b in
    context ctx contributes (w+ · cnt_b/Σcnt, w- · cnt_b/Σcnt) when b is the running top, and the miss
    mass to the non-top tokens. Summing these across orders is the revision; the pooled per-token
    (w+_b, w-_b) give a per-token frequency f_b = w+_b/(w+_b+w-_b), which we softmax into a log-dist.

    Concretely (and tractably): each order contributes a per-position 27-dim evidence vector
        e_b = cnt_b   (positive evidence the token gets), scaled by that context's confidence c.
    Summing c·cnt across orders is additive evidence weighted by confidence — the revision rule's
    additive-w with the c-weight folded in. Renormalize per row → log-dist. This is the (f,c)-revision
    combiner: a high-c order dominates the pooled evidence; a low-c (rare/unreliable) order barely moves
    it, which is precisely what bare-count pooling fails to do."""
    m = len(ctx_queries[0])
    acc = np.full((m, V), ALPHA)          # additive evidence with a smoothing floor
    for table, ctxq in zip(tables, ctx_queries):
        rows = table.lookup(ctxq)
        seen = rows >= 0
        if not seen.any():
            continue
        r = rows[seen]
        c = table.cnt[r]                                   # (s,27) raw counts
        wp = table.wp[r]; wm = table.wm[r]
        conf = (wp + wm) / (wp + wm + K)                   # NARS confidence per context (s,)
        acc[seen] += conf[:, None] * c                     # evidence-additive, confidence-weighted
    p = acc / acc.sum(1, keepdims=True)
    return np.log(p)


# ── per-position confidence / precision weights for weighted_pool ─────────────────────────────────

def confidence_weight(table, ctx_query):
    """Per-position NARS confidence c of one order (0 where unseen). Used as a voter weight."""
    m = len(ctx_query)
    rows = table.lookup(ctx_query)
    w = np.zeros(m)
    seen = rows >= 0
    if seen.any():
        r = rows[seen]
        wsum = table.wp[r] + table.wm[r]
        w[seen] = wsum / (wsum + K)
    return w


def precision_weights(order_lds, targets, gamma=0.9, eps=1e-3):
    """Predictive-coding PRECISION per order, online & leaky. For each order, the signed error at t is
    e_t = 1 - p_t(observed)  (0 when the order put all mass on the right token; →1 when surprised).
    Keep a leaky running MEAN and VARIANCE of e per order (causal: weight at t uses stats up to t-1).
    precision pi_t = 1/(var + eps). Returns a list of length-m weight arrays, one per order, each
    normalized so the per-position weights across orders sum to 1 (a precision-allocation).
    This is the inverse-variance gain of predictive coding, as a leaky accumulator — no batch stats."""
    n_orders = len(order_lds)
    m = order_lds[0].shape[0]
    # per-order observed-token prob at each position
    rows = np.arange(m)
    err = np.empty((n_orders, m))
    for i, ld in enumerate(order_lds):
        p_obs = np.exp(ld[rows, targets])
        err[i] = 1.0 - p_obs
    pi = np.empty((n_orders, m))
    for i in range(n_orders):
        e = err[i]
        mean = e[0]
        var = 0.1                       # prior variance
        for t in range(m):
            pi[i, t] = 1.0 / (var + eps)        # causal: precision at t from stats up to t-1
            d = e[t] - mean
            mean = gamma * mean + (1 - gamma) * e[t]
            var = gamma * var + (1 - gamma) * d * d
    # allocate: normalize precisions across orders per position
    pi = pi / pi.sum(0, keepdims=True)
    return [pi[i] for i in range(n_orders)]


# ── gates: tuned-threshold vs principled-c ────────────────────────────────────────────────────────

def tuned_threshold_gate(hi_ld, lo_ld, hi_conf, thresh):
    """Hand-tuned gate: use the HIGH-order distribution where its leaky/NARS confidence ≥ thresh, else
    the LOW-order backoff. hi_conf is a per-position confidence (m,). Returns (logdist, open_frac)."""
    open_mask = hi_conf >= thresh
    out = np.where(open_mask[:, None], hi_ld, lo_ld)
    return out, float(open_mask.mean())


def principled_gate(hi_table, lo_table, ctx_hi, ctx_lo, hi_ld, lo_ld):
    """Principled gate — NO tuned threshold. Open to the HIGH order exactly when its evidence says it is
    expected to do better than the backoff. Expected accuracy of an order at a context = its
    c-discounted frequency f·c (NARS: a high frequency you're unsure of buys you little). The gate
    opens when EA_hi > EA_lo. Unseen high context → closed. Returns (logdist, open_frac, decided_frac).

    This sets the operating point from the data: a high context only takes over once its own track
    record (f) AND its evidence mass (c) jointly beat the backoff's — the threshold is endogenous."""
    rows_hi = hi_table.lookup(ctx_hi)
    rows_lo = lo_table.lookup(ctx_lo)
    m = len(ctx_hi)
    ea_hi = np.zeros(m); ea_lo = np.zeros(m)
    sh = rows_hi >= 0
    if sh.any():
        r = rows_hi[sh]; f, c = truth_of(hi_table.wp[r], hi_table.wm[r])
        ea_hi[sh] = f * c
    sl = rows_lo >= 0
    if sl.any():
        r = rows_lo[sl]; f, c = truth_of(lo_table.wp[r], lo_table.wm[r])
        ea_lo[sl] = f * c
    open_mask = sh & (ea_hi > ea_lo)
    out = np.where(open_mask[:, None], hi_ld, lo_ld)
    return out, float(open_mask.mean()), float(sh.mean())


# ── metrics: bpc, perplexity, calibration ─────────────────────────────────────────────────────────

def decode_metrics(logdist, targets):
    """(accuracy, bpc) for a (m,27) log-distribution (natural log) and m true next-tokens."""
    pred = logdist.argmax(1)
    acc = float((pred == targets).mean())
    rows = np.arange(len(targets))
    bpc = float(-(logdist[rows, targets] / np.log(2)).mean())
    return acc, bpc


def perplexity(logdist, targets):
    rows = np.arange(len(targets))
    nll = -logdist[rows, targets].mean()       # natural-log NLL
    return float(np.exp(nll))


def calibration(conf, correct, n_bins=10):
    """Expected Calibration Error (ECE) + reliability table. conf in [0,1] is the model's stated
    confidence per prediction; correct is the 0/1 hit of that prediction. Bin by conf; ECE = Σ
    (bin_size/N)·|acc_bin − conf_bin|. Returns (ece, rows) where rows = (lo, hi, n, mean_conf, acc)."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    N = len(conf)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    table = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        nb = int(sel.sum())
        if nb == 0:
            table.append((lo, hi, 0, 0.0, 0.0))
            continue
        mc = float(conf[sel].mean())
        ac = float(correct[sel].mean())
        ece += nb / N * abs(ac - mc)
        table.append((lo, hi, nb, mc, ac))
    return float(ece), table


def stated_confidence(logdist):
    """The model's per-position stated confidence = max softmax prob (the prob it assigns its argmax)."""
    return np.exp(logdist).max(1)
