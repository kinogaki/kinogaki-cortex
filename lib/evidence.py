"""evidence.py — Exp R: leaky log-evidence accumulation (Thousand-Brains voting with decay).

The thesis: don't recompute the next-char belief fresh every step. Keep a LEAKY accumulator of
log-evidence so one noisy step can't tank the prediction, and read the accumulator's DROP as a
free boundary signal. The experts are plain char context-orders (the Column, reused).

Two pieces:
  - ExpertBank: learn count tables for a set of char orders; emit, for EVERY position, the full
    27-dim next-char log-prob distribution PER ORDER (vectorized — no per-position python).
  - Pooling: FRESH product-of-experts (sum of the per-order log-probs at this step) vs the
    leaky accumulator E_t = gamma*E_{t-1} + sum_k logp_k(.).  Both decode by argmax / give a bpc.
  - Confidence + boundary signals: running confidence on the OBSERVED char (its drop = a cut),
    and the classic forward branching-entropy RISE, scored head-to-head against true spaces.

Alphabet matches lib/cortex / lib/fastchar: a..z = 0..25, space = 26, V = 27.
"""
import numpy as np

V = 27
ALPHA = 0.05


def _ctx_ids(ids, k):
    """For every t in [k, n): the order-k context id (base-V encode of ids[t-k:t]). Vectorized."""
    n = len(ids)
    if k == 0:
        return np.zeros(n, np.int64)
    w = np.lib.stride_tricks.sliding_window_view(ids, k)[: n - k].astype(np.int64)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w @ powers  # length n-k, aligns to positions t = k..n-1


class ExpertBank:
    """A set of char context-order experts. learn() once on clean train; logp_orders() emits, for a
    (possibly noisy) eval stream, the per-position 27-dim log next-char distribution for each order.

    For order k we store a DENSE table: row = a seen context id, 27 columns of add-α smoothed
    probabilities. Contexts unseen at order k fall back to the next-lower order's distribution at the
    same position (true Katz-style backoff, vectorized: lower orders fill not-yet-resolved rows)."""

    def __init__(self, orders=(2, 3, 4, 5)):
        self.orders = tuple(sorted(orders))         # experts emitted by logp_orders()
        self.all_orders = tuple(range(1, max(self.orders) + 1))  # all needed for backoff
        self.tabs = {}      # k -> (sorted_ctx_ids[int64], logp[float32, (n_ctx,27)])
        self.uni = None     # order-0 log distribution (27,)

    def learn(self, ids):
        ids = np.ascontiguousarray(ids, np.int64)
        cnt0 = np.bincount(ids, minlength=V).astype(np.float64)
        p0 = (cnt0 + ALPHA) / (cnt0.sum() + ALPHA * V)
        self.uni = np.log(p0).astype(np.float32)
        for k in self.all_orders:
            ctx = _ctx_ids(ids, k)
            tok = ids[k:].astype(np.int64)
            key = ctx * V + tok
            ukey, ucnt = np.unique(key, return_counts=True)
            uctx = ukey // V
            utok = ukey % V
            # unique contexts, dense count matrix
            ctx_ids, inv = np.unique(uctx, return_inverse=True)
            mat = np.zeros((len(ctx_ids), V), np.float64)
            mat[inv, utok] = ucnt
            tot = mat.sum(axis=1, keepdims=True)
            logp = np.log((mat + ALPHA) / (tot + ALPHA * V)).astype(np.float32)
            self.tabs[k] = (ctx_ids, logp)
        return self

    def logp_orders(self, ids):
        """Return (orders_logp, positions) where orders_logp[i] is shape (m, 27): for each of the m
        predicted positions t (t = 1..n-1), the order-orders[i] backed-off log next-char distribution.
        positions = the absolute index t each row predicts (== arange(1,n))."""
        ids = np.ascontiguousarray(ids, np.int64)
        n = len(ids)
        m = n - 1
        out = []
        for k in self.orders:
            logp = np.tile(self.uni, (m, 1)).astype(np.float32)   # start at unigram everywhere
            resolved = np.zeros(m, bool)
            # high→low backoff: an order j<=k context that is SEEN fills positions not yet resolved
            for j in range(k, 0, -1):
                ctx_ids, table = self.tabs[j]
                ctx = _ctx_ids(ids, j)              # length n-j, position t = j..n-1
                off = j - 1                         # logp row index for position t is t-1
                pos = np.searchsorted(ctx_ids, ctx)
                seen = (pos < len(ctx_ids)) & (ctx_ids[np.minimum(pos, len(ctx_ids) - 1)] == ctx)
                rows = np.arange(off, off + len(ctx))
                newly = seen & ~resolved[rows]
                if not newly.any():
                    continue
                tgt_rows = rows[newly]
                logp[tgt_rows] = table[pos[newly]]
                resolved[tgt_rows] = True
            out.append(logp)
        return out, np.arange(1, n)


def fresh_pool_logp(orders_logp):
    """FRESH product-of-experts each step: sum the per-order log-probs, renormalize per row → (m,27)
    log-distribution. (Sum of logs = product of experts.)"""
    s = np.zeros_like(orders_logp[0])
    for lp in orders_logp:
        s = s + lp
    s = s - s.max(axis=1, keepdims=True)
    z = np.log(np.exp(s).sum(axis=1, keepdims=True))
    return s - z


def evidence_logp(orders_logp, gamma=0.8):
    """Leaky log-evidence accumulator. step_t = sum_k logp_k(.)  (the fresh pool's UNnormalized score).
    E_t = gamma*E_{t-1} + step_t, then renormalize each row → (m,27) log-distribution.
    Vectorized leaky cumulation via a recurrence (m is modest)."""
    step = np.zeros_like(orders_logp[0])
    for lp in orders_logp:
        step = step + lp
    # center each step (subtract row-max) so the accumulator doesn't drift to -inf; affects only the
    # additive constant per row, which cancels under per-row softmax.
    step = step - step.max(axis=1, keepdims=True)
    m = step.shape[0]
    E = np.empty_like(step)
    acc = step[0].copy()
    E[0] = acc
    for t in range(1, m):
        acc = gamma * acc + step[t]
        E[t] = acc
    # scale by the effective window mass (1-gamma) so E is an AVERAGE log-evidence, not a growing sum
    # — keeps softmax(E) calibrated/comparable to the fresh pool instead of over-sharpening.
    E = E * (1.0 - gamma)
    E = E - E.max(axis=1, keepdims=True)
    z = np.log(np.exp(E).sum(axis=1, keepdims=True))
    return E - z


def decode_metrics(logdist, targets):
    """Given (m,27) log-distribution and the m true next-chars, return (accuracy, bpc)."""
    pred = logdist.argmax(axis=1)
    acc = float((pred == targets).mean())
    rows = np.arange(len(targets))
    bpc = float(-(logdist[rows, targets] / np.log(2)).mean())
    return acc, bpc


# ── boundary signals ──────────────────────────────────────────────────────────────────────────

def running_confidence(logdist, targets, gamma=0.8):
    """conf_t = gamma*conf_{t-1} + logP(observed char | context). Returns conf over the m positions."""
    rows = np.arange(len(targets))
    obs = logdist[rows, targets]            # log-prob the model gave the char that actually came
    conf = np.empty_like(obs)
    acc = obs[0]
    conf[0] = acc
    for t in range(1, len(obs)):
        acc = gamma * acc + obs[t]
        conf[t] = acc
    return conf


def forward_entropy(logdist):
    """Forward predictive entropy (bits) per position from a log-distribution."""
    p = np.exp(logdist)
    return -(p * (logdist / np.log(2))).sum(axis=1)


def drop_signal(conf):
    """Boundary score = how sharply confidence DROPS at this step: max(0, conf_{t-1} - conf_t)."""
    d = np.zeros_like(conf)
    d[1:] = np.maximum(0.0, conf[:-1] - conf[1:])
    return d


def rise_signal(H):
    """Boundary score = how sharply entropy RISES: max(0, H_t - H_{t-1}). (Exp A's winner.)"""
    r = np.zeros_like(H)
    r[1:] = np.maximum(0.0, H[1:] - H[:-1])
    return r


def f1_at_rate(score, true_idx, n, tol=1, rate=None):
    """Threshold `score` (length n, position-aligned) at the true-boundary rate (or given rate),
    score precision/recall/F1 of the predicted cut positions against true_idx within ±tol.
    Greedy one-to-one matching so duplicates within a window aren't double-counted."""
    if rate is None:
        rate = len(true_idx) / n
    k = max(1, int(round(rate * n)))
    pred = np.sort(np.argsort(score)[::-1][:k])
    return _match_f1(pred, np.asarray(sorted(true_idx)), tol)


def _match_f1(pred, true, tol):
    if len(pred) == 0 or len(true) == 0:
        return 0.0, 0.0, 0.0
    tp = 0
    used = np.zeros(len(true), bool)
    j0 = 0
    for p in pred:
        # find nearest unused true within tol
        lo = np.searchsorted(true, p - tol)
        hi = np.searchsorted(true, p + tol + 1)
        best = -1
        for j in range(lo, hi):
            if not used[j]:
                best = j
                break
        if best >= 0:
            used[best] = True
            tp += 1
    prec = tp / len(pred)
    rec = tp / len(true)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def corrupt_context(ids, frac, rng, protect_targets=True):
    """Randomize a fraction of chars in the stream (simulating noisy CONTEXT). Each chosen position
    gets a uniformly random char in [0,V). protect_targets is informational only — corruption is in
    the stream the model reads as context; we score against the ORIGINAL next-chars (the clean truth),
    so the eval target array must come from the un-corrupted ids."""
    out = ids.copy()
    n = len(out)
    nflip = int(round(frac * n))
    if nflip <= 0:
        return out
    idx = rng.choice(n, size=nflip, replace=False)
    out[idx] = rng.integers(0, V, size=nflip).astype(out.dtype)
    return out
