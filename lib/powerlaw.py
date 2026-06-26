"""powerlaw.py — Exp AI: POWER-LAW memory (ACT-R base-level activation) for weighting AND budgeted eviction.

The strongest empirical result in the cognition sweep (Anderson & Schooler 1991, *Reflections of the Environment
in Memory*) is that human memory accessibility tracks the RECENCY, FREQUENCY and SPACING statistics of the real
environment — and the curve that fits all three is a POWER LAW, not an exponential. ACT-R's rational analysis
encodes this as the base-level activation of a memory chunk i:

    B_i = ln( Σ_k (t − t_k)^(−d) ),   d ≈ 0.5

  · t_k = the times the chunk was used; (t − t_k) = age of the k-th use.
  · FREQUENCY adds terms (more uses → more terms → higher B).
  · RECENCY weights them (recent uses have small age → large term).
  · DECAY d is the leak (older terms shrink as a power law, not exponentially).
  · need-odds (the prior probability a chunk is needed NOW) ∝ exp(B). So exp(B) is the natural retrieval weight.

The substrate today uses RAW COUNTS (frequency only, no recency) or an EXPONENTIAL EMA / leaky use-score (Exp AE's
FLAT/DUAL — recency that decays geometrically). The bet here: a power-law accumulator is the RIGHT recency
weighting, and it pays on two axes a memory budget makes visible — SPACING (a motif seen spaced should stay more
accessible than the same count massed; exponential recency can't represent this) and EVICTION-UNDER-A-CAP (evict
the lowest B → keep the long, sparse, repeated tail that a power law values and an EMA forgets).

── The incremental ACT-R approximation (no stored timestamps) ──────────────────────────────────────────────────
The exact B needs every t_k. We use Petrov's (2006) "hybrid"/optimized approximation, which is exact for the most
recent use and analytic for the rest, and is a per-entry O(1) recurrence — so it stays ONLINE and bounded-memory:

  Keep per entry only: n (use count), t_last (time of most recent use), and S_old = Σ_{k<last}(t − t_k)^(−d)
  carried forward via the approximation. On a new use at time t, with the previous most-recent use at t_prev:
    · the term for the previous most-recent use ages from 0 → (t − t_prev): add (t − t_prev)^(−d) into S_old
    · the older mass S_old decays: multiplied by ((t − t_prev_creation)... )  — we use the standard optimized
      decay that the bulk of (n−1) older uses, assumed spread over the lifetime, contributes
            S_old ≈ (n_old) · ( (t − t_create)^(1−d) − (t − t_last)^(1−d) ) / ((t_last − t_create)·(1−d))
      i.e. the integral approximation to the older-uses sum (Anderson et al. 1998, eq. for the "rest").
  Then B = ln( (t − t_last + 1)^(−d)  +  S_old ).   exp(B) = (t − t_last + 1)^(−d) + S_old  is the weight.

We keep it deliberately simple and FAITHFUL to the shape: the most-recent use is tracked exactly (recency), and
the older-use mass is the integral approximation (frequency × age), so a SPACED history (uses spread across the
lifetime) yields a larger older-mass integral than a MASSED history (all uses bunched early, then a long gap) —
which is exactly the spacing effect. No timestamps stored; one float of state (S_old) plus n, t_last, t_create.

HARD RULES honored: single streaming pass; no gradient descent; no batch optimization. Activation is a per-entry
O(1) recurrence on a global step clock (lazy — only recomputed when touched/queried); eviction is reservoir-style
sample-and-drop-lowest-weight (O(1) amortized, no global sort). Nothing iterates to convergence; nothing backprops.
Alphabet matches lib/fastchar / lib/retention: a..z = 0..25, space = 26, V = 27.
"""
import os, re
import numpy as np

V = 27
ALPHA = 0.1
A = "abcdefghijklmnopqrstuvwxyz "
_RNG = np.random.default_rng(0)


# ── the power-law (ACT-R) accumulator ───────────────────────────────────────────────────────────

def actr_weight(e, now, d):
    """exp(B) for entry e at time `now`, the incremental ACT-R approximation. e = [counts, n, t_create, t_last,
    s_old]. weight = recency term of the most-recent use + integral-approx of all older uses' mass.

    recent = (age_of_last_use + 1)^(−d).
    s_old (the older-uses mass) was frozen at t_last; it keeps decaying as a power law, so we re-evaluate the
    integral approximation at `now`. For n≤1 there are no older uses (s_old=0)."""
    age_last = now - e[3] + 1.0
    recent = age_last ** (-d)
    if e[1] <= 1:
        return recent
    # older-uses integral: (n-1) uses assumed spread over [t_create, t_last]; mean power-law mass at `now`.
    lo = now - e[3] + 1.0          # age of t_last
    hi = now - e[2] + 1.0          # age of t_create (oldest)
    span = max(e[3] - e[2], 1.0)   # lifetime over which the older uses were spread
    # ∫ age^(−d) over [lo,hi] / span  ×  (n−1) uses  → power-law-weighted average term, scaled by frequency.
    integ = (hi ** (1.0 - d) - lo ** (1.0 - d)) / (1.0 - d)
    s_old = (e[1] - 1.0) * integ / span
    return recent + max(s_old, 0.0)


# ── data: char registers + held-out slices (shared with Exp AE) ──────────────────────────────────

def _clean(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S)
    m2 = re.search(r"\*\*\* END OF", raw, re.S)
    if m1 and m2:
        raw = raw[m1.end():m2.start()]
    raw = raw.lower()
    raw = re.sub(r"[^a-z]+", " ", raw)
    raw = re.sub(r" +", " ", raw).strip()
    return np.array([ord(c) - 97 if c != " " else 26 for c in raw], dtype=np.int64)


def load_ids(path, n=None):
    ids = _clean(path)
    return np.ascontiguousarray(ids[:n] if n else ids)


def _ctx_key(ctx_ids):
    k = len(ctx_ids)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return int(ctx_ids @ powers)


def _sample_victim(d, score, n=24):
    """Reservoir eviction: sample ~n keys, drop the lowest score. O(n), online — no global sort."""
    keys = list(d.keys())
    if len(keys) <= n:
        cand = keys
    else:
        cand = [keys[i] for i in _RNG.integers(0, len(keys), size=n)]
    return min(cand, key=lambda kk: score(d[kk]))


# ── one bounded backoff model, four eviction policies + power-law weighting toggle ────────────────

class CountModel:
    """Bounded add-α highest-order backoff char model. Identical counting & prediction across policies, so PEAK
    quality is comparable — they differ ONLY in (a) what they EVICT when an order's table overflows, and (b)
    whether prediction is power-law WEIGHTED. Entry e = [counts(V), n, t_create, t_last, hit_count(for LFU/LRU)].

    policy:
      'powerlaw' — evict lowest ACT-R weight exp(B). The bet.
      'lru'      — evict largest age (oldest t_last).
      'lfu'      — evict smallest n (frequency).
      'ema'      — evict lowest exponential leaky use-score (the Exp AE FLAT recency baseline).
      'none'     — unbounded (cap ignored); the sanity-check upper bound.

    weighted: if True, the per-order backoff distributions are blended by exp(B) (power-law need-odds) instead of
    pure highest-order-seen — recency/spacing-aware prediction. If False, plain highest-order backoff."""

    def __init__(self, K=5, cap=3000, d=0.5, ema_leak=0.9995, policy="powerlaw", weighted=False):
        self.K = K; self.cap = cap; self.d = d; self.ema_leak = ema_leak
        self.policy = policy; self.weighted = weighted
        self.tab = [dict() for _ in range(K + 1)]
        self.clock = 0

    # eviction score per policy (higher = keep)
    def _score(self, e, now):
        p = self.policy
        if p == "powerlaw":
            return actr_weight(e, now, self.d)
        if p == "lru":
            return -(now - e[3])              # most-recent t_last kept
        if p == "lfu":
            return e[1]                       # highest count kept
        if p == "ema":
            return e[4] * (self.ema_leak ** (now - e[3]))   # leaky use-score
        return 0.0

    def _new_entry(self, t):
        # counts, n(uses), t_create, t_last, ema_use
        return [np.zeros(V, np.float64), 0.0, float(t), float(t), 0.0]

    def train_stream(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids)
        for ti in range(1, n):
            t = self.clock + ti
            tok = int(ids[ti]); kmax = min(self.K, ti)
            for k in range(kmax, 0, -1):
                key = _ctx_key(ids[ti - k:ti])
                d = self.tab[k]; e = d.get(key)
                if e is None:
                    if self.policy != "none" and len(d) >= self.cap:
                        sc = self._score
                        del d[_sample_victim(d, lambda x, now=t: sc(x, now))]
                    e = self._new_entry(t); d[key] = e
                e[0][tok] += 1.0
                e[1] += 1.0
                # ema use-score: decay-to-now then bump (the FLAT recency baseline state)
                e[4] = e[4] * (self.ema_leak ** (t - e[3])) + 1.0
                e[3] = float(t)                # t_last
        self.clock += n - 1

    def _dist(self, ctx):
        if not self.weighted:
            for k in range(min(self.K, len(ctx)), 0, -1):
                e = self.tab[k].get(_ctx_key(ctx[-k:]))
                if e is not None and e[0].sum() > 0:
                    c = e[0]; return (c + ALPHA) / (c.sum() + ALPHA * V)
            return np.full(V, 1.0 / V)
        # power-law-weighted blend over all matched orders: weight order k by exp(B) of its context entry.
        now = self.clock + 1
        num = np.full(V, ALPHA); den = ALPHA * V
        any_hit = False
        for k in range(min(self.K, len(ctx)), 0, -1):
            e = self.tab[k].get(_ctx_key(ctx[-k:]))
            if e is not None and e[0].sum() > 0:
                w = actr_weight(e, now, self.d) * (k * k)   # specificity × need-odds
                c = e[0]
                num = num + w * (c / c.sum()); den += w
                any_hit = True
        if not any_hit:
            return np.full(V, 1.0 / V)
        return num / den

    def eval_bpc(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); s = 0.0
        for t in range(1, n):
            p = self._dist(ids[max(0, t - self.K):t]); s += -np.log2(p[int(ids[t])] + 1e-12)
        return s / (n - 1)

    def size(self):
        return sum(len(d) for d in self.tab[1:])


# ── spacing-effect probe: one motif, spaced vs massed, same total count ──────────────────────────

def spacing_probe(d=0.5, ema_leak=0.9995, count=20, stream_len=20000, motif_len=6, seed=0):
    """Inject ONE rare motif (a fixed high-order context→token) `count` times into a noise stream, two ways:
       MASSED — all `count` uses bunched near the start, then a long gap to the end.
       SPACED — the same `count` uses spread evenly across the whole stream.
    Same total frequency, same final position budget. At the END measure each scheme's retrieval weight for the
    motif under POWER-LAW (exp B) vs EXPONENTIAL-EMA. Anderson-Schooler: spaced should be MORE accessible.
    Returns dict of final weights. (Pure accumulator probe — no eviction, isolates the recency curve.)"""
    rng = np.random.default_rng(seed)

    def run_weights(times):
        # times = sorted list of use-steps (1..stream_len). Build the ACT-R entry incrementally + an EMA.
        e = [np.zeros(V), 0.0, float(times[0]), float(times[0]), 0.0]
        for t in times:
            e[1] += 1.0
            e[4] = e[4] * (ema_leak ** (t - e[3])) + 1.0
            e[3] = float(t)
        now = stream_len
        return actr_weight(e, now, d), e[4] * (ema_leak ** (now - e[3]))

    # massed: first `count` steps. spaced: evenly spread.
    massed = list(range(1, count + 1))
    spaced = list(np.linspace(1, stream_len, count).astype(int))
    pl_m, ema_m = run_weights(massed)
    pl_s, ema_s = run_weights(spaced)
    return {
        "powerlaw_massed": pl_m, "powerlaw_spaced": pl_s,
        "ema_massed": ema_m, "ema_spaced": ema_s,
        "powerlaw_ratio": pl_s / pl_m, "ema_ratio": ema_s / ema_m,
    }


# ── budgeted eviction UNDER DOMAIN SHIFT: the regime where frequency-alone (LFU) should lose ──────

def shift_eviction(train_a, train_b, eval_a, K=5, cap=1500, d=0.5, ema_leak=0.9995, policies=None):
    """Stream register A then register B in ONE pass under a per-order cap, then measure held-out bpc on A.
    B floods the table and forces eviction. LFU keeps whatever A-context happened to be most FREQUENT even if A
    is now stale; LRU dumps all of A (recency only); the power law keeps A-contexts that were used REPEATEDLY and
    not-too-long-ago (frequency × recency × spacing) — the ones that actually still predict A. Returns {policy:
    bpc_on_A_after_B}. This is the bounded-memory, domain-shift regime Exp AE showed is where retention bites."""
    policies = policies or ["powerlaw", "lru", "lfu", "ema"]
    out = {}
    for pol in policies:
        m = CountModel(K=K, cap=cap, d=d, ema_leak=ema_leak, policy=pol, weighted=False)
        m.train_stream(train_a)
        m.train_stream(train_b)            # the flood — forces eviction of A's contexts
        out[pol] = m.eval_bpc(eval_a)      # how much of A survived
    return out
