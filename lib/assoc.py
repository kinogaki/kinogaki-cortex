"""Streaming-ASSOCIATION slot strength — ΔP / PPMI as the construction substrate. ONLINE, NO backprop.

AF (constructions.py) ranks an open slot's filler-categories by RAW commitment ratio r = c(f,s) / c(f,·)
— how big a share of the frame's tokens go to slot-category s. That is just the conditional P(s | f), and it
is base-rate-inflated: a category that follows EVERYTHING (the ubiquitous function-word cluster) earns a high
raw share in every frame, so raw counts over-generate (the 2025 "LLMs learn constructions humans don't know"
warning — raw co-occurrence licenses slots a human never would).

M5's fix: rank, freeze, and VETO with ASSOCIATION, not raw frequency. For every (frame f, slot-category s) keep
only four additive marginals — c(f,s), c(f,·), c(·,s), N — and derive two contingency scores ON DEMAND,
closed-form:

    ΔP   = P(s | f) − P(s | ¬f)                                            (Allan / Ellis cue contingency)
         = c(f,s)/c(f,·)  −  (c(·,s) − c(f,s)) / (N − c(f,·))
    PPMI = max(0, log( c(f,s) · N / (c(f,·) · c(·,s)) ))                   (positive pointwise MI)

Both DISCOUNT a category by its global base rate c(·,s)/N: a slot-category that is frequent everywhere gets a
small ΔP / PPMI even when its raw share is large. That is exactly the over-generation lever — association sees
through the base rate that fools the raw count.

ONLINE: the four marginals are additive leaky counts — accumulate in one streaming pass, identical to a
token-at-a-time update (the vectorized builder is order-independent batched accumulation). BOUNDED: association
PRUNES — categories with ΔP ≤ 0 or PPMI = 0 (no positive contingency) drop out of a frame's slot table, so the
table SHRINKS relative to raw counts. NO gradient descent, k-means, SVD, or eigen anywhere.

    AssocSlots(C).fit(frame_cat_counts, glob_cat, N)   four marginals per (frame, category)
      .score(frame, kind="dp"|"ppmi")                  -> dense (C,) association, ΔP/PPMI, ≤0 pruned to 0
      .ranked(frame, kind)                              -> categories ordered by association (skewed-input anchor)
      .productivity(frame)                              -> Baayen P = hapax / token (is the slot still productive?)
      .preempt_assoc(...)                               -> veto over-generation by relative ASSOCIATION (vs AF's r)

Drop-in for AF's commitment-ratio substrate: same frame→category counts in, an association-weighted slot table
out. AF answers "what share?"; AW answers "what share, ABOVE what you'd expect by chance?".
"""
import numpy as np


def four_marginals(frame_cat_counts, glob_cat_counts, N):
    """Assemble the four contingency marginals per (frame, category) from AF's frame→category counts.

    frame_cat_counts : dict frame_key -> dict{cat -> c(f,s)}   (a frame's per-category filler counts)
    glob_cat_counts  : dense (C,) c(·,s)                       (global per-category filler count, base rate)
    N                : float total filler tokens (Σ glob_cat_counts)

    Returns dict frame_key -> (cats np.array, cfs np.array, cf_dot float). c(·,s)=glob, N kept on the object.
    Order-independent batched accumulation == streaming additive counts (the M5 'four additive marginals')."""
    out = {}
    for fk, cc in frame_cat_counts.items():
        if not cc:
            continue
        cats = np.fromiter(cc.keys(), dtype=np.int64, count=len(cc))
        cfs = np.fromiter((cc[c] for c in cats), dtype=np.float64, count=len(cc))
        out[fk] = (cats, cfs, float(cfs.sum()))
    return out


class AssocSlots:
    """Association-scored open-slot table. Four marginals in, ΔP / PPMI / Baayen-P out — all closed-form on
    demand, so the stored state is only the (frame, category) co-counts plus the shared global base rate."""

    def __init__(self, C, eps=1e-12):
        self.C = C
        self.eps = eps
        self.fm = {}            # frame_key -> (cats, c(f,s), c(f,·))
        self.glob = None        # (C,) c(·,s)
        self.N = 0.0            # Σ c(·,s)

    def fit(self, frame_cat_counts, glob_cat_counts, N):
        self.fm = four_marginals(frame_cat_counts, glob_cat_counts, N)
        self.glob = np.asarray(glob_cat_counts, dtype=np.float64)
        self.N = float(N)
        return self

    # ── per-(frame,category) contingency scores, closed-form from the four marginals ──

    def _entry(self, frame):
        return self.fm.get(int(frame))

    def dp(self, frame):
        """ΔP(s | f) = P(s|f) − P(s|¬f) per category, dense (C,). Positive = the frame RAISES this category's
        odds above its base rate; ≤ 0 (no positive contingency / pruned) clamped to 0. Bounded: zeros drop."""
        e = self._entry(frame)
        v = np.zeros(self.C)
        if e is None:
            return v
        cats, cfs, cfd = e
        cdots = self.glob[cats]
        p_s_given_f = cfs / max(cfd, self.eps)
        denom = max(self.N - cfd, self.eps)
        p_s_given_notf = (cdots - cfs) / denom
        dpv = p_s_given_f - p_s_given_notf
        v[cats] = np.clip(dpv, 0.0, None)               # prune non-positive contingency
        return v

    def ppmi(self, frame):
        """PPMI(f, s) = max(0, log( c(f,s)·N / (c(f,·)·c(·,s)) )) per category, dense (C,). 0 = no positive
        association (pruned). Discounts a category by its global base rate c(·,s)/N — base-rate-blind raw share."""
        e = self._entry(frame)
        v = np.zeros(self.C)
        if e is None:
            return v
        cats, cfs, cfd = e
        cdots = self.glob[cats]
        num = cfs * self.N
        den = max(cfd, self.eps) * np.clip(cdots, self.eps, None)
        pmi = np.log(np.clip(num, self.eps, None) / np.clip(den, self.eps, None))
        v[cats] = np.clip(pmi, 0.0, None)               # POSITIVE pointwise MI: prune negative
        return v

    def score(self, frame, kind="dp"):
        return self.dp(frame) if kind == "dp" else self.ppmi(frame)

    def ranked(self, frame, kind="dp", k=None):
        """Categories ordered by association — the Casenhiser-Goldberg skewed-input anchor is argmax association
        (the prototype filler-class), NOT argmax raw count."""
        v = self.score(frame, kind)
        order = np.argsort(v)[::-1]
        order = order[v[order] > 0]
        return order if k is None else order[:k]

    def slot_dist(self, frame, kind="dp"):
        """An association-WEIGHTED open-slot distribution P(category | frame): the AF category head with each
        category re-weighted by its association (so base-rate-inflated categories are damped, pruned ones gone).
        Renormalized; falls back to None if the frame has no positively-associated category."""
        e = self._entry(frame)
        if e is None:
            return None
        cats, cfs, cfd = e
        a = self.score(frame, kind)
        w = cfs * a[cats]                                # raw count × association weight (base-rate corrected)
        if w.sum() <= 0:
            return None
        v = np.zeros(self.C)
        v[cats] = w
        return v / v.sum()

    def productivity(self, frame, fids_cnt=None):
        """Baayen's productivity P = hapax / token: the share of the frame's tokens carried by once-seen fillers.
        A slot still admitting fresh hapaxes is productive (open); one dominated by entrenched repeats is not.
        Needs the raw per-filler counts (fids_cnt = (ids, counts)); returns None if not supplied."""
        if fids_cnt is None:
            return None
        _, cnt = fids_cnt
        tok = float(cnt.sum())
        if tok <= 0:
            return 0.0
        hapax = float((cnt == 1).sum())
        return hapax / tok

    # ── association preemption: veto over-generation by relative ASSOCIATION (M5's swap of AF's veto) ──

    def preempt_assoc(self, open_frames, kind="ppmi"):
        """AF's preempt() compares frames by COMMITMENT RATIO r = c(f,s)/c(f,·). AW compares them by ASSOCIATION:
        for each category the strongest-ASSOCIATED frame leads; a competitor whose association is far weaker has
        its link down-weighted toward 0 — and a competitor with ZERO association (raw share but no contingency)
        is vetoed outright. Returns inhib dict {(frame, cat) -> mult in [0,1]}. Pure counting, no gradient."""
        # per category, every open frame's association to it
        by_cat = {}
        scores = {fk: self.score(fk, kind) for fk in open_frames}
        for fk in open_frames:
            a = scores[fk]
            for c in np.nonzero(a > 0)[0]:
                by_cat.setdefault(int(c), []).append((fk, float(a[c])))
        inhib = {}
        for c, lst in by_cat.items():
            if len(lst) < 2:
                continue
            amax = max(a for _, a in lst)
            if amax <= 0:
                continue
            for fk, a in lst:
                rel = a / amax
                if rel < 1.0:
                    inhib[(fk, c)] = rel                 # scale straight by relative association
        return inhib, scores
