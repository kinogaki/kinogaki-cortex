"""eventmodel.py — Exp AC: a discourse-coherence EVENT MODEL by Bayesian surprise. ONLINE, NO backprop.

The thesis (Zacks event-segmentation; Kumar 2023 Bayesian surprise; Franklin SEM sticky-CRP; Zwaan
situation models). The reader carries a persistent "event slot" = a leaky profile over the currently-active
phrase/topic clusters. That slot is a SOFT TOP-DOWN PRIOR on the next token. At every step we measure how much
seeing the actual token MOVED our belief about what comes next — the Bayesian surprise

    S_t = KL( P_t || P_{t-1} )

over the top-k next-token distribution, where P_{t-1} is the prediction made BEFORE the token and P_t the
distribution after conditioning on it (the one-step belief update). A leaky running mean/var normalizes S to a
z-score. When z crosses theta we declare an EVENT BOUNDARY: ARCHIVE the current slot into a non-forgetting
long-term store and SPAWN/SELECT a slot from a sticky-CRP-style bank — prefer the current slot (a stickiness
pseudocount), open a NEW slot only when no existing slot beats a new-slot pseudocount.

HARD RULE — everything here is ONLINE: a single streaming pass of leaky accumulators + online leader
clustering. No gradient descent, no k-means, no SVD. The slot bank is a leader clusterer with a stickiness
prior (sticky-CRP); slot profiles are leaky counters; the surprise normalizer is a leaky mean/var (Welford-ish
EMA). The per-token predictor it sits on is plain n-gram counts (a Column), reused unchanged.

  KLSurprise        — leaky running mean/var of the per-step KL; z-score + threshold crossing.
  SlotBank          — sticky-CRP slot bank: leaky cluster-profile slots, stickiness vs new-slot pseudocount,
                      non-forgetting archive of evicted slots.
  EventModel        — drives the two together over a token stream given a next-token predictor; returns the
                      per-step surprise/z, the committed slot id per step, and the boundary positions.
  topk_dist         — fast top-k renormalized next-token distribution from a sparse count expert.
"""
import numpy as np

EPS = 1e-9


# ───────────────────────────── Bayesian surprise (KL) + leaky z ─────────────────────────────

def kl_topk(p_prev, p_cur):
    """KL(P_cur || P_prev) over the UNION of the two top-k supports (Kumar 2023's Bayesian surprise =
    how much the posterior moved). Both are dicts {token: prob} already restricted to their top-k and
    renormalized. The union is small (≤2k); missing mass gets a tiny floor so the KL is finite."""
    keys = set(p_cur) | set(p_prev)
    if not keys:
        return 0.0
    # floor + renormalize each side over the shared support so both are proper distributions there
    qc = np.array([p_cur.get(k, 0.0) for k in keys]) + EPS
    qp = np.array([p_prev.get(k, 0.0) for k in keys]) + EPS
    qc /= qc.sum(); qp /= qp.sum()
    return float((qc * np.log2(qc / qp)).sum())


class KLSurprise:
    """Leaky running mean/variance of a scalar stream (the per-step KL), giving an online z-score and a
    threshold crossing. EMA mean/var (a leaky accumulator — order-dependent, single-pass, the canonical
    online normalizer); refractory period suppresses a burst of crossings inside one real boundary."""

    def __init__(self, halflife=4000.0, theta=3.0, refractory=200):
        self.decay = 0.5 ** (1.0 / halflife)
        self.theta = theta
        self.refractory = refractory
        self.mean = 0.0
        self.var = 1.0
        self.n = 0
        self._cooldown = 0

    def step(self, s):
        """Update with one KL value; return (z, fired). z uses the running mean/var BEFORE this sample is
        folded in (so a spike is measured against the recent baseline, not against itself)."""
        std = max(np.sqrt(self.var), EPS)
        z = (s - self.mean) / std
        fired = False
        if self._cooldown > 0:
            self._cooldown -= 1
        elif self.n > 50 and z > self.theta:
            fired = True
            self._cooldown = self.refractory
        # leaky update (EMA mean + EMA of squared deviation)
        d = s - self.mean
        self.mean += (1 - self.decay) * d
        self.var = self.decay * self.var + (1 - self.decay) * d * d
        self.n += 1
        return z, fired


# ───────────────────────────── sticky-CRP slot bank ─────────────────────────────

class SlotBank:
    """A bank of EVENT SLOTS. Each slot = a leaky counter profile over phrase/topic CLUSTERS (a normalized
    histogram of which clusters were active while the slot was live). Selection is sticky-CRP (Franklin SEM):

      - the CURRENT slot gets a stickiness pseudocount `stick` added to its match score (prefer to stay);
      - a brand-new slot is scored at a fixed `new_prior` pseudocount;
      - pick argmax over {existing slots' cosine-to-profile, current+stick, new_prior}; open a new slot only
        if the new-slot prior wins (capped at Cmax, after which we force the best existing).

    Slots are NEVER deleted: an evicted slot's profile is frozen into `archive` (non-forgetting long-term
    store), so a returning topic re-selects its old slot by similarity. Online: profiles are leaky counters,
    selection is a single nearest-prototype decision per boundary (leader clustering with a stickiness prior)."""

    def __init__(self, n_clusters, stick=1.5, new_prior=0.35, profile_halflife=300.0, Cmax=256):
        self.C = n_clusters
        self.stick = stick
        self.new_prior = new_prior
        self.pdecay = 0.5 ** (1.0 / profile_halflife)
        self.Cmax = Cmax
        self.profiles = np.zeros((1, n_clusters), np.float64)   # slot 0 starts empty
        self.nslots = 1
        self.current = 0
        self.live = np.zeros(n_clusters, np.float64)            # the live profile of the current slot
        self.archive = []                                       # frozen (slot_id, profile) — non-forgetting
        self.n_spawned = 0
        self.n_reselected = 0

    def observe(self, cluster):
        """Fold one active cluster id into the live profile of the current slot (leaky). cluster<0 ignored."""
        self.live *= self.pdecay
        if cluster >= 0:
            self.live[cluster] += 1.0

    def _unit(self, v):
        n = np.linalg.norm(v)
        return v / n if n > EPS else v

    def boundary(self):
        """An event boundary fired. ARCHIVE the live profile into the current slot + long-term store, then
        SELECT the next slot by sticky-CRP. Returns the newly committed slot id."""
        # 1) archive: commit the live profile into the current slot and freeze a copy
        self.profiles[self.current] = self.live.copy()
        self.archive.append((self.current, self.live.copy()))
        u_live = self._unit(self.live)

        # 2) score every existing slot by cosine to the just-closed live profile (what topic are we in now?)
        P = self.profiles[:self.nslots]
        norms = np.linalg.norm(P, axis=1)
        cos = (P @ u_live) / np.maximum(norms, EPS)             # similarity of each slot to the current topic
        score = cos.copy()
        score[self.current] += self.stick                      # stickiness: prefer to continue the current slot

        best = int(np.argmax(score)); best_score = float(score[best])

        # 3) sticky-CRP decision: open a NEW slot only if the new-slot pseudocount beats every existing match
        if best_score < self.new_prior and self.nslots < self.Cmax:
            self.profiles = np.vstack([self.profiles, np.zeros((1, self.C))])
            self.current = self.nslots
            self.nslots += 1
            self.n_spawned += 1
        else:
            self.current = best
            self.n_reselected += 1

        # 4) seed the new live profile from the selected slot's archived profile (a returning topic resumes)
        self.live = self.profiles[self.current].copy()
        return self.current

    def prior(self):
        """The current slot's normalized cluster profile = the soft TOP-DOWN PRIOR over clusters."""
        s = self.live.sum()
        return self.live / s if s > EPS else None


# ───────────────────────────── top-k next-token distribution ─────────────────────────────

_TOPK_MEMO = {}     # id(counts dict) -> top-k dist. The count dicts are frozen after fit (never mutated).

def topk_dist(counts, k, alpha, V):
    """counts: dict {token: count} (a sparse Column expert). Return the add-alpha smoothed next-token
    distribution restricted to its TOP-k tokens, renormalized — the P over which Bayesian surprise is taken.
    If counts is empty/None, return None (signals 'no local evidence'). Memoized by dict identity: each
    distinct context dict is sorted at most once, turning the streaming pass's millions of calls into one
    sort per seen context (the offsetattn trick — the dicts are immutable post-fit)."""
    if not counts:
        return None
    cached = _TOPK_MEMO.get(id(counts))
    if cached is not None:
        return cached
    tot = sum(counts.values()) + alpha * V
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
    d = {t: (c + alpha) / tot for t, c in items}
    z = sum(d.values())
    out = {t: p / z for t, p in d.items()}
    _TOPK_MEMO[id(counts)] = out
    return out


def blend(p, prior_over_tokens, w):
    """Soft top-down prior: mix the local next-token dist p with a token-level prior (from the event slot),
    weight w in [0,1]. Both dicts; union support; renormalized. prior None -> p unchanged."""
    if prior_over_tokens is None or p is None:
        return p
    keys = set(p) | set(prior_over_tokens)
    out = {kk: (1 - w) * p.get(kk, 0.0) + w * prior_over_tokens.get(kk, 0.0) for kk in keys}
    z = sum(out.values())
    return {kk: v / z for kk, v in out.items()} if z > 0 else p
