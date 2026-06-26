"""takethebest.py — Exp AJ: validity-ordered, noncompensatory, early-stopping inference.

The bet (Gigerenzer & Goldstein's *take-the-best*; the *recognition heuristic*; Simon's satisficing;
the bias–variance theorem). Our usual combiner pools EVERY cue — all D offset-experts, the soft
accumulated channel — with a geometric mean, paying full compute to weigh evidence that mostly does not
matter. Fast-and-frugal heuristics say the opposite: rank cues by their measured **validity**, consult
them ONE AT A TIME in descending validity, and **STOP at the first cue that discriminates** (clears a
margin). Lower-validity cues never get to override a higher one — the rule is *noncompensatory*. A count
model is high-bias / low-variance by construction (the bias–variance decomposition: total error =
bias² + variance + noise); when data is sparse and single-pass, that is the RIGHT place to be, and a
frugal rule that ignores weak channels trades a little bias for a lot less variance — the *less-is-more*
effect.

Cues, and how validity is read off counts (NO gradient, NO fit):

  - **Recognition cue** `count>0`: have we ever seen this context word at this offset? Bare recognition.
  - **Per-offset experts** (from `offsetattn`): each offset `d` is a cue. Its predicted token is its
    running argmax; we score that bet online against what actually came.
  - **Soft / accumulated cue**: the full geometric-mean pool over all offsets — the "integrate
    everything" channel, here demoted to ONE cue among many.

  validity v_j = hits / (hits + misses) for cue j   — Goldstein–Gigerenzer ecological validity, counted.

Take-the-best inference (`take_the_best`): walk cues high-validity-first; the first cue that fires AND
clears the **aspiration margin** decides; if none clears, fall back to the soft pool (or uniform). The
aspiration is **satisficing** (Simon): a leaky level that RISES when answers clear easily and FALLS when
they are scarce — the bar sets itself from the stream, no tuned threshold.

Less-is-more override (α>β): a soft/accumulated cue may OVERRIDE a crisp count/recognition cue ONLY when
its measured validity EXCEEDS the count cue's. If the weak channel's track record is worse, the rule
ignores it — and ignoring it is what *improves* accuracy on sparse/noisy contexts (the α>β prediction).

Base-rate guard on clustering (`guarded_assign`): leader-clustering by pure cosine similarity is the
representativeness heuristic with NO base rate — a rare prototype wins on a chance-high similarity. The
guard scores assignment = argmax( similarity × clusterCount^γ ) — a count prior folded in
(representativeness → approximate Bayes). γ is the single knob; γ=0 is the current pure-similarity rule.

Everything is a single online pass over counts. Validity is the online hit/miss of each cue's running
bet; the aspiration is a leaky accumulator; nothing iterates to convergence and nothing backprops.
"""
import math
import numpy as np

ALPHA = 0.05            # matches offsetattn / cortex — geometric-mean smoothing floor


# ── per-cue online validity ──────────────────────────────────────────────────────────────────────

class CueValidity:
    """Online hit/miss tally per cue → ecological validity v = hits/(hits+misses).

    A "cue" is anything that, given a context, emits a single discrete BET (its running top-1 token) and
    can ABSTAIN. We never look ahead: at each eval-ordered step a cue's bet is fixed by the counts it has
    accumulated from earlier positions, we score it against the token that actually came, then fold that
    token in. The recognition cue's bet = "the most-frequent token I've ever seen follow this word"; an
    offset cue's bet = the argmax of its offset table for the context word; the soft cue's bet = the
    argmax of the geometric-mean pool. With no evidence v defaults to 0.5 (max ignorance)."""

    def __init__(self, names):
        self.names = list(names)
        self.hits = {n: 0 for n in self.names}
        self.miss = {n: 0 for n in self.names}

    def score(self, name, bet, truth):
        """Record one online bet for cue `name`. bet=None means the cue abstained (not scored)."""
        if bet is None:
            return
        if bet == truth:
            self.hits[name] += 1
        else:
            self.miss[name] += 1

    def validity(self, name):
        h, m = self.hits[name], self.miss[name]
        return h / (h + m) if (h + m) else 0.5

    def table(self):
        """Sorted (name, validity, n) descending by validity — the cue ORDER take-the-best scans."""
        rows = [(n, self.validity(n), self.hits[n] + self.miss[n]) for n in self.names]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows


# ── satisficing aspiration (Simon): a leaky bar that rises/falls with how easily answers clear ─────

class Aspiration:
    """Simon's adaptive aspiration level as a leaky accumulator. The "discrimination" a cue achieves is
    its TOP-1 PROBABILITY — how confidently this one cue points at a single answer (the recognition-
    heuristic signal: a cue that fires hard is trusted, a hedged cue is not). We track a leaky mean of the
    discrimination of whatever cue ENDS UP deciding, and set the aspiration to that mean. When recent
    decisions are confident (high top-1) the bar RISES (be pickier — keep scanning for a sharp cue); when
    confident cues are scarce the bar FALLS (be satisfied sooner). No tuned threshold — the operating
    point is set by the stream. `tighten<1` makes the bar a fraction of the running mean, so a cue that
    is about as decisive as the recent average already clears (≈half stop at the first cue by design).

    Online by construction: the bar at step t reflects only discriminations observed before t (we read
    `.level`, then `.observe`)."""

    def __init__(self, init=0.3, leak=0.99, tighten=0.9):
        self.mean = init
        self.leak = leak
        self.tighten = tighten

    @property
    def level(self):
        return self.tighten * self.mean

    def observe(self, disc):
        self.mean = self.leak * self.mean + (1 - self.leak) * disc


# ── the cue family over offset tables ──────────────────────────────────────────────────────────────

def _argmax_dict(d):
    """Top-1 key of a {token:count} dict, or None if empty (canonical lowest-id tie-break)."""
    if not d:
        return None
    best_k, best_c = None, -1
    for k, c in d.items():
        if c > best_c or (c == best_c and (best_k is None or k < best_k)):
            best_k, best_c = k, c
    return best_k


def _dist_margin(dist):
    """(top_token, top_prob - second_prob) of a {token:prob} dist; margin in [0,1]. Empty → (None,0).
    Single linear scan for the two largest — no sort."""
    if not dist:
        return None, 0.0
    top_k = None; top_p = -1.0; second_p = 0.0
    for k, p in dist.items():
        if p > top_p:
            second_p = top_p if top_p > 0 else second_p
            top_p, top_k = p, k
        elif p > second_p:
            second_p = p
    return top_k, top_p - second_p


_TOPK_MEMO = {}         # id(count_dict) -> cached top-K key list (tables are immutable post-fit)


def _topk_keys(d, topk):
    """Top-`topk` keys of a {token:count} dict by count, memoized on the dict's identity. Low-count
    tokens never top the geometric-mean pool (a single tiny count is overwhelmed by the ALPHA floor of
    the others), so capping the candidate union is a near-lossless sparse approximation — same trick as
    offsetattn._weighted_pool, and the one thing that makes the per-position loop fast."""
    if len(d) <= topk:
        return d.keys()
    cached = _TOPK_MEMO.get(id(d))
    if cached is None:
        cached = [k for k, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:topk]]
        _TOPK_MEMO[id(d)] = cached
    return cached


_NORM_MEMO = {}         # id(count_dict) -> cached {token:prob} top-K normalized dist (tables immutable)


def _normalize_topk(dd, topk=48):
    """Count-normalized {token:prob} over a dict's top-`topk` tokens, memoized on dict identity.
    Returns None for an empty/missing dict. Probabilities sum over the kept tokens (the dropped tail is
    negligible mass and never the argmax)."""
    if not dd:
        return None
    cached = _NORM_MEMO.get(id(dd))
    if cached is not None:
        return cached
    if len(dd) > topk:
        items = sorted(dd.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    else:
        items = list(dd.items())
    tot = sum(c for _, c in items)
    out = {k: c / tot for k, c in items}
    _NORM_MEMO[id(dd)] = out
    return out


def _pool(experts, topk=48):
    """Unweighted log-linear (geometric-mean) pool over (count_dict) experts → {token:prob} or None.
    Candidates = union of each expert's TOP-`topk` tokens (sparse approximation). The 'soft/accumulated'
    cue and the full-integration baseline both use this."""
    if not experts:
        return None
    keys = set()
    for d in experts:
        keys.update(_topk_keys(d, topk))
    out = {}
    for k in keys:
        lp = sum(math.log(d.get(k, 0) + ALPHA) for d in experts)
        out[k] = lp
    m = max(out.values())
    z = sum(math.exp(v - m) for v in out.values())
    return {k: math.exp(v - m) / z for k, v in out.items()}


class TakeTheBest:
    """A fast-and-frugal predictor over the offset count tables from `offsetattn.OffsetAttn`.

    Cues (all read from the SAME tables — this is purely a different COMBINER, not new evidence):
      * "recog"  — the recognition cue: the bag-merged table (any offset). Bet = its argmax.
      * "off{d}" — offset d's table. Bet = argmax of that offset's row for the context word.
      * "soft"   — the geometric-mean pool over all firing offsets (full integration as one cue).

    `prime(stream, eval_start)` runs the ONLINE validity + aspiration pass on the train region so the
    cue ranking and bar are set before eval (still single-pass, causal). `predict(ctx)` returns
    (dist, n_cues_consulted): take-the-best scans cues high-validity-first and stops at the first that
    fires and clears the bar; n_cues_consulted is the compute actually spent."""

    def __init__(self, oa, less_is_more=True):
        self.oa = oa                       # a fitted OffsetAttn (has .tab[d], .bag, .D)
        self.D = oa.D
        self.less_is_more = less_is_more
        self.offset_names = [f"off{d}" for d in range(1, self.D + 1)]
        self.names = ["recog"] + self.offset_names + ["soft"]
        self.cv = CueValidity(self.names)
        self.asp = Aspiration()
        self.order = None                  # cue names in descending validity (set by prime)
        self.v = {}                        # cached validity per cue

    # -- per-cue bet / distribution from the (already counted) tables --

    def _recog_dist(self, ctx):
        """Recognition cue: bag table for the most recent context word. {token:prob} (count-normalized,
        top-K capped — the tail never tops the argmax or moves the top-1 margin)."""
        dd = self.oa.bag.get(int(ctx[-1]))
        return _normalize_topk(dd)

    def _offset_dist(self, ctx, d):
        return _normalize_topk(self.oa.tab[d].get(int(ctx[-d])))

    def _soft_dist(self, ctx):
        experts = []
        for d in range(1, self.D + 1):
            dd = self.oa.tab[d].get(int(ctx[-d]))
            if dd:
                experts.append(dd)
        return _pool(experts)

    def _cue_dist(self, name, ctx):
        if name == "recog":
            return self._recog_dist(ctx)
        if name == "soft":
            return self._soft_dist(ctx)
        return self._offset_dist(ctx, int(name[3:]))

    # -- online priming pass: validity + aspiration, causal over the train region --

    def prime(self, stream, eval_start, window=None):
        """Single causal pass over the train region: score every cue's running bet (for VALIDITY) and
        fold the deciding cue's discrimination into the satisficing aspiration. Bets come from the FINAL
        tables (the count a context has is fixed by fit); the online discipline is that we score each cue
        against the next token and never use eval data — `window` just bounds the pass to the last
        `window` train positions (ecological validity is a sample; the cue RANKING stabilizes early).

        Validity is read off EVERY cue at every position (each cue's running argmax vs the next token).
        The aspiration only watches the cue that would have DECIDED — to learn the bar at the operating
        point. We do two-stage priming: a short bootstrap to fix the cue order, then the rest to set the
        bar — but a single pass suffices since validity and the bar are independent running stats."""
        start = self.D if window is None else max(self.D, eval_start - window)
        crisp = ["recog"] + self.offset_names      # the noncompensatory scan set (soft is fallback only)
        for t in range(start, eval_start):
            ctx = stream[t - self.D:t]
            truth = int(stream[t])
            self.cv.score("recog", _argmax_dict(self.oa.bag.get(int(ctx[-1]))), truth)
            for d in range(1, self.D + 1):
                self.cv.score(f"off{d}", _argmax_dict(self.oa.tab[d].get(int(ctx[-d]))), truth)
            self.cv.score("soft", _dist_margin(self._soft_dist(ctx))[0], truth)
            # feed the bar the discrimination (top-1 prob) of the FIRST crisp cue that fires, in the
            # CURRENT ranking — this is what the take-the-best scan would actually act on
            order_crisp = sorted(crisp, key=lambda nm: self.cv.validity(nm), reverse=True)
            for nm in order_crisp:
                d = self._cue_dist(nm, ctx)
                if d:
                    self.asp.observe(max(d.values()))
                    break
        self.v = {n: self.cv.validity(n) for n in self.names}
        # the scan order take-the-best uses = crisp cues by descending validity (soft is the fallback)
        self.crisp_order = sorted(crisp, key=lambda nm: self.v[nm], reverse=True)
        self.order = sorted(self.names, key=lambda nm: self.v[nm], reverse=True)   # for display
        return self

    # -- take-the-best inference: validity-ordered, noncompensatory, early-stopping --

    def predict(self, ctx, aspiration=None):
        """Scan CRISP cues in descending validity; STOP at the first whose TOP-1 probability clears the
        satisficing bar (noncompensatory — a later, lower-validity cue never overrides an earlier one
        that already cleared). Returns ({token:prob} or None, n_cues_consulted).

        If no crisp cue clears, satisfice with the soft/accumulated pool — UNLESS less-is-more applies.
        less-is-more (α>β): the soft channel may take over from the best crisp cue that fired ONLY when
        its measured validity exceeds that crisp cue's (v[soft] > v[best_crisp]). When the soft channel
        is the weaker one, it is IGNORED and we keep the crisp cue's answer — ignoring the weak channel
        is the whole point, and the prediction is that on sparse/noisy contexts this is the better call."""
        bar = self.asp.level if aspiration is None else aspiration
        n = 0
        best_crisp = None        # (dist, validity) of the highest-validity crisp cue that fired
        for name in self.crisp_order:
            n += 1
            dist = self._cue_dist(name, ctx)
            if not dist:
                continue
            if best_crisp is None:
                best_crisp = (dist, self.v[name])    # crisp cues scanned high-validity-first → this is best
            if max(dist.values()) >= bar:            # this cue discriminates → STOP (noncompensatory)
                return dist, n
        # no crisp cue cleared the bar — fall back to the soft/accumulated pool
        soft = self._soft_dist(ctx); n += 1
        if soft is None:
            return (best_crisp[0] if best_crisp else None), n
        if self.less_is_more and best_crisp is not None and self.v["soft"] <= best_crisp[1]:
            return best_crisp[0], n - 1              # IGNORE the weak channel (didn't actually use soft)
        return soft, n


def full_integration(oa, ctx):
    """The BASELINE combiner: the full geometric-mean pool over ALL D offsets (every cue, every step).
    Returns ({token:prob} or None, n_cues_consulted) — n is the number of offsets that fired (the
    compute it always spends). This is Exp S's offset-attention without the IG weighting, so the
    take-the-best comparison isolates the COMBINER (frugal vs integrate-everything), not the evidence."""
    experts = []
    n = 0
    for d in range(1, oa.D + 1):
        dd = oa.tab[d].get(int(ctx[-d]))
        n += 1                              # full integration always consults every offset
        if dd:
            experts.append(dd)
    return _pool(experts), n


# ── base-rate-guarded leader clustering (representativeness → approximate Bayes) ───────────────────

def guarded_leader_cluster(sig, cnt, order, gamma=0.0, min_evidence=40, thresh=0.55, Cmax=400):
    """Online leader-clustering (a copy of jepa.leader_cluster's single-pass shape) with a BASE-RATE
    GUARD: assignment scores cosine_similarity × clusterCount^γ, not pure cosine. The clusterCount is the
    running member count of each prototype — a count prior. γ=0 reproduces jepa.leader_cluster exactly
    (pure representativeness); γ>0 tilts toward larger, better-attested clusters (approximate Bayes:
    posterior ∝ likelihood × prior). The SPAWN test still uses raw cosine vs `thresh` (the guard only
    breaks ties between EXISTING prototypes; a genuinely novel word still earns its own cluster).

    Single pass, no re-assignment. Returns clu:(N,) in [0,C) or -1, and C."""
    N, Dd = sig.shape
    unit = sig / np.maximum(np.linalg.norm(sig, axis=1, keepdims=True), 1e-9)
    proto = np.zeros((Cmax, Dd), np.float64)         # running SUM of member unit-sigs
    pcount = np.zeros(Cmax, np.float64)              # running member COUNT per prototype (the base rate)
    C = 0
    clu = -np.ones(N, np.int64)
    for w in order:
        if cnt[w] < min_evidence:
            continue
        u = unit[w]
        if C == 0:
            proto[0] = u; pcount[0] = 1.0; C = 1; clu[w] = 0
            continue
        dots = proto[:C] @ u
        cos = dots / np.maximum(np.linalg.norm(proto[:C], axis=1), 1e-9)
        prior = pcount[:C] ** gamma if gamma else np.ones(C)
        score = cos * prior                          # similarity × clusterCount^γ
        best = int(score.argmax()); bcos = float(cos[best])
        if bcos >= thresh or C >= Cmax:              # spawn test stays on RAW cosine
            proto[best] += u; pcount[best] += 1.0; clu[w] = best
        else:
            proto[C] = u; pcount[C] = 1.0; clu[w] = C; C += 1
    used = np.unique(clu[clu >= 0])
    relabel = -np.ones(Cmax, np.int64); relabel[used] = np.arange(len(used))
    clu = np.where(clu >= 0, relabel[clu], -1)
    return clu, len(used)


def cluster_stability(clu_a, clu_b):
    """Agreement between two clusterings of the same words (only words clustered in BOTH): the fraction
    of word-PAIRS that are same-cluster-in-both or different-in-both (Rand-index style). A higher value
    = a more stable assignment under perturbation."""
    both = (clu_a >= 0) & (clu_b >= 0)
    a = clu_a[both]; b = clu_b[both]
    n = len(a)
    if n < 2:
        return 1.0
    # sample pairs to keep it cheap at N~thousands
    rng = np.random.default_rng(0)
    m = min(200_000, n * (n - 1) // 2)
    i = rng.integers(0, n, m); j = rng.integers(0, n, m)
    ok = i != j
    i, j = i[ok], j[ok]
    same_a = a[i] == a[j]
    same_b = b[i] == b[j]
    return float((same_a == same_b).mean())
