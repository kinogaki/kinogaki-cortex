"""retention.py — Exp AE: online NON-FORGETTING under REAL domain shift, the right-axis machinery.

The headline axis is BACKWARD RETENTION: train register A, then B, then C in ONE streaming pass (no replay),
and ask how much earlier registers degrade. Plain additive count models never forget (they only accumulate),
so the only way forgetting is even POSSIBLE is under a MEMORY CAP — a bounded table that must EVICT. The whole
experiment then turns on the EVICTION/RETENTION policy.

Both substrates here share the EXACT same prediction math (add-α highest-order-seen backoff over a per-order
context table) and count ALL orders identically, so their PEAK quality is comparable and the only thing that
differs is WHAT THEY KEEP when the bounded table overflows:

  FlatCount — single-timescale recency. Each context carries one leaky `use` score (decays globally, bumped on
    use). Evict the lowest-`use` context. A new register's contexts are all 'freshly used', so they crowd out the
    earlier register's contexts → backward forgetting.
  DualCount — the brain-inspired stack (ECAN STI/LTI, CLS fast/slow, ART vigilance, LIDA broadcast). Each context
    carries STI (short leak, salience) AND LTI (long leak, importance). Evict the LOWEST LTI — so a context that
    mattered earlier survives a flood of new contexts (protects rare-but-important). LIDA broadcast: each step
    only the SINGLE most-salient (highest-STI) context in the active backoff chain gets its LTI reinforced (the
    'this still matters' write), so a register can't cheaply inflate the LTI of all its contexts. ART vigilance ρ:
    a FAMILIAR update (the next char was already well-predicted by this context, match ≥ ρ) reinforces LTI
    strongly (consolidated knowledge, CLS slow store); a NOVEL update (match < ρ) reinforces weakly (it must earn
    retention over time). Stability–plasticity in one knob.

HARD RULES honored: ONLINE single streaming pass; NO gradient descent; NO batch optimization (no k-means/SVD).
Decay/leak are per-step recurrences; eviction is reservoir-style random-sample-and-drop (O(1) amortized, itself
online — no global sort). Nothing iterates to convergence, nothing backprops. Alphabet matches lib/fastchar:
a..z=0..25, space=26, V=27.
"""
import os, re
import numpy as np

V = 27
ALPHA = 0.1
A = "abcdefghijklmnopqrstuvwxyz "
_RNG = np.random.default_rng(0)


# ── data: char-level registers + held-out eval slice ──────────────────────────────────────────

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


def load_registers(regs, data_dir, n_train, n_eval, seed=0):
    """For each (name, filename): a contiguous train slice (streamed in order) + a DISJOINT held-out eval slice
    later in the same file. Returns dict name -> (train_ids, eval_ids)."""
    out = {}
    for name, fn in regs:
        ids = _clean(os.path.join(data_dir, fn))
        assert len(ids) >= n_train + n_eval, f"{name}: only {len(ids)} chars"
        out[name] = (np.ascontiguousarray(ids[:n_train]),
                     np.ascontiguousarray(ids[n_train:n_train + n_eval]))
    return out


def _ctx_key(ctx_ids):
    k = len(ctx_ids)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return int(ctx_ids @ powers)


def _sample_victim(d, score, n=24):
    """Reservoir-style eviction: sample ~n keys, drop the one with the lowest `score(entry)`. O(n), online — no
    global sort, no full scan. (Random-replacement caches use exactly this; it approximates lowest-score eviction
    while staying streaming-cheap.)"""
    keys = list(d.keys())
    if len(keys) <= n:
        cand = keys
    else:
        cand = [keys[i] for i in _RNG.integers(0, len(keys), size=n)]
    return min(cand, key=lambda kk: score(d[kk]))


# ── FLAT baseline: single-timescale recency eviction ──────────────────────────────────────────

class FlatCount:
    """Bounded backoff counts, ONE timescale. Entry = [counts(27), use]. Global leaky `use` (decays each step,
    +1 on touch). Evict lowest `use`. cap is PER ORDER; it bites on the high orders where register n-grams live."""

    def __init__(self, K=5, cap=6000, leak=0.9995):
        self.K = K; self.cap = cap; self.leak = leak
        self.tab = [dict() for _ in range(K + 1)]
        self.clock = 0                                       # global monotonic step across phases

    def _touch(self, order, key, tok):
        d = self.tab[order]
        e = d.get(key)
        if e is None:
            if order >= 1 and len(d) >= self.cap:
                del d[_sample_victim(d, lambda x: x[1])]
            e = [np.zeros(V, np.float64), 0.0]; d[key] = e
        e[0][tok] += 1.0
        e[1] += 1.0

    def _decay(self):
        # global recency decay applied lazily to the touched chain is unfaithful for FLAT (recency must age ALL),
        # so FLAT pays a cheap global multiplicative decay on the high orders only (where the cap bites). Done as
        # a scalar 'time' the use-scores are compared against — see train_stream.
        pass

    def train_stream(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids)
        # recency via a global discount clock: store last-touch 'time'; use-score = count * leak**(now-last).
        # Cheaper + faithful: keep entry = [counts, use, last]; effective recency = use * leak**(now-last).
        for ti in range(1, n):
            t = self.clock + ti
            tok = int(ids[ti]); kmax = min(self.K, ti)
            for k in range(kmax, 0, -1):
                key = _ctx_key(ids[ti - k:ti])
                d = self.tab[k]; e = d.get(key)
                if e is None:
                    if len(d) >= self.cap:
                        del d[_sample_victim(d, lambda x, now=t: x[1] * (self.leak ** (now - x[2])))]
                    e = [np.zeros(V, np.float64), 0.0, t]; d[key] = e
                # bring use up to date with recency decay, then bump
                e[1] = e[1] * (self.leak ** (t - e[2])) + 1.0
                e[2] = t
                e[0][tok] += 1.0
        self.clock += n - 1

    def _dist(self, ctx):
        for k in range(min(self.K, len(ctx)), 0, -1):
            e = self.tab[k].get(_ctx_key(ctx[-k:]))
            if e is not None and e[0].sum() > 0:
                c = e[0]; return (c + ALPHA) / (c.sum() + ALPHA * V)
        return np.full(V, 1.0 / V)

    def eval_bpc(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); s = 0.0
        for t in range(1, n):
            p = self._dist(ids[max(0, t - self.K):t]); s += -np.log2(p[int(ids[t])] + 1e-12)
        return s / (n - 1)

    def size(self):
        return sum(len(d) for d in self.tab[1:])


# ── DUAL model: STI/LTI two-timescale + LIDA broadcast + ART vigilance ────────────────────────

class DualCount:
    """Same counting + backoff prediction as FlatCount, so PEAK quality matches. Every entry = [counts(27), sti,
    lti, last]. The mechanisms govern only RETENTION (what survives eviction):

      STI (fast leak)  — salience; decays quickly. Bumped on every touch. Picks the LIDA broadcast winner.
      LTI (slow leak)  — importance; decays slowly. EVICTION prunes lowest LTI → protects rare-but-important.
      LIDA broadcast   — each step, among the active backoff chain, only the SINGLE highest-STI context gets its
                         LTI reinforced (the 'this still matters' write). Others' LTI is left to its slow leak.
      ART vigilance ρ  — the winner's LTI write is STRONG when the incoming char was already well-predicted by it
                         (match = P(tok|ctx) ≥ ρ: consolidated/familiar) and WEAK when novel (< ρ). Stability when
                         the world is familiar, plasticity when it's new.

    Leaks are lazy per-entry recurrences keyed on a global step clock (sti/lti *= leak**(now-last)); eviction is
    reservoir-sampled lowest-LTI. All online, no backprop."""

    def __init__(self, K=5, cap=6000, sti_leak=0.97, lti_leak=0.999995, rho=0.4,
                 lti_init=0.0, fam_gain=2.0, nov_gain=0.2):
        self.K = K; self.cap = cap
        self.sti_leak = sti_leak       # half-life ~23 steps
        self.lti_leak = lti_leak       # half-life ~140k steps (spans the whole stream)
        self.rho = rho
        self.lti_init = lti_init       # new contexts start with NO importance — must EARN retention
        self.fam_gain = fam_gain       # LTI boost when the char was familiar (consolidated knowledge)
        self.nov_gain = nov_gain       # weak boost when novel (must prove itself over time)
        self.tab = [dict() for _ in range(K + 1)]
        self.clock = 0                                       # global monotonic step across phases

    def _eff_lti(self, e, now):
        return e[2] * (self.lti_leak ** (now - e[3]))

    def train_stream(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids)
        for ti in range(1, n):
            t = self.clock + ti
            tok = int(ids[ti]); kmax = min(self.K, ti)
            chain = []   # (order, key, entry-or-None, match-before-update)
            for k in range(kmax, 0, -1):     # chain is ordered HIGH order → low order (most specific first)
                key = _ctx_key(ids[ti - k:ti])
                e = self.tab[k].get(key)
                if e is not None and e[0].sum() > 0:
                    match = e[0][tok] / e[0].sum()           # P(tok | this context) BEFORE the update
                else:
                    match = -1.0                             # unseen context recognizes nothing
                chain.append((k, key, e, match))

            # ART + LIDA: the WINNER is the MOST SPECIFIC context that RECOGNIZED the input (match ≥ ρ). That is
            # the resonant category (ART) and the single item broadcast (LIDA). If nothing recognizes it (all
            # match < ρ), the input is NOVEL → the most specific context becomes the winner and is reinforced
            # only weakly (it must earn retention). Only the winner's LTI is written → less interference.
            winner_idx = 0
            for ci, (k, key, e, match) in enumerate(chain):  # high→low: first recognizer wins
                if match >= self.rho:
                    winner_idx = ci; break

            # count ALL orders (counts are pure association — same as FLAT, so PEAK quality is identical).
            for ci, (k, key, e, match) in enumerate(chain):
                d = self.tab[k]
                if e is None:
                    if len(d) >= self.cap:
                        del d[_sample_victim(d, lambda x, now=t: self._eff_lti(x, now))]
                    e = [np.zeros(V, np.float64), 0.0, self.lti_init, t]; d[key] = e
                    chain[ci] = (k, key, e, match)
                e[1] *= self.sti_leak ** (t - e[3])
                e[2] *= self.lti_leak ** (t - e[3])
                e[3] = t
                e[0][tok] += 1.0
                e[1] += 1.0                                   # STI bumps on every touch

            # broadcast LTI write: ONLY the winner. Familiar (recognized) → strong; novel → weak.
            wk, wkey, we, wmatch = chain[winner_idx]
            we[2] += (self.fam_gain if wmatch >= self.rho else self.nov_gain)
        self.clock += n - 1

    def _dist(self, ctx):
        for k in range(min(self.K, len(ctx)), 0, -1):
            e = self.tab[k].get(_ctx_key(ctx[-k:]))
            if e is not None and e[0].sum() > 0:
                c = e[0]; return (c + ALPHA) / (c.sum() + ALPHA * V)
        return np.full(V, 1.0 / V)

    def eval_bpc(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); s = 0.0
        for t in range(1, n):
            p = self._dist(ids[max(0, t - self.K):t]); s += -np.log2(p[int(ids[t])] + 1e-12)
        return s / (n - 1)

    def size(self):
        return sum(len(d) for d in self.tab[1:])


# ── the right axis: retention matrix + forgetting summary ─────────────────────────────────────

def retention_matrix(model, registers, order_names, log=print):
    """Stream the registers in `order_names` order (one pass each, NO replay). After EACH phase, eval bpc on the
    held-out slice of EVERY register. Returns M (P×P), M[i][j] = bpc on register j after training phase i."""
    P = len(order_names)
    M = np.full((P, P), np.nan)
    for i, name in enumerate(order_names):
        model.train_stream(registers[name][0])
        for j, ev in enumerate(order_names):
            M[i, j] = model.eval_bpc(registers[ev][1])
        log(f"    after [{name:<11}] table={model.size():>7,}  " +
            " ".join(f"{ev[:4]}={M[i, j]:.3f}" for j, ev in enumerate(order_names)))
    return M


def forgetting(M, order_names):
    """Per register: peak = bpc right after training it (diagonal); final = bpc at end (last row).
    Forgetting Δ = final − peak (positive = worse = forgot). Returns [(name, peak, final, delta)]."""
    P = len(order_names)
    return [(order_names[j], M[j, j], M[P - 1, j], M[P - 1, j] - M[j, j]) for j in range(P)]
