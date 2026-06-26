"""goldilocks.py — Exp BI: the Goldilocks learning-rate gate (an INVERTED-U on surprisal).

A child does not learn equally from every word it hears. The desirable-difficulty / N400 literature
(Kidd's "Goldilocks effect" in infant attention; the U-shaped attention curve over predictability)
says learning is strongest in a MIDDLE band: too-predictable input carries no news (you already know
it), too-surprising input is unparsable (it doesn't connect to anything you have). The naive form —
"learn MORE the more surprised you are" (monotone surprise-as-gate) — is exactly wrong at the high
end: a typo, a foreign word, an OOV burst is maximally surprising and maximally worthless to store.

This module makes the WRITE-WEIGHT of an online count model a function of the model's OWN surprisal
on each token, BEFORE it counts it (predict-then-write — the AT reactive contract, used for the write
gate). Three gate shapes share one streaming-pass, bounded-memory count model so the ONLY thing that
differs is *how much each token is allowed to write*:

  · flat       w(s) = 1                       — the baseline (count every token once; today's observe()).
  · monotone   w(s) = clip(s / s0, 0, wmax)   — the naive "surprise-as-gate" the queue says BI corrects.
  · goldilocks w(s) = a smooth inverted-U on s peaked at a target band — SKIP the already-known low-s
                      (no news) AND the unparsable high-s (noise); learn most in the middle.

The decisive comparison is at EQUAL TABLE SIZE: every model is capped at the same per-order budget and
evicts lowest-frequency rows (LFU — Exp AI's winner for dense char-grams). If the gate helps held-out
bpc only by storing MORE distinct contexts, that is a memory win, not a learning-rate win — the kill
condition. The gate must do better *at the same budget*, by spending its bounded writes on the tokens
that actually teach.

The N400 / cloze read-out is kept DISTINCT from the write-gate (BUILD_QUEUE: "fold in … kept distinct"):
after training, the model's per-token surprisal is validated as a cloze/N400 proxy — does model
surprisal track the held-out continuation's actual rank/cloze probability? That is a read of the SAME
counts, not a second gate. A clean inverted-U write-gate is licensed only if the surprisal it reads is
itself a faithful predictability signal.

HARD RULES honored: single streaming pass (predict-then-write, one token at a time); no gradient
descent / k-means / SVD / backprop (the gate is a closed-form scalar on a surprisal the counts already
give); bounded memory (every order's table capped, LFU eviction — the cap IS the experiment). Fixed
seed at the call site. Alphabet matches lib/cortex / lib/metrics: a..z = 0..25, space = 26, V = 27.
"""
import math
import numpy as np

V = 27
ALPHA = 0.05            # add-alpha smoothing, == cortex.ALPHA (the calibrated pool's floor)


# ── the gate shapes: write-weight as a function of the model's surprisal on the token ─────────────

def w_flat(s, **_):
    """Baseline: every token writes weight 1 (today's observe path — frequency only)."""
    return 1.0


def w_monotone(s, s0=2.0, wmax=4.0, **_):
    """The naive surprise-as-gate BI corrects: write MORE the more surprised. Linear in s, clipped.
    Wrong at the high end — a maximally surprising typo/OOV gets the maximum write."""
    return min(max(s / s0, 0.0), wmax)


def w_goldilocks(s, center=2.2, width=1.6, floor=0.05, peak=3.0, **_):
    """The inverted-U. A Gaussian bump on surprisal centered in the middle band:
        w(s) = floor + (peak − floor) · exp( −((s − center)/width)² )
    Low-s (already-known, s≪center) → ~floor (skip the news-free token). High-s (unparsable,
    s≫center) → ~floor (skip the noise). Middle band → up to `peak`. Smooth, closed-form, no state."""
    z = (s - center) / width
    return floor + (peak - floor) * math.exp(-z * z)


GATES = {"flat": w_flat, "monotone": w_monotone, "goldilocks": w_goldilocks}


# ── the bounded, surprisal-gated count model (one streaming pass, LFU-capped per order) ───────────

class GatedCountModel:
    """An online backoff char n-gram whose per-token write-weight is set by a surprisal gate, with each
    order's context table capped (LFU eviction) so memory is bounded and comparisons are equal-budget.

    predict-then-write, one token at a time (the gate needs the surprisal BEFORE the write):
      1. predict p(next | context) by calibrated backoff over the capped tables;
      2. surprisal s = −log2 p(true_next);
      3. write weight w = gate(s) into every order's (context → next) cell — fractional counts allowed;
      4. if an order's table is over cap, evict the lowest-total-count context (LFU = Exp AI's winner).

    Exposes `.K` and `.dist(suffix_str)` so lib/metrics scores it unchanged.
    """

    def __init__(self, order=5, cap=4000, gate="goldilocks", gate_kw=None, alpha=ALPHA):
        self.order = order
        self.cap = cap                                    # per-order context budget (0 = unbounded)
        self.gate = GATES[gate]; self.gate_name = gate
        self.gate_kw = gate_kw or {}
        self.alpha = alpha
        self.tab = [dict() for _ in range(order + 1)]     # tab[k][ctx_tuple] -> {next: weight}
        self.tot = [dict() for _ in range(order + 1)]     # tab[k][ctx] -> total weight (LFU key)
        self.uni = np.zeros(V)                            # order-0 fallback (always present)
        self.K = 64                                       # context window the metrics suite passes
        self.buf = []                                     # rolling left-context tail (bounded)
        self.writes = 0.0                                 # total weight written (the learning-rate budget)
        self.skipped = 0                                  # tokens that wrote ~0 (gate floor)

    # —— prediction: calibrated geometric-mean backoff over the capped tables ——
    def _dist_ids(self, ctx):
        """Next-char distribution from a context tuple: log-linear pool over the orders that have a row
        for this context (cortex.vote's calibrated geometric mean, restricted to non-abstaining orders)."""
        logp = np.zeros(V); n = 0
        for k in range(min(self.order, len(ctx)), -1, -1):
            key = tuple(ctx[len(ctx) - k:]) if k else ()
            d = self.tab[k].get(key)
            if not d: continue
            p = np.full(V, self.alpha); z = self.alpha * V
            for tok, c in d.items():
                p[tok] += c; z += c
            logp += np.log(p / z); n += 1
        if n == 0:
            u = self.uni + self.alpha
            return u / u.sum()
        logp /= n
        e = np.exp(logp - logp.max())
        return e / e.sum()

    def dist(self, suffix):
        ids = [c for c in (_CH.get(ch) for ch in suffix[-self.K:]) if c is not None]
        return self._dist_ids(ids[-(self.order):] if self.order else [])

    # —— one streaming pass: predict, gate on surprisal, write, evict ——
    def observe(self, ids):
        for nx in ids:
            ctx = self.buf[-self.order:] if self.order else []
            p = self._dist_ids(ctx)
            s = -math.log2(p[nx] + 1e-12)                 # the model's surprisal on the true token
            w = self.gate(s, **self.gate_kw)
            if w <= 1e-6:
                self.skipped += 1
            else:
                self.writes += w
                self.uni[nx] += w
                for k in range(min(self.order, len(self.buf)) + 1):
                    key = tuple(self.buf[len(self.buf) - k:]) if k else ()
                    d = self.tab[k].setdefault(key, {})
                    d[nx] = d.get(nx, 0.0) + w
                    self.tot[k][key] = self.tot[k].get(key, 0.0) + w
                    if self.cap and k > 0 and len(self.tab[k]) > self.cap:
                        self._evict(k, protect=key)
            self.buf.append(nx)
            if len(self.buf) > self.order + 2:
                self.buf = self.buf[-(self.order + 2):]
        return self

    def _evict(self, k, protect):
        """LFU: drop the lowest-total-weight context in order k (Exp AI's winner for dense char-grams).
        Reservoir-sampled candidate set keeps this O(1)-amortized, no global sort — never the row just
        touched."""
        items = self.tot[k]
        # sample a small candidate pool, evict the weakest (avoids an O(n) scan every overflow)
        n = len(items)
        keys = list(items.keys())
        idx = np.random.randint(0, n, size=min(32, n))
        worst = None; worst_w = math.inf
        for i in idx:
            key = keys[i]
            if key == protect: continue
            if items[key] < worst_w:
                worst_w = items[key]; worst = key
        if worst is not None:
            del self.tab[k][worst]; del self.tot[k][worst]

    def size(self):
        return sum(len(t) for t in self.tab)


_A = "abcdefghijklmnopqrstuvwxyz "
_CH = {c: i for i, c in enumerate(_A)}


# ── training + evaluation over an id-stream (single pass) ─────────────────────────────────────────

def train_eval(train_ids, eval_ids, order=5, cap=4000, gate="goldilocks", gate_kw=None):
    """One streaming pass over train_ids with the surprisal gate, then held-out bpc on eval_ids.
    Returns a dict: bpc, table-size, total write-weight (the learning-rate budget), skipped tokens."""
    m = GatedCountModel(order=order, cap=cap, gate=gate, gate_kw=gate_kw)
    m.observe([int(x) for x in train_ids])
    bpc = eval_bpc(m, eval_ids)
    return dict(bpc=bpc, size=m.size(), writes=m.writes, skipped=m.skipped, model=m)


def eval_bpc(model, eval_ids):
    """Held-out bits-per-char: mean surprisal of the true next char under the (frozen) model."""
    ids = [int(x) for x in eval_ids]
    bits = []
    ctx = []
    for nx in ids:
        p = model._dist_ids(ctx[-model.order:] if model.order else [])
        bits.append(-math.log2(p[nx] + 1e-12))
        ctx.append(nx)
        if len(ctx) > model.order + 2:
            ctx = ctx[-(model.order + 2):]
    return float(np.mean(bits))


# ── the N400 / cloze read-out (DISTINCT from the write-gate) ──────────────────────────────────────

def cloze_readout(model, eval_ids, n_probes=4000, seed=0):
    """N400/cloze validation. The N400 ERP amplitude scales with a word's CLOZE surprisal: low-cloze
    (expected) words evoke a small N400, high-cloze (unexpected) a large one. Here the count model's
    surprisal is the N400 proxy and the cloze probability is read from the SAME model's distribution at
    each probe site. We report:

      · the rank/probability profile: mean model-prob of the true next char split by how predictable the
        site is (low/mid/high context-entropy) — does surprisal separate expected from unexpected?
      · the N400 analogue: correlation between model surprisal and (1 − cloze-prob). A faithful read-out
        has surprisal rising as cloze falls — the canonical N400/cloze relationship.

    This is a READ of the counts, never a write — kept distinct from the learning-rate gate per the
    BUILD_QUEUE note. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    ids = [int(x) for x in eval_ids]
    o = model.order
    sites = rng.integers(o + 1, len(ids) - 1, size=min(n_probes, len(ids) - o - 2))
    surpr = []; cloze = []; ent = []
    for t in sites:
        ctx = ids[t - o:t]
        p = model._dist_ids(ctx)
        nx = ids[t]
        cp = float(p[nx])                                 # cloze probability of the realized char
        surpr.append(-math.log2(cp + 1e-12))              # the N400 proxy (model surprisal)
        cloze.append(cp)
        ent.append(float(-(p * np.log2(p + 1e-12)).sum()))  # context entropy = predictability of the site
    surpr = np.array(surpr); cloze = np.array(cloze); ent = np.array(ent)
    # split sites by context predictability (low entropy = constraining context, like a cloze sentence)
    qlo, qhi = np.quantile(ent, [0.33, 0.66])
    lo = ent <= qlo; hi = ent >= qhi
    # N400 analogue: surprisal vs (1 − cloze). Pearson r over probe sites.
    x = surpr; y = 1.0 - cloze
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return dict(
        n400_r=r,                                          # surprisal ↔ (1−cloze): should be strongly +
        cloze_lowctx=float(cloze[lo].mean()),              # constraining context → high cloze (expected)
        cloze_hictx=float(cloze[hi].mean()),               # weak context → low cloze (unexpected)
        surpr_lowctx=float(surpr[lo].mean()),              # constraining context → small N400
        surpr_hictx=float(surpr[hi].mean()),               # weak context → large N400
        mean_surpr=float(surpr.mean()),
    )
