"""contingency.py — G2: a contingency-gated learning rate (the temporal-contingency dial).

A child does not learn equally from every word it hears. A reply that *answers* you — one that lands
while your own utterance is still warm — teaches more than the same words overheard from across the
room (Goldstein & Schwade: contingent feedback beats yoked feedback even when the audio is identical).
The dial is **temporal contingency**: how soon after I spoke did this input arrive?

The mechanism, online and per-increment. Track `Δt` = steps since the agent last emitted a non-empty
`act()`. Each count increment is multiplied by a soft contingency gain

    g = exp(−Δt/τ)                       # 1.0 right after speaking, decaying as the world goes quiet

and routed into TWO bounded registers that PREDICTION POOLS together:

    tab_hot   — warm input (a reply that answered me): counted with weight g·HOT
    tab_cold  — background input (overheard / stale):  counted with weight g·COLD + a floor

The guard the spec insists on (joint-attention is contested, so the gate stays *soft*): cold input is
NEVER discarded — every token still updates `tab_cold` at a low weight. The hot table only ever *adds*
emphasis to genuinely contingent tokens; it never gates anything to zero. Hot is small and specific
(AE/ART-protectable), cold carries the bulk — bounded by the same context-tail trim as the base agent.

This module is a drop-in over the harness `CortexAgent`: same `observe / act / dist / .K`, so the
existing metrics suite and `harness.run` score it unchanged. The ONLY change is *how much* each
increment counts, and into which of the two tables.

The load-bearing control (registered before any run): **YOKED**. Feed the identical reply text with the
identical loop, but draw `g` from the *scrambled* timeline instead of the real one. Same tokens, same
total count mass — only the timing→gain alignment differs. If contingency-ON does not beat YOKED, the
dial is inert and the honest verdict is negative.

Rules: ONLINE (a per-increment scalar; no second pass), BOUNDED (two dicts, same tail trim, hot is
small), NO gradient/k-means/SVD/backprop. Reuses cortex.Column tables + the calibrated `vote` verbatim.
"""
import numpy as np
from cortex import Column, vote


class ContingencyAgent:
    """CortexAgent with a contingency-gated, two-register learning rate.

    Two parallel Column bands over the SAME token stream — `hot` and `cold` — distinguished only by how
    each increment is weighted. `observe()` weights every increment by the contingency gain g (and the
    register's base weight); `act()` resets the contingency clock (Δt → 0) so the NEXT input is warm.
    Prediction pools hot (weighted up) and cold into one distribution via the calibrated `vote`.

    `gain_fn` is the seam the YOKED ablation swaps: ON  → g = exp(−Δt/τ) from the real Δt;
    YOKED → g drawn from a scrambled timeline (same g-values, wrong tokens). Pass `gain_override` per
    observe() to inject a yoked gain and bypass the internal clock entirely.
    """

    def __init__(self, orders=(0, 1, 2, 3, 4, 5, 6), codec=None, seed=0,
                 tau=3.0, hot_w=1.0, cold_w=0.2, cold_floor=0.05, hot_pool=2.0):
        from harness import CharCodec
        self.hot = [Column(o) for o in orders]
        self.cold = [Column(o) for o in orders]
        self.maxord = max(orders)
        self.codec = codec or CharCodec(); self.vocab = self.codec.vocab
        self.K = 64
        self.buf = []
        self.rng = np.random.default_rng(seed)
        # contingency dial
        self.tau = tau                       # decay constant of the warmth window (steps)
        self.hot_w = hot_w                   # base weight of a (fully warm) hot increment
        self.cold_w = cold_w                 # base weight of a cold increment
        self.cold_floor = cold_floor         # cold input is NEVER zero (soft gate guard)
        self.hot_pool = hot_pool             # how much louder hot counts vote than cold
        self.dt = 10 ** 9                     # steps since last self-emission (cold to start)
        # bookkeeping (so the experiment can audit where mass landed)
        self.hot_mass = 0.0; self.cold_mass = 0.0

    # —— learning (the dial lives here) ——
    def observe(self, ids, gain_override=None):
        """Count a chunk online. Each new token is counted into BOTH registers, weighted by the
        contingency gain g (real, or `gain_override` for the yoked ablation) times the register weight.
        Advancing the world by `len(ids)` steps cools the clock for the NEXT observe."""
        ids = list(ids)
        if not ids:
            self.dt += 1
            return
        g = float(gain_override) if gain_override is not None else float(np.exp(-self.dt / self.tau))
        w_hot = self.hot_w * g                       # warm → strong; stale → ~0
        w_cold = self.cold_w * g + self.cold_floor   # always positive: cold never silenced

        start = len(self.buf); self.buf.extend(ids)
        for band, w, mtot in ((self.hot, w_hot, "hot"), (self.cold, w_cold, "cold")):
            if w <= 0:
                continue
            for col in band:
                tab = col.tab
                for t in range(start, len(self.buf)):
                    nx = self.buf[t]
                    for k in range(min(col.order, t) + 1):
                        ctx = tuple(self.buf[t - k:t])
                        d = tab[k].setdefault(ctx, {})
                        d[nx] = d.get(nx, 0.0) + w
        self.hot_mass += w_hot * len(ids)
        self.cold_mass += w_cold * len(ids)
        # the world advanced while NO self-emission happened → cool the clock
        self.dt += len(ids)
        if len(self.buf) > self.K + self.maxord:
            self.buf = self.buf[-(self.K + self.maxord):]

    # —— prediction: pool hot (loud) + cold ——
    def _pred_dicts(self, ctx):
        ctx = tuple(ctx)
        ds = []
        for col in self.hot:                          # hot voted hot_pool× louder
            p = col.predict(ctx)
            if p:
                ds.append({k: v * self.hot_pool for k, v in p.items()})
        for col in self.cold:
            p = col.predict(ctx)
            if p:
                ds.append(p)
        return ds

    def _dist_ids(self, ctx):
        return vote(self._pred_dicts(ctx), self.vocab)

    def dist(self, suffix):
        return self._dist_ids(self.codec.encode(suffix)[-self.K:])

    # —— generation: emitting RESETS the contingency clock (the next input is contingent) ——
    def act(self, k, temp=0.8):
        if not k:
            return ()
        out = []; ctx = list(self.buf[-self.K:])
        for _ in range(k):
            p = self._dist_ids(ctx[-self.K:]) ** (1.0 / temp); p = p / p.sum()
            tok = int(self.rng.choice(self.vocab, p=p)); out.append(tok); ctx.append(tok)
        self.dt = 0                                   # I just spoke → the world's reply will be warm
        return tuple(out)


def yoked_gains(real_gains, seed=0):
    """The control: the SAME multiset of gains, RE-PAIRED to the wrong tokens (timing scrambled).
    Same total count mass, same g-distribution — only the alignment to content is destroyed. If the
    dial is real, ON beats this; if not, ON == YOKED and the honest verdict is negative."""
    g = np.asarray(real_gains, dtype=float).copy()
    np.random.default_rng(seed).shuffle(g)
    return g
