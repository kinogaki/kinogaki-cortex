"""replay.py — Exp BG / M24: INVERSE-COUNT, SURPRISE-PRIORITIZED spindle replay.

Sleep doesn't replay everything evenly. Sharp-wave ripples preferentially reinstate the WEAK,
the RECENT, the SURPRISING (Schapiro 2018: spindles protect infrequent words; the hippocampus
re-fires what the cortex hasn't yet absorbed). AA's sleep pass replays its buffer UNIFORMLY — it
spends the same offline effort on a context it has seen ten-thousand times as on one it saw twice.
That is exactly backwards for the rare tail, which is where a count model is weakest and where a
night of consolidation should do the most good.

THE BUDGET. Offline time is scarce (the brain has ~hours of sleep, not infinite replay). So the
mechanism is defined against a FIXED increment budget B — a hard cap on how many reinforcement
counts the sleep pass may deposit. The only question is WHERE to spend them. This is the M24
mechanism: a bounded offline pass that decides its replay distribution online.

THE SUBSTRATE (unchanged, online). The same backoff count tables as Exp AA / consolidate.py:
tab[k] : ctx(k chars) -> dense (V,) next-char counts. We import learn_tables / score / memory_size
from consolidate so the comparison is apples-to-apples against the AA baseline.

THE REPLAY POLICIES (all online, single streaming pass over the buffer, bounded by B):

  uniform   — AA's policy. Every buffer position gets the same replay weight; the budget lands
              proportional to how OFTEN a context recurs => the head soaks up almost everything.

  invcount  — M24's policy. Replay weight for a (ctx, next) event ∝ 1/(1 + count) under the CURRENT
              table => rare contexts pull weight away from the head. The bounded budget concentrates
              on the tail. (count read from the longest matched backoff context, mirroring predict.)

  surprise  — M24 variant. Replay weight ∝ surprise = -log2 P(next | ctx) under the current model:
              re-fire the events the cortex predicted WORST. (Correlated with invcount but driven by
              local branching uncertainty, not raw frequency — the "uncertain" half of M24.)

Each policy produces a per-event weight; we normalize weights to a probability and DEPOSIT exactly B
increments by sampling-with-fractional-expectation (deterministic largest-remainder allotment given a
seed) onto the (ctx, next) entries those events touch. No gradient, no batch optimization: every
deposit is a +1 count. Bounded: total deposited == B regardless of buffer size.

THE STRATIFIED PROTECTION (the dual item-vs-regularity budget of M24). Because the prune step in a
sleep cycle drops contexts with total count < min_ctx, a rare context that recurred in the buffer but
sits just under the threshold is LOST under uniform replay (the budget never reached it) but SAVED
under invcount (the budget targeted it). We expose `protect_floor`: any context that receives a
replay deposit is lifted to at least the prune floor — the count-world version of "the spindle
protected that infrequent word from being forgotten tonight."

VOCAB: a..z=0..25, space=26, V=27 (matches corpus / consolidate). Char order small (<=6).
"""
import numpy as np
from consolidate import learn_tables, _ctx_key, _smooth, V


def _event_stats(tab, buffer_ids, order):
    """One streaming pass over the buffer. For every (ctx, next) EVENT at the highest order present,
    return parallel arrays: (k, ctxkey, next, count_now, surprise_now). count_now = the table count of
    that next-char under the longest matched context (the evidence the model currently has); surprise =
    -log2 P(next | longest matched ctx) under the current backoff model. These drive the replay weight.

    Grouped by unique (k, ctxkey, next) so a buffer event that recurs R times carries weight R (this is
    what makes UNIFORM replay frequency-proportional and lets INVCOUNT down-weight the head)."""
    ids = np.ascontiguousarray(buffer_ids, np.int64)
    n = len(ids)
    # Resolve each buffer position to the highest order whose context is present in tab (mirror backoff).
    m = n - order
    if m <= 0:
        return None
    res_k = np.zeros(m, np.int64)          # matched order per position (positions t=order..n-1)
    res_ctx = np.zeros(m, np.int64)        # matched ctxkey
    resolved = np.zeros(m, bool)
    nxt = ids[order:]                       # the realized next char at each position
    for k in range(order, 0, -1):
        ctx = _ctx_key(ids, k)              # contexts for positions t=k..n-1
        ctx = ctx[order - k:]               # align to positions t=order..n-1
        need = ~resolved
        uniq, inv = np.unique(ctx[need], return_inverse=True)
        present = np.array([int(c) in tab[k] for c in uniq], bool)
        pres_pos = present[inv]
        if pres_pos.any():
            gi = np.nonzero(need)[0][pres_pos]
            res_k[gi] = k
            res_ctx[gi] = ctx[need][pres_pos]
            resolved[gi] = True
        if resolved.all():
            break
    res_k[~resolved] = 0
    res_ctx[~resolved] = 0
    # Group identical (k, ctx, next) events; weight = multiplicity R.
    key = (res_k * (V ** (order + 1)) + res_ctx * V + nxt).astype(np.int64)
    uk, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    ek = res_k[idx]; ectx = res_ctx[idx]; en = nxt[idx]; mult = cnt.astype(np.float64)
    # current count + surprise of each unique event under the table
    cnow = np.zeros(len(uk)); snow = np.zeros(len(uk))
    for i in range(len(uk)):
        k = int(ek[i]); ck = int(ectx[i]); c = int(en[i])
        if k == 0:
            row = tab[0][0]
        else:
            row = tab[k][ck]
        cnow[i] = row[c]
        snow[i] = -np.log2(_smooth(row)[c])
    return dict(k=ek, ctx=ectx, nxt=en, mult=mult, count=cnow, surprise=snow)


def _allot(weights, budget, seed=0):
    """Deposit exactly `budget` integer increments across events in proportion to `weights`
    (largest-remainder / Hamilton apportionment, deterministic given seed). Returns int array."""
    w = np.asarray(weights, np.float64)
    s = w.sum()
    if s <= 0:
        return np.zeros(len(w), np.int64)
    exact = w / s * budget
    base = np.floor(exact).astype(np.int64)
    rem = budget - int(base.sum())
    if rem > 0:
        frac = exact - base
        # break ties deterministically with a seeded jitter
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(len(frac)), -frac))
        base[order[:rem]] += 1
    return base


def replay_sleep(tab, buffer_ids, order, policy="invcount", budget=200_000,
                 min_ctx=4, tail_mass=0.999, distill_tau=0.02, protect_floor=True,
                 alpha_inv=1.0, seed=0):
    """One PRIORITIZED sleep cycle at a FIXED increment budget. Returns (new_tab, stats).

    Steps:
      1. _event_stats over the buffer (online single pass) -> per-event (count, surprise, multiplicity).
      2. policy -> per-event replay weight:
           uniform   : weight = multiplicity                       (AA: frequency-proportional)
           invcount  : weight = multiplicity / (1 + count)^alpha_inv  (M24: protect the rare tail)
           surprise  : weight = multiplicity * surprise               (M24: re-fire worst-predicted)
      3. _allot deposits exactly `budget` +1 increments onto the touched (ctx, next) entries
         (a COPY of tab; we never mutate the input). protect_floor lifts any context that received a
         deposit to >= min_ctx so the subsequent prune cannot drop a context sleep just reinforced.
      4. prune + distill (identical to consolidate.sleep) so memory stays bounded and the comparison
         is the replay POLICY, holding the rest of the sleep pass fixed.

    All deposits are +1 counts (no gradient). Total deposited == budget (bounded)."""
    rng_seed = seed
    new_tab = [dict() for _ in range(order + 1)]
    for k in range(order + 1):
        for ck, row in tab[k].items():
            new_tab[k][ck] = row.copy()

    ev = _event_stats(tab, buffer_ids, order)
    deposited = 0
    touched_ctx = set()
    if ev is not None:
        mult = ev["mult"]; cnt = ev["count"]; sup = ev["surprise"]
        if policy == "uniform":
            w = mult
        elif policy == "invcount":
            w = mult / np.power(1.0 + cnt, alpha_inv)
        elif policy == "surprise":
            w = mult * sup
        else:
            raise ValueError(policy)
        alloc = _allot(w, budget, seed=rng_seed)
        deposited = int(alloc.sum())
        ek = ev["k"]; ectx = ev["ctx"]; en = ev["nxt"]
        for i in np.nonzero(alloc)[0]:
            k = int(ek[i]); ck = int(ectx[i]); c = int(en[i]); inc = int(alloc[i])
            if k == 0:
                new_tab[0][0][c] += inc
            else:
                new_tab[k][ck][c] += inc
                touched_ctx.add((k, ck))

    # prune + distill (held fixed across policies)
    pruned_ctx = 0; pruned_tail = 0; distilled = 0
    final = [dict() for _ in range(order + 1)]
    for k in range(order, -1, -1):
        for ck, row in new_tab[k].items():
            row = row.copy()
            tot = int(row.sum())
            protected = protect_floor and (k, ck) in touched_ctx
            if k >= 1 and tot < min_ctx and not protected:
                pruned_ctx += 1
                continue
            if k >= 1:
                ordr = np.argsort(row)[::-1]
                csum = np.cumsum(row[ordr]) / max(tot, 1)
                keep_n = int(np.searchsorted(csum, tail_mass) + 1)
                drop = ordr[keep_n:]
                if len(drop):
                    before = int((row > 0).sum())
                    row[drop] = 0
                    pruned_tail += before - int((row > 0).sum())
            if k >= 2:
                bck = int(ck) % (V ** (k - 1)) if k > 1 else 0
                brow = tab[k - 1].get(bck)
                if brow is not None and brow.sum() > 0 and not protected:
                    if float(np.sum(_smooth(row) * np.log2(_smooth(row) / _smooth(brow)))) < distill_tau:
                        distilled += 1
                        continue
            final[k][ck] = row

    stats = dict(policy=policy, budget=budget, deposited=deposited,
                 n_events=0 if ev is None else len(ev["mult"]),
                 touched_ctx=len(touched_ctx), pruned_ctx=pruned_ctx,
                 pruned_tail=pruned_tail, distilled=distilled)
    return final, stats
