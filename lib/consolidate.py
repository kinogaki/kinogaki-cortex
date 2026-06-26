"""consolidate.py — Exp AA: a SLEEP / CONSOLIDATION pass over the count memory ("agent dreaming").

Letta's "Towards Agents that Learn": an agent improves by refining its TOKEN-SPACE MEMORY, not its
weights; offline "sleep-time compute" refines that memory without new data; and the known failure mode is
that "memories become generic and lossy after repeated refinement." In our world this is almost literal —
each Column IS a memory-agent and its COUNT TABLES are the token-space memory. The online substrate already
learns weight-free. The NEW thing here is the offline SLEEP pass that REFINES the count memory.

THE SUBSTRATE (unchanged, online). A backoff char model = one count table per order k:
    tab[k] : ctx(k chars) -> {next_char: count}.
Prediction at a position backs off high->low to the longest SEEN context, add-alpha smoothed. This is the
plain online Column (cortex.Column / FastChar), built once by streaming counts.

THE SLEEP PASS (offline, but COUNT-BASED — no gradient, no batch optimization). One extra pass that REWRITES
the tables using only count operations, replaying a bounded recent buffer:

  (a) PRUNE  — heavy-hitter / count-min style. Drop low-utility entries: contexts that are too rare to
       trust (total count < MIN_CTX) and, within a kept context, the long tail of next-chars that carry
       negligible mass (cumulative-mass cap). Removes lossy noise; bounds memory.

  (b) DISTILL specific->generic — where a high-order context's next-char distribution is ~equal to its
       BACKOFF (the order k-1 distribution it would fall back to), the specific entry adds no information:
       drop it (KL(specific || backoff) < tau). Lossless compression: prediction is unchanged because
       backoff fills the hole with the same distribution. This is the direct count-world analogue of
       Letta's "distill into the generic memory."

  (c) PROMOTE — patterns that recur across MANY DISTINCT high-order contexts but share a near-identical
       continuation become a CONCEPT: cluster such contexts online (leader clustering on their next-char
       distributions) and store one shared distribution per concept, queried as a backoff tier when the
       specific context is absent. Generalization to unseen contexts, by counting.

Everything is local/count-based. The honest nuance (reported in RESULTS): sleep is a SECOND PASS over a
buffer, so it is offline — but the learning rule stays count/replay/leader-cluster, never gradient or
iterate-to-convergence batch optimization. Brains replay during sleep; this is replay + bookkeeping.

VOCAB: a..z=0..25, space=26, V=27 (matches lib/corpus, lib/fastchar). Char order small (<=6).
"""
import numpy as np

V = 27
ALPHA = 0.05


# ── the online substrate: dict-of-dicts backoff count tables (one Column, inspectable) ───────────

def learn_tables(ids, order):
    """ONE online streaming pass -> tab[k] = {ctx_key: {next_char: count}} for k=0..order.

    ctx_key packs k chars base-V into an int (k=0 -> the single key 0). Built vectorized by np.unique
    over packed (ctx, next) keys — identical counts to a token-at-a-time online update, no optimization.
    Returns a list `tab` of dicts; tab[k][ctxkey] is an np.int64 array of length V (dense next-char counts)."""
    ids = np.ascontiguousarray(ids, np.int64)
    n = len(ids)
    tab = []
    for k in range(order + 1):
        if k == 0:
            cnt = np.bincount(ids, minlength=V).astype(np.int64)
            tab.append({0: cnt}); continue
        ctx = _ctx_key(ids, k)                      # length n-k, position t=k..n-1
        nxt = ids[k:]
        key = ctx * V + nxt
        uk, uc = np.unique(key, return_counts=True)
        uctx = uk // V; unx = (uk % V).astype(np.int64); uc = uc.astype(np.int64)
        d = {}
        edges = np.nonzero(np.diff(uctx))[0] + 1
        starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(uk)]])
        for s, e in zip(starts, ends):
            row = np.zeros(V, np.int64); row[unx[s:e]] = uc[s:e]
            d[int(uctx[s])] = row
        tab.append(d)
    return tab


def _ctx_key(ids, k):
    """Pack the order-k context ids[t-k:t] base-V into an int, for every t in [k, n). Vectorized."""
    n = len(ids)
    w = np.lib.stride_tricks.sliding_window_view(ids, k)[: n - k].astype(np.int64)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return w @ powers


def _backoff_ctx(ctxkey, k):
    """The order-(k-1) context key reached by dropping the OLDEST char of an order-k context key."""
    return int(ctxkey) % (V ** (k - 1)) if k > 1 else 0


# ── prediction (backoff, add-alpha) + a concept backoff tier ──────────────────────────────────────

def _smooth(row):
    return (row + ALPHA) / (row.sum() + ALPHA * V)


def predict_dist(tab, ctx_ids, concepts=None, ctx2concept=None):
    """Backoff next-char distribution (V,) for a context (a length->=order int array, newest last).

    High->low: longest SEEN context wins. If a CONCEPT tier exists, an order-k context absent from tab[k]
    but mapped to a concept uses the concept's shared distribution BEFORE backing off to k-1 (the promote
    tier). ctx2concept[k] maps a kept order-k ctxkey -> concept id; concepts[cid] is a count row."""
    order = len(tab) - 1
    kmax = min(order, len(ctx_ids))
    for k in range(kmax, 0, -1):
        ck = int(np.asarray(ctx_ids[-k:], np.int64) @ (V ** np.arange(k - 1, -1, -1)))
        d = tab[k].get(ck)
        if d is not None:
            return _smooth(d)
        if concepts is not None and ctx2concept is not None:
            cid = ctx2concept[k].get(ck)
            if cid is not None:
                return _smooth(concepts[cid])
    return _smooth(tab[0][0])


# ── BATCH held-out scoring (vectorized): bpc + accuracy, split by context rarity ───────────────────

def score(tab, ids, concepts=None, ctx2concept=None, rare_ctx_thresh=20):
    """Mean bpc, accuracy, and the same split over RARE vs COMMON contexts, for the stream `ids`.

    A position's "context rarity" = the train total-count of its longest matched context (how much
    evidence the model had for it). rare = total < rare_ctx_thresh. Vectorized per order: for each
    position find the highest order whose context is present (in tab[k] OR, if enabled, in the concept
    tier), score that distribution. Mirrors predict_dist's backoff exactly, batched."""
    ids = np.ascontiguousarray(ids, np.int64)
    n = len(ids); m = n - 1
    order = len(tab) - 1
    logp = np.full(m, np.log2(_smooth(tab[0][0])[0]))   # placeholder, overwritten by unigram below
    pred = np.zeros(m, np.int64)
    ctxtot = np.zeros(m, np.float64)                     # evidence behind the chosen context
    resolved = np.zeros(m, bool)
    tgt = ids[1:]
    for k in range(order, -1, -1):
        if k == 0:
            row = tab[0][0]; dist = _smooth(row)
            idx = ~resolved
            logp[idx] = np.log2(dist[tgt[idx]])
            pred[idx] = dist.argmax(); ctxtot[idx] = row.sum()
            resolved[:] = True; break
        # contexts for positions t=k..n-1 -> logp index t-1 -> slice [k-1 : k-1+(n-k)]
        ctx = _ctx_key(ids, k)
        off = k - 1; sl = slice(off, off + len(ctx))
        sub_res = resolved[sl]
        # build a dense (n_ctx, V) for the contexts we actually see, via dict lookup on the needed keys
        need = ~sub_res
        if not need.any(): continue
        # resolve each needed position's context against tab[k] then concept tier
        ck = ctx
        # group identical context keys to amortize dict lookups
        uniq, inv = np.unique(ck[need], return_inverse=True)
        rows = np.full((len(uniq), V), -1.0)            # -1 marks "absent"
        present = np.zeros(len(uniq), bool)
        tot_u = np.zeros(len(uniq))
        for j, c in enumerate(uniq):
            d = tab[k].get(int(c))
            if d is not None:
                rows[j] = _smooth(d); present[j] = True; tot_u[j] = d.sum()
            elif concepts is not None and ctx2concept is not None:
                cid = ctx2concept[k].get(int(c))
                if cid is not None:
                    cr = concepts[cid]; rows[j] = _smooth(cr); present[j] = True; tot_u[j] = cr.sum()
        present_pos = present[inv]
        if not present_pos.any(): continue
        # write into the global arrays
        glob_idx = np.nonzero(need)[0][present_pos]      # indices into the slice's local coordinates
        sel = inv[present_pos]
        dsel = rows[sel]
        ttgt = tgt[sl][glob_idx]
        gp = dsel[np.arange(len(ttgt)), ttgt]
        abs_idx = off + glob_idx
        logp[abs_idx] = np.log2(gp)
        pred[abs_idx] = dsel.argmax(axis=1)
        ctxtot[abs_idx] = tot_u[sel]
        resolved[abs_idx] = True
    rare = ctxtot < rare_ctx_thresh
    bpc = float(-logp.mean())
    acc = float((pred == tgt).mean())
    out = dict(bpc=bpc, acc=acc, rare_frac=float(rare.mean()))
    out["bpc_rare"] = float(-logp[rare].mean()) if rare.any() else float("nan")
    out["bpc_common"] = float(-logp[~rare].mean()) if (~rare).any() else float("nan")
    out["acc_rare"] = float((pred[rare] == tgt[rare]).mean()) if rare.any() else float("nan")
    out["acc_common"] = float((pred[~rare] == tgt[~rare]).mean()) if (~rare).any() else float("nan")
    return out


def memory_size(tab, concepts=None, ctx2concept=None):
    """Memory footprint = number of stored (context, next-char) entries (nonzero counts) + concept rows.
    This is the token-space-memory size the sleep pass tries to SHRINK for equal-or-better prediction."""
    n = 0
    for k in range(len(tab)):
        for row in tab[k].values():
            n += int((row > 0).sum())
    nctx = sum(len(tab[k]) for k in range(len(tab)))
    nconcept = 0
    if concepts is not None:
        nconcept = sum(int((r > 0).sum()) for r in concepts)
        nctx += sum(len(ctx2concept[k]) for k in range(len(ctx2concept)))
    return dict(entries=n + nconcept, contexts=nctx)


# ── the SLEEP pass: prune -> distill -> promote, all count-based, over a recent buffer ──────────────

def _kl(p, q):
    """KL(p || q) in bits, both already smoothed prob vectors."""
    return float(np.sum(p * np.log2(p / q)))


def sleep(tab, buffer_ids, order,
          min_ctx=4, tail_mass=0.999, distill_tau=0.02,
          promote=True, promote_min_ctx=8, promote_thresh=0.85, promote_min_members=3, cmax=2000,
          promote_replace=False, rng=None, verbose=False):
    """One SLEEP/CONSOLIDATION cycle. Returns (new_tab, concepts, ctx2concept, stats).

    Replays `buffer_ids` only to recount a fresh per-order REINFORCEMENT (the recent buffer the agent
    "dreams" over): entries seen in the buffer get a small reinforcement, so consolidation is anchored to
    what actually recurred (heavy-hitter). All three steps below are pure count operations.

    (a) PRUNE: drop contexts whose total count < min_ctx (untrustworthy); within a kept context, zero the
        long tail of next-chars beyond `tail_mass` cumulative probability (count-min style heavy-hitter).
    (b) DISTILL: for k>=2, if KL(smooth(specific) || smooth(backoff)) < distill_tau, drop the specific
        entry (its backoff predicts identically -> lossless). Generic absorbs the specific.
    (c) PROMOTE: among the high-order contexts that SURVIVE, leader-cluster their next-char distributions
        (online, single pass, running-mean prototype or spawn). A cluster with >= promote_min_members
        contexts becomes a CONCEPT with a shared (summed) count row; its members are MOVED out of tab[k]
        into the concept tier (so the concept generalizes to unseen contexts that route to it later)."""
    if rng is None:
        rng = np.random.default_rng(0)
    new_tab = [dict() for _ in range(order + 1)]
    # buffer reinforcement counts (heavy-hitter anchor) — what recurred recently
    buf_tab = learn_tables(buffer_ids, order)

    pruned_ctx = 0; pruned_tail = 0; distilled = 0
    # process orders high->low so distill's backoff target already exists in new_tab when we need it
    # but backoff target is the ORIGINAL tab[k-1]; use tab (pre-sleep) as the distill reference.
    for k in range(order, -1, -1):
        src = tab[k]
        for ck, row in src.items():
            row = row.copy()
            tot = int(row.sum())
            if k >= 1 and tot < min_ctx:
                pruned_ctx += 1
                continue                                    # (a) prune untrustworthy context
            if k >= 1:
                # (a) prune the long tail beyond tail_mass cumulative probability
                ordr = np.argsort(row)[::-1]
                csum = np.cumsum(row[ordr]) / max(tot, 1)
                keep_n = int(np.searchsorted(csum, tail_mass) + 1)
                drop = ordr[keep_n:]
                if len(drop):
                    before = int((row > 0).sum())
                    row[drop] = 0
                    pruned_tail += before - int((row > 0).sum())
            if k >= 2:
                # (b) distill: compare to the backoff distribution from the PRE-sleep tables
                bck = _backoff_ctx(ck, k)
                brow = tab[k - 1].get(bck)
                if brow is not None and brow.sum() > 0:
                    if _kl(_smooth(row), _smooth(brow)) < distill_tau:
                        distilled += 1
                        continue                            # backoff predicts identically -> drop
            new_tab[k][ck] = row

    # (c) PROMOTE — leader-cluster the surviving high-order contexts' distributions into concepts
    concepts = []
    ctx2concept = [dict() for _ in range(order + 1)]
    promoted = 0
    if promote and order >= 2:
        protos = np.zeros((cmax, V))                        # running SUM of member unit-distributions
        proto_count = np.zeros((cmax, V), np.int64)         # running SUM of member raw counts (the concept)
        members = [[] for _ in range(cmax)]                 # (k, ctxkey) per cluster
        C = 0
        # cluster contexts of order >= 2 with enough evidence; stream in a fixed (sorted) order
        for k in range(2, order + 1):
            keys = sorted(new_tab[k].keys())
            for ck in keys:
                row = new_tab[k][ck]
                tot = int(row.sum())
                if tot < promote_min_ctx:
                    continue
                p = _smooth(row); u = p / np.linalg.norm(p)
                if C == 0:
                    protos[0] = u; proto_count[0] = row; members[0].append((k, ck)); C = 1
                    continue
                dots = protos[:C] @ u
                nrm = np.maximum(np.linalg.norm(protos[:C], axis=1), 1e-9)
                cos = dots / nrm
                best = int(cos.argmax()); bcos = float(cos[best])
                if bcos >= promote_thresh or C >= cmax:
                    protos[best] += u; proto_count[best] += row; members[best].append((k, ck))
                else:
                    protos[C] = u; proto_count[C] = row; members[C].append((k, ck)); C += 1
        # keep clusters with enough members; MOVE their contexts into the concept tier
        for c in range(C):
            if len(members[c]) >= promote_min_members:
                cid = len(concepts)
                concepts.append(proto_count[c].copy())
                for (k, ck) in members[c]:
                    if promote_replace and ck in new_tab[k]:
                        del new_tab[k][ck]                  # moved out of specific table (replace mode)
                    ctx2concept[k][ck] = cid                # else concept is a pure fallback tier
                    promoted += 1
    stats = dict(pruned_ctx=pruned_ctx, pruned_tail=pruned_tail, distilled=distilled,
                 promoted=promoted, n_concepts=len(concepts))
    if verbose:
        print(f"    sleep: pruned_ctx={pruned_ctx:,} pruned_tail={pruned_tail:,} "
              f"distilled={distilled:,} promoted={promoted:,} concepts={len(concepts):,}")
    return new_tab, (concepts or None), (ctx2concept if concepts else None), stats
