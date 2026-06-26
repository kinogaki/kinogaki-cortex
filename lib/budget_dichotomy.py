"""budget_dichotomy.py — Exp AS: re-run "what survives scale" UNDER A MEMORY BUDGET.

The capstone (overnight) measured every "vanished" mechanism with UNBOUNDED memory: np.unique over the whole
stream, every count kept forever. Under infinite storage, MORE DATA SUBSUMES any mechanism that merely
re-predicts what raw high-order counts already hold — so the top-down topic prior (ignition), word-concept
generalization, and consolidation/sleep all read out ≈0 gain at scale. They were flat because nothing was
ever thrown away.

THE BOUNDED-MEMORY RULE predicts this FLIPS. When you cannot keep every count you must discard most of them,
and the only way to stay good after discarding is to have GENERALIZED first — turned a million specific
high-order counts into a few reusable abstractions (a topic prior, a concept, a distilled generic). So under
a FIXED MEMORY CAP the "vanished" mechanisms should RETURN: each should EARN ITS KEEP (improve bpc at equal
final budget) where it was flat unbounded.

This file gives the apparatus to measure, per mechanism, two deltas at one data scale:
    Δ_unbounded  = quality(mechanism ON) − quality(mechanism OFF), full tables   (expect ≈ 0, the prior finding)
    Δ_bounded    = same, but BOTH sides capped to the SAME final #entries         (predicted > 0 if it flips)

A "flip" = Δ_unbounded ≈ 0 AND Δ_bounded > 0 (mechanism on the right side of the cap).

HARD RULES honored: single streaming pass to build counts (vectorized np.unique == token-at-a-time online
update); capping is a heavy-hitter keep-top-B (count-min style), not gradient/iterate-to-convergence; sleep is
count-based replay (lib/consolidate). Fixed seed. Reuses lib/consolidate (sleep + score + memory_size + the
backoff count tables) and lib/ignition (G-conditioned backoff). Alphabet a..z=0..25, space=26, V=27.
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lib import consolidate as C
from lib.ignition import TopicCoder, commit_G, GCondChar, _ctx_tok
from lib.corpus import split_words

V = 27
SPACE = 26


# ─────────────────────────── the budget operation: heavy-hitter cap ───────────────────────────────

def cap_tables(tab, budget, protect_low_orders=True):
    """Return a copy of the backoff tables (consolidate.learn_tables format: list of {ctxkey:(V,) counts}) with
    at most `budget` total stored CONTEXTS, keeping the highest-TOTAL-COUNT contexts (heavy-hitter / LFU keep).

    This is the bounded-memory operation: you cannot keep every count, so you keep the ones with the most
    evidence and DROP the long, sparse tail — exactly the entries an unbounded model relies on. Prediction
    then BACKS OFF through whatever survives (lower-order tables / concept tiers), so a model that GENERALIZED
    its sparse counts into a few dense survivors loses less.

    protect_low_orders: orders 0 and 1 are tiny (≤ V and ≤ V² contexts) and always kept — capping them would
    just remove the backoff floor and confound the test. The budget is spent on orders ≥ 2 (where the sparse
    high-order tail lives). Returns a new tab; inputs untouched."""
    order = len(tab) - 1
    new = [dict() for _ in range(order + 1)]
    # always keep the cheap low-order floor
    floor = 1 if protect_low_orders else 0
    for k in range(0, min(floor, order) + 1):
        new[k] = {ck: row.copy() for ck, row in tab[k].items()}
    # rank every capped-order context by total count, keep the top `budget`
    pool = []                                            # (total, k, ctxkey)
    for k in range(floor + 1, order + 1):
        for ck, row in tab[k].items():
            pool.append((int(row.sum()), k, ck))
    pool.sort(key=lambda x: (-x[0], x[1], x[2]))         # deterministic: by total desc, then k, ck
    for tot, k, ck in pool[:budget]:
        new[k][ck] = tab[k][ck].copy()
    return new


def n_entries(tab, concepts=None, ctx2concept=None):
    """Total stored entries = nonzero (context,next-char) cells + concept rows. The memory the budget bounds."""
    return C.memory_size(tab, concepts, ctx2concept)["entries"]


# ─────────────────────── mechanism 1: consolidation / sleep under a cap ────────────────────────────

def mech_consolidation(train, evl, order=5, budget=8000, buffer_frac=0.25, seed=0):
    """Sleep/consolidation. OFF = raw counts; ON = one count-based sleep pass (prune→distill→promote concepts).

    Unbounded: full tables both sides, compare bpc (sleep is near-lossless by design → Δ≈0, prior finding).
    Bounded:   cap BOTH sides to the same final #entries. OFF caps raw counts (drops the sparse tail outright).
               ON sleeps FIRST (distilling specific→generic, promoting recurring contexts into concepts) THEN
               caps to the same entry budget — so the survivors include reusable concepts, not just frequent
               literals. If generalize-before-discard helps, ON beats OFF at equal budget.
    Returns dict with unbounded/bounded bpc for ON and OFF, the two deltas, and entry counts."""
    rng = np.random.default_rng(seed)
    tab = C.learn_tables(train, order)

    # ---- unbounded ----
    raw_full = C.score(tab, evl)["bpc"]
    buf = train[-int(len(train) * buffer_frac):]
    s_tab, s_con, s_c2c, sstat = C.sleep(tab, buf, order, rng=rng, promote=True, promote_replace=False)
    sleep_full = C.score(s_tab, evl, s_con, s_c2c)["bpc"]
    d_unb = raw_full - sleep_full                        # positive = sleep better

    # ---- bounded: cap both to the same FINAL entry budget ----
    # express the budget as a CONTEXT cap, then measure realized entries so both sides match.
    raw_cap = cap_tables(tab, budget)
    raw_cap_bpc = C.score(raw_cap, evl)["bpc"]
    raw_cap_E = n_entries(raw_cap)
    # ON side: sleep, then cap the surviving specific tables to the SAME entry budget (concepts kept — they are
    # the generalization the cap is meant to reward; count their entries against the budget too).
    s_cap = cap_tables(s_tab, max(1, budget))
    # shrink/grow the ON specific cap until total entries (specific + concepts) ≈ raw_cap_E (fair budget)
    target = raw_cap_E
    lo, hi = 1, budget * 4
    for _ in range(18):
        mid = (lo + hi) // 2
        cand = cap_tables(s_tab, mid)
        if n_entries(cand, s_con, s_c2c) <= target:
            s_cap = cand; lo = mid + 1
        else:
            hi = mid - 1
    sleep_cap_bpc = C.score(s_cap, evl, s_con, s_c2c)["bpc"]
    sleep_cap_E = n_entries(s_cap, s_con, s_c2c)
    d_bnd = raw_cap_bpc - sleep_cap_bpc                  # positive = sleep better under budget

    return dict(name="consolidation/sleep", metric="bpc (lower better)",
                off_unb=raw_full, on_unb=sleep_full, d_unb=d_unb,
                off_bnd=raw_cap_bpc, on_bnd=sleep_cap_bpc, d_bnd=d_bnd,
                off_E=raw_cap_E, on_E=sleep_cap_E, budget=budget,
                concepts=sstat["n_concepts"], promoted=sstat["promoted"], distilled=sstat["distilled"])


# ──────────────────── mechanism 2: top-down topic prior (ignition) under a cap ─────────────────────
#
# Design: build TWO consolidate-format backoff tabs over the SAME char stream, scored by ONE G-aware
# backoff scorer (`score_with_G`) so ON and OFF are apples-to-apples and `cap_tables` applies unchanged.
#   OFF tab : plain order-k char contexts (== FastChar).
#   ON  tab : the top `g_orders` are keyed by (G, ctx) — a SEPARATE order-tier above the plain ones — where
#             G is the committed global topic broadcast onto every char. Lower orders are plain (shared).
# The scorer tries, at each position, the longest matched G-context first (ON only), then plain contexts,
# then the floor. Capping ON keeps the few dense (G,short-ctx) survivors; capping OFF keeps frequent literal
# high-order contexts. Equal entry budget → does the topic prior buy back what the cap removed?

def mech_ignition(train, evl, order=5, g_orders=(5, 4), K=64, budget=12000, seed=0):
    """Top-down topic/ignition prior under a budget. Returns the two deltas + entry counts."""
    tr = np.asarray(train, np.int64); ev = np.asarray(evl, np.int64)
    # committed global topic G over chars, train + eval (eval uses the TRAIN-fit topic coder)
    tc, Gchar_tr = _fit_Gchar(tr, K, seed)
    Gchar_ev = _commit_Gchar(tc, ev)

    off_tab = C.learn_tables(tr, order)                  # plain char backoff (FastChar)
    g_tab = _build_g_tab(tr, Gchar_tr, order, g_orders)  # (G,ctx) tiers, keyed by order+order_max packing

    # ---- unbounded ----
    off_unb = score_with_G(off_tab, None, ev, None, order, g_orders)
    on_unb = score_with_G(off_tab, g_tab, ev, Gchar_ev, order, g_orders)
    d_unb = off_unb - on_unb

    # ---- bounded: BOTH sides get the SAME total high-order context budget ----
    # OFF spends all `budget` contexts on plain high-order ctx. ON SPLITS the budget: it keeps fewer plain
    # high-order contexts and spends the freed half on a G-tier — so ON never gets more memory than OFF, the
    # test is "is a slice of the budget better spent on a topic prior than on more literal contexts?"
    off_cap = cap_tables(off_tab, budget)
    target_E = n_entries(off_cap)                         # the fair budget, measured in ENTRIES
    # ON keeps fewer literal high-order contexts + a G-tier, binary-searched so its TOTAL entries ≤ target_E.
    best = (off_tab[:1], [dict() for _ in g_tab])
    lo, hi = 1, budget
    for _ in range(16):
        mid = (lo + hi) // 2
        on_plain = cap_tables(off_tab, max(1, budget - mid))
        g_cap = _cap_g_tab(g_tab, mid)
        e = n_entries(on_plain) + sum(int((r > 0).sum()) for d in g_cap for r in d.values())
        if e <= target_E:
            best = (on_plain, g_cap); lo = mid + 1
        else:
            hi = mid - 1
    on_plain, g_cap = best
    off_bnd = score_with_G(off_cap, None, ev, None, order, g_orders)
    on_bnd = score_with_G(on_plain, g_cap, ev, Gchar_ev, order, g_orders)
    d_bnd = off_bnd - on_bnd
    off_E = target_E
    on_E = n_entries(on_plain) + sum(int((r > 0).sum()) for d in g_cap for r in d.values())

    return dict(name="top-down topic prior (ignition)", metric="bpc (lower better)",
                off_unb=off_unb, on_unb=on_unb, d_unb=d_unb,
                off_bnd=off_bnd, on_bnd=on_bnd, d_bnd=d_bnd,
                off_E=off_E, on_E=on_E, budget=budget, K=tc.K)


def _word_ids(ids):
    """Vectorized per-word ids = base-V hash of each word's first ≤6 chars, folded into a fixed prime vocab.
    Returns (wids, words) where words is the split_words span list aligned to wids. Collision-tolerant — only
    used to cluster words into topics, so a few hash collisions just merge rare words."""
    ids = np.asarray(ids, np.int64)
    words = split_words(ids)
    starts = np.array([a for a, b in words], np.int64)
    lens = np.array([min(b - a, 6) for a, b in words], np.int64)
    h = np.zeros(len(words), np.int64)
    for off in range(6):                                 # 6 vectorized steps, not a per-word Python loop
        m = lens > off
        idx = starts[m] + off
        h[m] = h[m] * V + ids[idx] + 1
    return (h % 200003).astype(np.int64), words


def _fit_Gchar(ids, K, seed):
    wids, words = _word_ids(ids)
    vocab = int(wids.max()) + 1 if len(wids) else 1
    tc = TopicCoder(K=K, seed=seed).fit(wids, vocab)
    Gw = commit_G(tc.topic_of[wids], tc.K)
    return tc, _broadcast(ids, words, Gw)


def _commit_Gchar(tc, ids):
    wids, words = _word_ids(ids)
    topic = np.where(wids < len(tc.topic_of), tc.topic_of[np.clip(wids, 0, len(tc.topic_of) - 1)], -1)
    Gw = commit_G(topic, tc.K)
    return _broadcast(ids, words, Gw)


def _broadcast(ids, words, Gw):
    """Every char gets the G committed at its word (topic held across the word + its trailing space)."""
    G = np.zeros(len(ids), np.int64)
    for (a, b), g in zip(words, Gw):
        G[a:min(b + 1, len(ids))] = g
    return G


# G-tier key packing: for an order-k G context we store key = G * (V**k) + ctx, in g_tab[k].
def _build_g_tab(ids, Gchar, order, g_orders):
    """Build (G,ctx) count tables for the top `g_orders`, vectorized (np.unique == online counting)."""
    ids = np.ascontiguousarray(ids, np.int64)
    g_tab = [dict() for _ in range(order + 1)]
    for k in g_orders:
        ctx, tok = _ctx_tok(ids, k)
        g = Gchar[k:].astype(np.int64)
        gkey = g * (V ** k) + ctx                        # fold G into the context (== GCondChar)
        full = gkey * V + tok
        uk, uc = np.unique(full, return_counts=True)
        gc = uk // V; nx = (uk % V).astype(np.int64); uc = uc.astype(np.int64)
        edges = np.nonzero(np.diff(gc))[0] + 1
        starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(uk)]])
        for s, e in zip(starts, ends):
            row = np.zeros(V, np.int64); row[nx[s:e]] = uc[s:e]
            g_tab[k][int(gc[s])] = row
    return g_tab


def _cap_g_tab(g_tab, budget):
    """Keep the top-`budget` (G,ctx) entries across all G-orders by total count (heavy-hitter)."""
    new = [dict() for _ in range(len(g_tab))]
    pool = [(int(r.sum()), k, ck) for k in range(len(g_tab)) for ck, r in g_tab[k].items()]
    pool.sort(key=lambda x: (-x[0], x[1], x[2]))
    for tot, k, ck in pool[:budget]:
        new[k][ck] = g_tab[k][ck].copy()
    return new


def score_with_G(plain_tab, g_tab, evl, Gchar, order, g_orders):
    """Vectorized G-aware backoff bpc. At each position: try the longest matched (G,ctx) survivor among
    g_orders (if g_tab given), then fall back to the plain char backoff (consolidate.score's exact ladder).
    Mirrors predict order high→low. plain_tab/g_tab are consolidate-format. Returns mean bpc."""
    ids = np.ascontiguousarray(evl, np.int64); n = len(ids); m = n - 1
    logp = np.zeros(m); resolved = np.zeros(m, bool); tgt = ids[1:]

    def fill_order(d, keyfn, k):
        if not d:
            return
        ctx, _ = _ctx_tok(ids, k); keys = keyfn(ctx, k)
        off = k - 1; sl = slice(off, off + len(ctx)); need = ~resolved[sl]
        if not need.any():
            return
        uniq, inv = np.unique(keys[need], return_inverse=True)
        rows = np.zeros((len(uniq), V)); pres = np.zeros(len(uniq), bool)
        for j, c in enumerate(uniq):
            r = d.get(int(c))
            if r is not None:
                rows[j] = C._smooth(r); pres[j] = True
        pp = pres[inv]
        if not pp.any():
            return
        gi = np.nonzero(need)[0][pp]; sel = inv[pp]; ds = rows[sel]
        tt = tgt[sl][gi]; ai = off + gi
        logp[ai] = np.log2(ds[np.arange(len(tt)), tt]); resolved[ai] = True

    # Backoff ladder, highest evidence first. At each order k (high→low) we try the PLAIN ctx FIRST (it has
    # the most evidence for that exact context); only where the plain ctx is UNSEEN do we consult the
    # G-conditioned tier for that order — G is a top-down PRIOR that fills holes the literal context can't,
    # not a replacement for a well-attested literal context. (Trying G first would override good literals
    # with a sparser, noisier (G,ctx) row and lose bits — the wrong test for "does the prior earn its keep".)
    for k in range(order, 0, -1):
        fill_order(plain_tab[k], lambda ctx, kk: ctx, k)
        if g_tab is not None and Gchar is not None and k in g_orders:
            g = Gchar[k:].astype(np.int64)
            fill_order(g_tab[k], lambda ctx, kk, gg=g: gg * (V ** kk) + ctx, k)
    # 3) unigram floor
    if not resolved.all():
        dist = C._smooth(plain_tab[0][0]); idx = ~resolved
        logp[idx] = np.log2(dist[tgt[idx]])
    return float(-logp.mean())


# ──────────────── mechanism 3: word-concept generalization vs raw word counts ──────────────────────

def mech_concepts(train, evl, budget=4000, order=2, seed=0):
    """Word-level concept/construction generalization. OFF = raw word n-gram counts; ON = open-slot
    constructions that predict the filler through its CATEGORY (a concept), so they score words a frame has
    never hosted.

    Unbounded: raw word counts already cover the seen pairs; the category head re-predicts them → Δ≈0 (prior).
    Bounded:   cap the raw (frame→filler) table. OFF loses the sparse frame-filler pairs outright. ON keeps a
               few CATEGORY heads + a category lexicon (orders of magnitude smaller) that still place mass on
               the right fillers via the slot's category → generalizes past the cap. If concepts earn keep,
               ON beats OFF at equal budget on word-prediction log-loss.
    Metric: bits/word on held-out next-word (lower better). Scoring is precompute-then-vectorize: each side
    builds one prediction per DISTINCT eval frame, then bits are read out by (frame,filler) lookup."""
    from lib.constructions import build_frame_counts, ConstructionGrammar
    from lib.jepa import online_signatures, leader_cluster

    seq, _ = _word_ids(train)
    Vw = int(seq.max()) + 1
    unigram = np.bincount(seq, minlength=Vw).astype(np.float64)
    uni_p = (unigram + 0.5) / (unigram.sum() + 0.5 * Vw)         # smoothed unigram floor (dense)

    # online filler categories (jepa leader clustering); visit words most-frequent-first (a stable order)
    sig, cnt = online_signatures(seq, Vw, D=64, window=5, seed=seed)
    visit = np.argsort(cnt)[::-1]
    clu_of, Ccl = leader_cluster(sig, cnt, order=visit, min_evidence=20, thresh=0.5, Cmax=300)

    fc = build_frame_counts(seq, order=1)
    g = ConstructionGrammar(clu_of, Ccl, min_token=20).fit(fc)
    g.classify(); g.build_category_lexicon(fc)

    eseq, _ = _word_ids(evl)
    frames = eseq[:-1]; fillers = eseq[1:]
    ufr = np.unique(frames)                                       # distinct eval frames — predict each once

    def cap_frames(keys, keyscore, cap):
        if cap is None:
            return set(keys)
        return set(sorted(keys, key=lambda k: -keyscore(k))[:cap])

    def cap_frames_by_cells(keys, keyscore, cellfn, cell_budget):
        """Keep the highest-scoring frames until their stored CELLS would exceed `cell_budget` (fair memory)."""
        out = set(); used = 0
        for k in sorted(keys, key=lambda k: -keyscore(k)):
            c = cellfn(k)
            if used + c > cell_budget:
                break
            out.add(k); used += c
        return out

    # honest memory unit = stored CELLS. A raw frame stores one cell per distinct seen filler; a concept head
    # stores one (C,) category vector (its nonzero cats); the category lexicon P(w|c) is a shared fixed cost.
    lex_cells = sum(len(d) for d in getattr(g, "_cat_word_prob", {}).values())

    def bits_raw(cap=None, cell_budget=None):
        """Plain frame→filler counts; cap to top-`cap` frames by token OR to a CELL budget; back off to unigram.
        Returns (bpc, stored_cells) — cells = Σ distinct fillers over kept frames (the real memory)."""
        if cell_budget is not None:
            kept = cap_frames_by_cells(list(fc.keys()), lambda k: fc[k][1].sum(),
                                       lambda k: len(fc[k][0]), cell_budget)
        else:
            kept = cap_frames(list(fc.keys()), lambda k: fc[k][1].sum(), cap)
        bits = np.empty(len(frames)); bits[:] = -np.log2(uni_p[fillers])
        cells = 0
        for fr in ufr:
            fr = int(fr)
            if fr not in kept or fr not in fc:
                continue
            fids, cs = fc[fr]; tot = cs.sum()
            pos = np.nonzero(frames == fr)[0]; fl = fillers[pos]
            seen = np.isin(fl, fids)
            cmap = {int(w): float(c) for w, c in zip(fids, cs)}
            pv = np.array([(cmap.get(int(x), 0.0) + 0.1) / (tot + 0.1 * Vw) for x in fl])
            bits[pos[seen]] = -np.log2(pv[seen])                 # only override where the pair was attested
        cells = sum(len(fc[fr][0]) for fr in (kept & set(fc.keys())))
        return float(bits.mean()), cells

    def bits_concept(cap=None, cell_budget=None, with_raw_backoff=None):
        """Compositional head P(w|frame)=Σ_c P(c|frame)P(w|c) for open-slot frames (cap by count OR cell budget);
        these generalize to fillers the frame never hosted. `with_raw_backoff`: raw frames consulted FIRST
        (hybrid: literal where attested, category where not). Misses → unigram.
        Returns (bpc, stored_cells) — cells = head-cats + shared lexicon (+ raw cells if hybrid)."""
        headcells = lambda k: int((g.cat_tab[k] > 0).sum()) if k in g.cat_tab else 0
        if cell_budget is not None:
            heads = cap_frames_by_cells(list(g.cat_tab.keys()),
                                        lambda k: g.frames[k].token if k in g.frames else 0.0,
                                        headcells, max(1, cell_budget - lex_cells))
        else:
            heads = cap_frames(list(g.cat_tab.keys()),
                               lambda k: g.frames[k].token if k in g.frames else 0.0, cap)
        bits = np.empty(len(frames)); bits[:] = -np.log2(uni_p[fillers])
        for fr in ufr:
            fr = int(fr)
            pos = np.nonzero(frames == fr)[0]; fl = fillers[pos]
            done = np.zeros(len(pos), bool)
            if with_raw_backoff is not None and fr in with_raw_backoff and fr in fc:
                fids, cs = fc[fr]; tot = cs.sum()
                seen = np.isin(fl, fids)
                cmap = {int(w): float(c) for w, c in zip(fids, cs)}
                pv = np.array([(cmap.get(int(x), 0.0) + 0.1) / (tot + 0.1 * Vw) for x in fl])
                bits[pos[seen]] = -np.log2(pv[seen]); done |= seen
            if fr in heads:
                pred = g.predict_filler_via_category(fr)
                if pred:
                    pv = np.array([pred.get(int(x)) for x in fl], dtype=object)
                    hit = np.array([x is not None for x in pv]) & ~done
                    if hit.any():
                        bits[pos[hit]] = -np.log2(np.maximum(np.array([p for p in pv[hit]], float), 1e-12))
        head_cells = sum(int((g.cat_tab[fr] > 0).sum()) for fr in heads if fr in g.cat_tab)
        raw_cells = (sum(len(fc[fr][0]) for fr in with_raw_backoff if fr in fc)
                     if with_raw_backoff is not None else 0)
        return float(bits.mean()), head_cells + lex_cells + raw_cells

    # UNBOUNDED: full raw counts vs full concept heads (the prior-finding axis).
    off_unb, _ = bits_raw(); on_unb, _ = bits_concept()
    d_unb = off_unb - on_unb

    # BOUNDED: a FIXED CELL budget. OFF spends it all on raw frame→filler cells. ON spends HALF on raw cells and
    # the rest on concept heads + the shared category lexicon (hybrid: literal where attested, category where
    # not). Fair memory in stored cells. Does buying concepts beat buying more literal frames at equal cells?
    cell_budget = budget
    off_bnd, off_E = bits_raw(cell_budget=cell_budget)
    raw_half = cap_frames_by_cells(list(fc.keys()), lambda k: fc[k][1].sum(),
                                   lambda k: len(fc[k][0]), cell_budget // 2)
    raw_half_cells = sum(len(fc[k][0]) for k in raw_half)
    on_bnd, on_E = bits_concept(cell_budget=cell_budget - raw_half_cells, with_raw_backoff=raw_half)
    d_bnd = off_bnd - on_bnd

    return dict(name="word-concept generalization", metric="bits/word (lower better)",
                off_unb=off_unb, on_unb=on_unb, d_unb=d_unb,
                off_bnd=off_bnd, on_bnd=on_bnd, d_bnd=d_bnd,
                off_E=off_E, on_E=on_E, budget=budget,
                n_categories=Ccl, n_open_slot=sum(1 for v in g.label.values() if v == "open-slot"))
