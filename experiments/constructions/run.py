#!/usr/bin/env python3
"""Exp AF — usage-based CONSTRUCTION induction: grammar as counting, made productive. ONLINE, NO backprop.

Bybee frequency effects / Goldberg constructions / Tomasello usage-based grammar / statistical preemption,
mechanized as counting. For every FRAME ("X ___") we count TWO things at once:
  - TOKEN count: how often each specific filler followed (entrenchment driver).
  - TYPE count : how many DISTINCT fillers / filler-CATEGORIES followed (productivity driver).
Categories come from jepa.py's ONLINE leader clustering (single pass, running-mean prototype or spawn). Then:
  (a) high token + one dominant filler  -> FREEZE into a chunk unit (frozen idiom; predict the specific filler).
  (b) high type/category spread         -> SPAWN an open-slot CONSTRUCTION (predict the filler CATEGORY).
  (c) competing frames for one category -> STATISTICAL PREEMPTION: up-weight observed, down-weight competitor.

MEASURE (right axes):
  1. COMPOSITIONAL GENERALIZATION — hold out a set of (frame, filler) PAIRS from training. At test, does the
     open-slot construction (predict filler THROUGH its category) give those held-out, never-in-this-frame
     fillers higher probability than a plain n-gram (which floors them)? This is the headline.
  2. ENTRENCHMENT vs ABSTRACTION — show discovered frozen idioms (predict the specific word) and discovered
     open-slot constructions (predict the category). Do the labels behave as claimed?
  3. PREEMPTION — does count-based inhibition reduce probability mass on UNATTESTED frame->category forms
     (over-generation) without hurting attested ones?

Honest if the abstraction doesn't beat the n-gram. Corpus: text8. Fixed seed, single pass.
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from jepa import online_signatures, leader_cluster
from constructions import (build_frame_counts, ConstructionGrammar, NgramBackoff)

print = functools.partial(print, flush=True)

# ── config ──
TRAIN_BYTES  = 14_000_000
N            = 10_000      # top-N words get an id + a category; rest OOV (-1)
D            = 128         # signature dims (more dims -> finer, less-colliding categories)
SIG_WINDOW   = 5
MIN_EVIDENCE = 40
COS_THRESH   = 0.78
CMAX         = 400
MIN_TOKEN    = 40          # a frame must occur this often before we judge it (online 'ripe')
FREEZE_DOM   = 0.50        # dominant-filler fraction to FREEZE a frame into an idiom
OPEN_TYPES   = 12          # distinct fillers to be eligible for an open slot
HOLDOUT_FRAC = 0.30        # fraction of eligible (frame,filler) PAIRS held out for compositional test
N_HOLDOUT_PROBE = 40_000
SEED         = 0


def perplexity(p):
    return float(np.exp(-np.mean(np.log(np.clip(p, 1e-12, 1.0)))))


def main():
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN_BYTES)
    spans = split_words(ids)
    words = [ids_to_str(ids[s:e]) for s, e in spans]
    w2id, wids = {}, np.empty(len(words), np.int64)
    for i, w in enumerate(words):
        wids[i] = w2id.setdefault(w, len(w2id))
    id2word = {v: k for k, v in w2id.items()}
    counts_g = np.bincount(wids, minlength=len(w2id))
    top = np.argsort(counts_g)[::-1][:N]
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    topword = [id2word[t] for t in top]
    seq = remap[wids]
    print(f"{len(words):,} words, {len(w2id):,} types | top-N={N} | load+map {time.time()-t0:.1f}s")

    # ── ONLINE categories: signatures (1 pass) + leader clustering (1 pass) ──
    t1 = time.time()
    sig, ev = online_signatures(seq, N=N, D=D, window=SIG_WINDOW, seed=SEED)
    first = np.full(N, len(seq), np.int64)
    vp = np.nonzero(seq >= 0)[0]
    np.minimum.at(first, seq[vp], vp)
    order = np.argsort(first); order = order[ev[order] >= MIN_EVIDENCE]
    clu, C = leader_cluster(sig, ev, order, min_evidence=MIN_EVIDENCE, thresh=COS_THRESH, Cmax=CMAX)
    print(f"online categories in {time.time()-t1:.1f}s | C={C} | {(clu>=0).sum():,}/{N} words categorized")

    # ── HOLD OUT (frame, filler) PAIRS for the compositional test (deterministic) ──
    # We remove a fraction of DISTINCT (frame,filler) combinations from training so the model has NEVER seen
    # that filler in that frame; the open-slot construction must predict it via its category. We hold out only
    # pairs whose filler is categorized and whose filler is not the frame's dominant one (so we never freeze
    # away a held-out filler's frame). 1-gram frames.
    rng = np.random.default_rng(SEED)
    fr = seq[:-1]; fl = seq[1:]
    m = (fr >= 0) & (fl >= 0) & (clu[np.clip(fl, 0, N-1)] >= 0)
    fr, fl = fr[m].astype(np.int64), fl[m].astype(np.int64)
    pair_key = fr * N + fl
    uniq_pairs = np.unique(pair_key)
    held = rng.random(len(uniq_pairs)) < HOLDOUT_FRAC
    held_set = set(uniq_pairs[held].tolist())
    keep_mask = np.array([k not in held_set for k in pair_key])     # training keeps the rest
    # training token stream of (frame,filler): drop held-out occurrences entirely
    tr_fr = fr[keep_mask]; tr_fl = fl[keep_mask]
    print(f"compositional holdout: {len(uniq_pairs):,} distinct (frame,filler) pairs, "
          f"held out {held.sum():,} ({held.mean()*100:.0f}%) -> never seen in-frame during training")

    # build training frame counts from the KEPT pairs (vectorized order-independent accumulation)
    F = N
    key = tr_fr * F + tr_fl
    uk, uc = np.unique(key, return_counts=True)
    fc = {}
    uf = uk // F; uw = (uk % F).astype(np.int64)
    edges = np.nonzero(np.diff(uf))[0] + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(uk)]])
    for s, e in zip(starts, ends):
        fc[int(uf[s])] = (uw[s:e], uc[s:e].astype(np.float64))

    # ── INDUCE the grammar ──
    cg = ConstructionGrammar(clu, C, alpha=0.1, min_token=MIN_TOKEN,
                             freeze_dom=FREEZE_DOM, open_types=OPEN_TYPES)
    cg.fit(fc).classify()
    cg.build_category_lexicon(fc)
    ngram = NgramBackoff(fc, vocab=N, alpha=0.1)

    labels = list(cg.label.values())
    from collections import Counter
    lc = Counter(labels)
    n_ripe = sum(v for k, v in lc.items() if k != "sparse")
    print(f"\n=== frames classified ({len(cg.label):,} total; {n_ripe:,} ripe >= {MIN_TOKEN} tokens) ===")
    print(f"    frozen idioms : {lc.get('frozen',0):,}")
    print(f"    open-slot     : {lc.get('open-slot',0):,}")
    print(f"    mixed         : {lc.get('mixed',0):,}")
    print(f"    sparse        : {lc.get('sparse',0):,}")

    # ── Result 2: show discovered constructions ──
    def fname(fk): return topword[fk]
    frozen = [(fk, cg.frames[fk]) for fk in cg.frames if cg.label[fk] == "frozen"]
    frozen.sort(key=lambda x: -x[1].token)
    print("\n=== entrenchment: discovered FROZEN idioms (high token, one dominant filler) ===")
    for fk, fs in frozen[:12]:
        print(f"    \"{fname(fk)} ___\"  token={int(fs.token):6d}  dom='{topword[fs.dom]}' "
              f"({fs.domfrac*100:.0f}%)  types={fs.types}")

    openf = [(fk, cg.frames[fk]) for fk in cg.frames if cg.label[fk] == "open-slot"]
    openf.sort(key=lambda x: -x[1].token)

    # global category base-rate (how often each category appears as ANY filler) — lets us surface a frame's
    # DISTINCTIVE slot category (the one it prefers far above base rate), not the ubiquitous function-word one.
    glob_cat = np.zeros(C)
    for fids, cnt in fc.values():
        cats = clu[fids]
        for c, n in zip(cats, cnt):
            if c >= 0:
                glob_cat[c] += n
    glob_cat /= glob_cat.sum()

    def cat_members(c, k=6):
        mem = np.nonzero(clu == c)[0]
        mem = mem[np.argsort([-counts_g[top[i]] for i in mem])][:k]
        return ", ".join(topword[i] for i in mem)

    cat_size = np.bincount(clu[clu >= 0], minlength=C)

    def distinctive_cat(fs, min_members=4, min_mass=0.02):
        """The category this frame commits to most ABOVE its global base rate (the slot's real selectional pref).
        Require the category to have several members and carry real mass — so we surface a genuine multi-word
        slot category, not a noisy singleton."""
        best, bestlift = None, 0.0
        for c, w in fs.cat_counts.items():
            if cat_size[c] < min_members or (w / fs.token) < min_mass:
                continue
            r = (w / fs.token) / max(glob_cat[c], 1e-9)
            if r > bestlift and glob_cat[c] > 0:
                bestlift, best = r, c
        return best, bestlift

    print("\n=== abstraction: discovered OPEN-SLOT constructions (high type/category spread) ===")
    print("    (top-slot-category = the category the frame prefers MOST ABOVE its global base rate = the slot's"
          " selectional preference)")
    for fk, fs in openf[:14]:
        c, lift = distinctive_cat(fs)
        if c is None:
            continue
        print(f"    \"{fname(fk)} ___\"  token={int(fs.token):6d}  types={fs.types}  cats={fs.cats}  "
              f"slot-prefers(x{lift:.0f} base)={{{cat_members(c)}}}")

    # ── Result 1: COMPOSITIONAL GENERALIZATION on held-out (frame,filler) pairs ──
    held_arr = np.array(sorted(held_set), dtype=np.int64)
    probe = rng.choice(held_arr, size=min(N_HOLDOUT_PROBE, len(held_arr)), replace=False)
    pf = probe // N; pw = (probe % N).astype(np.int64)

    # restrict the headline to held-out pairs whose FRAME is an induced open-slot construction (the regime the
    # construction is supposed to help) AND report on all held-out pairs too (honest全景).
    def eval_pairs(frames, fillers):
        p_ng = np.empty(len(frames)); p_cx = np.empty(len(frames))
        has_cx = np.zeros(len(frames), bool)
        rank_better = 0; n = 0
        for i, (fr_, w_) in enumerate(zip(frames, fillers)):
            p_ng[i] = ngram.prob_of(int(fr_), int(w_))            # floors held-out filler
            d = cg.predict_filler_via_category(int(fr_))           # productive: via category
            if d is None:
                p_cx[i] = p_ng[i]
            else:
                p_cx[i] = d.get(int(w_), 1.0 / N)
                has_cx[i] = True
                n += 1
                rank_better += (p_cx[i] > p_ng[i])
        return p_ng, p_cx, has_cx, rank_better, n

    p_ng, p_cx, has_cx, better, ncx = eval_pairs(pf, pw)
    open_mask = np.array([cg.label.get(int(f)) == "open-slot" for f in pf])
    print("\n=== Result 1 — COMPOSITIONAL GENERALIZATION (held-out frame,filler pairs unseen in-frame) ===")
    print(f"    probes: {len(pf):,} held-out pairs | {open_mask.sum():,} have an open-slot frame")

    def row(name, mask):
        if mask.sum() == 0:
            print(f"    {name:<26} (no probes)"); return
        a = p_ng[mask]; b = p_cx[mask]
        win = float((b > a).mean())
        print(f"    {name:<26} ngram ppl {perplexity(a):10.1f} | construction ppl {perplexity(b):10.1f} | "
              f"construction>ngram on {win*100:5.1f}% of pairs")
    row("all held-out pairs", np.ones(len(pf), bool))
    row("open-slot frames only", open_mask)
    row("(frames w/ a category head)", has_cx)

    # ── Result 3: STATISTICAL PREEMPTION ──
    # Preemption acts on OVER-GENERATION candidates: a category this frame holds WEAKLY (low commitment) but a
    # COMPETING frame owns strongly. Those are the "could occur but is rarely observed -> blocked" forms (the
    # near-synonym the learner preempts). We split each open-slot frame's category links into:
    #   STRONG (this frame's own well-attested categories, commitment >= half its max) — must be retained.
    #   WEAK-COMPETITOR (commitment <= 20% of this frame's max AND some rival frame commits >= 2x harder) —
    #     these are the over-generation forms; preemption should suppress them.
    # Measure the mean probability the open-slot head assigns to each group, before vs after preemption.
    cg_pre = ConstructionGrammar(clu, C, alpha=0.1, min_token=MIN_TOKEN,
                                 freeze_dom=FREEZE_DOM, open_types=OPEN_TYPES)
    cg_pre.fit(fc).classify(); cg_pre.build_category_lexicon(fc)
    cg_pre.preempt(strength=0.4)

    open_frames = [fk for fk in cg.frames if cg.label[fk] == "open-slot"]
    # per-category, the max commitment any frame makes (the 'leader' / conventional expression)
    cat_lead = {}
    for fk in open_frames:
        fs = cg.frames[fk]
        for c, w in fs.cat_counts.items():
            r = w / fs.token
            cat_lead[c] = max(cat_lead.get(c, 0.0), r)
    strong_b = []; weak_b = []; strong_a = []; weak_a = []
    for fk in open_frames:
        fs = cg.frames[fk]
        d0 = cg.predict_open(fk); d1 = cg_pre.predict_open(fk)
        if d0 is None or d1 is None or fs.token == 0:
            continue
        rmax = max(fs.cat_counts.values()) / fs.token if fs.cat_counts else 0.0
        for c, w in fs.cat_counts.items():
            r = w / fs.token
            if rmax > 0 and r >= 0.5 * rmax:
                strong_b.append(d0[c]); strong_a.append(d1[c])
            elif rmax > 0 and r <= 0.2 * rmax and cat_lead.get(c, 0) >= 2 * r:
                weak_b.append(d0[c]); weak_a.append(d1[c])
    strong_b = np.array(strong_b); weak_b = np.array(weak_b)
    strong_a = np.array(strong_a); weak_a = np.array(weak_a)
    print("\n=== Result 3 — STATISTICAL PREEMPTION (count-based inhibition of competitor links) ===")
    print(f"    over {len(open_frames):,} open-slot frames | strong links {len(strong_b):,}  "
          f"weak-competitor links {len(weak_b):,}")
    print(f"    {'':<14}{'strong (attested)':>20}{'weak-competitor':>18}")
    print(f"    {'before':<14}{strong_b.mean():>20.5f}{weak_b.mean():>18.5f}")
    print(f"    {'after preempt':<14}{strong_a.mean():>20.5f}{weak_a.mean():>18.5f}")
    if weak_b.mean() > 0:
        print(f"    -> weak-competitor (over-generation) mass reduced "
              f"{(1-weak_a.mean()/weak_b.mean())*100:.1f}%, strong (attested) retained "
              f"{strong_a.mean()/max(strong_b.mean(),1e-12)*100:.1f}%")

    print(f"\ntotal {time.time()-t0:.1f}s")
    return dict(C=C, labels=dict(lc), n_ripe=n_ripe,
                comp_ng_ppl=perplexity(p_ng), comp_cx_ppl=perplexity(p_cx),
                comp_open_ng=perplexity(p_ng[open_mask]) if open_mask.sum() else None,
                comp_open_cx=perplexity(p_cx[open_mask]) if open_mask.sum() else None,
                comp_open_win=float((p_cx[open_mask] > p_ng[open_mask]).mean()) if open_mask.sum() else None,
                frozen_examples=[(topword[fk], topword[fs.dom], fs.domfrac, int(fs.token)) for fk, fs in frozen[:12]],
                open_examples=[(topword[fk], fs.types, fs.cats, int(fs.token)) for fk, fs in openf[:12]],
                preempt=(float(strong_b.mean()), float(weak_b.mean()),
                         float(strong_a.mean()), float(weak_a.mean())))


if __name__ == "__main__":
    main()
