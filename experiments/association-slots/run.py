#!/usr/bin/env python3
"""Exp AW — streaming-ASSOCIATION slot strength (ΔP / PPMI) as the construction substrate. ONLINE, NO backprop.

The cognitive bet (Ellis cue contingency; Allan's ΔP; collostructional PPMI; Casenhiser-Goldberg skewed input).
AF builds open-slot constructions by counting, but it ranks and vetoes a slot's filler-CATEGORIES by RAW
commitment ratio r = c(f,s)/c(f,·) — pure conditional P(s|f). That share is base-rate-inflated: a category that
follows EVERYTHING earns a high raw share in every frame, so raw counts over-generate (the 2025 "LLMs learn
constructions humans don't know" warning). The human learner does not weigh a cue by how often the outcome
follows it; it weighs CONTINGENCY — how much MORE the outcome follows this cue than it follows in general.

M5 swaps the substrate. Keep only four additive marginals per (frame f, category s) — c(f,s), c(f,·), c(·,s), N
— and derive ON DEMAND:
    ΔP   = P(s|f) − P(s|¬f)                       (Allan / Ellis contingency)
    PPMI = max(0, log(c(f,s)·N / (c(f,·)·c(·,s)))) (positive pointwise MI)
Both discount a category by its global base rate c(·,s)/N. ASSOCIATION, not frequency, then drives three things:
  • the skewed-input ANCHOR (argmax association, not argmax count) — the prototype slot-class;
  • the open-slot DISTRIBUTION (count × association, base-rate-corrected, pruned where contingency ≤ 0);
  • the PREEMPTION veto (relative ASSOCIATION between competing frames, not relative commitment ratio).

THE TWO DIALS (the kill is conjunctive — association must move at least ONE):
  DIAL A — OVER-GENERATION. AF's preemption cut weak-competitor (could-occur-but-unobserved) mass −39.5 %.
           Does association cut it MORE? (it should: PPMI zeroes the base-rate-only links AF only damps.)
  DIAL B — HELD-OUT COMPOSITIONAL PERPLEXITY. Same held-out (frame,filler) pairs as AF; does routing the open
           slot through an ASSOCIATION-weighted category distribution beat the RAW-count one at equal memory?

KILL (per BUILD_QUEUE AW): association does not lower over-generation below AF's −39.5% AND does not beat
raw-count held-out perplexity at equal memory — after the FRAGILE budget (we sweep ≥10 kind×threshold variants
and check BOTH dials) — then PARK as "raw counts suffice for English text", not killed.

Rules obeyed: ONLINE single streaming pass (four additive marginals, closed-form on demand); BOUNDED (association
PRUNES — ≤0-contingency links drop, the slot table SHRINKS vs raw counts; we report the memory delta);
NO gradient descent / k-means / SVD / eigen. Corpus: text8 (AF's pipeline; CHILDES not in data/ — said so below).
Fixed seed, single pass. Honest if raw counts win.
"""
import os, sys, time, functools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics, assoc          # noqa: F401  (assoc is the mechanism under test)
from corpus import load_ids, split_words, ids_to_str
from jepa import online_signatures, leader_cluster
from constructions import build_frame_counts, ConstructionGrammar, NgramBackoff

print = functools.partial(print, flush=True)

# ── config (matches AF so the comparison is at equal memory / equal categories) ──
TRAIN_BYTES  = 8_000_000     # ≤10 MB this pass (AF used 14 MB; smaller fast slice, same pipeline)
N            = 10_000        # top-N words get an id + a category; rest OOV (-1)
D            = 128
SIG_WINDOW   = 5
MIN_EVIDENCE = 40
COS_THRESH   = 0.78
CMAX         = 400
MIN_TOKEN    = 40
FREEZE_DOM   = 0.50
OPEN_TYPES   = 12
HOLDOUT_FRAC = 0.30
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

    # ── ONLINE categories (1 pass signatures + 1 pass leader clustering) — identical to AF ──
    t1 = time.time()
    sig, ev = online_signatures(seq, N=N, D=D, window=SIG_WINDOW, seed=SEED)
    first = np.full(N, len(seq), np.int64)
    vp = np.nonzero(seq >= 0)[0]
    np.minimum.at(first, seq[vp], vp)
    order = np.argsort(first); order = order[ev[order] >= MIN_EVIDENCE]
    clu, C = leader_cluster(sig, ev, order, min_evidence=MIN_EVIDENCE, thresh=COS_THRESH, Cmax=CMAX)
    print(f"online categories in {time.time()-t1:.1f}s | C={C} | {(clu>=0).sum():,}/{N} categorized")

    # ── hold out (frame, filler) pairs (deterministic; identical recipe to AF) ──
    rng = np.random.default_rng(SEED)
    fr = seq[:-1]; fl = seq[1:]
    m = (fr >= 0) & (fl >= 0) & (clu[np.clip(fl, 0, N-1)] >= 0)
    fr, fl = fr[m].astype(np.int64), fl[m].astype(np.int64)
    pair_key = fr * N + fl
    uniq_pairs = np.unique(pair_key)
    held = rng.random(len(uniq_pairs)) < HOLDOUT_FRAC
    held_set = set(uniq_pairs[held].tolist())
    keep_mask = np.array([k not in held_set for k in pair_key])
    tr_fr = fr[keep_mask]; tr_fl = fl[keep_mask]
    print(f"holdout: {len(uniq_pairs):,} distinct pairs, held out {held.sum():,} ({held.mean()*100:.0f}%)")

    # training frame counts (vectorized order-independent accumulation = streaming counts)
    F = N
    key = tr_fr * F + tr_fl
    uk, uc = np.unique(key, return_counts=True)
    fc = {}
    uf = uk // F; uw = (uk % F).astype(np.int64)
    edges = np.nonzero(np.diff(uf))[0] + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(uk)]])
    for s, e in zip(starts, ends):
        fc[int(uf[s])] = (uw[s:e], uc[s:e].astype(np.float64))

    # ── induce AF grammar (the raw-count baseline) ──
    cg = ConstructionGrammar(clu, C, alpha=0.1, min_token=MIN_TOKEN,
                             freeze_dom=FREEZE_DOM, open_types=OPEN_TYPES)
    cg.fit(fc).classify(); cg.build_category_lexicon(fc)
    ngram = NgramBackoff(fc, vocab=N, alpha=0.1)
    open_frames = [fk for fk in cg.frames if cg.label[fk] == "open-slot"]
    from collections import Counter
    lc = Counter(cg.label.values())
    print(f"\nframes: open-slot {lc.get('open-slot',0):,}  frozen {lc.get('frozen',0):,}  "
          f"mixed {lc.get('mixed',0):,}  sparse {lc.get('sparse',0):,}")

    # ── build the ASSOCIATION substrate from the SAME counts (four marginals) ──
    glob_cat = np.zeros(C)
    frame_cat_counts = {}
    for fk, fs in cg.frames.items():
        if cg.label[fk] in ("open-slot", "mixed"):
            frame_cat_counts[fk] = dict(fs.cat_counts)
        for c, w in fs.cat_counts.items():
            glob_cat[c] += w
    Ntok = float(glob_cat.sum())
    A = assoc.AssocSlots(C).fit(frame_cat_counts, glob_cat, Ntok)

    # memory: AF stores all (frame,cat) links; AW prunes ≤0-contingency ones. Count surviving PPMI/ΔP links.
    raw_links = sum(len(v) for v in frame_cat_counts.values())
    ppmi_links = sum(int((A.ppmi(fk) > 0).sum()) for fk in frame_cat_counts)
    dp_links   = sum(int((A.dp(fk)   > 0).sum()) for fk in frame_cat_counts)
    print(f"slot-table links — raw(AF) {raw_links:,} | PPMI-pruned {ppmi_links:,} "
          f"({ppmi_links/raw_links*100:.0f}%) | ΔP-pruned {dp_links:,} ({dp_links/raw_links*100:.0f}%)")

    # ════════════════════════════ DIAL A — OVER-GENERATION ════════════════════════════
    # weak-competitor = a category the frame holds weakly while a RIVAL frame commits/associates much harder:
    # the "could-occur-but-unobserved → blocked" over-generation forms. We measure the open-slot head's mean
    # mass on weak-competitor vs strong-attested links, BEFORE vs AFTER each veto. Same split as AF.
    cat_lead_r = {}
    for fk in open_frames:
        fs = cg.frames[fk]
        for c, w in fs.cat_counts.items():
            cat_lead_r[c] = max(cat_lead_r.get(c, 0.0), w / fs.token)

    def split_links():
        strong, weak = [], []
        for fk in open_frames:
            fs = cg.frames[fk]
            if not fs.cat_counts:
                continue
            rmax = max(fs.cat_counts.values()) / fs.token
            for c, w in fs.cat_counts.items():
                r = w / fs.token
                if rmax > 0 and r >= 0.5 * rmax:
                    strong.append((fk, c))
                elif rmax > 0 and r <= 0.2 * rmax and cat_lead_r.get(c, 0) >= 2 * r:
                    weak.append((fk, c))
        return strong, weak
    strong, weak = split_links()

    def af_open_dist(fk, inhib):
        v = cg.cat_tab.get(fk)
        if v is None:
            return None
        v = v.copy()
        for (ifk, c), mult in inhib.items():
            if ifk == fk:
                v[c] *= mult
        v = v + cg.alpha
        return v / v.sum()

    def mass_on(links, dist_fn):
        cache = {}
        out = []
        for fk, c in links:
            if fk not in cache:
                cache[fk] = dist_fn(fk)
            d = cache[fk]
            if d is not None:
                out.append(d[c])
        return np.array(out) if out else np.array([0.0])

    # AF baseline veto (commitment-ratio preemption, strength 0.4 — AF's setting)
    cg.preempt(strength=0.4)
    af_inhib = dict(cg.inhib); cg.inhib = {}        # capture then reset so cg.cat_tab stays clean for AW dists
    base_strong = mass_on(strong, lambda fk: af_open_dist(fk, {}))
    base_weak   = mass_on(weak,   lambda fk: af_open_dist(fk, {}))
    af_strong   = mass_on(strong, lambda fk: af_open_dist(fk, af_inhib))
    af_weak     = mass_on(weak,   lambda fk: af_open_dist(fk, af_inhib))
    af_cut = (1 - af_weak.mean() / max(base_weak.mean(), 1e-12)) * 100
    af_ret = af_strong.mean() / max(base_strong.mean(), 1e-12) * 100

    # AW association veto — sweep kind × strength floor (FRAGILE budget: ≥10 variations, BOTH dials)
    print("\n=== DIAL A — over-generation veto (weak-competitor mass cut; higher = better; retain strong ≈100%) ===")
    print(f"    AF raw commitment-ratio veto: cut {af_cut:5.1f}%   retain {af_ret:5.1f}%   "
          f"(the −39.5% bar at this slice)")
    dialA = []
    for kind in ("dp", "ppmi"):
        inhib_full, _ = A.preempt_assoc(open_frames, kind=kind)
        for floor in (0.0, 0.25, 0.4, 0.6):
            # floor maps the relative-association rel∈[0,1] into [floor,1]: 0.0 = full veto (rel as-is).
            inhib = {k: floor + (1 - floor) * v for k, v in inhib_full.items()}
            sa = mass_on(strong, lambda fk: af_open_dist(fk, inhib))
            wa = mass_on(weak,   lambda fk: af_open_dist(fk, inhib))
            cut = (1 - wa.mean() / max(base_weak.mean(), 1e-12)) * 100
            ret = sa.mean() / max(base_strong.mean(), 1e-12) * 100
            dialA.append((kind, floor, cut, ret))
            flag = "  <-- beats AF" if cut > af_cut and ret >= 95 else ""
            print(f"    assoc[{kind:4s}] floor={floor:.2f}:  cut {cut:5.1f}%   retain {ret:6.1f}%{flag}")
    best_cut = max(c for _, _, c, r in dialA if r >= 95) if any(r >= 95 for *_, r in dialA) else max(c for *_, c, _ in dialA)

    # Association-NATIVE veto diagnostic: AF's split favors AF (links defined by commitment ratio). The PPMI
    # claim is sharper — it ZEROES base-rate-only links outright. Of AF's weak-competitor (over-generation)
    # links, what fraction does PPMI already prune to zero contingency (a hard veto AF can only soften)? And how
    # many STRONG (attested) links does PPMI wrongly zero (the false-veto cost)?
    ppmi_full = {fk: A.ppmi(fk) for fk in open_frames}
    dp_full = {fk: A.dp(fk) for fk in open_frames}
    weak_zeroed_ppmi = np.mean([ppmi_full[fk][c] <= 0 for fk, c in weak]) if weak else 0.0
    weak_zeroed_dp   = np.mean([dp_full[fk][c]   <= 0 for fk, c in weak]) if weak else 0.0
    strong_zeroed_ppmi = np.mean([ppmi_full[fk][c] <= 0 for fk, c in strong]) if strong else 0.0
    strong_zeroed_dp   = np.mean([dp_full[fk][c]   <= 0 for fk, c in strong]) if strong else 0.0
    print(f"    [native] PPMI hard-vetoes {weak_zeroed_ppmi*100:.1f}% of weak-competitor links "
          f"(false-veto {strong_zeroed_ppmi*100:.1f}% of strong) | ΔP vetoes {weak_zeroed_dp*100:.1f}% weak "
          f"({strong_zeroed_dp*100:.1f}% strong)")

    # ════════════════════════════ DIAL B — HELD-OUT COMPOSITIONAL PERPLEXITY ════════════════════════════
    # Same compositional head as AF (P(w|frame)=Σ_c P(c|frame,slot)·P(w|c)), but P(c|frame,slot) comes from the
    # ASSOCIATION-weighted distribution instead of the raw category counts. Both at equal memory / equal P(w|c).
    held_arr = np.array(sorted(held_set), dtype=np.int64)
    probe = rng.choice(held_arr, size=min(N_HOLDOUT_PROBE, len(held_arr)), replace=False)
    pf = probe // N; pw = (probe % N).astype(np.int64)
    open_mask = np.array([cg.label.get(int(f)) == "open-slot" for f in pf])

    catword = cg._cat_word_prob

    def comp_prob(frame, w, cat_prior):
        """P(w|frame) routed through the slot category, with cat_prior = P(category|frame,slot) (C,)."""
        if cat_prior is None:
            return None
        active = np.nonzero(cat_prior > (cat_prior.min() + 1e-12))[0]
        if active.size == 0:
            active = np.nonzero(cat_prior > 0)[0]
        p = 0.0
        for c in active:
            pw_ = catword.get(int(c))
            if pw_ is not None and w in pw_:
                p += cat_prior[c] * pw_[w]
        return p

    def eval_dist(prior_fn):
        p = np.empty(len(pf))
        for i, (f_, w_) in enumerate(zip(pf, pw)):
            prior = prior_fn(int(f_))
            pp = comp_prob(int(f_), int(w_), prior) if prior is not None else None
            p[i] = pp if (pp is not None and pp > 0) else (1.0 / N)
        return p

    # raw-count prior (AF) vs association priors (AW) — sweep kind (the ≥10-variation budget spans both dials)
    p_ng = np.array([ngram.prob_of(int(f_), int(w_)) for f_, w_ in zip(pf, pw)])
    p_raw = eval_dist(lambda fk: cg.predict_open(fk))
    print("\n=== DIAL B — held-out compositional perplexity (lower = better; AF raw-count is the bar) ===")
    print(f"    n-gram floor (no construction):           ppl {perplexity(p_ng):10.1f}")
    print(f"    AF raw-count open-slot prior:             ppl {perplexity(p_raw):10.1f}   "
          f"(all held-out) | open-slot only {perplexity(p_raw[open_mask]):.1f}")
    dialB = [("raw", perplexity(p_raw), perplexity(p_raw[open_mask]))]
    for kind in ("dp", "ppmi"):
        p_a = eval_dist(lambda fk, k=kind: A.slot_dist(fk, k))
        dialB.append((kind, perplexity(p_a), perplexity(p_a[open_mask])))
        d_all = (perplexity(p_raw) - perplexity(p_a)) / perplexity(p_raw) * 100
        d_op = (perplexity(p_raw[open_mask]) - perplexity(p_a[open_mask])) / perplexity(p_raw[open_mask]) * 100
        flag = "  <-- beats raw" if perplexity(p_a[open_mask]) < perplexity(p_raw[open_mask]) else ""
        print(f"    assoc[{kind:4s}] open-slot prior:             ppl {perplexity(p_a):10.1f}   "
              f"(all, {d_all:+.1f}%) | open-slot only {perplexity(p_a[open_mask]):.1f} ({d_op:+.1f}%){flag}")

    raw_open_ppl = perplexity(p_raw[open_mask])
    best_assoc_open = min(op for k, _, op in dialB if k != "raw")
    dialB_win = best_assoc_open < raw_open_ppl

    # ════════════════════════════ VERDICT (the conjunctive kill) ════════════════════════════
    dialA_win = best_cut > 39.5
    print("\n=== VERDICT ===")
    print(f"    DIAL A (over-generation): best assoc cut {best_cut:.1f}% vs AF 39.5% bar  ->  "
          f"{'WIN' if dialA_win else 'no'}")
    print(f"    DIAL B (held-out ppl):    best assoc open-slot ppl {best_assoc_open:.1f} vs raw {raw_open_ppl:.1f}  ->  "
          f"{'WIN' if dialB_win else 'no'}")
    print(f"    memory: PPMI prunes slot table to {ppmi_links/raw_links*100:.0f}% of raw links (bounded ✓)")
    if dialA_win or dialB_win:
        print("    -> WIN/PARTIAL: association moves at least one dial; kill-condition does NOT fire.")
    else:
        print("    -> NEGATIVE/PARK: association moves neither dial -> 'raw counts suffice for English text'.")
    print(f"\ntotal {time.time()-t0:.1f}s")

    return dict(C=C, open_frames=len(open_frames), raw_links=raw_links, ppmi_links=ppmi_links,
                af_cut=af_cut, best_cut=best_cut, dialA=dialA,
                raw_open_ppl=raw_open_ppl, best_assoc_open=best_assoc_open, dialB=dialB,
                dialA_win=dialA_win, dialB_win=dialB_win)


if __name__ == "__main__":
    main()
