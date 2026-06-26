#!/usr/bin/env python3
"""Exp AX — the function-word anchor voter: a free top-k frequency category bootstrap. ONLINE, NO backprop.

Cognitive bet (Zipf closed-class; the determiner as the cheapest POS seed; Mintz's frequent frames). The
toddler is handed FREQUENCY RANK for free. The most-frequent handful of tokens IS the closed class — the,
of, to, and, a — and the closed class is exactly the set whose RIGHT NEIGHBOUR'S CATEGORY it predicts
("the ___" → noun-ish, "to ___" → verb-ish). So mine the top-k word-ids by raw count as ANCHORS (no
labels), keep each anchor's right/left-neighbour CATEGORY tally, and feed "follows-anchor-a" as one
counted cue (validity v = hits/(hits+misses)) into take-the-best (AJ) beside the AF frame cue.

We ask the three questions the spec names:
  1. POS-CLUSTER PURITY of the anchor voter vs the AF frame voter vs the two combined under AJ.
     (gold POS = a small hand-built English closed/open-class lexicon — there is no POS tagger in the
     substrate, and a hand lexicon is a legitimate GOLD for purity; see GOLD below.)
  2. NEXT-WORD PERPLEXITY where the anchor cue FIRES (prev is an anchor) — anchor vs AF vs combined.
  3. THE HONEST NEGATIVE. The spec asks for a German slice (anchors should degrade). data/ has no German,
     so we SUBSTITUTE a principled within-English control: a WORD-ORDER-SHUFFLED text8 stream. Shuffling
     destroys adjacency while preserving the SAME frequency ranks, so the anchor band is identical but its
     right-neighbour category signal is GONE — if the anchor cue is real it must collapse to chance here
     and the AJ-combined result must NOT degrade (AF/anchor down-validitied → AJ ignores the dead cue).
     Reported as a SUBSTITUTION, not the German test the spec literally names.

Kill (per BUILD_QUEUE AX): anchor adds <2% purity over AF on English AND degrades the AJ-combined result
on either language. Per FRAGILE we sweep k (the anchor-band size) before judging — a mis-sized band, not
a dead idea, is the first thing to rule out. Corpus text8. Fixed seed, single streaming pass.
"""
import os, sys, time, math, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus, harness, metrics            # noqa: F401  (substrate imports the spec asks for)
import cortex                              # noqa: F401
from jepa import online_signatures, leader_cluster
from constructions import build_frame_counts, ConstructionGrammar
import funcanchor as fa

print = functools.partial(print, flush=True)

# ── config ──
TRAIN_BYTES  = 14_000_000     # text8 prefix (~2.5M words) — fast first-pass slice, <10MB on disk
N            = 10_000         # top-N words get a dense id + a category; rest OOV (-1)
D            = 128            # signature dims
SIG_WINDOW   = 5
MIN_EVIDENCE = 40
COS_THRESH   = 0.78
CMAX         = 400
MIN_TOKEN    = 40            # AF: a frame must occur this often to be 'ripe'
FREEZE_DOM   = 0.50
OPEN_TYPES   = 12
K_SWEEP      = [5, 10, 20, 30, 50, 80, 120]   # FRAGILE: sweep the anchor-band size before judging
K_MAIN       = 20            # the spec's "~20 tokens" band, for the firing-perplexity table
EVAL_FRAC    = 0.15          # last 15% of the word stream is held-out eval
SEED         = 0


# ── GOLD POS lexicon (hand-built; the only ground truth, used solely to MEASURE purity) ───────────
# A legitimate gold for clustering purity: a few hundred unambiguous-enough common English words tagged by
# their dominant class. NOT used to train anything — anchors and categories are induced label-free; this
# only scores how cleanly the induced categories map onto real POS. Coarse 8-way tag set.
GOLD = {}
def _tag(words, pos):
    for w in words.split():
        GOLD[w] = pos
_tag("the a an this that these those some any each every all both no another my your his her its our their", "DET")
_tag("of in to for with on at by from as into about over under between through during before after against "
     "within without upon toward towards among across behind beyond near", "PREP")
_tag("i you he she it we they me him her us them who which what whom whose someone anyone everyone nobody", "PRON")
_tag("and or but nor so yet because although though while whereas unless if when whenever since until", "CONJ")
_tag("is are was were be been being am has have had do does did will would can could shall should may might must", "AUX")
_tag("man men woman people child children world time year years day days life water work house water hand "
     "head body country state city war god king church land sea fire book name part way thing things place "
     "system group number area money family school student company government law history science nature force", "NOUN")
_tag("make made take took give gave find found think thought know knew see saw come came go went say said "
     "use used call called become became leave left feel felt show showed run ran move moved live lived "
     "believe seem seemed grow grew build built bring brought produce produced develop create form", "VERB")
_tag("great good new old high large small long little young important different early late large early "
     "general public human national social political economic possible whole common natural certain free "
     "true real full main major modern ancient strong weak central royal foreign", "ADJ")
_tag("not very also more most so too then there here now well often always never sometimes usually however "
     "thus therefore indeed perhaps almost rather quite still even just only much less", "ADV")


def perplexity(p):
    return float(np.exp(-np.mean(np.log(np.clip(p, 1e-12, 1.0)))))


def load_stream(nbytes, shuffle=False, rng=None):
    """text8 → dense top-N word id stream (-1 = OOV). If shuffle, randomly permute WORD ORDER (frequency
    ranks unchanged, adjacency destroyed) — the honest-negative control substituting for German."""
    ids = corpus.load_ids("text8", nbytes=nbytes)
    spans = corpus.split_words(ids)
    words = [corpus.ids_to_str(ids[s:e]) for s, e in spans]
    w2id = {}
    wids = np.empty(len(words), np.int64)
    for i, w in enumerate(words):
        wids[i] = w2id.setdefault(w, len(w2id))
    id2word = {v: k for k, v in w2id.items()}
    counts_g = np.bincount(wids, minlength=len(w2id))
    top = np.argsort(counts_g)[::-1][:N]
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    topword = [id2word[t] for t in top]
    seq = remap[wids]
    if shuffle:
        seq = seq.copy(); rng.shuffle(seq)
    return seq, topword


def build_categories(seq):
    """ONLINE categories: 1-pass hashed signatures + 1-pass leader clustering (the AF/jepa substrate)."""
    sig, ev = online_signatures(seq, N=N, D=D, window=SIG_WINDOW, seed=SEED)
    first = np.full(N, len(seq), np.int64)
    vp = np.nonzero(seq >= 0)[0]
    np.minimum.at(first, seq[vp], vp)
    order = np.argsort(first); order = order[ev[order] >= MIN_EVIDENCE]
    clu, C = leader_cluster(sig, ev, order, min_evidence=MIN_EVIDENCE, thresh=COS_THRESH, Cmax=CMAX)
    return clu, C


def gold_of(topword):
    """word-id → gold POS (or None) using the dense top-id → surface-word map."""
    return np.array([GOLD.get(w) for w in topword], dtype=object)


# ── voters: for each eval position predict the NEXT word's category, collect (predicted_cat, gold_pos) ─

def purity_run(label, seq, eval_start, anchor, cg, gpos, k_anchor_silent_to_af=True):
    """Walk eval positions. For each, a voter predicts a (C,) category vote for the NEXT word; we read the
    next word's gold POS and record (argmax_category, gold_pos) for purity. Three voters in one pass:
      anchor  — fires only when prev is an anchor (predict_right)
      frame   — AF open-slot head for the 1-gram frame = prev word (predict_open)
      combo   — AJ take-the-best between them: the higher-VALIDITY cue (counted hit/miss on category) wins,
                noncompensatory; if the winner abstains, fall back to the other.
    Validity here = online category-prediction hit/miss of each cue, accumulated causally over eval (the
    cue's running argmax-category vs the next word's actual category)."""
    # 'frame_on_anchor' = the FAIR control: AF's own prediction, scored on the SAME firing positions as the
    # anchor (purity is biased toward fewer-category / lower-entropy subsets, so anchor's full-eval purity
    # is NOT comparable to frame's full-eval purity — they must be read on the same positions).
    votes = {"anchor": [], "frame": [], "frame_on_anchor": [], "combo": []}
    # online validity tallies (hits/misses on CATEGORY) for the AJ combine
    hit = {"anchor": 0, "frame": 0}; mis = {"anchor": 0, "frame": 0}
    for t in range(eval_start, len(seq) - 1):
        p = int(seq[t]); nxt = int(seq[t + 1])
        if p < 0 or nxt < 0:
            continue
        gp = gpos[nxt] if nxt < len(gpos) else None
        truecat = anchor.clu[nxt] if nxt < len(anchor.clu) else -1

        a_dist = anchor.predict_right(p)
        f_dist = cg.predict_open(p)

        a_cat = int(a_dist.argmax()) if a_dist is not None else None
        f_cat = int(f_dist.argmax()) if f_dist is not None else None

        if a_cat is not None:
            votes["anchor"].append((a_cat, gp))
            if f_cat is not None:
                votes["frame_on_anchor"].append((f_cat, gp))   # AF on the anchor's positions only
        if f_cat is not None:
            votes["frame"].append((f_cat, gp))

        # AJ take-the-best: pick the cue with higher running validity that FIRES; noncompensatory
        va = hit["anchor"] / (hit["anchor"] + mis["anchor"]) if (hit["anchor"] + mis["anchor"]) else 0.5
        vf = hit["frame"]  / (hit["frame"]  + mis["frame"])  if (hit["frame"]  + mis["frame"])  else 0.5
        order = ("anchor", "frame") if va >= vf else ("frame", "anchor")
        chosen = None
        for nm in order:
            d = a_cat if nm == "anchor" else f_cat
            if d is not None:
                chosen = d; break
        if chosen is not None:
            votes["combo"].append((chosen, gp))

        # update validities (causal — after we used them)
        if truecat >= 0:
            if a_cat is not None:
                (hit if a_cat == truecat else mis)["anchor"] += 1
            if f_cat is not None:
                (hit if f_cat == truecat else mis)["frame"] += 1

    out = {nm: fa.category_pos_purity(v, None) for nm, v in votes.items()}
    return out, (hit, mis)


def firing_perplexity(seq, eval_start, anchor, cg, C):
    """Next-word-CATEGORY perplexity restricted to positions where the ANCHOR cue fires (prev is anchor).
    anchor vs frame vs combined — does the anchor cue help where it speaks? (category-level ppl: the
    voter's (C,) dist scored against the next word's actual category.)"""
    res = {"anchor": [], "frame": [], "combo": []}
    hit = {"anchor": 0, "frame": 0}; mis = {"anchor": 0, "frame": 0}
    for t in range(eval_start, len(seq) - 1):
        p = int(seq[t]); nxt = int(seq[t + 1])
        if p < 0 or nxt < 0:
            continue
        if not anchor.fires_right(p):
            continue
        tc = anchor.clu[nxt]
        if tc < 0:
            continue
        a = anchor.predict_right(p); f = cg.predict_open(p)
        af = np.ones(C) / C if a is None else a
        ff = np.ones(C) / C if f is None else f
        res["anchor"].append(af[tc]); res["frame"].append(ff[tc])
        va = hit["anchor"] / (hit["anchor"] + mis["anchor"]) if (hit["anchor"] + mis["anchor"]) else 0.5
        vf = hit["frame"]  / (hit["frame"]  + mis["frame"])  if (hit["frame"]  + mis["frame"])  else 0.5
        c = af if va >= vf else (ff if f is not None else af)
        res["combo"].append(c[tc])
        ac = int(af.argmax()); fc = int(ff.argmax())
        (hit if ac == tc else mis)["anchor"] += 1
        (hit if fc == tc else mis)["frame"] += 1
    return {k: (perplexity(np.array(v)) if v else float("nan"), len(v)) for k, v in res.items()}


def fit_af(seq, clu, C):
    fc = build_frame_counts(seq, order=1)
    cg = ConstructionGrammar(clu, C, alpha=0.1, min_token=MIN_TOKEN,
                             freeze_dom=FREEZE_DOM, open_types=OPEN_TYPES)
    cg.fit(fc).classify()
    cg.build_category_lexicon(fc)
    return cg


def run_lang(label, shuffle, rng):
    print(f"\n########## {label} ##########")
    t0 = time.time()
    seq, topword = load_stream(TRAIN_BYTES, shuffle=shuffle, rng=rng)
    gpos = gold_of(topword)
    n_gold = int(sum(g is not None for g in gpos))
    clu, C = build_categories(seq)
    cg = fit_af(seq, clu, C)
    eval_start = int(len(seq) * (1 - EVAL_FRAC))
    print(f"  {len(seq):,} words | C={C} categories | {(clu>=0).sum():,}/{N} categorized | "
          f"gold-tagged top words={n_gold} | eval words={len(seq)-eval_start:,} | setup {time.time()-t0:.1f}s")

    # FRAGILE: sweep the anchor band size. The FAIR comparison is anchor vs frame_on_anchor (AF scored on
    # the SAME firing positions) — full-eval frame purity is on a different, higher-entropy subset.
    print(f"  {'k':>4} | {'anchorPty':>9} {'(nCat,n)':>11} | {'AFsamePos':>9} | {'anchor-AF':>9} | {'comboPty':>9}")
    sweep = {}
    for k in K_SWEEP:
        anchor = fa.AnchorVoter(clu, C, k=k, alpha=0.1).fit(seq)
        out, _ = purity_run(label, seq, eval_start, anchor, cg, gpos)
        ap, an, ac = out["anchor"]; foa, fon, foc = out["frame_on_anchor"]; cp, cn, cc = out["combo"]
        sweep[k] = (ap, foa, cp)        # (anchor, AF-on-same-positions, combo)
        print(f"  {k:>4} | {ap:9.3f} {('('+str(ac)+','+str(an)+')'):>11} | {foa:9.3f} | {ap-foa:+9.3f} | {cp:9.3f}")

    # firing-perplexity at the spec's ~20-token band
    anchor = fa.AnchorVoter(clu, C, k=K_MAIN, alpha=0.1).fit(seq)
    fp = firing_perplexity(seq, eval_start, anchor, cg, C)
    print(f"  next-CATEGORY perplexity where anchor fires (k={K_MAIN}):")
    for nm in ("anchor", "frame", "combo"):
        ppl, n = fp[nm]; print(f"      {nm:<8} ppl={ppl:8.3f}  (n={n})")

    # the headline numbers at K_MAIN
    out, _ = purity_run(label, seq, eval_start, anchor, cg, gpos)
    print(f"  headline (k={K_MAIN}): anchor purity {out['anchor'][0]:.3f} | "
          f"AF-on-anchor-positions {out['frame_on_anchor'][0]:.3f} | AF-full {out['frame'][0]:.3f} | "
          f"combo {out['combo'][0]:.3f}")
    return dict(label=label, C=C, sweep=sweep,
                purity_anchor=out["anchor"][0], purity_frame_fair=out["frame_on_anchor"][0],
                purity_frame_full=out["frame"][0], purity_combo=out["combo"][0],
                fire_ppl={k: v[0] for k, v in fp.items()})


def main():
    rng = np.random.default_rng(SEED)
    eng = run_lang("ENGLISH (text8)", shuffle=False, rng=rng)
    neg = run_lang("SHUFFLED (negative control; substitutes German)", shuffle=True, rng=rng)

    print("\n==================== VERDICT INPUTS ====================")
    # FAIR delta: anchor vs AF scored on the SAME firing positions (purity is subset-dependent).
    d_eng = eng["purity_anchor"] - eng["purity_frame_fair"]
    d_combo_eng = eng["purity_combo"] - eng["purity_frame_full"]
    print(f"  ENGLISH : anchor {eng['purity_anchor']:.3f} | AF-same-positions {eng['purity_frame_fair']:.3f} | "
          f"AF-full {eng['purity_frame_full']:.3f} | combo {eng['purity_combo']:.3f}")
    print(f"            FAIR anchor-AF(same pos) = {d_eng:+.3f}   combo-AF(full) = {d_combo_eng:+.3f}")
    print(f"  SHUFFLED: anchor {neg['purity_anchor']:.3f} | AF-same-positions {neg['purity_frame_fair']:.3f} | "
          f"combo {neg['purity_combo']:.3f}  (anchor-AF = {neg['purity_anchor']-neg['purity_frame_fair']:+.3f})")
    best_k = max((eng["sweep"][k][0] - eng["sweep"][k][1], k) for k in eng["sweep"])
    print(f"  best FAIR anchor-over-AF gain across k-sweep: {best_k[0]:+.3f} at k={best_k[1]}")
    kill_purity = d_eng < 0.02
    kill_degrade = (neg["purity_combo"] < neg["purity_frame_full"] - 1e-9)
    print(f"  KILL test: anchor<+2% purity over AF (same positions) on English? {kill_purity}  "
          f"AND combo degrades on a language? {kill_degrade}  -> KILL FIRES: {kill_purity and kill_degrade}")
    return eng, neg


if __name__ == "__main__":
    main()
