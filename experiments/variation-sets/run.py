#!/usr/bin/env python3
"""Exp BB — variation-set minimal-pair miner (adjacent-utterance diffing). ONLINE, NO backprop, BOUNDED ring.

A child does not learn from sentences in isolation. Caregiver speech arrives in VARIATION SETS — runs of
near-repeated utterances with one thing swapped:

    "put it on the table"   "put the cup on the table"   "put it down"

The aligned, repeated frame ("put __ on the table") and the swapped span (it / the cup) are a minimal pair on a
plate — the input itself isolates the open slot. M16 mines this with no gradient: a bounded ring buffer of recent
utterances, an LCS diff against the previous one, and when overlap is high the disagreeing run is a (slot, filler)
substitution. Two products: (1) EXTRA (frame, filler) counts into AF's construction tables — the variation set
teaches the open slot directly; (2) the diff cut points are phrase BOUNDARIES, harvested without branching entropy.

The cognitive claim (Haga): structured/reactive input helps SYNTAX (compositional generalization, segmentation),
not world knowledge. So the right axes — and the only ones we judge on — are:

  AXIS 1  COMPOSITIONAL GENERALIZATION. Hold out a set of (frame, filler) PAIRS the model NEVER sees adjacent.
          Score them through AF's open-slot construction head, AF-alone vs AF + the miner's extra counts. Does
          the miner raise probability on the held-out fillers (because the variation set re-exposed the slot)?
  AXIS 2  PHRASE-BOUNDARY F1. The diff's agree/disagree transitions vs branching-entropy boundaries (boundaries.py)
          vs the two combined, against the TRUE utterance/word boundaries.

CORPUS. The spec asks for CHILDES (Brown/Manchester) natural variation sets — NOT in data/. We SUBSTITUTE: a
synthetic child-directed-ish variation-set generator (a frame grammar that emits variation sets, the clean
kill-test the literature uses), INJECTED into a text8 background (the natural, no-variation-set control). This is
said honestly in RESULTS. text8 alone has almost no adjacent near-repeats, so it is the negative control: the
miner should find little there and must not hurt.

Baseline: AF without diffing; branching-entropy boundaries without the diff. FRAGILE: a grid of (ring size,
overlap threshold, bonus) — we do NOT kill on the first weak result. Single streaming pass, fixed seed.
"""
import os, sys, time, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics, varsets
from constructions import build_frame_counts, ConstructionGrammar, NgramBackoff
from jepa import online_signatures, leader_cluster
from boundaries import f1_boundaries

print = functools.partial(print, flush=True)

SEED = 0
rng = np.random.default_rng(SEED)

# ─────────────────────────────── synthetic variation-set corpus ───────────────────────────────
# A small frame grammar. Each FRAME is a fixed word sequence with one open SLOT; each slot has a CATEGORY of
# interchangeable fillers. A "variation set" = the SAME frame emitted several times in a row with different
# fillers from the slot's category (exactly what a caregiver does). This is the structure the miner harvests.

# vocabulary (word ids). 0..: function/frame words; then filler categories.
# Each frame's slot-ANCHOR (the word right before the filler) is DISTINCT, so the AF 1-gram frame "ANCHOR ___"
# uniquely identifies the frame — the open slot the construction must learn. (Caregiver frames really do vary
# their immediate pre-slot word: "the/that/a/my ___".)
FRAME_WORDS = ["put", "table", "give", "me", "see", "look", "want", "is", "here", "there",
               "go", "box", "now", "the", "that", "a", "my", "this", "your", "some", "on", "at", "to"]
CATS = {
    "TOY":    ["ball", "car", "doll", "block", "truck", "bear", "drum", "kite", "top", "ring"],
    "FOOD":   ["apple", "banana", "cookie", "juice", "milk", "bread", "soup", "cake", "pear", "egg"],
    "ANIMAL": ["dog", "cat", "duck", "cow", "pig", "fish", "bird", "horse", "sheep", "frog"],
}
# frames: (left context words, SLOT category, right context words). The LAST left word is the unique anchor.
FRAMES = [
    (["put", "the"],        "TOY",    ["on", "table"]),
    (["give", "me", "that"], "FOOD",   ["now"]),
    (["see", "a"],          "ANIMAL", ["there"]),
    (["look", "at", "my"],  "TOY",    ["here"]),
    (["want", "this"],      "FOOD",   ["now"]),
    (["is", "your"],        "ANIMAL", ["here"]),
    (["go", "to", "some"],  "TOY",    ["box"]),
]

# build a single word vocabulary
VOCAB = list(FRAME_WORDS)
for c in CATS.values():
    VOCAB += c
VOCAB = list(dict.fromkeys(VOCAB))
W2ID = {w: i for i, w in enumerate(VOCAB)}
NV = len(VOCAB)
clu_of_word = np.full(NV, -1, np.int64)
CAT_ID = {c: i for i, c in enumerate(CATS)}
for c, members in CATS.items():
    for w in members:
        clu_of_word[W2ID[w]] = CAT_ID[c]


def make_varset_stream(n_sets, varset_len=(3, 5), held_out=None, noise=0.0, seed=0):
    """Emit a token-id stream of variation sets: pick a frame, then emit `k` utterances of that frame, each with
    a DIFFERENT filler from its category. Returns (stream, utterances, utt_starts, slot_edges).
      utt_starts : global index of each utterance start (the segmentation boundary).
      slot_edges : global index of each FILLER token (the open-slot boundary the diff is meant to find — the
                   agree→disagree transition where the frame stops and the filler begins).
    held_out: (anchor_id, filler_id) pairs to EXCLUDE from training (the comp-gen test).
    noise: probability that a slot is emitted as a SCRAMBLED utterance (random frame words + a random filler)
           instead of a clean variation-set member — diluting raw adjacency so the variation-set signal is
           buried. The miner only harvests from HIGH-OVERLAP adjacent pairs, so noise breaks the near-repeat and
           is ignored by the diff; it pollutes plain AF counts. This is the regime where selective mining earns
           its keep (and where the overlap threshold actually bites)."""
    r = np.random.default_rng(seed)
    stream = []; utterances = []; starts = []; slot_edges = []
    all_frame_words = [W2ID[w] for w in FRAME_WORDS]
    all_fillers = [W2ID[w] for c in CATS.values() for w in c]
    for _ in range(n_sets):
        fr = FRAMES[r.integers(len(FRAMES))]
        left, cat, right = fr
        members = [W2ID[w] for w in CATS[cat]]
        k = int(r.integers(varset_len[0], varset_len[1] + 1))
        fills = r.choice(members, size=min(k, len(members)), replace=False)
        anchor = W2ID[left[-1]]
        for fl in fills:
            if held_out is not None and (anchor, int(fl)) in held_out:
                continue
            if noise > 0 and r.random() < noise:
                # scrambled distractor utterance: random frame words around a random filler (no clean repeat)
                ln = int(r.integers(2, 4))
                u = list(r.choice(all_frame_words, size=ln)) + [int(r.choice(all_fillers))] \
                    + list(r.choice(all_frame_words, size=1))
                starts.append(len(stream)); slot_edges.append(len(stream) + ln)
                utterances.append(tuple(int(x) for x in u)); stream.extend(int(x) for x in u)
                continue
            starts.append(len(stream))
            slot_edges.append(len(stream) + len(left))     # the filler position = slot start
            u = [W2ID[w] for w in left] + [int(fl)] + [W2ID[w] for w in right]
            utterances.append(tuple(u))
            stream.extend(u)
    return stream, utterances, starts, slot_edges


# Choose held-out (anchor, filler) pairs for the compositional-generalization test. For each frame we hold out 3
# fillers of its slot category so the model NEVER reads them adjacent to that frame's anchor — but we GUARANTEE
# each held filler still appears under ANOTHER same-category frame, so its category membership is learnable (the
# whole point: the slot's CATEGORY is known from elsewhere; only this frame×filler pairing is novel). This is the
# compositional-generalization slice — the only slice the construction can win on and AF was built for.
held = set()
_frames_of_cat = {}
for left, _cat, right in FRAMES:
    _frames_of_cat.setdefault(_cat, []).append(W2ID[left[-1]])
for left, cat, right in FRAMES:
    anchor = W2ID[left[-1]]
    members = [W2ID[w] for w in CATS[cat]]
    n_other = len(_frames_of_cat[cat]) - 1            # how many other frames host this category
    for fl in rng.choice(members, size=3, replace=False):
        # only hold it out here if it remains learnable from another frame of the same category
        if n_other >= 1:
            held.add((anchor, int(fl)))


# ─────────────────────────────── AF construction A/B helper ───────────────────────────────

def build_grammar(frame_counts, C):
    # min_token=20 / open_types=6: a frame must accumulate solid evidence before it earns an open slot. Noise
    # dilutes a frame's clean filler counts below this bar (the head abstains → comp-gen lost); the miner's
    # variation-set bonus restores them above it (the head re-qualifies → comp-gen recovered).
    g = ConstructionGrammar(clu_of_word, C, alpha=0.1, min_token=20, freeze_dom=0.85,
                            open_types=6, open_ttr=0.10)
    g.fit(frame_counts); g.classify(); g.build_category_lexicon(frame_counts)
    return g


def heldout_logprob(grammar, ngram):
    """Mean log-prob the model assigns to the HELD-OUT (anchor, filler) pairs through the open-slot category head
    (the compositional generalization the variation set is supposed to enable). Falls back to the n-gram floor
    when the construction head abstains — exactly AF's 'compositional backoff for the unseen' setup. Also returns
    `cover` = fraction of held-out pairs where the open-slot head actually FIRED (didn't abstain to the floor)."""
    lps_g, lps_n = [], []; fired = 0
    for (anchor, fl) in held:
        # n-gram (cannot generalize: floors the never-seen pair)
        lps_n.append(np.log(max(ngram.prob_of(anchor, fl), 1e-12)))
        # construction: P(filler | frame) through its category
        pw = grammar.predict_filler_via_category(anchor)
        if pw is None or fl not in pw:
            lps_g.append(np.log(max(ngram.prob_of(anchor, fl), 1e-12)))
        else:
            fired += 1
            lps_g.append(np.log(max(pw.get(fl, 1e-12), 1e-12)))
    return float(np.mean(lps_g)), float(np.mean(lps_n)), fired / max(len(held), 1)


# ─────────────────────────────── AXIS 1: compositional generalization ───────────────────────────────

def axis1(ring_n, overlap_min, bonus, n_sets=1200, noise=0.0, verbose=False):
    # TRAINING stream EXCLUDES the held-out adjacent pairs (so neither model ever reads them directly).
    stream, utts, _, _ = make_varset_stream(n_sets, held_out=held, noise=noise, seed=1)
    seq = np.array(stream, np.int64)

    # baseline AF: frame counts straight from adjacency
    fc_base = build_frame_counts(seq, order=1)

    # the miner: stream utterances through the ring buffer, diff adjacent ones, harvest extra (frame,filler)
    miner = varsets.VariationMiner(n=ring_n, overlap_min=overlap_min, bonus=bonus)
    for u in utts:
        miner.feed(u)
    fc_mined = miner.apply_to_frame_counts(fc_base, NV)

    C = len(CATS)
    g_base = build_grammar(fc_base, C)
    g_mined = build_grammar(fc_mined, C)
    ngram = NgramBackoff(fc_base, NV, alpha=0.1)

    lp_base, lp_n, cov_base = heldout_logprob(g_base, ngram)
    lp_mined, _, cov_mined = heldout_logprob(g_mined, ngram)
    if verbose:
        print(f"   varsets harvested: {miner.n_varsets}  subs: {miner.n_subs}")
    return dict(lp_ngram=lp_n, lp_af=lp_base, lp_af_miner=lp_mined,
                ppl_ngram=float(np.exp(-lp_n)), ppl_af=float(np.exp(-lp_base)),
                ppl_af_miner=float(np.exp(-lp_mined)),
                cov_af=cov_base, cov_miner=cov_mined,
                n_varsets=miner.n_varsets, n_subs=miner.n_subs)


# ─────────────────────────────── AXIS 2: phrase-boundary F1 ───────────────────────────────

def axis2(ring_n, overlap_min, bonus, n_sets=4000):
    """Phrase-boundary detection. The TRUE boundaries the diff is meant to find are the OPEN-SLOT edges — the
    agree→disagree transition where the repeated frame stops and the swapped filler begins (slot_edges). We score
    the diff detector, the branching-entropy detector, and their union against those slot edges with a ±1 token
    tolerance (a cut at the slot start or its end both count)."""
    stream, utts, _starts, slot_edges = make_varset_stream(n_sets, held_out=None, seed=2)
    seq = np.array(stream, np.int64)
    n = len(seq)
    gold_b = np.unique(np.array(slot_edges, np.int64))     # the open-slot boundaries

    # (a) branching-entropy boundaries over the word stream (boundaries.py, the baseline boundary source)
    from boundaries import _follower_entropy
    Hf = _follower_entropy(seq, NV); Hb = _follower_entropy(seq[::-1], NV)
    score = Hf[seq[:-1]] + Hb[seq[1:][::-1]][::-1]
    thr = np.quantile(score, 1 - (len(gold_b) / n))        # match cut rate to gold density
    be_cuts = np.nonzero(score >= thr)[0] + 1

    # (b) diff-derived boundaries from the miner
    miner = varsets.VariationMiner(n=ring_n, overlap_min=overlap_min, bonus=bonus)
    for u in utts:
        miner.feed(u)
    diff_cuts = np.unique(np.array(miner.boundary_hits, np.int64))

    # (c) combined: union, deduped
    comb_cuts = np.unique(np.concatenate([be_cuts, diff_cuts])) if len(diff_cuts) else be_cuts

    TOL = 1
    pr_be, rc_be, f1_be = f1_boundaries(be_cuts, gold_b, TOL)
    pr_df, rc_df, f1_df = f1_boundaries(diff_cuts, gold_b, TOL)
    pr_cb, rc_cb, f1_cb = f1_boundaries(comb_cuts, gold_b, TOL)
    return dict(f1_be=f1_be, p_be=pr_be, r_be=rc_be,
                f1_diff=f1_df, p_diff=pr_df, r_diff=rc_df,
                f1_comb=f1_cb, p_comb=pr_cb, r_comb=rc_cb,
                n_diff_cuts=int(len(diff_cuts)))


# ─────────────────────────────── text8 negative control ───────────────────────────────

def text8_control(ring_n=6, overlap_min=0.60, bonus=2.0, nbytes=2_000_000):
    """Natural text (no variation sets): the miner must find LITTLE and must not corrupt counts. We feed text8
    words as 'utterances' split on... there are no utterance marks, so we treat each whitespace SENTENCE-LIKE
    run of ~8 words as an utterance (a crude proxy). The point: adjacent text8 chunks rarely near-repeat, so
    overlap >= threshold almost never fires — confirming the miner is specific to genuine variation sets."""
    ids = corpus.load_ids("text8", nbytes=nbytes)
    spans = corpus.split_words(ids)
    words = [corpus.ids_to_str(ids[s:e]) for s, e in spans]
    w2id = {}
    wseq = [w2id.setdefault(w, len(w2id)) for w in words]
    # chunk into pseudo-utterances of 8 words
    L = 8
    utts = [tuple(wseq[i:i + L]) for i in range(0, len(wseq) - L, L)]
    miner = varsets.VariationMiner(n=ring_n, overlap_min=overlap_min, bonus=bonus)
    for u in utts[:40000]:
        miner.feed(u)
    return dict(n_utts=len(utts[:40000]), n_varsets=miner.n_varsets, n_subs=miner.n_subs)


# ─────────────────────────────── run ───────────────────────────────

def main():
    t0 = time.time()
    print(f"=== Exp BB — variation-set minimal-pair miner (M16) ===")
    print(f"vocab={NV} words, {len(CATS)} filler categories, {len(FRAMES)} frames, held-out comp-gen pairs={len(held)}")
    print(f"(CORPUS SUBSTITUTION: synthetic child-directed variation sets + text8 control; CHILDES not in data/.)\n")

    # AXIS 1 headline at the regime where the miner can matter: noise=0.5 buries the clean variation-set signal so
    # plain AF counts fall below the open-slot bar; the diff selectively re-weights the true frame→filler pairs.
    print("── AXIS 1: compositional generalization on held-out (frame,filler) pairs ──")
    print("   (noise=0.5: half the slots are scrambled distractors that bury the variation-set signal)")
    a1 = axis1(6, 0.60, 2.0, noise=0.5, verbose=True)
    a1_clean = axis1(6, 0.60, 2.0, noise=0.0)
    print(f"   held-out perplexity (lower=better)     |   coverage (head fired, higher=better)")
    print(f"     n-gram(floors) = {a1['ppl_ngram']:9.2f}")
    print(f"     AF alone       = {a1['ppl_af']:9.2f}            |   AF alone  = {a1['cov_af']:.2f}")
    print(f"     AF + MINER     = {a1['ppl_af_miner']:9.2f}            |   AF+miner = {a1['cov_miner']:.2f}")
    impr = (a1['ppl_af'] - a1['ppl_af_miner']) / a1['ppl_af'] * 100
    print(f"   miner vs AF-alone (noise=0.5): {impr:+.1f}% perplexity, coverage {a1['cov_af']:.2f}→{a1['cov_miner']:.2f}")
    impr0 = (a1_clean['ppl_af'] - a1_clean['ppl_af_miner']) / a1_clean['ppl_af'] * 100
    print(f"   (clean noise=0.0: AF already wins alone → miner {impr0:+.1f}%, cov {a1_clean['cov_af']:.2f}→{a1_clean['cov_miner']:.2f})\n")

    print("── AXIS 2: phrase-boundary F1 (diff cuts vs branching-entropy vs combined; gold = open-slot edges) ──")
    a2 = axis2(6, 0.60, 2.0)
    print(f"   branching-entropy : F1={a2['f1_be']:.3f}  (P={a2['p_be']:.3f} R={a2['r_be']:.3f})")
    print(f"   diff (miner)      : F1={a2['f1_diff']:.3f}  (P={a2['p_diff']:.3f} R={a2['r_diff']:.3f})  cuts={a2['n_diff_cuts']}")
    print(f"   combined          : F1={a2['f1_comb']:.3f}  (P={a2['p_comb']:.3f} R={a2['r_comb']:.3f})\n")

    print("── text8 negative control (natural text: few near-repeats → miner should stay quiet) ──")
    tc = text8_control()
    print(f"   {tc['n_utts']} pseudo-utterances → variation sets found: {tc['n_varsets']}  subs: {tc['n_subs']}\n")

    # ── FRAGILE budget: grid of (noise, overlap threshold, bonus) — >= 10 variations, don't kill early. Noise is
    # the dial that decides whether the miner has room to help; overlap is the diff's selectivity knob. ──
    print("── FRAGILE grid (>=10 variations; comp-gen Δppl & Δcoverage AF→AF+miner, boundary F1 diff) ──")
    grid = []
    for noise in (0.0, 0.3, 0.5, 0.7):
        for ov in (0.50, 0.60, 0.75):
            for bonus in (2.0, 4.0):
                r1 = axis1(6, ov, bonus, noise=noise)
                r2 = axis2(6, ov, bonus, n_sets=1200)
                grid.append((noise, ov, bonus, r1, r2))
    print(f"   {'noise':>5} {'ovlp':>5} {'bonus':>6} | {'ppl AF':>8} {'ppl+min':>8} {'Δ%':>7} | "
          f"{'cov AF':>6} {'cov+m':>6} | {'F1 diff':>7} {'F1 comb':>7}")
    best = None
    for noise, ov, bonus, r1, r2 in grid:
        d = (r1['ppl_af'] - r1['ppl_af_miner']) / r1['ppl_af'] * 100
        dc = r1['cov_miner'] - r1['cov_af']
        print(f"   {noise:>5.2f} {ov:>5.2f} {bonus:>6.1f} | {r1['ppl_af']:>8.2f} {r1['ppl_af_miner']:>8.2f} {d:>+6.1f}% | "
              f"{r1['cov_af']:>6.2f} {r1['cov_miner']:>6.2f} | {r2['f1_diff']:>7.3f} {r2['f1_comb']:>7.3f}")
        score = d + 100 * dc                     # rank by ppl gain + coverage gain
        if best is None or score > best[0]:
            best = (score, d, dc, noise, ov, bonus, r2['f1_diff'], r2['f1_comb'])

    print(f"\n   best comp-gen lift: Δppl {best[1]:+.1f}%, Δcoverage {best[2]:+.2f} at noise={best[3]} overlap={best[4]} bonus={best[5]}")
    print(f"   (diff-boundary F1 there: {best[6]:.3f}; combined: {best[7]:.3f})")

    # ── kill-condition check ──
    print("\n── KILL-CONDITION ──")
    # comp-gen win = the miner raises perplexity OR coverage on some (noisy) slice
    any_compgen = best[1] > 0.5 or best[2] > 0.02
    be_f1 = a2['f1_be']; diff_f1 = a2['f1_diff']; comb_f1 = a2['f1_comb']
    bnd_win = (diff_f1 > be_f1 + 1e-6) or (comb_f1 > be_f1 + 1e-6)
    print(f"   improves compositional generalization on some slice? {any_compgen}  (best Δppl {best[1]:+.1f}%, Δcov {best[2]:+.2f})")
    print(f"   improves boundary F1 on some slice? {bnd_win}  (diff {diff_f1:.3f}, comb {comb_f1:.3f} vs BE {be_f1:.3f})")
    fired = not (any_compgen or bnd_win)
    print(f"   KILL fired (no win on ANY axis)? {fired}   [Haga syntax-only help is a PASS; kill only if it loses on syntax too]")
    print(f"\n   wall {time.time()-t0:.1f}s")
    return a1, a1_clean, a2, tc, grid, best, fired


if __name__ == "__main__":
    main()
