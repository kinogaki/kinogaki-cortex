#!/usr/bin/env python3
"""Exp BH — BLiMP / minimal-pair grammaticality Probe + impossible-language ablation.

THE HONESTY BAR. Every acquisition claim upstream (boundaries, chunks, categories,
constructions) has so far been scored on bpc / real-word-% — the modelling axes. The field
scores GRAMMAR differently: BLiMP (Warstadt 2020) gives a model a minimal pair (s+ grammatical,
s- minimally ungrammatical) and counts it right iff the model assigns s+ the LOWER surprisal.
This experiment puts the existing char vote on THAT axis — read-side only, the eval never
learns — so the rest of the queue can be judged the way the field judges, not on bpc.

THE COGNITIVE FRAME. A learner with the right inductive bias should (a) find grammatical
strings less surprising than their ungrammatical minimal twins, and (b) acquire a NATURAL
language more easily than an IMPOSSIBLE one (Kallini 2024 — a scrambled counterfactual). Our
counter geometry is backoff over char-tails: it bakes in a LOCALITY bias (recent context
predicts the next char). The question is whether that locality bias is enough to (a) order
minimal pairs and (b) prefer natural English over a position-scramble — and crucially whether
any natural>scramble gap is STRUCTURAL or just ENTROPY-driven (the spec's required control).

CORPUS NOTE / SUBSTITUTIONS (declared up front):
  - The spec names a CDS/transcribed (CHILDES-style) train mix and the BLiMP 67 sets. Neither
    is on disk. We SUBSTITUTE: train on a text8 slice (the standing char corpus), and BUILD the
    minimal pairs in-process from high-frequency English templates across 6 phenomena (the same
    construction families BLiMP probes: agreement, det-noun number, anaphor, neg-polarity, wh /
    island, argument structure). This is a smaller, hand-built BLiMP analogue — honest about it.

Questions:
  Q1  Does the count band order minimal pairs above chance, per-phenomenon and macro?
  Q2  BASELINE: does it beat a bigram (order-1) at the SAME budget? (the kill axis)
  Q3  ABLATION: trained on natural vs position-scrambled English of EQUAL bytes — is the
      grammar Probe higher for natural, and is the gap structural (survives a LOCAL, entropy-
      matched scramble) or just entropy (the confound)?

Online single streaming pass (each band .fit once); bounded memory; no gradients/k-means/SVD.
Fixed seed.
"""
import os, sys, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics, blimp

SEED = 0
NBYTES = 5_000_000          # ~5MB text8 slice — fast first pass, ≤10MB as instructed
TRAIN_CHARS = None          # use the whole slice for training


# ───────────────────────── the minimal-pair set (a BLiMP analogue) ─────────────────────────
# Built from common, lowercase, in-[a-z ] words so a text8 char model sees them in-distribution.
# Each phenomenon is a template family; we expand over fillers to get many items per phenomenon.

SING_N = ["dog", "cat", "man", "king", "child", "woman", "boy", "girl", "book", "house",
          "tree", "city", "river", "horse", "bird", "ship", "road", "star", "door", "field"]
PLUR_N = ["dogs", "cats", "men", "kings", "children", "women", "boys", "girls", "books", "houses",
          "trees", "cities", "rivers", "horses", "birds", "ships", "roads", "stars", "doors", "fields"]
SING_M = {"dog": "himself", "cat": "itself", "man": "himself", "king": "himself", "child": "itself",
          "woman": "herself", "boy": "himself", "girl": "herself", "book": "itself", "house": "itself"}
VERB_S = ["runs", "walks", "sleeps", "eats", "sees", "knows", "moves", "falls", "stands", "waits"]
VERB_P = ["run", "walk", "sleep", "eat", "see", "know", "move", "fall", "stand", "wait"]


def build_pairs():
    """A hand-built minimal-pair set across 6 phenomena. Each triple = (phenomenon, good, bad);
    good is grammatical, bad is a minimal ungrammatical twin (one local violation)."""
    rng = np.random.default_rng(SEED)
    pairs = []

    # 1. SUBJECT-VERB AGREEMENT (number): "the dog runs" vs "the dog run"
    for n, vs, vp in zip(SING_N[:10], VERB_S, VERB_P):
        pairs.append(("agreement", f"the {n} {vs} away", f"the {n} {vp} away"))
    for n in PLUR_N[:10]:
        v_i = SING_N.index  # not used; pair plural subj with plural verb (good) vs singular (bad)
    for i, n in enumerate(PLUR_N[:10]):
        pairs.append(("agreement", f"the {n} {VERB_P[i]} away", f"the {n} {VERB_S[i]} away"))

    # 2. DETERMINER-NOUN NUMBER: "this dog" vs "this dogs" ; "these dogs" vs "these dog"
    for s, p in zip(SING_N[:12], PLUR_N[:12]):
        pairs.append(("det_noun", f"i saw this {s} today", f"i saw this {p} today"))
        pairs.append(("det_noun", f"i saw these {p} today", f"i saw these {s} today"))

    # 3. ANAPHOR AGREEMENT (reflexive): "the woman saw herself" vs "the woman saw himself"
    refs = list(SING_M.items())
    for n, good_m in refs:
        bad_m = "herself" if good_m != "herself" else "himself"
        pairs.append(("anaphor", f"the {n} saw {good_m} there", f"the {n} saw {bad_m} there"))

    # 4. NEGATIVE POLARITY: "no king has ever" vs "the king has ever" (ever needs a licensor)
    for n in SING_N[:12]:
        pairs.append(("npi", f"no {n} has ever been here", f"the {n} has ever been here"))

    # 5. WH / ISLAND (simple filler-gap): "what did the man see" vs "what did the man see it"
    for n in SING_N[:12]:
        pairs.append(("wh_gap", f"what did the {n} see", f"what did the {n} see it"))

    # 6. ARGUMENT STRUCTURE / word order: "the dog ate the food" vs "the dog ate food the"
    objs = ["food", "meat", "bread", "fish", "fruit", "water", "milk", "corn", "rice", "soup"]
    for n, o in zip(SING_N[:10], objs):
        pairs.append(("arg_struct", f"the {n} ate the {o}", f"the {n} ate {o} the"))

    rng.shuffle(pairs)
    return pairs


# ───────────────────────── models (each a char band; read-side only) ─────────────────────────

def fit_band(text, orders):
    return cortex.Cortex(char_orders=tuple(orders), word_orders=()).fit(text)


def main():
    t0 = time.time()
    print(f"Exp BH — BLiMP-analogue grammaticality Probe + impossible-language ablation")
    print(f"  corpus: text8 {NBYTES//1_000_000}MB slice (SUBSTITUTE for CDS); pairs: hand-built (SUBSTITUTE for BLiMP-67)\n")

    ids = corpus.load_ids("text8", NBYTES)
    text = corpus.ids_to_str(ids)
    n = len(text)
    train = text[: int(n * 0.95)]
    held = text[int(n * 0.95):]                 # held-out natural for bpc
    print(f"  chars: {n:,}   train {len(train):,}   held {len(held):,}   ({time.time()-t0:.1f}s)")

    pairs = build_pairs()
    phen_counts = {}
    for p, _, _ in pairs:
        phen_counts[p] = phen_counts.get(p, 0) + 1
    print(f"  minimal pairs: {len(pairs)}   phenomena: {phen_counts}\n")

    # ── the count band vs the bigram baseline, both natural-trained ──
    print("fitting natural bands ...")
    band = fit_band(train, (1, 2, 3, 4, 5, 6))      # the full count band
    bigram = fit_band(train, (1,))                  # the baseline (order-1)
    print(f"  fit done ({time.time()-t0:.1f}s)")

    r_band = blimp.evaluate(band, pairs)
    r_bi = blimp.evaluate(bigram, pairs)

    print("\n=== Q1/Q2: minimal-pair accuracy (count band vs bigram baseline) ===")
    print(f"  {'phenomenon':<14} {'band-acc':>9} {'bigram-acc':>11} {'band-margin(bits)':>18} {'n':>5}")
    for phen in sorted(r_band["per_phen"]):
        ba, cnt, mg = r_band["per_phen"][phen]
        bia = r_bi["per_phen"][phen][0]
        print(f"  {phen:<14} {ba*100:8.1f}% {bia*100:10.1f}% {mg:17.2f} {cnt:>5}")
    print(f"  {'MACRO':<14} {r_band['macro']*100:8.1f}% {r_bi['macro']*100:10.1f}% "
          f"{r_band['mean_margin']:17.2f} {r_band['n']:>5}")
    print(f"  held-out bpc: band {metrics.bpc(band, held):.3f}   bigram {metrics.bpc(bigram, held):.3f}")

    # ── Q3: impossible-language ablation (natural vs scrambles, equal bytes) ──
    print("\n=== Q3: impossible-language ablation (equal-byte trains) ===")
    nat_train = train
    glob_train = blimp.scramble_global(train, seed=SEED)
    loc_train = blimp.scramble_local(train, window=4, seed=SEED)
    rev_train = blimp.scramble_reverse(train)

    nat = band                                       # already fit on natural
    glob = fit_band(glob_train, (1, 2, 3, 4, 5, 6))
    loc = fit_band(loc_train, (1, 2, 3, 4, 5, 6))
    rev = fit_band(rev_train, (1, 2, 3, 4, 5, 6))
    print(f"  scramble bands fit ({time.time()-t0:.1f}s)")

    # held-out of each KIND, so bpc measures how surprised each learner is by its OWN language.
    nat_held = held
    glob_held = blimp.scramble_global(held, seed=SEED + 1)
    loc_held = blimp.scramble_local(held, window=4, seed=SEED + 1)
    rev_held = blimp.scramble_reverse(held)

    # grammar Probe: the SAME natural minimal pairs, scored by each learner. A learner that
    # acquired natural grammar should order natural pairs well; a scramble-trained one should not.
    r_nat = blimp.evaluate(nat, pairs)
    r_glob = blimp.evaluate(glob, pairs)
    r_loc = blimp.evaluate(loc, pairs)
    r_rev = blimp.evaluate(rev, pairs)

    rows = [
        ("natural", nat, nat_held, r_nat),
        ("global-scramble", glob, glob_held, r_glob),
        ("local-scramble(w4)", loc, loc_held, r_loc),
        ("reverse-scramble", rev, rev_held, r_rev),
    ]
    print(f"  {'train':<20} {'own-bpc':>8} {'pair-acc(nat)':>14} {'margin(bits)':>13}")
    bpcs = {}
    for name, m, hh, r in rows:
        b = metrics.bpc(m, hh)
        bpcs[name] = b
        print(f"  {name:<20} {b:8.3f} {r['macro']*100:13.1f}% {r['mean_margin']:12.2f}")

    nat_bpc = bpcs["natural"]
    print(f"\n  natural-minus-scramble grammar gap (pair-acc): "
          f"global {(r_nat['macro']-r_glob['macro'])*100:+.1f}pp   "
          f"local {(r_nat['macro']-r_loc['macro'])*100:+.1f}pp   "
          f"reverse {(r_nat['macro']-r_rev['macro'])*100:+.1f}pp")
    print(f"  CONFOUND check — own-bpc (entropy of each language under its own learner):")
    print(f"    natural {nat_bpc:.3f}   global {bpcs['global-scramble']:.3f}   "
          f"local {bpcs['local-scramble(w4)']:.3f}   reverse {bpcs['reverse-scramble']:.3f}")
    print(f"    -> if local-scramble bpc ≈ natural bpc but grammar gap persists, the gap is STRUCTURAL.")

    # machine-readable
    dump = {
        "n_pairs": len(pairs),
        "band": {"macro": round(r_band["macro"], 4),
                 "per_phen": {k: round(v[0], 4) for k, v in r_band["per_phen"].items()},
                 "held_bpc": round(metrics.bpc(band, held), 4)},
        "bigram": {"macro": round(r_bi["macro"], 4),
                   "per_phen": {k: round(v[0], 4) for k, v in r_bi["per_phen"].items()}},
        "ablation": {name: {"own_bpc": round(bpcs[name], 4), "pair_acc": round(r["macro"], 4),
                            "margin": round(r["mean_margin"], 3)}
                     for name, _, _, r in rows},
    }
    print("\nRESULTS_DICT = " + repr(dump))
    print(f"\n(total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
