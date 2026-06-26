#!/usr/bin/env python3
"""Exp AO — Cue-based retrieval with similarity-based (fan) interference, for LONG-DISTANCE binding.

Lineage: Exp S built offset-attention — count tables keyed by relative POSITION, weighted by each
offset's information gain. That key decays with distance and cannot reach a far antecedent. Here we
generalise the key from a POSITION to a {feature bundle} and weight by FAN instead of offset-IG:
content-addressable retrieval, à la Lewis & Vasishth (2005) / Jaeger-Engelmann-Vasishth (2017).

    activation(item) = leaked_base(item) / FAN(cue)

The task is LONG-DISTANCE subject-verb agreement. A number-marked verb (is/are/was/were/has/have) must
retrieve its subject across an embedded clause. We construct probes from text8: a number-marked subject
noun, an agreeing verb at distance d, with or without an intervening DISTRACTOR noun of the OPPOSITE
(or SAME) number. We ask:

  Q1  RETRIEVAL ACCURACY vs DISTANCE — does cue-retrieval pick the correct-number subject across the
      clause, where offset-attention (a fixed position key) cannot reach?
  Q2  INTERFERENCE SIGNATURE (the fan effect) — does accuracy DEGRADE, and does fan RISE, when a
      distractor shares the verb's number cue (similarity-based interference)?
  Q3  vs OFFSET-ATTENTION head-to-head — same probes, same retrieval target; honest if offset wins.

Count-based, online single-pass, bounded memory, fixed seed. No gradients, no GPU.
"""
import os, sys, math, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus
from offsetattn import build_word_stream
from cueretrieval import (RetrievalStore, number_of, verb_number, word_class,
                          SING_DET, PLUR_DET)

VOCAB = 40000
NBYTES = 40_000_000        # text8 prefix (~7M words)
HALF_LIFE = 25.0           # recency leak: base activation halves every 25 words
SEED = 0
MAX_PROBES = 60000         # cap probes for runtime; sampled deterministically across the stream


def build_label_tables(vocab_list, UNK):
    """Precompute per-WORD-ID labels once (the heavy string work happens here, not in the scan loop):
       noun_num[wid] : 0 none, 1 singular, 2 plural   (content nouns; function words -> 0)
       verb_num[wid] : 0 none, 1 singular, 2 plural   (agreeing copula/aux: is/are/was/were/has/have)
       det_num[wid]  : 0 none, 1 singular, 2 plural   (a/an/this -> 1; these/those/many -> 2)
    Returns int8 arrays indexable by the stream; UNK rows are 0."""
    V = UNK + 1
    nn = np.zeros(V, dtype=np.int8)
    vn = np.zeros(V, dtype=np.int8)
    dn = np.zeros(V, dtype=np.int8)
    code = {None: 0, "singular": 1, "plural": 2}
    for wid, w in enumerate(vocab_list):
        if word_class(w) == "content":
            nn[wid] = code[number_of(w)]
        vn[wid] = code[verb_number(w)]
        if w in SING_DET:
            dn[wid] = 1
        elif w in PLUR_DET:
            dn[wid] = 2
    decode = {i: w for i, w in enumerate(vocab_list)}
    decode[UNK] = "<unk>"
    return nn, vn, dn, decode


def build_probes(stream, noun_num, verb_num_tab, det_num):
    """Construct LONG-DISTANCE subject-verb agreement probes — Lewis & Vasishth style.

    The SUBJECT is pinned by a DETERMINER, independently of the verb: a determiner det at position i fixes
    the number of the head noun at i+1 (a/an/this dog -> singular; these/those/many dogs -> plural). We then
    look FORWARD up to WIN words for the FIRST agreeing VERB whose number matches the subject — a real
    agreement dependency. Every number-marked CONTENT noun strictly between the head noun and the verb is a
    DISTRACTOR; we record whether a SAME-number or an OPPOSITE-number distractor intervenes. Because the
    determiner pins the subject far from the verb, distance varies and BOTH distractor conditions occur
    (a same-number distractor is a same-CUE competitor — the fan interferer; an opposite-number distractor
    is a mismatching item that should NOT be retrieved). The retrieval target is the head-noun instance."""
    WIN = 40
    probes = []
    n = len(stream)
    nn = noun_num[stream]
    vn_arr = verb_num_tab[stream]
    dn_arr = det_num[stream]
    for i in range(n - 2):
        dnum = int(dn_arr[i])
        if dnum == 0:
            continue
        head = i + 1                                  # the determined head noun
        hnum = int(nn[head])
        if hnum == 0 or hnum != dnum:                 # head must be a number-marked noun matching the det
            continue
        # find the first agreeing verb after the head, within the window, that matches the head's number
        verb_pos = -1
        same_between = False
        opp_between = False
        for j in range(head + 1, min(head + 1 + WIN, n)):
            vnj = int(vn_arr[j])
            if vnj != 0:                              # an agreeing verb
                if vnj == hnum:
                    verb_pos = j
                break                                 # first agreeing verb settles the dependency
            num = int(nn[j])                          # else accumulate intervening noun distractors
            if num == hnum:
                same_between = True
            elif num != 0:
                opp_between = True
        if verb_pos < 0:
            continue
        probes.append({
            "t": verb_pos, "subj_pos": head, "verb_num": "singular" if hnum == 1 else "plural",
            "subj": int(stream[head]), "verb": int(stream[verb_pos]),
            "dist": verb_pos - head,
            "opp_distractor": opp_between,
            "same_distractor": same_between,
        })
    if len(probes) > MAX_PROBES:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(probes), MAX_PROBES, replace=False))
        probes = [probes[i] for i in idx]
    return probes


# ---- the two retrieval strategies, evaluated on the SAME probes ----------------------------------

def eval_cue_retrieval(stream, probes, noun_num):
    """Online single pass: stream the words into a leaky RetrievalStore; AT each probe verb, fire the
    retrieval cue {(word_class, content), (number, verb_num)} and check whether the top-ranked retrieved
    NOUN is the true subject (or at least carries the correct number). Also record the fan of the number
    cue at retrieval time (the interference load). Bounded memory, no gradients.

    noun_num: per-WORD-ID int8 number label (0 none, 1 sing, 2 plur), restricted to content nouns."""
    NUMSTR = {1: "singular", 2: "plural"}
    store = RetrievalStore(half_life=HALF_LIFE)
    probe_at = {p["t"]: p for p in probes}
    nn_pos = noun_num[stream]                          # per-position number label (vectorized)
    # results bucketed by distance and by distractor condition
    by_dist = defaultdict(lambda: [0, 0])            # dist_bucket -> [correct_subject, total]
    cond = {"no_distractor": [0, 0], "opp_distractor": [0, 0], "same_distractor": [0, 0]}
    num_correct_cond = {"no_distractor": [0, 0], "opp_distractor": [0, 0], "same_distractor": [0, 0]}
    fan_by_cond = defaultdict(list)
    # distance-CONTROLLED interference: (cond, dist_bucket) -> [correct, total]  and fan samples.
    # Isolates the fan effect from the distance effect (same-number distractors also tend to lie farther).
    cond_dist = defaultdict(lambda: [0, 0])
    fan_cond_dist = defaultdict(list)
    attract_cond = {"no_distractor": [0, 0], "opp_distractor": [0, 0], "same_distractor": [0, 0]}
    n = len(stream)
    correct_subj = 0
    correct_num = 0
    total = 0
    for t in range(n):
        if t in probe_at:
            p = probe_at[t]
            # The verb retrieves its subject. It does NOT get the answer for free: number is a SOFT cue,
            # not a hard filter (using the verb's own number as a hard filter would be circular and score
            # a trivial 100%). The cue bundle = {must be a content noun} + {prefer the verb's number}. A
            # very recent OPPOSITE-number content noun can then out-activate the correct subject on the
            # class cue alone — that misretrieval IS agreement attraction (Lewis & Vasishth / Wagers).
            cue = [("class", "content"), ("number", p["verb_num"])]
            ranked = store.retrieve(cue, t=t, soft=True, topn=4)
            fan = store.fan(("number", p["verb_num"]), t=t)
            total += 1
            cond_key = ("same_distractor" if p["same_distractor"]
                        else "opp_distractor" if p["opp_distractor"]
                        else "no_distractor")
            db = min(p["dist"] // 4 * 4, 36)         # distance buckets of 4, capped at 36+
            by_dist[db][1] += 1
            cond[cond_key][1] += 1
            num_correct_cond[cond_key][1] += 1
            attract_cond[cond_key][1] += 1
            fan_by_cond[cond_key].append(fan)
            cond_dist[(cond_key, db)][1] += 1
            fan_cond_dist[(cond_key, db)].append(fan)
            if ranked:
                top_item, _ = ranked[0]
                top_word = top_item[0]               # item id = (word_id, position)
                top_pos = top_item[1]
                top_num = NUMSTR.get(int(noun_num[top_word]))
                # correct SUBJECT: retrieved the exact determiner-pinned head instance
                if top_pos == p["subj_pos"]:
                    correct_subj += 1
                    by_dist[db][0] += 1
                    cond[cond_key][0] += 1
                    cond_dist[(cond_key, db)][0] += 1
                # correct NUMBER: retrieved a noun carrying the verb's number (the agreement decision)
                if top_num == p["verb_num"]:
                    correct_num += 1
                    num_correct_cond[cond_key][0] += 1
                else:
                    # ATTRACTION ERROR: bound to a WRONG-number noun (the interference failure mode)
                    attract_cond[cond_key][0] += 1
        # online store update: index every content noun with a clear number as an item
        num = int(nn_pos[t])
        if num != 0:
            feats = (("class", "content"), ("number", NUMSTR[num]))
            store.observe(feats, (int(stream[t]), t), t=t)
        store.tick()
    return {
        "acc_subject": correct_subj / total,
        "acc_number": correct_num / total,
        "total": total,
        "by_dist": dict(by_dist),
        "cond": cond,
        "num_cond": num_correct_cond,
        "attract_cond": attract_cond,
        "fan_by_cond": {k: (float(np.mean(v)) if v else 0.0) for k, v in fan_by_cond.items()},
        "cond_dist": dict(cond_dist),
        "fan_cond_dist": {k: (float(np.mean(v)) if v else 0.0) for k, v in fan_cond_dist.items()},
    }


def eval_offset_attention(stream, probes, noun_num):
    """Offset-attention baseline on the SAME retrieval target. Offset-attention keys by POSITION, so to
    'retrieve the subject' it can only point at a FIXED relative offset. The fairest count-based form:
    pick, among the look-back window, the position whose offset carries the most information gain about
    the verb's number — but the offset key is the same for every probe, so it cannot adapt to where the
    subject actually sits. We give it its BEST shot: at each probe, it 'retrieves' the noun at the single
    offset that, over training, most often held the correct-number subject (the modal subject offset).
    That fixed offset is the position-key analogue of the content cue."""
    # learn the modal distance of the matching subject (the best a fixed-offset model can do)
    dists = defaultdict(int)
    for p in probes:
        dists[p["dist"]] += 1
    best_offset = max(dists, key=dists.get)
    NUMSTR = {1: "singular", 2: "plural"}
    by_dist = defaultdict(lambda: [0, 0])
    cond = {"no_distractor": [0, 0], "opp_distractor": [0, 0], "same_distractor": [0, 0]}
    num_correct_cond = {"no_distractor": [0, 0], "opp_distractor": [0, 0], "same_distractor": [0, 0]}
    correct_subj = 0
    correct_num = 0
    total = 0
    for p in probes:
        t = p["t"]
        pos = t - best_offset                      # fixed-offset 'retrieval' — same key for every probe
        total += 1
        cond_key = ("same_distractor" if p["same_distractor"]
                    else "opp_distractor" if p["opp_distractor"]
                    else "no_distractor")
        db = min(p["dist"] // 4 * 4, 36)
        by_dist[db][1] += 1
        cond[cond_key][1] += 1
        num_correct_cond[cond_key][1] += 1
        if 0 <= pos < t:
            if pos == p["subj_pos"]:
                correct_subj += 1
                by_dist[db][0] += 1
                cond[cond_key][0] += 1
            if NUMSTR.get(int(noun_num[stream[pos]])) == p["verb_num"]:
                correct_num += 1
                num_correct_cond[cond_key][0] += 1
    return {
        "best_offset": best_offset,
        "acc_subject": correct_subj / total,
        "acc_number": correct_num / total,
        "total": total,
        "by_dist": dict(by_dist),
        "cond": cond,
        "num_cond": num_correct_cond,
    }


def fmt_bucketed(by_dist):
    """{dist_bucket: [correct, total]} -> {dist_bucket: accuracy} for buckets with enough support."""
    return {db: c / n for db, (c, n) in by_dist.items() if n >= 20}


def main():
    print(f"Exp AO — cue-based retrieval with fan interference   "
          f"(vocab={VOCAB}, {NBYTES//1_000_000}MB text8, half-life={HALF_LIFE})")
    print("loading text8 ...")
    ids = corpus.load_ids("text8", NBYTES)
    spans = corpus.split_words(ids)
    stream, vocab_list, UNK = build_word_stream(ids, spans, VOCAB)
    print(f"words: {len(stream):,}   vocab {VOCAB}+UNK   OOV {np.mean(stream==UNK):.3f}")

    noun_num, verb_num_tab, det_num, decode = build_label_tables(vocab_list, UNK)
    print("constructing agreement probes ...")
    t0 = time.time()
    probes = build_probes(stream, noun_num, verb_num_tab, det_num)
    nopp = sum(p["opp_distractor"] and not p["same_distractor"] for p in probes)
    nsame = sum(p["same_distractor"] for p in probes)
    nnone = sum(not p["opp_distractor"] and not p["same_distractor"] for p in probes)
    dmean = float(np.mean([p["dist"] for p in probes]))
    print(f"  {len(probes):,} probes ({time.time()-t0:.1f}s)   mean subject->verb distance {dmean:.1f} words")
    print(f"  conditions: no-distractor {nnone:,}   opp-number distractor {nopp:,}   "
          f"same-number competitor {nsame:,}")

    print("\nrunning CUE-RETRIEVAL (content-addressable, fan-weighted) ...")
    t0 = time.time()
    cr = eval_cue_retrieval(stream, probes, noun_num)
    print(f"  done ({time.time()-t0:.1f}s)")

    print("running OFFSET-ATTENTION baseline (fixed position key) ...")
    oa = eval_offset_attention(stream, probes, noun_num)
    print(f"  best fixed offset = {oa['best_offset']} words")

    # === Q3 head-to-head (print first; it frames everything) ===
    print("\n=== Q3: cue-retrieval vs offset-attention (same probes, same target) ===")
    print(f"  {'model':<24} {'subject-acc':>12} {'number-acc':>12}")
    print(f"  {'cue-retrieval':<24} {cr['acc_subject']*100:11.2f}% {cr['acc_number']*100:11.2f}%")
    print(f"  {'offset-attention':<24} {oa['acc_subject']*100:11.2f}% {oa['acc_number']*100:11.2f}%")

    # === Q1 accuracy vs distance ===
    print("\n=== Q1: retrieval accuracy vs subject->verb distance (subject-exact) ===")
    print(f"  {'dist':>6} {'cue-retr':>10} {'offset':>10} {'n':>8}")
    crd = dict(fmt_bucketed(cr["by_dist"]))
    oad = dict(fmt_bucketed(oa["by_dist"]))
    for db in sorted(set(crd) | set(oad)):
        cn = cr["by_dist"].get(db, [0, 0])[1]
        cacc = crd.get(db)
        oacc = oad.get(db)
        cs = f"{cacc*100:9.2f}%" if cacc is not None else "      -- "
        os_ = f"{oacc*100:9.2f}%" if oacc is not None else "      -- "
        print(f"  {db:>4}+ {cs} {os_} {cn:>8}")

    # === Q2 interference signature ===
    print("\n=== Q2: similarity-based (fan) interference ===")
    print(f"  {'condition':<22} {'num-acc':>10} {'attract-err':>12} {'subj-exact':>11} {'mean-fan':>10} {'n':>8}")
    for k in ("no_distractor", "opp_distractor", "same_distractor"):
        c, n = cr["cond"][k]
        nc, nn = cr["num_cond"][k]
        ac, an = cr["attract_cond"][k]
        fan = cr["fan_by_cond"].get(k, 0.0)
        if n:
            print(f"  {k:<22} {nc/nn*100:9.2f}% {ac/an*100:11.2f}% {c/n*100:10.2f}% {fan:10.2f} {n:>8}")
    print("  (attract-err = top retrieval has the WRONG number = agreement attraction;")
    print("   opp_distractor has a recent wrong-number competitor, so it should attract MOST.)")

    # distance-CONTROLLED: same vs no distractor WITHIN each distance band (isolates fan from distance)
    print("\n  distance-controlled (subj-acc | mean-fan) — same-number competitor vs none, per band:")
    print(f"  {'dist':>6} {'none-acc':>9} {'same-acc':>9} {'none-fan':>9} {'same-fan':>9} {'n_none':>8} {'n_same':>8}")
    cd = cr["cond_dist"]; fcd = cr["fan_cond_dist"]
    for db in sorted({k[1] for k in cd}):
        nc = cd.get(("no_distractor", db), [0, 0])
        sc = cd.get(("same_distractor", db), [0, 0])
        if nc[1] >= 20 and sc[1] >= 20:
            print(f"  {db:>4}+ {nc[0]/nc[1]*100:8.2f}% {sc[0]/sc[1]*100:8.2f}% "
                  f"{fcd.get(('no_distractor',db),0):9.2f} {fcd.get(('same_distractor',db),0):9.2f} "
                  f"{nc[1]:>8} {sc[1]:>8}")

    # machine-readable dump
    dump = {
        "cue": {"acc_subject": round(cr["acc_subject"], 5), "acc_number": round(cr["acc_number"], 5),
                "cond": {k: (round(v[0]/v[1], 5) if v[1] else None) for k, v in cr["cond"].items()},
                "num_cond": {k: (round(v[0]/v[1], 5) if v[1] else None) for k, v in cr["num_cond"].items()},
                "attract_cond": {k: (round(v[0]/v[1], 5) if v[1] else None) for k, v in cr["attract_cond"].items()},
                "fan": {k: round(v, 3) for k, v in cr["fan_by_cond"].items()},
                "by_dist": {k: (round(v[0]/v[1], 5) if v[1] else None) for k, v in cr["by_dist"].items()}},
        "offset": {"best_offset": oa["best_offset"], "acc_subject": round(oa["acc_subject"], 5),
                   "acc_number": round(oa["acc_number"], 5),
                   "cond": {k: (round(v[0]/v[1], 5) if v[1] else None) for k, v in oa["cond"].items()},
                   "by_dist": {k: (round(v[0]/v[1], 5) if v[1] else None) for k, v in oa["by_dist"].items()}},
        "n_probes": len(probes), "mean_dist": round(dmean, 2),
    }
    print("\nRESULTS_DICT = " + repr(dump))


if __name__ == "__main__":
    main()
