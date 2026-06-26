#!/usr/bin/env python3
"""Exp V — Trajectory / change memory & affordances.

Operationalizes TBP's "Trajectory Memory for Behavior Models" on the text stream:
  (a) a behavior = a SEQUENCE OF CHANGES, learned INDEPENDENT of the object, SHARED across objects
      ("the movements are shared, the locations are unique");
  (b) trajectories are DIRECTIONAL ("you can't write your signature backwards");
  (c) AFFORDANCES — a keyframe feature predicts which behavior/trajectory is beginning.

Three measured tests. ONLINE ONLY: every model is a single-pass count table (np.unique counting, the
streaming-equivalent of leaky counters). NO gradient descent, NO k-means/SVD/eigendecomposition.

Judged on the RIGHT AXIS (fragile-ideas): TRANSFER gap, DIRECTIONAL asymmetry, PRIMING lift — not raw bpc.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words
from trajectory import (CountColumn, class_change, NCHG, char_class, NCLS,
                        word_arrays, first_class, V, SPACE)

TRAIN = 12_000_000
EVAL  = 1_000_000
rng = np.random.default_rng(0)

def hr(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — CHANGE/TRAJECTORY THAT GENERALIZES ACROSS CONTENT (transfer)
# ════════════════════════════════════════════════════════════════════════════
# Claim: "movements shared, locations unique". Split the VOCABULARY into two disjoint halves by word
# identity (hash of the word). Train a content char n-gram AND a change-trajectory model on half-A words
# only; evaluate BOTH on UNSEEN half-B words. The content model memorizes locations (specific spellings);
# the change model learns object-independent moves (vowel/consonant transition trajectories).
def word_hash(ids, spans):
    h = np.zeros(len(spans), np.int64)
    for i, (s, e) in enumerate(spans):
        v = 0
        for j in range(s, e): v = (v * 31 + int(ids[j]) + 1) & 0x7fffffff
        h[i] = v
    return h

def concat_words(ids, spans, mask):
    """Glue the selected words back into one stream separated by spaces (keeps within-word + boundary moves)."""
    parts = []
    for keep, (s, e) in zip(mask, spans):
        if keep:
            parts.append(ids[s:e]); parts.append(np.array([SPACE], np.int8))
    return np.concatenate(parts).astype(np.int8)

def test1(ids):
    hr("TEST 1 — CHANGE/TRAJECTORY GENERALIZATION ACROSS UNSEEN CONTENT  (transfer)")
    spans = split_words(ids[:TRAIN])
    h = word_hash(ids[:TRAIN], spans)
    inA = (h % 2) == 0                                   # disjoint vocab halves by word identity
    # word streams
    streamA = concat_words(ids[:TRAIN], spans, inA)
    streamB = concat_words(ids[:TRAIN], spans, ~inA)
    # disjointness check
    setA = set(map(int, h[inA])); setB = set(map(int, h[~inA]))
    print(f"  vocab split: |A_words|={inA.sum()}  |B_words|={(~inA).sum()}  hash-overlap={len(setA & setB)}")
    print(f"  streamA chars={len(streamA):,}  streamB chars={len(streamB):,}")

    # CONTENT model: char backoff (order 5) — memorizes spellings (locations)
    content = CountColumn(base=V, order=5).learn(streamA)
    # CHANGE model: class-transition trajectory (order 4 over the 9-symbol move alphabet) — moves
    chgA = class_change(streamA); chgB = class_change(streamB)
    change = CountColumn(base=NCHG, order=4).learn(chgA)

    c_self = content.batch_logloss(streamA); c_xfer = content.batch_logloss(streamB)
    g_self = change.batch_logloss(chgA);     g_xfer = change.batch_logloss(chgB)
    # normalize transfer degradation to each model's own seen baseline (different alphabets/bpc scales)
    print(f"\n  {'model':28s} {'bpsym(seen A)':>14s} {'bpsym(unseen B)':>16s} {'degradation':>12s}")
    print(f"  {'content char-5 (locations)':28s} {c_self:14.3f} {c_xfer:16.3f} {(c_xfer-c_self)/c_self*100:+10.1f}%")
    print(f"  {'change traj-4 (movements)':28s} {g_self:14.3f} {g_xfer:16.3f} {(g_xfer-g_self)/g_self*100:+10.1f}%")
    print(f"\n  VERDICT axis = TRANSFER: change model degrades {(g_xfer-g_self)/g_self*100:+.1f}% vs "
          f"content {(c_xfer-c_self)/c_self*100:+.1f}% on unseen vocab.")
    return dict(c_self=c_self, c_xfer=c_xfer, g_self=g_self, g_xfer=g_xfer)

# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — FORWARD-DIRECTIONALITY ("you can't speak backwards")
# ════════════════════════════════════════════════════════════════════════════
# Train a FORWARD char n-gram (predict next | prev k). Measure:
#   (a) forward prediction quality;
#   (b) the SAME forward model abused to predict the PREVIOUS char from the FOLLOWING context (reverse);
#   (c) a SEPARATELY-trained reverse model (count on the reversed stream).
# A directional trajectory should give forward >> forward-used-backward, recovered by a separate reverse memory.
def test2(ids):
    hr('TEST 2 — FORWARD DIRECTIONALITY  ("you can\'t speak backwards")')
    train = ids[:TRAIN]; ev = ids[TRAIN:TRAIN + EVAL]
    fwd = CountColumn(base=V, order=5).learn(train)
    rev_train = train[::-1].copy()
    rev = CountColumn(base=V, order=5).learn(rev_train)        # separate reverse memory

    # (a) forward: predict ev[t] | ev[t-k:t]
    a_bpc = fwd.batch_logloss(ev); a_acc = fwd.acc(ev)
    # (c) separate reverse model: predict ev[t] | ev[t+1:t+1+k]  == forward-loss on reversed eval
    ev_rev = ev[::-1].copy()
    c_bpc = rev.batch_logloss(ev_rev); c_acc = rev.acc(ev_rev)
    # (b) forward model USED backward: predict previous char ev[t-1] from following context ev[t:t+k].
    #     The forward table maps ctx=ids[t-k:t] -> ids[t]. To reverse-query it we feed the FOLLOWING chars
    #     as if they were a preceding context (the model has no separate reverse memory). Equivalent:
    #     score the reversed eval stream with the FORWARD-trained table.
    b_bpc = fwd.batch_logloss(ev_rev); b_acc = fwd.acc(ev_rev)

    print(f"  {'direction':38s} {'bpc':>8s} {'next-acc':>9s}")
    print(f"  {'(a) forward  (next | prev)':38s} {a_bpc:8.3f} {a_acc:9.3f}")
    print(f"  {'(b) FORWARD model used BACKWARD':38s} {b_bpc:8.3f} {b_acc:9.3f}")
    print(f"  {'(c) SEPARATE reverse model':38s} {c_bpc:8.3f} {c_acc:9.3f}")
    print(f"\n  asymmetry: fwd-used-backward is {b_bpc-a_bpc:+.3f} bpc worse than forward; "
          f"a dedicated reverse memory recovers to {c_bpc:.3f} bpc ({b_bpc-c_bpc:+.3f} better than abuse).")
    return dict(a_bpc=a_bpc, a_acc=a_acc, b_bpc=b_bpc, b_acc=b_acc, c_bpc=c_bpc, c_acc=c_acc)

# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — AFFORDANCES (keyframe feature -> which trajectory begins)
# ════════════════════════════════════════════════════════════════════════════
# At a WORD BOUNDARY, does a keyframe feature (the trigger seen AT the boundary) predict the upcoming
# word's class, giving a top-down prime that improves the first chars of the word?
# Trigger = the LAST char of the PREVIOUS word (the feature present as the new trajectory begins).
# We measure prediction of the first char of each word: baseline P(first) vs primed P(first | trigger),
# and also the first-char class (vowel/consonant) lift. This is the affordance: "I see this feature, it
# suggests which behavior begins."
def _affordance_id(prev_first, prev_last, prev_lbucket):
    """Keyframe trigger = a compact CLUSTER of the PREVIOUS word (the feature present as the new
    trajectory begins): (its first char, last char, length-bucket). Object-independent register cue.
    Built online by counting — no clustering optimisation."""
    return (prev_first * V + prev_last) * 8 + prev_lbucket           # cardinality V*V*8

def test3(ids):
    hr("TEST 3 — AFFORDANCES  (boundary keyframe feature primes the next trajectory)")
    train = ids[:TRAIN]; ev = ids[TRAIN:TRAIN + EVAL]
    sp_tr = split_words(train); sp_ev = split_words(ev)
    fa, la, lena, lba = word_arrays(train, sp_tr)
    fe, le, lene, lbe = word_arrays(ev, sp_ev)

    # Two trigger strengths: (1) bare boundary char = last char of prev word; (2) prev-word cluster.
    def shift(prev): return np.concatenate([[0], prev[:-1]]).astype(np.int64)
    trig1_tr = shift(la); trig1_ev = shift(le); CARD1 = V
    aff_tr = _affordance_id(fa, la, lba); aff_ev = _affordance_id(fe, le, lbe)
    trig2_tr = shift(aff_tr); trig2_ev = shift(aff_ev); CARD2 = V * V * 8

    def primed_bits_acc(trig_tr, trig_ev, card, target_tr, target_ev, tcard):
        tab = np.full((card, tcard), 0.5)
        np.add.at(tab, (trig_tr, target_tr), 1.0)
        p = tab / tab.sum(1, keepdims=True)
        bits = -np.log2(p[trig_ev, target_ev]).mean()
        acc = (np.argmax(tab, 1)[trig_ev] == target_ev).mean()
        return bits, acc
    def base_bits_acc(target_tr, target_ev, tcard):
        cnt = np.bincount(target_tr, minlength=tcard).astype(np.float64) + 0.5; p = cnt / cnt.sum()
        return -np.log2(p[target_ev]).mean(), (np.argmax(cnt) == target_ev).mean()

    # --- predict first char of upcoming word (the trajectory's opening keyframe) ---
    b_bits, b_acc = base_bits_acc(fa, fe, V)
    p1_bits, p1_acc = primed_bits_acc(trig1_tr, trig1_ev, CARD1, fa, fe, V)
    p2_bits, p2_acc = primed_bits_acc(trig2_tr, trig2_ev, CARD2, fa, fe, V)
    print(f"  {'predict WORD-INITIAL char':40s} {'bits':>8s} {'top1':>6s} {'lift':>7s}")
    print(f"  {'baseline  P(first)':40s} {b_bits:8.3f} {b_acc:6.3f} {'':>7s}")
    print(f"  {'primed by boundary char (weak trig)':40s} {p1_bits:8.3f} {p1_acc:6.3f} {b_bits-p1_bits:+7.3f}")
    print(f"  {'primed by prev-word cluster (afford)':40s} {p2_bits:8.3f} {p2_acc:6.3f} {b_bits-p2_bits:+7.3f}")

    # --- predict first-char CLASS (vowel/consonant register of the upcoming trajectory) ---
    fc_tr = first_class(fa); fc_ev = first_class(fe)
    bc_bits, _ = base_bits_acc(fc_tr, fc_ev, 2)
    pc_bits, _ = primed_bits_acc(trig2_tr, trig2_ev, CARD2, fc_tr, fc_ev, 2)
    print(f"\n  first-char CLASS:  baseline {bc_bits:.3f} -> afford-primed {pc_bits:.3f} bits ({pc_bits-bc_bits:+.3f})")

    # --- predict the first 1..3 chars as a JOINT trajectory opening (does the prime help the whole onset?) ---
    print(f"\n  {'opening trajectory P(first n chars)':40s} {'base':>8s} {'primed':>8s} {'lift':>7s}")
    for n in (1, 2, 3):
        # encode first-n chars as one symbol; only words with len>=n
        def firstn(ids_, spans, n):
            ok = [(s, e) for s, e in spans if e - s >= n]
            arr = np.array([[ids_[s + j] for j in range(n)] for s, e in ok], np.int64)
            code = np.zeros(len(arr), np.int64)
            for j in range(n): code = code * V + arr[:, j]
            return code, ok
        ct, ok_t = firstn(train, sp_tr, n); ce, ok_e = firstn(ev, sp_ev, n)
        # rebuild aligned triggers for the filtered word sets
        idx_t = {id(o): k for k, o in enumerate(sp_tr)}
        # simpler: recompute trigger per kept word via position in original span list
        keep_t = np.array([e - s >= n for s, e in sp_tr]); keep_e = np.array([e - s >= n for s, e in sp_ev])
        tcard = V ** n
        bb, _ = base_bits_acc(ct, ce, tcard)
        pp, _ = primed_bits_acc(trig2_tr[keep_t], trig2_ev[keep_e], CARD2, ct, ce, tcard)
        print(f"  {'  n=%d' % n:40s} {bb:8.3f} {pp:8.3f} {bb-pp:+7.3f}")

    print(f"\n  PRIMING lift (best, prev-word affordance): {b_bits-p2_bits:+.3f} bits / first char "
          f"({(b_bits-p2_bits)/b_bits*100:.1f}% reduction); acc {b_acc:.3f}->{p2_acc:.3f}.")
    return dict(b_bits=b_bits, p1_bits=p1_bits, p2_bits=p2_bits, b_acc=b_acc, p2_acc=p2_acc,
                bc=bc_bits, pc=pc_bits)

if __name__ == "__main__":
    t0 = time.time()
    print(f"loading {TRAIN+EVAL:,} chars of text8 ...")
    ids = load_ids("text8", TRAIN + EVAL + 100)
    print(f"  loaded {len(ids):,} ids in {time.time()-t0:.1f}s")
    r1 = test1(ids)
    r2 = test2(ids)
    r3 = test3(ids)
    print(f"\n[done in {time.time()-t0:.1f}s]")
