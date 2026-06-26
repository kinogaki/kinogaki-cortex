#!/usr/bin/env python3
"""Exp BF — Margin-gated production: read the same counts the hard way (G6).

Cognitive frame. A child UNDERSTANDS far more words than she SAYS — comprehension runs ahead of
production for months ("understands but won't say it yet"). The standard story makes these two
faculties two separate systems. G6's claim is leaner: they are ONE count table read in two
DIRECTIONS. The spine learns a single (cue -> label) co-occurrence table. Then

  COMPREHENSION = read cue -> label, ONE-to-MANY, FORGIVING: is the heard label compatible with
    the cue? No gate — any plausible label is recognised. (Comprehension is cheap; it comes early.)
  PRODUCTION    = read cue -> the ONE label, MANY-to-ONE, COMMITTING: to SAY a word you must beat
    its competitors. Read the SAME counts through a MARGIN gate:
        activation(label|cue) = count * AB-frequency / FAN(cue)
        emit top  iff  activation(top)/activation(2nd) >= theta_emit
    else back off to a generic label or DEFER (stay silent).

The structural prediction we test (the kill axis): the SAME table, read the two ways, reproduces
the C>P GAP and the gap SHRINKS as evidence (per-cue count) grows — and gated production is MORE
PRECISE than ungated argmax at MATCHED RECALL. If gating buys no precision at matched recall, or
the gap never appears / never shrinks, the structural-gap claim fails.

Corpus: text8 word-level (small slice this pass). A cue = the LEFT context (previous word, a slot's
left bracket à la AF); a label = the content word that fills the slot. Held-out probe positions
score the two read directions. Online single pass, bounded per-cue store, fixed seed, no gradients.

Baselines: (1) ungated argmax production (emit the top label always — theta=1); (2) offset/raw
production WITHOUT the fan division and WITHOUT the AB frequency (raw-count argmax + raw-count
margin) — isolates what fan + AB buy. FRAGILE budget: theta swept over >=10 values.
"""
import os, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus, margingen
from offsetattn import build_word_stream
from cueretrieval import FUNCTION

VOCAB = 20000
NBYTES = 10_000_000        # ~1.7M words of text8 — a small fast slice for a first pass
CAP = 32                   # per-cue bounded label store
SEED = 0
TRAIN_FRAC = 0.85          # online stream split: learn on the prefix, probe on the held-out tail
THETAS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 1e9]   # >=10: the FRAGILE sweep


def is_content(word):
    """Content-word filter: a label worth producing is a content word (drop closed-class function
    words and very short tokens). Production targets are the words a child would 'say'."""
    return len(word) >= 3 and word not in FUNCTION


def build_stream():
    ids = corpus.load_ids("text8", NBYTES)
    spans = corpus.split_words(ids)
    stream, vocab_list, UNK = build_word_stream(ids, spans, VOCAB)
    return stream, vocab_list, UNK


def run():
    print(f"Exp BF — margin-gated production   (vocab={VOCAB}, {NBYTES//1_000_000}MB text8, cap={CAP})")
    print("loading text8 ...")
    t0 = time.time()
    stream, vocab_list, UNK = build_stream()
    content = np.zeros(UNK + 1, dtype=bool)
    for wid, w in enumerate(vocab_list):
        content[wid] = is_content(w)
    n = len(stream)
    split = int(n * TRAIN_FRAC)
    print(f"words: {n:,}   vocab {VOCAB}+UNK   OOV {np.mean(stream==UNK):.3f}   "
          f"train {split:,} / probe {n-split:,}   ({time.time()-t0:.1f}s)")

    # ---- ONE online pass building the (cue -> label) table; cue = previous word ------------------
    # We also record, for each PROBE position, the cue's count AT PROBE TIME (evidence) so we can
    # bucket the C>P gap by how much the model has seen — testing whether the gap shrinks.
    store = margingen.CueLabelStore(cap=CAP)
    cue_count_at = {}                 # probe index -> #times the cue had been seen before the probe
    cue_seen = defaultdict(int)
    probes = []                       # (probe_index, cue, true_label)
    for i in range(1, n):
        cue = int(stream[i - 1])
        lab = int(stream[i])
        if i >= split and lab != UNK and content[lab] and cue != UNK:
            # held-out probe: a content label with a real left-context cue
            cue_count_at[len(probes)] = cue_seen[cue]
            probes.append((i, cue, lab))
        if i < split:
            store.observe(cue, lab, t=i)     # learn only on the training prefix (clean held-out)
            cue_seen[cue] += 1
        store.tick()
    print(f"probes: {len(probes):,} held-out content-label positions")

    # ---- the two read directions over the SAME frozen table at probe time ------------------------
    # Comprehension (no gate) vs Production (margin gate), per theta. Recall = fraction of probes the
    # producer EMITS on (margin clears theta); precision = of those, fraction correct. Comprehension
    # has no gate, so its "recall" is 1 wherever the cue is known; we report its accuracy among known.
    T = split  # freeze-time: read the table as of the end of training (causal: probes are after it)

    # comprehension: among probes whose cue is known, is the TRUE label recognised (present), and
    # is it the cue's top-1 (a strict comprehension-accuracy)? one-to-many => present is enough, but
    # we also report top-1 to compare apples-to-apples with production's argmax.
    comp_known = 0
    comp_present = 0          # true label present among the cue's labels (forgiving recognition)
    comp_top1 = 0            # true label is the cue's argmax (strict)
    for pi, cue, lab in probes:
        rec, rank, nlab = store.comprehend(cue, lab, t=T)
        if nlab > 0:
            comp_known += 1
            if rec:
                comp_present += 1
                if rank == 0:
                    comp_top1 += 1
    comp_present_acc = comp_present / max(1, comp_known)
    comp_top1_acc = comp_top1 / max(1, comp_known)

    # production: for each theta, emit-or-defer; precision among emitted, recall over all probes.
    def production_pass(theta, use_fan_ab=True):
        emitted = 0
        correct = 0
        for pi, cue, lab in probes:
            if use_fan_ab:
                out, margin, info = store.produce(cue, t=T, theta=theta)
            else:
                out, margin, info = produce_raw(store, cue, t=T, theta=theta)
            if out is not None:
                emitted += 1
                if out == lab:
                    correct += 1
        prec = correct / max(1, emitted)
        recall = emitted / len(probes)            # recall = fraction of probes spoken on
        cov_correct = correct / len(probes)       # absolute correct-emission rate (precision*recall-ish)
        return prec, recall, cov_correct, emitted

    sweep = []
    for th in THETAS:
        prec, recall, cov, emitted = production_pass(th, use_fan_ab=True)
        sweep.append((th, prec, recall, cov, emitted))

    # ungated argmax baseline = theta=1 with fan/AB (emit always when a label exists)
    ungated_prec, ungated_recall, _, _ = production_pass(1.0, use_fan_ab=True)

    # raw baseline (no fan, no AB): margin on raw counts only — what fan+AB buy
    raw_sweep = []
    for th in THETAS:
        prec, recall, cov, emitted = production_pass(th, use_fan_ab=False)
        raw_sweep.append((th, prec, recall, cov, emitted))

    # ---- the C>P gap, and whether it SHRINKS with evidence ---------------------------------------
    # Bucket probes by the cue's count at probe time (how much evidence). In each bucket measure
    # comprehension-present-acc and gated-production precision at a FIXED theta. The gap = comp - prod
    # should be large at low evidence and shrink as evidence grows.
    GATE_THETA = 2.0
    ev_buckets = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 256), (256, 10**9)]
    bucket_stats = []
    for lo, hi in ev_buckets:
        c_known = c_present = c_top1 = 0
        p_emit = p_correct = 0
        for k, (pi, cue, lab) in enumerate(probes):
            ev = cue_count_at[k]
            if not (lo <= ev < hi):
                continue
            rec, rank, nlab = store.comprehend(cue, lab, t=T)
            if nlab > 0:
                c_known += 1
                if rec:
                    c_present += 1
                    if rank == 0:
                        c_top1 += 1
            out, margin, info = store.produce(cue, t=T, theta=GATE_THETA)
            if out is not None:
                p_emit += 1
                if out == lab:
                    p_correct += 1
        if c_known >= 30:
            comp_acc = c_present / c_known            # forgiving recognition (one-to-many)
            comp_top1 = c_top1 / c_known              # strict comprehension (argmax) — production's twin
            prod_prec = p_correct / max(1, p_emit)
            prod_recall = p_emit / c_known
            bucket_stats.append((lo, hi, c_known, comp_acc, comp_top1, prod_prec, prod_recall,
                                 comp_acc - prod_prec))

    # ---- report ----------------------------------------------------------------------------------
    print(f"\n=== comprehension (cue->label, NO gate) over {comp_known:,} known-cue probes ===")
    print(f"  recognised (present, one-to-many forgiving): {comp_present_acc*100:6.2f}%")
    print(f"  strict top-1 (cue's argmax == true label)  : {comp_top1_acc*100:6.2f}%")

    print(f"\n=== production (cue->the one label, MARGIN gated) — theta sweep [fan+AB] ===")
    print(f"  {'theta':>7} {'precision':>10} {'recall':>9} {'correct/all':>12} {'emitted':>9}")
    for th, prec, recall, cov, emitted in sweep:
        ths = "inf" if th >= 1e8 else f"{th:.2f}"
        print(f"  {ths:>7} {prec*100:9.2f}% {recall*100:8.2f}% {cov*100:11.2f}% {emitted:>9}")
    print(f"  [ungated argmax (theta=1)] precision {ungated_prec*100:.2f}% @ recall {ungated_recall*100:.2f}%")

    print(f"\n=== baseline: RAW-count production (no fan, no AB) — theta sweep ===")
    print(f"  {'theta':>7} {'precision':>10} {'recall':>9} {'correct/all':>12}")
    for th, prec, recall, cov, emitted in raw_sweep:
        ths = "inf" if th >= 1e8 else f"{th:.2f}"
        print(f"  {ths:>7} {prec*100:9.2f}% {recall*100:8.2f}% {cov*100:11.2f}%")

    # gated precision at recall MATCHED to ungated: find the theta whose recall is closest to a target
    # and compare precision. Cleanest summary: does ANY theta beat ungated precision while keeping
    # recall? Report the precision LIFT at the highest theta whose recall >= 50% of ungated recall.
    target_recall = 0.5 * ungated_recall
    best = None
    for th, prec, recall, cov, emitted in sweep:
        if recall >= target_recall:
            if best is None or prec > best[1]:
                best = (th, prec, recall)
    print(f"\n=== the gate's value: precision LIFT at matched recall ===")
    print(f"  ungated argmax           : precision {ungated_prec*100:6.2f}%  recall {ungated_recall*100:6.2f}%")
    if best:
        th, prec, recall = best
        ths = "inf" if th >= 1e8 else f"{th:.2f}"
        print(f"  gated (theta={ths}, recall>={target_recall*100:.1f}%): "
              f"precision {prec*100:6.2f}%  recall {recall*100:6.2f}%   "
              f"LIFT {(prec-ungated_prec)*100:+.2f} pts")
        gate_lift = prec - ungated_prec
    else:
        gate_lift = None
        print("  (no theta retained >=50% of ungated recall — gate too aggressive on this slice)")

    print(f"\n=== C>P gap vs evidence (cue count at probe), gate theta={GATE_THETA} ===")
    print(f"  {'cue-count':>12} {'n':>7} {'comp-any':>9} {'comp-top1':>10} {'prod-prec':>10} "
          f"{'p-recall':>9} {'gap(any-P)':>11}")
    for lo, hi, nk, ca, ct, pp, pr, gap in bucket_stats:
        rng = f"{lo}-{hi if hi < 10**8 else 'inf'}"
        print(f"  {rng:>12} {nk:>7} {ca*100:8.2f}% {ct*100:9.2f}% {pp*100:9.2f}% {pr*100:8.2f}% {gap*100:+10.2f}")
    # the STRICT gap (top-1 comprehension vs gated production) is the apples-to-apples twin:
    print("  (comp-any = forgiving one-to-many recognition; comp-top1 = strict argmax comprehension,")
    print("   the apples-to-apples twin of gated production. strict gap = comp-top1 - prod-prec.)")

    gap_first = bucket_stats[0][7] if bucket_stats else None
    gap_last = bucket_stats[-1][7] if bucket_stats else None
    gap_shrinks = (gap_first is not None and gap_last is not None and gap_last < gap_first)
    # strict-gap trajectory (comp-top1 - prod-prec)
    strict_first = (bucket_stats[0][4] - bucket_stats[0][5]) if bucket_stats else None
    strict_last = (bucket_stats[-1][4] - bucket_stats[-1][5]) if bucket_stats else None
    strict_shrinks = (strict_first is not None and strict_last is not None and strict_last < strict_first)
    print(f"  strict gap (comp-top1 - prod-prec): first bucket {strict_first*100:+.2f} -> "
          f"last bucket {strict_last*100:+.2f}   {'SHRINKS' if strict_shrinks else 'does NOT shrink'}")

    # machine-readable dump
    dump = {
        "n_probes": len(probes),
        "comp_present_acc": round(comp_present_acc, 5),
        "comp_top1_acc": round(comp_top1_acc, 5),
        "ungated_prec": round(ungated_prec, 5),
        "ungated_recall": round(ungated_recall, 5),
        "sweep": [(("inf" if th >= 1e8 else round(th, 2)), round(p, 5), round(r, 5)) for th, p, r, c, e in sweep],
        "raw_sweep": [(("inf" if th >= 1e8 else round(th, 2)), round(p, 5), round(r, 5)) for th, p, r, c, e in raw_sweep],
        "gate_lift_at_matched_recall": (round(gate_lift, 5) if gate_lift is not None else None),
        "gap_buckets": [(lo, (hi if hi < 10**8 else "inf"), nk, round(ca, 5), round(ct, 5), round(pp, 5),
                         round(pr, 5), round(gap, 5)) for lo, hi, nk, ca, ct, pp, pr, gap in bucket_stats],
        "gap_first": (round(gap_first, 5) if gap_first is not None else None),
        "gap_last": (round(gap_last, 5) if gap_last is not None else None),
        "gap_shrinks": bool(gap_shrinks),
        "strict_gap_first": (round(strict_first, 5) if strict_first is not None else None),
        "strict_gap_last": (round(strict_last, 5) if strict_last is not None else None),
        "strict_gap_shrinks": bool(strict_shrinks),
    }
    print("\nRESULTS_DICT = " + repr(dump))
    return dump


def produce_raw(store, cue, t, theta):
    """Baseline production read WITHOUT fan division and WITHOUT the AB frequency: rank labels by
    RAW leaked count only, and gate on the raw-count margin. Isolates what fan + AB buy over a plain
    frequency argmax with a frequency margin."""
    post = store.cues.get(cue)
    info = {"deferred": True}
    if not post:
        return None, 0.0, info
    ranked = sorted(((lab, rec["n"]) for lab, rec in post.items()), key=lambda x: x[1], reverse=True)
    if not ranked:
        return None, 0.0, info
    top, a_top = ranked[0]
    if len(ranked) == 1:
        margin = float("inf")
    else:
        a_2nd = ranked[1][1]
        margin = (a_top / a_2nd) if a_2nd > 0 else float("inf")
    if margin >= theta:
        return top, margin, {"deferred": False}
    return None, margin, info


if __name__ == "__main__":
    np.random.seed(SEED)
    run()
