#!/usr/bin/env python3
"""Exp AB — NARS-style calibrated confidence + precision-weighting on count-based associations.

From Pei Wang's NARS (truth = frequency + confidence), predictive-coding precision, and ACT-R. A raw
count says how OFTEN a context fired, never whether it was RIGHT. So in a single online pass we split
each context's count into prediction HITS w+ and MISSES w- (its running top-1 scored against the next
char), giving a NARS truth value: frequency f = w+/w, confidence c = w/(w+1). Three things to test:

  Q1  PREDICTION (bpc / perplexity). Does (f,c)-revision or precision-weighted pooling beat a bare
      product-of-experts over the same char orders? (Honest expectation: revision is about CALIBRATION,
      not sharpness — bpc may be flat or slightly worse.)
  Q2  CALIBRATION (the real axis). When the model says confidence is high, is it right more often?
      Measure Expected-Calibration-Error (ECE) + a reliability table for bare-count vs (f,c)-revision
      vs precision-weighted. This is where calibrated truth values should earn their keep.
  Q3  THE GATE. A principled c-based gate (open the high order exactly when its c-discounted accuracy
      f·c beats the backoff's) vs a hand-tuned confidence threshold swept over many values. Does the
      parameter-free gate match or beat the best tuned threshold?

Char level on text8. Train on a prefix, eval on a held-out suffix; single causal pass; fixed seed.
No gradients, no batch optimization (no k-means/SVD) — only counting + leaky accumulators.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids
from confidence import (CountTruth, truth_of, unigram_log, order_logdist,
                        bare_count_pool, weighted_pool, revision_truth,
                        confidence_weight, precision_weights,
                        tuned_threshold_gate, principled_gate,
                        decode_metrics, perplexity, calibration, stated_confidence,
                        _ctx_ids, V, K)

ORDERS = (2, 3, 4, 5)
TRAIN = 12_000_000
EVAL = 300_000
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN + EVAL + 1_000_000)
    train = ids[:TRAIN].astype(np.int64)
    ev = ids[TRAIN:TRAIN + EVAL].astype(np.int64)
    targets = ev[1:]
    m = len(targets)
    print(f"loaded {len(ids):,} chars  train {len(train):,}  eval {len(ev):,}  "
          f"orders {ORDERS}  ({time.time()-t0:.1f}s)")

    # ── single online pass per order: counts + NARS hit/miss split ──
    tables = {}
    for k in ORDERS:
        tables[k] = CountTruth(k).online_pass(train)
    uni = unigram_log(train)
    print(f"online pass done: per-order (w+,w-) accumulated ({time.time()-t0:.1f}s)")
    print(f"{'order':>5} | {'#ctx':>9} | {'hit-rate f̄':>10} | {'mean c':>7}")
    for k in ORDERS:
        tb = tables[k]
        f, c = truth_of(tb.wp, tb.wm)
        w = tb.wp + tb.wm
        fbar = float(tb.wp.sum() / max(1.0, w.sum()))     # overall top-1 hit-rate of this order
        print(f"{k:>5} | {len(tb.ctx_ids):>9,} | {fbar:>10.4f} | {float(c.mean()):>7.4f}")
    print()

    # ── eval-side per-order log-dists + matched contexts ──
    ctx_q = {k: _ctx_ids(ev, k) for k in ORDERS}          # context id at each eval position (t=k..)
    # align every order's per-position log-dist to the SAME m positions (predict t=1..n-1).
    # order k yields rows for t=k..n-1; pad the first k-1 positions with the unigram.
    order_lds = []
    for k in ORDERS:
        ld_k, _ = order_logdist(tables[k], ctx_q[k], uni)   # length n-k, aligns to t=k..n-1
        full = np.tile(uni, (m, 1))
        full[k - 1:] = ld_k
        order_lds.append(full)

    # ── Q1: prediction (bpc / perplexity) ──
    print("=== Q1: prediction — bare-count vs (f,c)-revision vs precision-weighted ===")
    bare = bare_count_pool(order_lds)

    # (f,c)-revision: evidence-additive, confidence-weighted (recompute on padded full ctx queries)
    ctx_full = []
    for k in ORDERS:
        cq = np.zeros(m, np.int64); cq[:] = -1
        cq[k - 1:] = ctx_q[k]
        ctx_full.append(cq)
    rev = revision_truth([tables[k] for k in ORDERS], ctx_full, uni)

    # confidence-weighted product-of-experts (each order scaled by its NARS c at this position)
    cw = []
    for k in ORDERS:
        w = np.zeros(m); w[k - 1:] = confidence_weight(tables[k], ctx_q[k])
        cw.append(w)
    conf_pool = weighted_pool(order_lds, cw)

    # precision-weighted (inverse running error-variance per order, leaky)
    pw = precision_weights(order_lds, targets)
    prec_pool = weighted_pool(order_lds, pw)

    rows_q1 = []
    for name, ld in [("bare-count pool", bare), ("conf-weighted pool", conf_pool),
                     ("(f,c)-revision", rev), ("precision-weighted", prec_pool)]:
        acc, bpc = decode_metrics(ld, targets)
        ppl = perplexity(ld, targets)
        rows_q1.append((name, acc, bpc, ppl))
        print(f"  {name:>22} | acc {acc:.4f} | bpc {bpc:.4f} | ppl {ppl:8.2f}")
    print()

    # ── Q2: calibration (ECE + reliability) ──
    print("=== Q2: calibration — does high stated confidence mean more-often-right? ===")
    rows_q2 = []
    reliab = {}
    for name, ld in [("bare-count pool", bare), ("conf-weighted pool", conf_pool),
                     ("(f,c)-revision", rev), ("precision-weighted", prec_pool)]:
        conf = stated_confidence(ld)
        correct = (ld.argmax(1) == targets).astype(float)
        ece, table = calibration(conf, correct, n_bins=10)
        rows_q2.append((name, ece, float(correct.mean())))
        reliab[name] = table
        print(f"  {name:>22} | ECE {ece:.4f} | overall acc {float(correct.mean()):.4f}")
    print()
    print("  reliability (bare-count vs (f,c)-revision): bin = stated-conf range")
    print(f"  {'bin':>10} | {'bare n':>7} {'bare conf':>9} {'bare acc':>8} | "
          f"{'rev n':>7} {'rev conf':>9} {'rev acc':>8}")
    for (lo, hi, nb, mc, ac), (lo2, hi2, nb2, mc2, ac2) in zip(reliab["bare-count pool"],
                                                                reliab["(f,c)-revision"]):
        print(f"  {lo:.1f}-{hi:.1f}  | {nb:>7,} {mc:>9.3f} {ac:>8.3f} | "
              f"{nb2:>7,} {mc2:>9.3f} {ac2:>8.3f}")
    print()

    # ── Q3: gate — principled c vs tuned threshold ──
    print("=== Q3: gate — principled (f·c) vs hand-tuned confidence threshold ===")
    # high = order 5, backoff = order 2 (the spread that makes routing matter)
    hi_k, lo_k = max(ORDERS), min(ORDERS)
    hi_ld = order_lds[ORDERS.index(hi_k)]
    lo_ld = order_lds[ORDERS.index(lo_k)]
    # full-length (m,) context-id arrays, -1 where the order has no context yet (treated as unseen)
    ctx_hi_full = ctx_full[ORDERS.index(hi_k)]
    ctx_lo_full = ctx_full[ORDERS.index(lo_k)]
    hi_conf = confidence_weight(tables[hi_k], ctx_hi_full)

    # tuned threshold sweep
    print(f"  high=order{hi_k} backoff=order{lo_k}")
    print(f"  {'gate':>26} | {'thresh':>7} | {'open%':>6} | {'acc':>7} | {'bpc':>7} | {'ppl':>9}")
    rows_q3 = []
    best_tuned = None
    for th in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        ld, openf = tuned_threshold_gate(hi_ld, lo_ld, hi_conf, th)
        acc, bpc = decode_metrics(ld, targets); ppl = perplexity(ld, targets)
        rows_q3.append(("tuned thresh", th, openf, acc, bpc, ppl))
        print(f"  {'tuned threshold':>26} | {th:>7.2f} | {openf:>6.2%} | {acc:>7.4f} | "
              f"{bpc:>7.4f} | {ppl:>9.2f}")
        if best_tuned is None or bpc < best_tuned[4]:
            best_tuned = ("tuned thresh", th, openf, acc, bpc, ppl)

    pld, popenf, pdecided = principled_gate(tables[hi_k], tables[lo_k], ctx_hi_full, ctx_lo_full,
                                            hi_ld, lo_ld)
    pacc, pbpc = decode_metrics(pld, targets); pppl = perplexity(pld, targets)
    rows_q3.append(("principled f·c", None, popenf, pacc, pbpc, pppl))
    print(f"  {'PRINCIPLED f·c (no knob)':>26} | {'--':>7} | {popenf:>6.2%} | {pacc:>7.4f} | "
          f"{pbpc:>7.4f} | {pppl:>9.2f}   (decided={pdecided:.2%})")
    # reference floors
    a2, b2 = decode_metrics(lo_ld, targets); p2 = perplexity(lo_ld, targets)
    a5, b5 = decode_metrics(hi_ld, targets); p5 = perplexity(hi_ld, targets)
    print(f"  {'(always backoff o%d)'%lo_k:>26} | {'--':>7} | {0.0:>6.2%} | {a2:>7.4f} | {b2:>7.4f} | {p2:>9.2f}")
    print(f"  {'(always high o%d)'%hi_k:>26} | {'--':>7} | {1.0:>6.2%} | {a5:>7.4f} | {b5:>7.4f} | {p5:>9.2f}")
    print()
    print(f"  best tuned: thresh={best_tuned[1]} bpc={best_tuned[4]:.4f} | "
          f"principled bpc={pbpc:.4f}  (Δ={pbpc-best_tuned[4]:+.4f})")
    print()

    # ── Q3b: the gate where it MATTERS — the RARE/UNRELIABLE high-order slice ──
    # On clean text the high order wins nearly everywhere, so "always open" is hard to beat. The gate
    # earns its keep where the high order is SPARSE: its 5-gram seen but with little evidence (low c).
    # Restrict to positions whose order-5 context confidence c < 0.5 (a thin, unreliable high context).
    # c = w/(w+1): c<0.8 ⇔ evidence mass w ≤ ~4 — a thin, unreliable high-order context.
    print("=== Q3b: gate on the RARE high-order slice (order5 c < 0.8, w≤~4 — where routing matters) ===")
    rare = (hi_conf > 0) & (hi_conf < 0.8)
    nr = int(rare.sum())
    if nr > 0:
        tr = targets[rare]
        a_hi = float((hi_ld[rare].argmax(1) == tr).mean())
        a_lo = float((lo_ld[rare].argmax(1) == tr).mean())
        b_hi = float(-(hi_ld[rare][np.arange(nr), tr] / np.log(2)).mean())
        b_lo = float(-(lo_ld[rare][np.arange(nr), tr] / np.log(2)).mean())
        # principled routing restricted to this slice
        ea_hi_r = hi_conf[rare] * 0  # placeholder; recompute f·c on the slice
        f5, c5 = truth_of(tables[hi_k].wp, tables[hi_k].wm)
        # already have pld for full; just slice it
        a_pr = float((pld[rare].argmax(1) == tr).mean())
        b_pr = float(-(pld[rare][np.arange(nr), tr] / np.log(2)).mean())
        # best-tuned gate sliced
        ld_bt, _ = tuned_threshold_gate(hi_ld, lo_ld, hi_conf, best_tuned[1])
        a_bt = float((ld_bt[rare].argmax(1) == tr).mean())
        b_bt = float(-(ld_bt[rare][np.arange(nr), tr] / np.log(2)).mean())
        print(f"  rare slice: {nr:,} positions ({nr/m:.1%} of eval)")
        print(f"  {'always high (o5)':>26} | acc {a_hi:.4f} | bpc {b_hi:.4f}")
        print(f"  {'always backoff (o2)':>26} | acc {a_lo:.4f} | bpc {b_lo:.4f}")
        print(f"  {'best tuned thresh':>26} | acc {a_bt:.4f} | bpc {b_bt:.4f}")
        print(f"  {'PRINCIPLED f·c':>26} | acc {a_pr:.4f} | bpc {b_pr:.4f}")
    print(f"done ({time.time()-t0:.1f}s)")
    return rows_q1, rows_q2, reliab, rows_q3, best_tuned, (a2, b2, p2, a5, b5, p5)


if __name__ == "__main__":
    main()
