#!/usr/bin/env python3
"""Exp AZ — reliability-gated boundary detectors → the head-final drift (M3, scoped down).

The cognition. An infant tracks several boundary cues at once (Saffran forward-TP, Pelucchi
backward-TP, Harris branching-entropy) and learns WHICH to trust from how often each cue's boundary
coincides with a unit that turns out to be a real word. The ~13-month forward→backward-TP shift
(thin, contested) is the developmental drift this tests: on a HEAD-FINAL language — where the
case-marker / modifier comes BEFORE the head, so the end of a unit is predictable from the right —
the reliability of the backward-TP detector should rise relative to forward-TP.

The mechanism (count-native, online single pass, bounded, no backprop):
  - three detectors over a char stream: forward-TP-dip, backward-TP-dip, branching-entropy-rise;
  - each carries an Exp-AB hit/miss tally vs the eventually-stable chunk (= the gold word boundary);
  - reliability = f·c (c-discounted frequency); detectors are combined by Exp-AJ take-the-best.

The deliverable (the ONE falsifiable claim): the DRIFT Δ = reliability(backward) − reliability(forward)
rises going head-initial → head-final. We have no Japanese/Korean corpus, so we SYNTHESIZE a
frequency-matched mirror-image pair (head-initial vs head-final, identical lexical stats, order is
the only difference) and anchor with REAL English text8 (a natural head-initial language). FRAGILE:
≥10 variations of rate / tolerance / seed / vocab before any verdict. A clean negative is acceptable.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics
import boundsdrift as bd

SEED = 0
N_WORDS = 40_000          # synthetic stream length (per language)
TEXT8_BYTES = 4_000_000   # real English anchor (~4 MB, fast)


def measure(ids, rate, tol, label):
    """Run the three detectors on a char stream, return (drift, ranked, reliabilities, winner)."""
    dets = bd.run_detectors(ids, rate=rate, tol=tol)
    rel = {n: d.reliability() for n, d in dets.items()}
    ranked, winner = bd.take_the_best(dets)
    dr = bd.drift(dets)
    return dr, ranked, rel, winner


def main():
    print(f"Exp AZ — reliability-gated boundary detectors → head-final drift  (seed={SEED})")
    print("=" * 78)

    # ── the real-English anchor (head-initial natural language) ────────────────────────────────
    print(f"\nloading text8 ({TEXT8_BYTES//1_000_000}MB, natural head-initial English) ...")
    eng = corpus.load_ids("text8", TEXT8_BYTES)
    print(f"  {len(eng):,} chars")

    # ── the FRAGILE budget: sweep rate × tol × seed × vocab over the mirror-image pair ──────────
    print("\n--- DRIFT sweep: Δ = reliability(backward_TP) − reliability(forward_TP) ---")
    print("  (M3 prediction: Δ rises going head-initial → head-final)\n")
    print(f"  {'variation':<34}{'Δ_init':>9}{'Δ_final':>9}{'drift↑':>9}{'winner_init':>14}{'winner_final':>14}")

    rows = []
    variations = []
    # 1..6: rate sweep at tol=1, seed 0, default vocab
    for rate in (0.30, 0.40, 0.50, 0.60, 0.70):
        variations.append(dict(rate=rate, tol=1, seed=SEED, stems=60, marks=6, tag=f"rate={rate}"))
    # 7..9: tolerance sweep
    for tol in (0, 2):
        variations.append(dict(rate=0.50, tol=tol, seed=SEED, stems=60, marks=6, tag=f"tol={tol}"))
    # 10..12: seed sweep (robustness)
    for sd in (1, 2, 3):
        variations.append(dict(rate=0.50, tol=1, seed=sd, stems=60, marks=6, tag=f"seed={sd}"))
    # 13..14: vocab / marker-class sweep
    variations.append(dict(rate=0.50, tol=1, seed=SEED, stems=120, marks=4,  tag="stems=120,marks=4"))
    variations.append(dict(rate=0.50, tol=1, seed=SEED, stems=40,  marks=10, tag="stems=40,marks=10"))

    drifts = []
    for vrec in variations:
        ids_init  = bd.synth_corpus(N_WORDS, head_final=False, seed=vrec["seed"],
                                    vocab_stems=vrec["stems"], vocab_marks=vrec["marks"])
        ids_final = bd.synth_corpus(N_WORDS, head_final=True,  seed=vrec["seed"],
                                    vocab_stems=vrec["stems"], vocab_marks=vrec["marks"])
        di, ri, reli, wi = measure(ids_init,  vrec["rate"], vrec["tol"], "init")
        df, rf, relf, wf = measure(ids_final, vrec["rate"], vrec["tol"], "final")
        drift_up = df - di              # did backward-TP gain reliability going head-final?
        drifts.append(drift_up)
        rows.append((vrec["tag"], di, df, drift_up, wi, wf, reli, relf))
        print(f"  {vrec['tag']:<34}{di:>+9.3f}{df:>+9.3f}{drift_up:>+9.3f}{wi:>14}{wf:>14}")

    drifts = np.array(drifts)
    pos = int((drifts > 0).sum())
    print(f"\n  drift>0 in {pos}/{len(drifts)} variations   "
          f"mean drift={drifts.mean():+.4f}  median={np.median(drifts):+.4f}  "
          f"min={drifts.min():+.4f}  max={drifts.max():+.4f}")

    # ── per-detector reliability table at the canonical operating point ─────────────────────────
    print("\n--- per-detector reliability (f·c) at rate=0.5, tol=1, seed=0 ---")
    print(f"  {'corpus':<22}{'forward_TP':>12}{'backward_TP':>13}{'entropy':>10}{'  winner':>16}")
    for label, ids in [("synth head-INITIAL", bd.synth_corpus(N_WORDS, False, SEED, 60, 6)),
                       ("synth head-FINAL",   bd.synth_corpus(N_WORDS, True,  SEED, 60, 6)),
                       ("text8 (English)",    eng)]:
        dets = bd.run_detectors(ids, rate=0.5, tol=1)
        ranked, winner = bd.take_the_best(dets)
        rel = {n: d.reliability() for n, d in dets.items()}
        print(f"  {label:<22}{rel['forward_tp']:>12.4f}{rel['backward_tp']:>13.4f}"
              f"{rel['entropy']:>10.4f}{winner:>16}")

    # ── secondary check (NOT a kill axis): frequency-vs-TP discrimination ────────────────────────
    # On head-final the marker (high-freq, low-entropy) precedes the stem; a pure-frequency cut would
    # mis-place boundaries. Report whether reliabilities differ from the head-initial baseline at all.
    print("\n--- secondary (not a kill axis): does head-direction move any reliability? ---")
    di_dets = bd.run_detectors(bd.synth_corpus(N_WORDS, False, SEED, 60, 6), rate=0.5, tol=1)
    df_dets = bd.run_detectors(bd.synth_corpus(N_WORDS, True,  SEED, 60, 6), rate=0.5, tol=1)
    for n in ("forward_tp", "backward_tp", "entropy"):
        a = di_dets[n].reliability(); b = df_dets[n].reliability()
        print(f"  {n:<14} init={a:.4f}  final={b:.4f}  Δ={b-a:+.4f}")

    # machine-readable dump for RESULTS.md
    print("\nDRIFTS = " + repr([round(float(x), 4) for x in drifts]))
    print("DRIFT_POS_FRACTION = " + repr((pos, len(drifts))))
    print("MEAN_DRIFT = " + repr(round(float(drifts.mean()), 4)))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n(total {time.time()-t0:.1f}s)")
