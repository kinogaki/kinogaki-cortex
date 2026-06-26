#!/usr/bin/env python3
"""Experiment AK — memory-budget-as-curriculum ("starting small"), count-native.

Elman 1993: a recurrent net learned complex EMBEDDED structure ONLY if it started small — staged data, OR a
memory that started short and grew. Full complexity at full capacity from step one failed entirely. Vygotsky's
ZPD says the same from the other side: spend effort at the edge of the masterable.

Our substrate carries a bounded-memory RULE — we always have a budget. This asks whether GROWING the budget on
a schedule is itself a curriculum that beats a fixed budget, and it stages MEMORY not data (teacher-free, the
count-native "starting small"): the per-context leaky-accumulator leak-horizon H starts short and grows.

Three regimes on ONE stream, single pass, fixed seed:
  GROW   — H small → large on a linear schedule (start short: local first; grow: long-range composes on top).
  FULL   — H = the final large value from char one (Elman's full-from-start).
  FIXED  — H stays small the whole pass (a permanently short memory).

RIGHT AXIS — long-range structure. We measure perplexity on the cue-distant AGREEMENT tokens alone (the char
at the end of an embedded clause whose only cue is the subject marker `gap` chars back), not flat bpc. Elman's
prediction: GROW acquires the long-range agreement that FULL fails to; GROW > FULL and GROW > FIXED.

Plus a ZPD overlay on GROW: confidence-weighted exposure (up-weight near-threshold contexts, down-weight
mastered, defer unreachable). Honest if growing doesn't beat full-from-start.

HARD RULES: online single streaming pass; no gradients; no batch optimization; bounded memory; fixed seed.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lib.curriculum import (make_agreement_corpus, LeakyCountModel,
                            grow_schedule, fixed_schedule)

SEED = 0
np.random.seed(SEED)

N_SENT = 12000         # sentences in the single-pass stream
GAP_LO, GAP_HI = 18, 34  # embedded-clause length: the key→cue span is unique per sentence (unmemorizable)
N_KEYS = 120           # distinct subject keys → each "<key> run" context recurs only ~every 120 sentences
K = 8                  # char-context order: enough to span the full "<key> run" cue (3+1+3 chars + ' ')
ALPHA = 0.05
H_SHORT = 3.0          # short leak-horizon: ~3 recent visits of a context survive (leaks across the sparse gap)
H_LONG = 600.0         # long horizon: effectively non-leaky over the stream (accumulates the sparse cue)


def run():
    t0 = time.time()
    ids, tgt_idx, tgt_true = make_agreement_corpus(N_SENT, GAP_LO, GAP_HI, seed=SEED, n_keys=N_KEYS)
    print(f"corpus: {len(ids):,} chars, {len(tgt_idx):,} cue-distant target tokens, "
          f"{N_KEYS} keys (each '<key> run' context recurs ~every {N_KEYS} sentences — sparse), K={K}")
    print(f"regimes: GROW H {H_SHORT:g}->{H_LONG:g} | FULL H={H_LONG:g} | FIXED H={H_SHORT:g}  "
          f"(single pass, seed={SEED})\n")

    regimes = [
        ("FIXED  (small, constant)", fixed_schedule(H_SHORT), False),
        ("FULL   (large from start)", fixed_schedule(H_LONG), False),
        ("GROW   (small -> large)",   grow_schedule(H_SHORT, H_LONG), False),
        ("GROW+ZPD (conf-weighted)",  grow_schedule(H_SHORT, H_LONG), True),
    ]

    print(f"  (chance on the agreement token = ppl 2.0 / acc 0.50: a memory that loses the key sees a 50/50 coin)\n")
    rows = []
    for name, sched, zpd in regimes:
        m = LeakyCountModel(K=K, alpha=ALPHA, H_schedule=sched, zpd=zpd)
        r = m.online_pass(ids, target_idx=tgt_idx)
        rows.append((name, r["bpc"], r["target_nll"], r["target_ppl"], r["target_acc"], r["n_target"]))
        print(f"  {name:<26} bpc={r['bpc']:.3f}  target_ppl={r['target_ppl']:.3f}  "
              f"target_acc={r['target_acc']:.3f}  (n={r['n_target']})  [{time.time()-t0:.0f}s]")

    # ── verdict on the right axis: target perplexity (long-range), lower = better ──
    by = {n: ppl for n, _, _, ppl, _, _ in rows}
    fixed = by["FIXED  (small, constant)"]
    full = by["FULL   (large from start)"]
    grow = by["GROW   (small -> large)"]
    growz = by["GROW+ZPD (conf-weighted)"]

    print("\n=== RIGHT AXIS: perplexity on cue-distant agreement tokens (lower = better) ===")
    print(f"  {'regime':<26}{'target ppl':>12}{'target acc':>12}{'vs FULL':>12}")
    for n, _, _, ppl, acc, _ in rows:
        rel = (ppl - full) / full * 100.0
        print(f"  {n:<26}{ppl:>12.3f}{acc:>12.3f}{rel:>+11.1f}%")

    grow_beats_full = grow < full - 1e-3
    grow_beats_fixed = grow < fixed - 1e-3
    zpd_helps = growz < grow - 1e-3
    print()
    print(f"  GROW vs FULL : {grow:.3f} vs {full:.3f}  -> "
          f"{'GROW WINS (Elman confirmed)' if grow_beats_full else 'full-from-start ties/wins (Elman NOT reproduced)'}")
    print(f"  GROW vs FIXED: {grow:.3f} vs {fixed:.3f}  -> "
          f"{'GROW wins' if grow_beats_fixed else 'fixed ties/wins'}")
    print(f"  ZPD overlay  : {growz:.3f} vs {grow:.3f}  -> "
          f"{'ZPD helps' if zpd_helps else 'ZPD does not help'}")
    print(f"\n  ({time.time()-t0:.0f}s, single pass, no gradients, no batch opt, seed={SEED})")
    return rows


if __name__ == "__main__":
    run()
