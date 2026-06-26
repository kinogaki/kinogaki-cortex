#!/usr/bin/env python3
"""Experiment AE — online NON-FORGETTING under REAL domain shift (the transformer-differentiator).

Exp B tried to show catastrophic forgetting in ENGLISH-ONLY registers and found there was nothing to forget —
four English registers are ~one distribution at the char level. The honest fix (Exp B's own decision): test with
GENUINE register shift. We now stream truly different registers — Darwin (Victorian science) → Shakespeare (Early
Modern verse) → KJV Bible (archaic scripture) — ONE pass, NO replay of earlier text, and measure BACKWARD
RETENTION: after each phase, bpc on a held-out slice of EVERY register.

Plain count models are additive → never forget. So forgetting is only POSSIBLE under a MEMORY CAP that forces
eviction. We cap both models to the same per-order table size and compare the eviction/retention policy:

  FLAT  — single-timescale recency. Evict the least-used context. (lib.retention.FlatCount)
  DUAL  — ECAN STI/LTI + CLS fast/slow + LIDA broadcast + ART vigilance. Evict the lowest-LTI (importance)
          context; only the single most-salient winner learns each step; spawn vs refine under vigilance ρ.
          (lib.retention.DualCount)

Hypothesis: under the cap, FLAT forgets earlier registers (its contexts get evicted by the new register's flood);
DUAL retains them at comparable peak. Honest either way.

HARD RULES: online single streaming pass; no gradient descent; no batch optimization. Fixed seed.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lib.retention import (load_registers, FlatCount, DualCount,
                           retention_matrix, forgetting)

DATA = os.path.join(HERE, "..", "..", "data")
SEED = 0
np.random.seed(SEED)

# genuinely different registers, streamed in this order (no replay)
REGS = [("darwin", "darwin.txt"), ("shakespeare", "shakespeare.txt"), ("bible", "bible.txt")]
ORDER = [n for n, _ in REGS]

N_TRAIN = 120_000     # chars trained per register (one streaming pass)
N_EVAL = 10_000       # held-out chars per register
K = 5                 # max context order
CAP = 3_000           # PER-ORDER table cap — small enough that a register flood forces eviction → forgetting
RHO = 0.15            # ART vigilance: recognize at P(char|ctx) ≥ ρ; below = novel, must earn retention


def run():
    t0 = time.time()
    print(f"loading registers (char-level, K={K}, train={N_TRAIN:,}/reg, eval={N_EVAL:,}/reg, cap={CAP:,}/order)")
    regs = load_registers(REGS, DATA, N_TRAIN, N_EVAL, seed=SEED)
    for n in ORDER:
        print(f"  {n:<12} train={len(regs[n][0]):>8,}  eval={len(regs[n][1]):>7,}")

    print("\n[FLAT] single-timescale recency, evict least-used")
    flat = FlatCount(K=K, cap=CAP)
    Mf = retention_matrix(flat, regs, ORDER)

    print("\n[DUAL] STI/LTI + LIDA broadcast + ART vigilance, evict lowest-LTI")
    dual = DualCount(K=K, cap=CAP, rho=RHO, fam_gain=2.0, nov_gain=0.1, lti_init=0.0)
    Md = retention_matrix(dual, regs, ORDER)

    # ── report ──
    def show(M, tag):
        print(f"\n=== RETENTION MATRIX [{tag}]  M[after i][eval j] bpc (lower=better) ===")
        print("  after \\ eval " + "".join(f"{n[:5]:>9}" for n in ORDER))
        for i, n in enumerate(ORDER):
            print(f"  {n[:11]:<12}" + "".join(f"{M[i, j]:>9.3f}" for j in range(len(ORDER))))

    show(Mf, "FLAT"); show(Md, "DUAL")

    print("\n=== BACKWARD FORGETTING (Δ bpc on each register: final − peak; + = forgot) ===")
    print(f"  {'register':<12}{'FLAT peak':>10}{'FLAT final':>11}{'FLAT Δ':>9}"
          f"{'DUAL peak':>11}{'DUAL final':>11}{'DUAL Δ':>9}")
    ff = {n: (pk, fn, d) for n, pk, fn, d in forgetting(Mf, ORDER)}
    fd = {n: (pk, fn, d) for n, pk, fn, d in forgetting(Md, ORDER)}
    tot_f = tot_d = 0.0
    for n in ORDER[:-1]:   # earlier registers are the ones that CAN be forgotten
        pf, nf, df = ff[n]; pd, nd, dd = fd[n]
        tot_f += df; tot_d += dd
        print(f"  {n:<12}{pf:>10.3f}{nf:>11.3f}{df:>+9.3f}{pd:>11.3f}{nd:>11.3f}{dd:>+9.3f}")
    print(f"  {'TOTAL fgt':<12}{'':>10}{'':>11}{tot_f:>+9.3f}{'':>11}{'':>11}{tot_d:>+9.3f}")

    print("\n=== FINAL bpc on each register (end of stream) ===")
    print(f"  {'register':<12}{'FLAT':>9}{'DUAL':>9}")
    for j, n in enumerate(ORDER):
        print(f"  {n:<12}{Mf[-1, j]:>9.3f}{Md[-1, j]:>9.3f}")

    # peak quality (diagonal mean) — are they comparable, so retention is a fair comparison?
    peak_f = np.nanmean([Mf[i, i] for i in range(len(ORDER))])
    peak_d = np.nanmean([Md[i, i] for i in range(len(ORDER))])
    print(f"\n  mean PEAK bpc (diagonal):  FLAT={peak_f:.3f}  DUAL={peak_d:.3f}")
    print(f"  mean FINAL bpc (last row): FLAT={np.nanmean(Mf[-1]):.3f}  DUAL={np.nanmean(Md[-1]):.3f}")

    verdict = "DUAL retains better" if tot_d < tot_f - 0.02 else (
        "comparable" if abs(tot_d - tot_f) <= 0.02 else "FLAT retains better")
    print(f"\n  total backward forgetting: FLAT={tot_f:+.3f}  DUAL={tot_d:+.3f}  →  {verdict}")
    print(f"\n  ({time.time()-t0:.0f}s, seed={SEED}, single pass, no replay)")
    return Mf, Md


if __name__ == "__main__":
    run()
