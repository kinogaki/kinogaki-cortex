#!/usr/bin/env python3
"""Exp BC — the Rescorla-Wagner recovery loop: recovery WITHOUT feedback (M20).

The developmental puzzle: a child over-regularizes ("mouses", "goed"), then quietly stops — with
*no one correcting them* (Marcus 1992; the no-negative-evidence problem). A passive count reader
can't explain this: hearing "mice" only ever INCREMENTS count("mice"); the over-applied "mouses" is
frozen, never decremented, and only loses ground if "mice" out-frequencies it. So recovery would be
"just frequency."

Rescorla-Wagner gives the missing organ, and it is non-gradient by construction (it is THE classical
associative rule). The cue is the stem; the competing outcomes are its inflected forms. PREDICT the
form before observing, then on observing UPDATE every outcome of the cue:
    V[o] += α·(λ·1[o==heard] − ΣV)
The heard form rises; every still-EXPECTED-but-ABSENT form is pushed DOWN by α·ΣV — decremented
purely from being predicted and not seen. That is cue competition / blocking, and it is the first
acquisition use of AT's reactive contract (predict-then-update). Recovery becomes impossible for a
passive increment-only reader by construction — so we pit the two on the SAME stream.

CORPUS: a synthetic, frequency-matched two-phase morphology stream (the M20 spec asks for CHILDES
order for the realistic version; CHILDES is NOT in data/ — we SUBSTITUTE a hand-built
child-directed-ish synthetic stream and a text8 frequency check, and say so in RESULTS). Phase 1:
over-generalization (regulars + the over-applied "mouses"). Phase 2: the correct "mice" arrives with
ZERO correction. We track P("mouses") and ask: does it decay below "mice" PURELY from prediction
error, and does it beat the increment-only baseline (the kill test)?

ONLINE single streaming pass; no gradient descent / k-means / SVD / backprop; bounded (decrement
frees budget, floor-prune). Fixed seed. Reuses lib/rescorla.py; lib/corpus for the freq sanity check.
"""
import os, sys, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus  # noqa: F401 (frequency sanity check)
from rescorla import RWCompetition, CountOnly, recovery_step, recovery_slope

print = functools.partial(print, flush=True)
SEED = 0
rng = np.random.default_rng(SEED)

# ─────────────────────────────────── the synthetic stream ───────────────────────────────────
# A child-directed-ish plural stream. Many REGULAR noun stems take "-s"; ONE irregular stem
# ("mouse") is the focus. In Phase 1 the learner only ever hears the OVER-APPLIED regular form
# ("mouses") for that stem (it has generalized the +s rule). In Phase 2 the ambient input switches
# to the correct irregular ("mice") — with NO correction, NO label, NO reward. The regulars keep
# flowing so the +s rule stays alive (and so a "regular exposure matures the cue" / Ramscar check
# is possible). Each stem is its OWN cue; outcomes are that stem's surface plural forms.

REG_STEMS = ["cat", "dog", "ball", "tree", "car", "cup", "book", "hat", "bird", "shoe",
             "star", "frog", "duck", "boat", "fish_n"]   # regulars: stem -> stem+"s"
TARGET = "mouse"                                          # the irregular under study
OVERREG = "mouses"                                        # over-applied regular plural
IRREG = "mice"                                            # correct irregular plural


def make_stream(n_phase1, n_phase2, irr_ratio, target_share=0.12, rng=rng):
    """Return a list of (cue, outcome) events. `irr_ratio` = fraction of Phase-2 TARGET tokens that
    are the correct "mice" (the corrective signal strength); the rest stay "mouses" (lingering
    over-application heard in the wild). `target_share` = fraction of all tokens that are the target
    stem (vs regulars). Phase boundary at n_phase1."""
    ev = []
    def emit(n, phase):
        for _ in range(n):
            if rng.random() < target_share:
                if phase == 1:
                    ev.append((TARGET, OVERREG))                 # only over-applied form heard
                else:
                    ev.append((TARGET, IRREG if rng.random() < irr_ratio else OVERREG))
            else:
                s = REG_STEMS[rng.integers(len(REG_STEMS))]
                ev.append((s, s + "s"))                          # a correctly-regular plural
    emit(n_phase1, 1); emit(n_phase2, 2)
    return ev


def run_model(model, events, target=TARGET, probe_every=200):
    """Stream events once through `model`; record the target cue's outcome distribution over time.
    The R-W gate uses the model's OWN prediction error as the contingency signal (AT contract):
    a fully-expected observation teaches little, a surprising one teaches hard."""
    traj = []  # (event_index, {outcome: prob})
    for i, (cue, outcome) in enumerate(events):
        pred = model.predict(cue)                                # predict-then-update
        tot = sum(max(0.0, v) for v in pred.values()) or 1.0
        p_obs = max(0.0, pred.get(outcome, 0.0)) / tot if pred else 0.0
        gate = 1.0 - 0.5 * p_obs                                 # surprise gate in [0.5,1] (pure-ish)
        model.observe(cue, outcome, gate=gate)
        if i % probe_every == 0 or i == len(events) - 1:
            traj.append((i, model.p(target)))
    return traj


def target_series(traj, form):
    return [p.get(form, 0.0) for _, p in traj]


# ─────────────────────────────────── the sweep (FRAGILE: ≥10 variations) ───────────────────────────────────
print("=" * 92)
print("Exp BC — Rescorla-Wagner recovery loop: does an over-applied form decay WITHOUT correction,")
print("         and does R-W competition beat increment-only counting (the kill test)?")
print("=" * 92)

N1, N2 = 6000, 9000           # phase lengths (events); target_share ~12% => ~720 / ~1080 target tokens
PHASE_BOUNDARY = N1

# variations: corrective-signal strength (irr_ratio) × R-W learning rate alpha.
IRR_RATIOS = [0.3, 0.5, 0.7, 0.9]
ALPHAS     = [0.05, 0.15, 0.30]

rows = []
print(f"\n{'irr_ratio':>9} {'alpha':>6} | {'RW xover':>9} {'CO xover':>9} | "
      f"{'RW slope':>9} {'CO slope':>9} | {'RW Δp1':>7} {'CO Δp1':>7} | winner")
print("-" * 92)

variation = 0
for irr in IRR_RATIOS:
    events = make_stream(N1, N2, irr_ratio=irr, rng=np.random.default_rng(SEED + int(irr * 100)))
    # which probe indices fall in the corrective phase
    for alpha in ALPHAS:
        variation += 1
        rw = RWCompetition(alpha=alpha, lam=1.0, floor=1e-3)
        co = CountOnly()
        rw_traj = run_model(rw, events)
        co_traj = run_model(co, events)

        # restrict the recovery analysis to the corrective phase (after the boundary)
        rw_phase2 = [(i, p) for i, p in rw_traj if i >= PHASE_BOUNDARY]
        co_phase2 = [(i, p) for i, p in co_traj if i >= PHASE_BOUNDARY]

        # SUSTAINED crossover (3 consecutive probes): single-cue R-W oscillates, so we require the
        # irregular to stay on top — the honest recovery marker, not a lucky single-event swing.
        rw_x = recovery_step([p for _, p in rw_phase2], IRREG, OVERREG, sustain=3)
        co_x = recovery_step([p for _, p in co_phase2], IRREG, OVERREG, sustain=3)
        # absolute crossover event index (None => never crossed)
        rw_xi = rw_phase2[rw_x][0] - PHASE_BOUNDARY if rw_x is not None else None
        co_xi = co_phase2[co_x][0] - PHASE_BOUNDARY if co_x is not None else None

        rw_over = target_series(rw_phase2, OVERREG)
        co_over = target_series(co_phase2, OVERREG)
        rw_sl = recovery_slope(rw_over)
        co_sl = recovery_slope(co_over)
        # net change in p(over-applied) across phase 2 (positive = it fell)
        rw_dp = rw_over[0] - rw_over[-1]
        co_dp = co_over[0] - co_over[-1]

        # winner = recovers (crosses) earlier; a non-crosser loses to a crosser
        def score(xi):
            return float("inf") if xi is None else xi
        if score(rw_xi) < score(co_xi):   win = "RW"
        elif score(co_xi) < score(rw_xi): win = "CO"
        else:                              win = "tie"
        rows.append(dict(irr=irr, alpha=alpha, rw_xi=rw_xi, co_xi=co_xi,
                         rw_sl=rw_sl, co_sl=co_sl, rw_dp=rw_dp, co_dp=co_dp, win=win))
        fx = lambda v: f"{v:>9}" if v is not None else f"{'never':>9}"
        print(f"{irr:>9.1f} {alpha:>6.2f} | {fx(rw_xi)} {fx(co_xi)} | "
              f"{rw_sl:>9.4f} {co_sl:>9.4f} | {rw_dp:>7.3f} {co_dp:>7.3f} | {win}")

# ─────────────────────────────────── the decisive contrast ───────────────────────────────────
# The cleanest M20 test: a WEAK corrective signal (irr_ratio just barely above 0.5). Increment-only
# needs "mice" to out-frequency "mouses" to ever cross; R-W can cross even when "mouses" is still
# heard, because the cue's budget is competitively reallocated.
print("\n" + "=" * 92)
print("DECISIVE CONTRAST — weak corrective signal (irr_ratio=0.5: 'mice' and 'mouses' near-balanced)")
print("=" * 92)
events = make_stream(N1, N2, irr_ratio=0.5, rng=np.random.default_rng(SEED + 50))
rw = RWCompetition(alpha=0.15); co = CountOnly()
rw_traj = run_model(rw, events); co_traj = run_model(co, events)
print("  p(mouses) / p(mice) trajectory through the corrective phase (event index from boundary):")
print(f"  {'event':>7} | {'RW mouses':>10} {'RW mice':>8} | {'CO mouses':>10} {'CO mice':>8}")
for (i, rp), (_, cp) in zip(rw_traj, co_traj):
    if i < PHASE_BOUNDARY: continue
    print(f"  {i-PHASE_BOUNDARY:>7} | {rp.get(OVERREG,0):>10.3f} {rp.get(IRREG,0):>8.3f} | "
          f"{cp.get(OVERREG,0):>10.3f} {cp.get(IRREG,0):>8.3f}")

# memory footprint (bounded-memory claim: R-W PRUNES, count-only only grows)
print(f"\n  live associations  —  RW: {rw.n_outcomes()}   CountOnly: {co.n_outcomes()}   "
      f"(R-W floor-prunes the bled-out form; count-only keeps it forever)")

# ─────────────────────────────────── Ramscar check (count-maturity) ───────────────────────────────────
# Ramscar's sign: exposure to OTHER REGULARS should affect the irregular's recovery by count-maturity
# — a learner who has heard many regular +s plurals has a MORE-committed +s expectation, so the same
# corrective "mice" must compete harder. We vary how mature the regular system is at the phase boundary.
print("\n" + "=" * 92)
print("RAMSCAR CHECK — does a more-mature regular system slow the irregular's recovery? (count-maturity)")
print("=" * 92)
print(f"  {'reg_warmup':>10} | {'RW xover (events into phase2)':>30}")
for warm in [0, 4000, 12000]:
    # warm = extra regular-only events BEFORE phase 1, maturing the +s expectation
    base = make_stream(N1, N2, irr_ratio=0.6, rng=np.random.default_rng(SEED + 7))
    warmup = []
    wr = np.random.default_rng(SEED + 99)
    for _ in range(warm):
        s = REG_STEMS[wr.integers(len(REG_STEMS))]; warmup.append((s, s + "s"))
    ev = warmup + base
    rw = RWCompetition(alpha=0.15)
    traj = run_model(rw, ev)
    bnd = warm + PHASE_BOUNDARY
    ph2 = [(i, p) for i, p in traj if i >= bnd]
    x = recovery_step([p for _, p in ph2], IRREG, OVERREG)
    xi = ph2[x][0] - bnd if x is not None else None
    print(f"  {warm:>10} | {str(xi) if xi is not None else 'never':>30}")

# ─────────────────────────────────── verdict tally ───────────────────────────────────
rw_wins = sum(1 for r in rows if r["win"] == "RW")
co_wins = sum(1 for r in rows if r["win"] == "CO")
ties    = sum(1 for r in rows if r["win"] == "tie")
rw_cross = sum(1 for r in rows if r["rw_xi"] is not None)
co_cross = sum(1 for r in rows if r["co_xi"] is not None)
# does R-W cross EARLIER on average where both cross?
both = [r for r in rows if r["rw_xi"] is not None and r["co_xi"] is not None]
mean_gap = np.mean([r["co_xi"] - r["rw_xi"] for r in both]) if both else float("nan")

print("\n" + "=" * 92)
print(f"TALLY over {len(rows)} variations: RW recovers-first {rw_wins}  |  CountOnly-first {co_wins}  |  tie {ties}")
print(f"  crossed at all — RW: {rw_cross}/{len(rows)}   CountOnly: {co_cross}/{len(rows)}")
print(f"  mean (CO_xover − RW_xover) where both cross: {mean_gap:.1f} events "
      f"({'R-W recovers earlier' if mean_gap > 0 else 'no R-W advantage'})")
print("=" * 92)

# KILL CONDITION (M20 / BUILD_QUEUE BC): increment-only passive reading recovers JUST AS FAST
# => recovery is just frequency => keep the simpler loop. It FIRES if CountOnly matches/beats R-W.
killed = (rw_wins <= co_wins) or (mean_gap <= 0)
print(f"\nKILL CONDITION fired: {killed}   "
      f"({'increment-only recovers as fast — recovery is just frequency' if killed else 'R-W recovers without correction faster than frequency alone — survives'})")
