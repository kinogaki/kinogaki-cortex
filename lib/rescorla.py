"""rescorla.py — the Rescorla-Wagner recovery loop: recovery without feedback (M20 / BC).

The count-native answer to "why does *mouses* disappear without anyone correcting the child."

A plain count reader only ever INCREMENTS: hear "mice", bump count("mice"). Under that rule the
over-applied "mouses" never *leaves* — its absolute count is frozen the moment the corrective input
("mice") starts, and it only loses *relative* probability if "mice" out-frequencies it. There is no
mechanism by which being *expected and not seen* costs an association anything.

Rescorla-Wagner supplies exactly that mechanism, and it is non-gradient by construction (it is THE
canonical associative learning rule, predating backprop). For a cue C with competing outcomes
{o_1..o_n} and total expectation V_tot = Σ_o V[C,o], on observing outcome y:

    for every outcome o of cue C:
        V[C,o] += α * ( λ * 1[o == y]  −  V_tot )

The observed outcome y is pulled UP toward λ; every other outcome that the cue currently predicts is
pulled DOWN by α·V_tot — **decremented purely from being expected and absent**. That is cue
competition / blocking: once "mice" reliably follows the mouse-cue, the cue's budget is spent on
"mice" and the still-predicted "mouses" bleeds out with no label and no reward. The decrement also
*frees* the budget it reclaims (bounded memory: prune outcomes whose V falls below a floor).

This module is encoding-agnostic over an OUTCOME alphabet per cue. `RWCompetition` is the mechanism;
`CountOnly` is the load-bearing increment-only baseline (the thing M20 must beat). Both expose the
same `.observe(cue, outcome)` / `.p(cue)` surface so an experiment scores them on the same axis.

Online single streaming pass; no gradient descent / k-means / SVD; bounded (decrement + floor prune).
"""
from collections import defaultdict


class RWCompetition:
    """Predict-then-update associative table with competitive decrement (the M20 mechanism).

    One shared elemental cue per `cue` key (the stem); outcomes are the competing inflected forms.
    `alpha` = learning rate, `lam` = the asymptote each confirmed association climbs toward, `floor`
    = the strength below which an outcome is pruned (frees budget — the bounded-memory coping move).
    `gate` optionally scales the update by a surprise/contingency signal in [0,1] (AT's reactive
    contract: a more-surprising observation teaches harder); default 1.0 keeps it pure R-W.
    """
    def __init__(self, alpha=0.15, lam=1.0, floor=1e-3):
        self.alpha = alpha; self.lam = lam; self.floor = floor
        self.V = defaultdict(dict)                       # V[cue] -> {outcome: strength}

    def predict(self, cue):
        """The strengths the cue currently expects, BEFORE this observation (predict-then-update)."""
        return dict(self.V.get(cue, {}))

    def observe(self, cue, outcome, gate=1.0):
        row = self.V[cue]
        if outcome not in row: row[outcome] = 0.0
        V_tot = sum(row.values())                        # combined expectation of ALL cue outcomes
        a = self.alpha * gate
        for o in list(row):
            target = self.lam if o == outcome else 0.0
            row[o] += a * (target - V_tot)               # R-W: observed up, expected-but-absent DOWN
            if o != outcome and row[o] < self.floor:     # decrement frees budget (bounded memory)
                del row[o]

    def p(self, cue):
        """Outcome distribution for a cue — strengths clamped ≥0 and normalized."""
        row = self.V.get(cue, {})
        pos = {o: max(0.0, v) for o, v in row.items()}
        z = sum(pos.values())
        if z <= 0: return {o: 1.0 / len(row) for o in row} if row else {}
        return {o: v / z for o, v in pos.items()}

    def strength(self, cue, outcome):
        return self.V.get(cue, {}).get(outcome, 0.0)

    def n_outcomes(self):
        return sum(len(r) for r in self.V.values())      # live associations = memory footprint


class CountOnly:
    """The load-bearing baseline: passive increment-only reading (the spine M20 must beat).

    Hear an outcome, bump its count. Same `.observe`/`.p` surface as RWCompetition so the experiment
    scores them identically. By construction the absolute count of an over-applied form NEVER falls —
    recovery, if any, is purely the corrective form out-frequencying it. No decrement, no competition.
    """
    def __init__(self):
        self.C = defaultdict(dict)

    def predict(self, cue):
        return dict(self.C.get(cue, {}))

    def observe(self, cue, outcome, gate=1.0):
        row = self.C[cue]; row[outcome] = row.get(outcome, 0) + 1

    def p(self, cue):
        row = self.C.get(cue, {})
        z = sum(row.values())
        return {o: c / z for o, c in row.items()} if z else {}

    def strength(self, cue, outcome):
        return float(self.C.get(cue, {}).get(outcome, 0))

    def n_outcomes(self):
        return sum(len(r) for r in self.C.values())


def recovery_step(traj, irregular="mice", overreg="mouses", sustain=1):
    """First step index from which p(correct irregular) STAYS above p(over-applied regular) for
    `sustain` consecutive probes (sustain=1 = first crossing). `traj` is a list of per-step
    {outcome: prob} dicts for the target cue. Single-cue R-W oscillates event-to-event, so a
    sustained crossing is the honest recovery marker."""
    n = len(traj)
    for i in range(n):
        if all(traj[j].get(irregular, 0.0) > traj[j].get(overreg, 0.0)
               for j in range(i, min(i + sustain, n))) and i + sustain <= n:
            return i
    return None


def recovery_slope(over_traj):
    """Mean per-step drop in p(over-applied form) across the corrective phase (>0 = recovering)."""
    if len(over_traj) < 2: return 0.0
    return (over_traj[0] - over_traj[-1]) / (len(over_traj) - 1)
