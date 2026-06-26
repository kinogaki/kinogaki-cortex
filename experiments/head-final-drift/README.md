# Exp AZ — reliability-gated boundary detectors → head-final drift

Three online boundary detectors (forward-TP-dip, backward-TP-dip, branching-entropy-rise) each
carry an Exp-AB hit/miss reliability tally (f·c) against the eventually-stable chunk boundary, and
are combined by Exp-AJ take-the-best. The one falsifiable claim (M3): the **drift** Δ =
reliability(backward) − reliability(forward) **rises** going head-initial → head-final. With no
Japanese/Korean corpus on disk, we synthesize a frequency-matched mirror-image pair (head-initial
STEM·MARK vs head-final MARK·STEM) and anchor with real English text8. **Verdict: PARTIAL PASS** —
the drift fires (positive in 10/12 FRAGILE-budget variations, mean +0.093; backward-TP gains +0.10
reliability on head-final text), the kill-condition did not fire, but `entropy` remains the strongest
single cue so the gate *shifts* without usually changing its top pick. Run:
`.../exp_a_boundary/.venv/bin/python exp_az_drift/run.py`. Mechanism module: `lib/boundsdrift.py`.
