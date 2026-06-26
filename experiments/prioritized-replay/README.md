# Exp BG / M24 — Inverse-count / surprise-prioritized spindle replay

Sleep doesn't replay uniformly: spindles preferentially reinstate the weak and the surprising
(Schapiro 2018 — they protect *infrequent* words). Exp AA's sleep pass replays its buffer
uniformly, so at a fixed offline effort almost all of it lands on the head the model already nails.
This experiment holds AA's sleep pass and a fixed reinforcement BUDGET fixed and changes only the
replay distribution: `uniform` (AA), `invcount` (deposit ∝ 1/(1+count) — protect the rare tail), and
`surprise` (deposit ∝ −log₂P(next|ctx) — re-fire worst-predicted events). It scores held-out
text8, split rare-context vs common-context, and asks the M24 question: does prioritized replay beat
uniform on **rare-context bpc at equal budget** without overtrading common? The mechanism lives in
`lib/replay.py` (online single buffer pass, every deposit a +1 count, total deposited == budget).
Outcome: the literal inverse-count rule is a **clean negative** (its kill fired), but the
surprise-driven variant **wins** at mid/high budget — see RESULTS.md.
