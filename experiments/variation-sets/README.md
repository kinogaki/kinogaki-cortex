# Exp BB — Variation-set minimal-pair miner (M16)

Caregiver speech arrives in **variation sets** — runs of near-repeated utterances with one thing swapped
("put the ball on table" / "put the car on table" / "put the dog on table"). The repeated frame and the one
swapped span are a minimal pair handed to the child for free. This experiment mines them online with **no
gradient**: a bounded ring buffer of recent utterances, a token-level LCS diff against the previous one, and —
when overlap is high — the disagreeing run is a (slot, filler) substitution. Two products fall out: extra
(frame, filler) counts into AF's construction tables, and the agree→disagree transition as a phrase **boundary**
(a new boundary source for `boundaries.py`, alongside branching entropy). Reusable mechanism in
`lib/varsets.py`; judged on the two axes the cognitive literature (Haga) says it can win — compositional
generalization and segmentation — not flat bpc. **Corpus substitution:** CHILDES is not in `data/`, so we use a
synthetic child-directed variation-set generator plus a text8 negative control (stated in `run.py`/RESULTS).
