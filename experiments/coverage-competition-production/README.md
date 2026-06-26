# Exp BD — Coverage-competition production (G1, the generation turn)

The harness's `act()` samples a flat geometric-mean vote over chars → gibberish. BD replaces that with
construction-driven production: retrieve the constructions keyed by the left word, score each by
**coverage × frequency** (coverage = the slot's category mass, gated by AW ΔP/PPMI association so a
category followed only at base rate is vetoed *before* emission), let constructions **compete** by AJ
take-the-best, and fill the winning slot — a frozen frame emits its idiom verbatim, an open-slot frame
samples productively from the category lexicon. Every emitted word is articulated through AU's chunk
lexicon (the emission vocabulary). It is read-only over the grammar AU/AW/AF already counted: no learning
step, online single pass, bounded, no backprop. Judged on the **constructional battery** (well-formedness
/ over-generation against a NON-circular held-out attestation oracle), not raw perplexity. Verdict:
**PARTIAL** — wins the primary axis (+18.5 pt well-formedness, −28.5 % over-generation vs the flat floor),
the merged Levelt frame-survival sub-claim stays weak (61 % vs the 80 % target).
