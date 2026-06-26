# Exp BH — BLiMP / minimal-pair Probe + impossible-language ablation

The honesty bar for the acquisition queue: score grammaticality **the field's way** — by minimal
pairs (Warstadt 2020), preferring the sentence the model finds less surprising — over the existing
char vote, read-side only (the eval never learns). Bundled with the Kallini (2024)
impossible-language ablation: train the same char band on natural English vs a position-scramble of
equal bytes, and ask whether the counter geometry's locality bias acquires natural grammar more
easily, **controlling for the entropy confound** (a window-4 local scramble whose bpc matches
natural). Result: the count band beats a bigram baseline (60.2% vs 53.4% macro, carried by
word-order phenomena; agreement / det-noun / NPI are the expected weak slices), and natural beats
every scramble on grammar — the local-scramble gap (+6.8pp at near-matched entropy) makes the
preference at least partly **structural**, not entropy-driven. PARTIAL (lean win); kill-condition
did not fire. text8 substitutes for CDS and the pairs are a hand-built BLiMP analogue (the
benchmark + CDS are not on disk) — declared in RESULTS. Reusable Probe + scrambles in
`lib/blimp.py`.
