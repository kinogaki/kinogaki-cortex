# Exp AX — Function-word anchor voter (free top-k frequency bootstrap)

Mine the top-k word-ids by raw frequency as a label-free closed-class **anchor** set (the/of/to/a/and),
keep each anchor's right-neighbour **category** tally (jepa online clusters), and feed "follows-anchor-a"
as one counted cue into take-the-best (AJ) beside the AF frame cue. The cheapest possible POS bootstrap:
a frequency threshold plus adjacency counts. Measured against AF frame-voter purity on text8 + a
word-order-shuffled negative control (substituting for the German slice the spec names, which data/
lacks). ONLINE single pass, bounded (~k anchors), no backprop. **Negative** — the anchor cue is a strict
subset of AF (both count P(right-category | preceding word)), so it adds exactly +0.000 purity over AF
on the same firing positions; AJ correctly ignores the redundant cue (combo never degrades).
