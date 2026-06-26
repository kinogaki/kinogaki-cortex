# Exp BC — Rescorla-Wagner recovery loop (recovery without feedback)

Children over-regularize ("mouses", "goed") then stop — with no one correcting them. A passive
count reader can't explain it: hearing "mice" only ever *increments* count("mice"), so the
over-applied "mouses" is frozen and only loses ground if "mice" out-frequencies it ("recovery is
just frequency"). This experiment builds the M20 mechanism — a count-native Rescorla-Wagner table
where the stem is a cue and its inflected forms compete: predict the form, then on observing, push
the heard form up *and decrement every still-expected-but-absent form* by `α·ΣV` (cue
competition / blocking — the first acquisition use of AT's predict-then-update contract). It pits
R-W against the increment-only baseline on the same two-phase synthetic morphology stream
(over-generalization → corrective input with **no** correction signal) across 12 variations and
asks the kill question: does increment-only recover just as fast? It does not. Online single pass,
no gradients, bounded (decrement + floor-prune). `lib/rescorla.py`, `run.py`.
