# Exp BA — dual-route inflection head (words-and-rules as one tunable gate)

Fuses **AB** (f·c memory split), **AF** (open-slot +ed default), and **AJ** (take-the-best blocking)
into ONE inflection head with a single gate. A verb's past tense is produced from a leaky per-stem
memory (Route A) iff its `f·c` clears the gate; otherwise the graded "+ed" default fires (Route B). On a
synthetic frequency-matched verb stream with a developmental ramp (the +ed rule's productivity grows as
the regular vocabulary comes online), we ask whether over-regularization (*goed*) falls out **rare,
item-specific, and frequency-graded** — a per-verb micro-U, never a synchronized macro-U — and whether
the gate is a clean Pinker↔Rumelhart knob. CHILDES isn't in `data/`, so the stream is synthetic (the spec
authorizes this). Mechanism module: `lib/dualroute.py`. Run: `python exp_ba_dualroute/run.py` (~20s, seed 0).
