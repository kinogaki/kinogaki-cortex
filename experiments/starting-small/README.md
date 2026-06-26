# Starting small, on purpose

**Experiment AK · 2026-06-26 · honest negative**

Growing the memory budget on a schedule does NOT beat full-from-start (FULL 2.744
vs GROW 2.751 target perplexity, robust across seeds and schedule shapes); only a
permanently small budget loses, decisively (FIXED +30.1%). Elman's "starting
small" was a property of the gradient *optimizer* — gradient descent locks in bad
early guesses, so it needs a gentle on-ramp — not of learning itself. A count
learner cannot get stuck (counts are additive and self-correcting), so it needs no
curriculum, only enough *final* memory. The ZPD overlay hurt (−5.8%). The
bounded-memory rule stands; scheduling it is a no-op.

The corpus is synthetic (Elman's embedded-agreement paradigm, count-native) — no
data download needed.

**Run it** (from the repo root):

```sh
python experiments/starting-small/run.py
```

**Blog post:** ["Starting small, on purpose"](https://cortex.kinogaki.com/blog/starting-small/)
