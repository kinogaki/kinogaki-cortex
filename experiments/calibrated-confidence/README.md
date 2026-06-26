# How sure is a count?

**Experiment AB · 2026-06-26 · qualified win**

A raw count says how *often* a context fired, never whether it was *right*. Split
it: count hits `w+` and misses `w-`, and attach Pei Wang's NARS truth value,
frequency `f = w+/w` and confidence `c = w/(w+1)`. The truth value calibrates the
model for free, ECE 0.280 → 0.027 (10×) with a near-diagonal reliability curve,
and as an expert weight it cuts perplexity threefold (12.4 → 4.3). The knob-free
`f·c` gate loses to a tuned threshold on clean text, where the long context wins
almost everywhere, but it is the only policy that beats always-open on the rare,
unreliable slice. Everything is read off counts, no fitted parameter.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/calibrated-confidence/run.py
```

**Blog post:** ["How sure is a count?"](https://cortex.kinogaki.com/calibrated-confidence/)
