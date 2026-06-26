# The shape of forgetting

**Experiment AI · 2026-06-26 · honest negative**

ACT-R's power law (B = ln Σ tₖ⁻ᵈ) is the right *shape* of forgetting — the only
curve that represents spacing (spaced 8.96× more accessible than massed at equal
count, where the exponential EMA simply collapses). But as a budgeted-eviction
policy for dense char-grams, raw-count LFU wins at every cap, because LFU is the
power law's d→0 limit and a char-gram's value is pure frequency; a decay sweep
degrades quality monotonically as d grows. Power-law-*weighted* prediction loses
outright (+0.68 bpc). Verdict: right shape, wrong place — keep LFU for char-grams,
reach for the power law at the sparse word and concept level.

**Run it** (from the repo root, after `bash data/get-data.sh` — needs
`darwin.txt` and `shakespeare.txt`):

```sh
python experiments/shape-of-forgetting/run.py
```

**Blog post:** ["The shape of forgetting"](https://cortex.kinogaki.com/blog/shape-of-forgetting/)
