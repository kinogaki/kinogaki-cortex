# Attention, but counted instead of trained

**Experiment S · 2026-06-26 · clear win**

Asks for a count-based answer to attention: one count table per relative position (next word | word d steps back), pooled by per-position information gain — no queries, keys, values, or gradients. Headline: it cut perplexity threefold under a bigram (4,318 vs 13,477) while tying on top-1 accuracy, the only model that is both accurate and calibrated.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/offset-attention/run.py
```

**Blog post:** ["Attention, but counted instead of trained"](https://cortex.kinogaki.com/offset-attention/)
