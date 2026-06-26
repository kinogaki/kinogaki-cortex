# Predicting the kind, not the word

**Experiment U · 2026-06-26 · qualified result**

A count-world JEPA: hide a word and predict its CLASS instead of its spelling. Word signatures are counted, then grouped into 400 clusters in one streaming leader-clustering pass (no k-means, no gradients, so no collapse). Headline: the cluster head towers over the token head (65.4% vs 6.7% overall; 11.2% vs 0.0% on rare words) — the class is predictable where the exact word is hopeless.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/jepa/run.py
```

**Blog post:** ["Predicting the kind, not the word"](https://cortex.kinogaki.com/jepa/)
