# Associative attention vs fixed n-grams

**Experiment L · 2026-06-25 · predecessor to Exp S**

Predecessor to the offset-attention win (Exp S), kept for the record. Tests content-based, variable-distance, count-learned attention (skip-gram associations + a learned per-word attention weight + a long context window) against fixed n-grams, with the same calibrated pooling and char bridge. The cleaner formulation that followed is `offset-attention`.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/associative-attention/run.py
```

**Blog post:** none (in-tree result; see RESULTS.md).
