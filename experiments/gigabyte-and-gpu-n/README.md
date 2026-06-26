# More data helps, all the way to a gigabyte (N)

**Experiment NO · 2026-06-26 · win**

The scale half (Exp N): feeds one character column the whole 1 GB enwik9 dump (827M chars) and measures cost at growing sizes. Headline: cost keeps falling all the way to the gigabyte (1.997 at 10 MB → 1.744 at 822 MB) — an earlier 'deeper level didn't pay off' was a 2 MB data-starvation artifact, not a wall. Pairs with `gigabyte-and-gpu-o`.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/gigabyte-and-gpu-n/run.py
```

**Blog post:** ["More data helps, all the way to a gigabyte (N)"](https://cortex.kinogaki.com/gigabyte-and-gpu/)
