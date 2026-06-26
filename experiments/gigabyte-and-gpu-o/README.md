# More data helps, all the way to a gigabyte (O)

**Experiment NO · 2026-06-26 · win**

The speed half (Exp O): builds the same character column three ways — sorted-count, dense histogram, and a Metal GPU path — and shows where the GPU actually helps. Same model, same answer, three engines, making the gigabyte affordable. Pairs with `gigabyte-and-gpu-n`.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/gigabyte-and-gpu-o/run.py
```

**Blog post:** ["More data helps, all the way to a gigabyte (O)"](https://cortex.kinogaki.com/gigabyte-and-gpu/)
