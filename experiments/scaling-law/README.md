# How big a brain the data wants

**Experiment F · 2026-06-25 · win**

Maps how the counting model scales along two knobs — how much text it reads and how much capacity it has — on a 100 MB corpus grid. Headline: a small model saturates early; a large one keeps learning but only once it has enough data, and the best capacity grows with the corpus (the large model overfits at 1 MB, wins at 3 MB+).

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/scaling-law/run.py
```

**Blog post:** ["How big a brain the data wants"](https://cortex.kinogaki.com/scaling-law/)
