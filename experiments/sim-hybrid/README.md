# Use the map to read, not to walk

**Experiment Z · 2026-06-26 · qualified win**

Proximity failed twice as a next-word predictor, so we stopped asking the
similarity map to *walk* and used it to *read*. Each word gets an online
co-occurrence signature, leader-clustered into concept clusters, and phrases one
level up. We pour the map into a bigram counter as a backoff prior: a rare
context borrows its cluster's pooled next-word counts. On the fifth of
predictions where the exact pair was never counted, the cluster cuts perplexity
about twentyfold (19M → 965k), and the phrase hierarchy shaves a little more. The
map prices the tail without ever changing the single best guess. A backoff prior,
counted, never trained against the thing it helps.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/sim-hybrid/run.py
```

**Blog post:** ["Use the map to read, not to walk"](https://cortex.kinogaki.com/sim-hybrid/)
