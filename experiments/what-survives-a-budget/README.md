# What survives a budget

**Experiment AS · 2026-06-26 · theory update**

The bounded-memory rule's prediction, confirmed: mechanisms that vanish at unbounded
scale **return** under a memory budget. Consolidation flips −0.006 bpc (unbounded) →
+0.144 → **+0.307 bpc** as the cap tightens — a monotone curve that vanishes exactly
when the budget stops binding. Lossless generalization (consolidation) wins broadly;
lossy (concepts) only as tight as the budget forces; the topic prior stays neutral.
Generalization is not optional — it is how a bounded model approximates the unbounded
one.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/what-survives-a-budget/run.py            # 60M default; writes the sweep
AS_TRAIN=10000000 AS_EVAL=2000000 python experiments/what-survives-a-budget/run.py   # the 10M concept-flip regime
```

**Blog post:** ["What survives a budget"](https://cortex.kinogaki.com/blog/what-survives-a-budget/)
