# Is the analogy already in the counts?

**Experiment AD · 2026-06-26 · qualified win**

*king is to queen as man is to ?* The famous answer lives in a trained vector
space. We do not train. But the parallelogram turns out to be already in the raw
counts: 3CosAdd over raw PPMI co-occurrence profiles solves `a:b::c:?` at 56%
top-1 and 94% top-5 (about 4× baseline, plurals 89%), no SVD, no word2vec, one
pass. Two honest negatives. Our only legal smoother, online leader-clustering,
cannot substitute for SVD: it *blurs* the relation axes (paris and tokyo
co-cluster, so the paris−france axis is destroyed), flat-to-worse at every
strength. And NARS transitive induction spreads its mass too broadly to beat a
direct counter. The representation is in the counts; we still lack a count-native
combiner that sharpens without blurring.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/analogy-in-counts/run.py
```

**Blog post:** ["Is the analogy already in the counts?"](https://cortex.kinogaki.com/analogy-in-counts/)
