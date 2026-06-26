# Phrases that rhyme share their counts

**Experiment AP · 2026-06-26 · negative**

Permutation-bound + FlyHash phrase addressing: similar phrases pool counts (beats a
floored literal on **67%** of unseen-in-form probes) AND preserves order (×2.29 under
scramble vs the bag's ×1.00). But FlyHash crosstalk loses the tail (aggregate ppl 1206
vs literal 831; poor exact memory 423 vs 2.2). The fix: use it as a **backoff layer**
under the literal table.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/phrases-that-rhyme/run.py
```

**Blog post:** ["Phrases that rhyme share their counts"](https://cortex.kinogaki.com/blog/phrases-that-rhyme/)

Credit: Kanerva (HDC); Dasgupta/Stevens/Navlakha (FlyHash).
