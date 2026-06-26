# When the letters lie, it leans on the idea

**Experiment Y · 2026-06-26 · qualified result**

Pour noise on the input the model reads, at perception time, on the input only,
and score against the clean target. Headline: the gated `char → word → phrase →
topic` stack from Exp X degrades about 2.7x slower than a flat bigram under
surface noise, and the confidence gate routes prediction mass from letters to
concepts (86% → 95% as the scramble probability rises 0 → 0.3) with no signal
ever telling it the input is noisy. Two parked negatives: training on noised
text hurt clean rare-context accuracy (count tables do not overfit like nets —
the fix is consistency-counting), and the word→topic crossover under word noise
is trending but has not landed by q ≈ 0.3.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/noise-concepts/run.py
```

**Blog post:** ["When the letters lie, it leans on the idea"](https://cortex.kinogaki.com/noise-concepts/)
