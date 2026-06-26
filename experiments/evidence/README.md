# A vote that remembers what it just saw

**Experiment R · 2026-06-26 · qualified result**

Replaces the fresh-every-step vote with a leaky evidence accumulator and tests robustness under corrupted context. Headline: on clean text the fresh vote wins (3.67 bits); past 10% noise the accumulator wins and degrades gracefully (rose 2.4 bits across the range vs the fresh vote's 9.1). Remembering is the right move exactly when input is unreliable.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/evidence/run.py
```

**Blog post:** ["A vote that remembers what it just saw"](https://cortex.kinogaki.com/evidence/)
