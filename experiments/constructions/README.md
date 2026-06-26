# Grammar is just counting, made productive

**Experiment AF · 2026-06-26 · qualified win**

A grammar is what falls out of counting usage, if for every frame you count two
things: token frequency, which entrenches a frame into a frozen idiom, and type
frequency, which abstracts it into an open slot that predicts the *category* and
so can fire for a filler it never saw there. On held-out, never-seen
(frame, filler) pairs the open-slot construction beats the n-gram 4.3× on
perplexity (5405 vs 23461) and wins on 80% of them. It froze idioms (*such as*,
*based on*, *part of*) and abstracted a NUMBER + UNIT slot ("two ___" → km, per,
miles, ft, square), and statistical preemption cut over-generation 39.5%. It is a
backoff for the unseen, not a replacement: when the exact pair was seen, the
specific count is sharper.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/constructions/run.py
```

**Blog post:** ["Grammar is just counting, made productive"](https://cortex.kinogaki.com/constructions/)
