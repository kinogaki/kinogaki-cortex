# Where forgetting's shape finally pays

**Experiment AR · 2026-06-26 · win (the parked resurrection from AI)**

Power-law (ACT-R) eviction **beats LFU** at the word level under non-stationarity and a
tight budget (cap 10k −0.008, 30k −0.006 bpw; LFU re-wins once the cap is loose). The
sign flipped from AI's char-gram result, exactly as AI predicted — it wins by serving
the present, not protecting the past. Three registers streamed in one pass (Darwin →
Shakespeare → KJV Bible), the same `actr_weight` code as AI, only the tokens changed.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/forgetting-at-the-word-level/run.py
```

**Blog post:** ["Where forgetting's shape finally pays"](https://cortex.kinogaki.com/blog/forgetting-at-the-word-level/)

Credit: Anderson & Schooler (the environmental power law of memory).
