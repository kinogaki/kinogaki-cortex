# When combining the experts made it worse

**Experiment D · 2026-06-25 · negative result**

Combines five experts (characters, word lexicon, phrase, topic cache) into one voting cortex two ways — gated linear and product-of-experts — and asks whether the full ensemble beats the simple char+lexicon mix. Headline: both lost (1.71 / 1.77 vs 1.653), because character prediction saturates. The right combiner is the product of experts; the lesson reshaped everything after.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/voting/run.py
```

**Blog post:** ["When combining the experts made it worse"](https://cortex.kinogaki.com/voting/)
