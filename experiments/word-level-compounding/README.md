# The hierarchy pays off at the right altitude

**Experiment E · 2026-06-25 · win**

Tests the prior negative's diagnosis directly: if higher experts barely move character cost because characters saturate, they should pay off when measured at the WORD level. Headline: a product-of-experts over word/phrase experts nearly halved perplexity (476 to 247), with almost all the weight landing on the phrase experts.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/word-level-compounding/run.py
```

**Blog post:** ["The hierarchy pays off at the right altitude"](https://cortex.kinogaki.com/word-level-compounding/)
