# Words that lower the cost of letters

**Experiment C · 2026-06-25 · win**

Tests the first rung of the concept hierarchy: does a layer of word concepts above the character model help predict the next character, by pure counting? Headline: word concepts cut cost 22% with a given lexicon and 17% with words discovered unsupervised from the surprise signal — and the discovered concept store is a readable document full of real English words.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/concepts/run.py
```

**Blog post:** ["Words that lower the cost of letters"](https://cortex.kinogaki.com/concepts/)
