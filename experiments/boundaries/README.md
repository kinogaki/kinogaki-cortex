# Finding phrases the way you'd guess them

**Experiment M · 2026-06-26 · mixed**

Climbs the surprise signal one level up: does branching entropy over the WORD stream discover phrases, and over a document discover topic boundaries? Headline: phrases are a clear unsupervised win ("united states", "such as", "see also", "according to"); topic boundaries are real but weak. One signal carves much of the hierarchy.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/boundaries/run.py
```

**Blog post:** ["Finding phrases the way you'd guess them"](https://cortex.kinogaki.com/boundaries/)
