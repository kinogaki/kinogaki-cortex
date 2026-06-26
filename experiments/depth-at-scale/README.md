# The level that reaches past the last few words

**Experiment K · 2026-06-25 · clarifying**

Asks whether stacking a 4th and 5th fixed level keeps paying once there is enough data. Headline: data was the lever (10→40 MB dropped cost 1.94→1.77); more fixed local levels were not (a 4th longer-word-context level saturated like the character level). The payoff is a level that reaches PAST the local context the lower levels already cover.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/depth-at-scale/run.py
```

**Blog post:** ["The level that reaches past the last few words"](https://cortex.kinogaki.com/depth-at-scale/)
