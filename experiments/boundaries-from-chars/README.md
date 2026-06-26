# Finding where one word ends

**Experiment A · 2026-06-25 · qualified win**

Tests whether prediction error alone, reading raw space-stripped characters with no labels, can find where one word ends and the next begins. Headline: a branching-entropy signal hits F1 0.775 against true word boundaries — a quality the literature respects — while the fashionable Bayesian-surprise signal scores below random at the character scale (wrong altitude). This experiment reads its own `data/raw.txt`; drop a plain-English Project Gutenberg book there.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/boundaries-from-chars/run.py
```

**Blog post:** ["Finding where one word ends"](https://cortex.kinogaki.com/boundaries-from-chars/)
