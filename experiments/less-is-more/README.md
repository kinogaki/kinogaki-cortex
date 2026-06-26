# Less is more, and you can prove it

**Experiment AJ · 2026-06-26 · clear win**

Validity-ordered, noncompensatory, early-stopping take-the-best beats full
geometric-mean integration on every axis at once: accuracy 15.00% vs 9.71%,
perplexity 1,918 vs 7,160, at 4.56 cues per step instead of 8. The less-is-more
effect (α>β) holds where the theory predicts — ignoring the weak soft channel
wins overall and most sharply on sparse contexts (11.68% vs 10.96%). One honest
negative: a base-rate prior γ>0 on the single-pass leader-clusterer *lowers*
stability (γ=0 stays the operating point at 0.982). This revises the standing
combiner: validity-ordered take-the-best and early-stop, not full pooling. A count
model is high-bias by construction, which is exactly where stopping early wins.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/less-is-more/run.py
```

**Blog post:** ["Less is more, and you can prove it"](https://cortex.kinogaki.com/blog/less-is-more/)
