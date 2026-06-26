# What an agent learns while it dreams

**Experiment AA · 2026-06-26 · qualified win**

A count memory never forgets, which means it keeps the junk it saw once in a
moment of noise. We built a sleep pass: one offline pass over the count tables
that prunes untrustworthy contexts, distills specifics that only echo their
backoff, and tries to promote recurring patterns. One gentle sleep cuts the
memory by 37% (3.26M → 2.06M entries) and *improves* prediction on the rare tail
(rare-context bpc 2.592 → 2.456), with no new data. Keep sleeping and Letta's
failure mode arrives on schedule: the memory goes generic and lossy, common
contexts ruined to flatter the tail. Dream once, then stop.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/sleep-consolidation/run.py
```

**Blog post:** ["What an agent learns while it dreams"](https://cortex.kinogaki.com/sleep-consolidation/)
