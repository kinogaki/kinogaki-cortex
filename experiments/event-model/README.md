# What the model thinks is happening

**Experiment AC · 2026-06-26 · qualified win**

A reader feels the jolt when one article ends and the next begins. We measure it
as Bayesian surprise, the KL between the next-word belief before a word and after
it, `KL(Pt ‖ Pt-1)`. On 36 MB of enwik9 with 4,598 real `<page>` boundaries, KL
beats per-token surprisal 5.7× and branching-entropy about 120× at locating those
boundaries (F1 0.154 at ±25 words), and the lead grows with scale. The
Kumar-2023 claim reproduces on real text. Honest negative: as a hard segmenter it
is precision-only, so the win is KL-as-ranked-signal. The persistent event-slot
prior it drives helps prediction only on the ~1% backoff slice (+0.143 bpw),
re-confirming the Exp T altitude law by a different road.

**Run it** (from the repo root, after `bash data/get-data.sh`; needs enwik9):

```sh
python experiments/event-model/run.py
```

**Blog post:** ["What the model thinks is happening"](https://cortex.kinogaki.com/event-model/)
