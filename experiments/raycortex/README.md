# Ray-cortex — offset attention + proximity graph + topic state

**Experiment W · 2026-06-26 · negative result**

The fair rematch for the parked proximity/raytracing idea (Exp P). We gave
proximity its best shot: the graph form (spreading activation over a PMI
association graph, the form the sources endorse), inside the best stack we have
(offset-keyed count-attention core from Exp S, leaky-evidence pooling from Exp R,
online-clustered topic prior from Exp T, swept weights), judged on the axis it
was always meant for — rare/unseen contexts. Headline: the offset core is the
workhorse (322.7 ppl vs bigram 694.5); accumulated evidence **earns its keep** on
rare contexts (+12.5 ppl, 95% CI [6.7, 20.6], significant), standalone and in
combination; proximity, given a real supported shot, still has no prediction
niche (rare gap −4.8 ppl, not significant) because a thin context means the
preceding word is rare and rare words are absent from the PMI graph. Parked
deeper, with a mechanism named; the live use for the association graph is
inspection and similarity, not prediction.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/raycortex/run.py
```

**Blog post:** ["We gave the map its best shot"](https://cortex.kinogaki.com/raycortex/)
