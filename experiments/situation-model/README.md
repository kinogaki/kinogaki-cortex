# Keeping track of what's going on

**Experiment AM · 2026-06-26 · negative**

A persistent situation model (Chambers-Jurafsky event chains + Zwaan who/where/topic
slots) does **not** predict over long spans. The +0.55 bpw "everywhere" win was pure
smoothing repair: a **static** frozen unigram beats the live situation by −0.07 bpw on
the 99% non-backoff slice; the situation helps only the same 0.9% backoff slice (the
third mechanism to land there). Methodological lesson: measure a top-down prior against
a *static* prior, never against no prior.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/situation-model/run.py
```

**Blog post:** ["Keeping track of what's going on"](https://cortex.kinogaki.com/blog/situation-model/)

Credit: Chambers & Jurafsky (narrative event chains); Zwaan (event-indexing model).
