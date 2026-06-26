# Ray-cortex — offset attention + proximity graph + topic state

**Experiment W · 2026-06-26 · in progress**

Combines three earlier pieces — offset-keyed count-attention (Exp S), a
proximity/meaning graph (Exp P), and an online topic state (Exp T) — into one
ray-cortex over the word stream, asking whether routing prediction through a
graph of related words plus a committed topic buys coherence the offset model
alone does not. This experiment was still running in the source tree at the time
this repository was assembled, so its `RESULTS.md` is not yet included; the
catalog row is marked **in progress** and the slot is left open.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/raycortex/run.py
```

**Blog post:** none yet (pending results).
