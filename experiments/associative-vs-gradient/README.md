# The counter beat the neural net

**Experiment B · 2026-06-25 · win**

Hunts for online learning without forgetting by streaming four registers of English in sequence and re-testing every register after each. The forgetting never appeared (character-level English is nearly one stationary distribution), but the important result fell out: a plain online counter beat both a dense and a sparse gradient network (2.3 vs ~3.5 bits per char). The counter became the substrate for everything after.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/associative-vs-gradient/run.py
```

**Blog post:** ["The counter beat the neural net"](https://cortex.kinogaki.com/associative-vs-gradient/)
