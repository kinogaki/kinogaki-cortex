# Reaching back by the right cue

**Experiment AO · 2026-06-26 · win**

Content-addressable cue retrieval binds long-distance subject–verb agreement to a
correct-number antecedent (subject ~15 words back, across a clause) **99.96%** of
the time vs offset-attention's **65.26%** — a wall past its one modal offset, where
a fixed position key scores flat zero. It also reproduces the human agreement-
attraction interference signature (0.00% → 2.11% → 0.01% across no-distractor /
opposite-number distractor / same-number competitor), because activation divides by
the **fan**. New primitive: long-distance binding by content cue + fan, generalising
offset-attention's *position* key to a *feature* key.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/cue-retrieval/run.py
```

**Blog post:** ["Reaching back by the right cue"](https://cortex.kinogaki.com/blog/cue-retrieval/)

Credit: Lewis & Vasishth (cue-based retrieval); Anderson (the fan effect).
