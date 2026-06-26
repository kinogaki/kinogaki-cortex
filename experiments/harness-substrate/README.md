# The harness substrate

**Experiment AT · 2026-06-26 · substrate**

The shell that turns offline modelling into a reactive learning loop: **read, speak, hear a reply that depends
on what you said, learn from it.** Four encoding-agnostic pieces in `lib/harness.py` — **Codec** (the only
text-aware part), **Environment** (reactive world; L0 passive `CorpusEnv` + L2 `InterlocutorEnv` seam with a
pluggable responder where a live model like Haiku slots in), **Agent** (the cortex as a policy: online
`observe`, minimal `act`), and **`run()`** (drives them, fires metric probes into a report card).

Substrate check passes: online reading drops held-out bpc **4.63 → 2.29**; the reactive contract holds and the
agent's surprise at its interlocutor falls **2.36 → 1.77**. Generation is still gibberish and comprehension
benchmarks aren't built — both deliberately deferred; this is the shell they plug into.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/the-harness/run.py
```

Online throughout: Columns + calibrated vote, no gradients, no new model.
