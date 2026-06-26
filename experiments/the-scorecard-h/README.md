# Building the dials that bits-per-char hides (H)

**Experiment GH · 2026-06-25 · win**

The second half of the scorecard work (Exp H): scores characters-only vs +word-concepts vs +phrases on the new dials. Headline: word concepts halved the overfit gap and lifted real-word rate 77%→89%; phrases lifted phrase coherence to 82%. The work was being done all along; bits-per-char simply could not see it.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/the-scorecard-h/run.py
```

**Blog post:** ["Building the dials that bits-per-char hides (H)"](https://cortex.kinogaki.com/the-scorecard/)
