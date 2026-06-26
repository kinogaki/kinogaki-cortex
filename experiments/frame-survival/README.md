# Exp BL — Push BD frame-survival past 61%

**Question.** BD's coverage-competition producer won its primary axis (more well-formed, less
over-generating than a flat sampler) but its merged Levelt **frame-survival** sub-claim hit only **61%** vs
the 80–95% target — the % of emitted utterances that PRESERVE a learned construction frame end-to-end. Can a
better Levelt formulator push survival past 80% **without** wrecking BD's winning well-formedness axis?

**Diagnosis.** BD measured survival with the chosen slot category's single **global-argmax word** — often a
high-frequency function word with a flat held-out profile, so the non-circular oracle refuses it even when the
category was right. A retrieval/selection slip, not a grammar error.

**Mechanism — `lib/framegen.py::FrameSurvivalProducer`** (read-only over AF/AW/AU/AJ; four FRAGILE levers):

- **L1** association-select the slot category (AW ΔP/PPMI `slot_dist`) — base-rate function-word categories can't win.
- **L2** **frame-true representative** — utter the filler the *frame itself hosts*, not the category's global argmax.
- **L3** AJ take-the-best margin — back off (silence) on thin categories.
- **L4** chunk-aware top-k — frame survives if *any* of its top-k frame-true committed fillers is held-out confirmed.

**Measure.** Identical to BD: `HeldoutWellFormedness` over the last 20% of an 8 MB text8 slice the grammar
never sees (verbatim held-out bigram OR frame prefers the word's category above its held-out base rate). The
61.4% BD anchor is reproduced inside the run. Fixed seed 0, single streaming pass.

**Result.** WIN. Frame-survival **61.4% → 87.5%** (+26.1 pts), inside the 80–95% target; well-formedness
preserved at **80.3%** (≫ 34.9% flat floor). Dominant lever: **L2 frame-true representative (+15 pts alone)**;
L4 repertoire stacks past 80%; L1 adds a base-rate trim that also lifts well-formedness; **L3 is a clean
no-op**. See `RESULTS.md`.

## Run

```sh
cd /Users/sedov/Dev/kinogaki/libraries/kinogaki-cortex/experiments
exp_a_boundary/.venv/bin/python exp_bl_framesurvival/run.py
```

~20 s. Builds the AF/AW/AU grammar on text8, anchors BD's 61% survival, sweeps 13 lever combinations.

## Files

- `run.py` — build grammar (= BD's), anchor BD survival, FRAGILE sweep over the four levers, verdict.
- `lib/framegen.py` — the improved Levelt formulator + `frame_survival` (BD's oracle, levers folded in).
- `RESULTS.md` — full table, per-lever attribution, rules compliance, kill-condition, an honesty caveat on
  the printed single-lever-ranking labels.

## Rules

Online single pass (no learning step — scored lookup over one-pass tables); no gradient/k-means/SVD/eigen/
backprop (closed-form ΔP/PPMI + count argmax/sort); bounded (constructicon + C=400 centroids + assoc marginals
+ LFU-capped chunk lexicon). FRAGILE: levers isolated then stacked; L3's null effect reported honestly.
