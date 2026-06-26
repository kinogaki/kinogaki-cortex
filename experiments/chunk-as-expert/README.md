# Exp BK — chunk-as-expert in the pool (close AU's bpc gap)

**Question.** AU's chunk lexicon won the splice axis but lost held-out bpc by +0.20 because its agent
*replaced* the calibrated backoff with raw chunk-completion. Can we close that gap by adding the
chunk-completion distribution as ONE EXPERT in `cortex.vote`'s geometric-mean pool alongside the char
Columns — never replacing the backoff?

**Answer: no (clean negative).** On one text8 slice (2 MB train / 200 KB held-out), the chunk-as-expert
blend never beats the plain n-gram for any positive chunk weight (bpc monotonically *worse*; only
break-even is weight 0 = the n-gram itself). Ungated, the chunk expert fires at 100% of positions and is
25.6% argmax-correct (noise); confidence-gated, it fires at 0.00% and changes bpc by exactly +0.000. A
calibrated order-0–6 backoff already contains the lexicon's confident completions. AU's splice win
(within-word B–C 0.0013 vs pure-TP 1.000) is left intact — the fix never touches the lexicon.

The chunk lexicon's value is **segmentation + an emission vocabulary**, not next-char prediction.

## Files
- `run.py` — trains one frozen `ChunkLexicon`, compares NgramAgent / ChunkOnlyAgent / ChunkVoteAgent
  (sweeping `chunk_w`) on held-out bpc, re-confirms the splice axis.
- `../lib/chunkvote.py` — the three agents + `vote_weighted` (`cortex.vote` with per-expert weights).
  Reuses `cortex` and AU's `chunklex` verbatim; READ, not modified.
- `RESULTS.md` — full numbers, the why-diagnosis, the verdict.

## Run
```sh
cd /Users/sedov/Dev/kinogaki/libraries/kinogaki-cortex/experiments
./exp_a_boundary/.venv/bin/python exp_bk_chunkvote/run.py
```
~120 s on CPU, fixed seed 0, online single pass.
