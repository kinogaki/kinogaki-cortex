# Exp BK — chunk-as-expert in the calibrated pool (close AU's bpc gap) — 2026-06-26

**The bet (follow-up to AU).** AU's `ChunkLexicon` WON the splice axis (Isbilen sub-unit interference,
count-native) but LOST held-out bpc by +0.20, because its `ChunkAgent` REPLACED the calibrated backoff
n-gram with raw chunk-completion. The frame: the parser and the sequence-predictor are not rivals —
they are two experts the cortex consults at once. So **add** the chunk-completion next-char
distribution as ONE EXPERT into `cortex.vote`'s geometric-mean pool ALONGSIDE the char Columns, never
replacing the backoff. The lexicon was supposed to help where confident (mid-committed-chunk) and
abstain elsewhere; the backoff carries the rest. (lib/chunkvote.py — reuses `cortex.Column`,
`cortex.vote`, AU's `chunklex.ChunkLexicon` verbatim; READ, not modified.)

**Setup.** One text8 slice (`corpus.load_ids`): **2 MB train / 200 KB held-out**. One `ChunkLexicon`
(AU's strong config: `cover=longest, decay=0.5, mint_thresh=4`) trained single-pass on the first 1 MB,
then **frozen**. The char band is orders 0–6 (the same band `harness.CortexAgent` uses), pooled by the
calibrated geometric mean. Held-out bpc compared across three agents that share that one lexicon and
that one slice:

  1. **NgramAgent** — the plain backoff n-gram (orders 0–6), `cortex.vote` verbatim. The number to beat.
  2. **ChunkOnlyAgent** — AU's chunk-completion-ONLY shape (completion blended with a low-order char
     floor, `lam=0.6`), the +0.20 loser, reproduced over the same lexicon.
  3. **ChunkVoteAgent** — the FIX: the chunk-completion dist as one expert in the pool, sweeping the
     chunk-expert weight `chunk_w ∈ {0, .1, .25, .5, 1, 2}` (FRAGILE).

The splice axis is re-confirmed from a fresh lexicon on the synthesized Saffran stream, so the win the
fix must PRESERVE is on the table. Fixed seed (0), online single streaming pass. ~120 s on CPU.

---

## Result — SPLICE (the win that must be preserved): INTACT

| signal | lexicon within-word B–C | pure-TP B–C |
|--------|-------------------------|-------------|
| value  | **0.0013**              | 1.0000      |

The committed within-word transition still decays to 0.0013 vs pure-TP's 1.000 (AU's Result 1
reproduced). The fix does not touch the lexicon, so the splice win is untouched. Good — the axis the
idea actually wins is still won.

## Result — HELD-OUT BPC: the gap does NOT close (clean negative)

| agent | held-out bpc | Δ vs n-gram |
|-------|--------------|-------------|
| (1) NgramAgent (orders 0–6) | **2.165** | — (the number to beat) |
| (2) ChunkOnlyAgent (AU's loser) | 3.500 | +1.336 |
| (3) ChunkVoteAgent `chunk_w=0.00` | 2.165 | +0.000 |
| (3) ChunkVoteAgent `chunk_w=0.10` | 2.171 | +0.006 |
| (3) ChunkVoteAgent `chunk_w=0.25` | 2.180 | +0.016 |
| (3) ChunkVoteAgent `chunk_w=0.50` | 2.198 | +0.033 |
| (3) ChunkVoteAgent `chunk_w=1.00` | 2.236 | +0.071 |
| (3) ChunkVoteAgent `chunk_w=2.00` | 2.322 | +0.157 |

**The chunk expert helps at no positive weight.** Every `chunk_w > 0` makes held-out bpc *worse*,
monotonically. The only "gap-closed" row is `chunk_w = 0`, which is literally the n-gram with the
chunk expert switched off — not a blend. **The bet fails: adding the chunk vote to a properly
calibrated backoff cannot beat the backoff alone.**

(ChunkOnlyAgent here is +1.34, not AU's +0.20, because AU's n-gram baseline was a single order-4 model
(bpc 2.256) and its chunk agent leaned on a weak order-2 floor; my baseline is the full 0–6 band in the
geometric pool (2.165), against which raw chunk-completion is far worse. Same direction, harsher gap —
the better the backoff, the more the chunk completion only hurts.)

## Why — diagnosed, not hand-waved

The chunk-completion read-out has **no real abstain path**, and when forced to abstain it **never
fires**:

- **Ungated** (the spec's design): the lexicon seeds every single token, so for *any* trailing
  character there is always a length-1 prefix whose "completions" are *every chunk that starts with
  that char*. The expert therefore fires at **100% of held-out positions** and is only **25.6%
  argmax-correct** — a noisy near-unigram vote that dilutes the calibrated pool everywhere. That is
  exactly the monotone harm above.
- **Confidence-gated** (require partial length ≥2 AND low-entropy completions, swept
  `min_partial ∈ {2,3} × max_H ∈ {1,2} × w ∈ {.5,1,2}` per FRAGILE — don't kill at the baseline gate):
  the chunk expert fires on **0.00%** of held-out positions and changes bpc by **exactly +0.000**
  across the entire sweep. Its genuinely-confident completions are a strict subset of what the 0–6
  char band *already* models (the n-gram has the same high-order contexts), so a gated chunk expert
  contributes nothing the backoff doesn't already have.

There is no middle setting: ungated → noise that hurts; gated → silence that does nothing. The +0.20
gap in AU was an artifact of AU's *weak* backoff floor, not a real signal the lexicon could add to a
calibrated one.

---

## Verdict — **NEGATIVE (gap not closed; the splice win survives)**

- **Held-out bpc: the fix FAILS.** Chunk-as-expert is strictly ≥ the n-gram's bpc at every positive
  weight; the only break-even is the degenerate `chunk_w=0`. Confidence-gating it to be safe makes it
  fire 0% and contribute 0.000. The hoped-for "lexicon sharpens mid-chunk, backoff carries the rest"
  does not materialize, because a calibrated order-0–6 backoff already contains the lexicon's
  confident completions.
- **Splice axis: still INTACT** (0.0013 vs 1.000) — the fix doesn't touch the lexicon, so AU's real
  win is preserved, exactly as required.
- **Honest read:** the +0.20 "gap" was never a property of the chunk organ; it was AU's read-out
  leaning on a deliberately weak floor. Pool the chunk vote against a *strong* backoff and it has
  nothing to add. The chunk lexicon's value is **segmentation and an emission vocabulary** (AU's
  Results 1–2), **not next-char prediction** — and that conclusion is now nailed down rather than
  hoped around.

### Kill-condition
The implicit kill for this follow-up ("the blend's bpc ≤ the n-gram's") **fires** as a negative: the
blend never beats the n-gram for any `chunk_w > 0`. Reported honestly; a clean negative is a real
result.

### Rules compliance (confirmed)
- **Online single pass:** yes — the lexicon covers/counts/mints in one streaming pass; the char
  Columns count online; held-out scoring is a frozen forward pass. No second pass, no convergence loop.
- **No gradient / k-means / SVD / eigen / word2vec / backprop:** none — pure counts, greedy cover,
  leader minting, geometric-mean pooling (`vote_weighted` is `cortex.vote` with per-expert weights).
- **Bounded memory:** yes — the lexicon caps + LFU-evicts (AU); the char band is a fixed bounded set
  of Columns; the prefix index is built once over the bounded lexicon.

### What this tells the next round
- Stop trying to make the chunk lexicon a *predictor*. Its measured value is the **segmenter +
  variable-length emission vocabulary** (AU Results 1–2). The BUILD_QUEUE payoff is wiring the
  committed chunks into `CortexAgent.act()` as emission units and measuring type/token + generation
  validity — NOT bpc.
- If a chunk-aware predictor is ever wanted, it must contribute something the char band lacks
  (e.g. a *boundary* signal — "a word ends here" — feeding a top-down word prior à la `char_prior`),
  not a redundant next-char completion the backoff already owns.
