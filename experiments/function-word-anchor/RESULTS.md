# Exp AX — Function-word anchor voter — RESULTS

**Verdict: NEGATIVE (clean, well-diagnosed).** The anchor cue is *mathematically redundant* with the AF
frame voter. It adds **+0.000** POS-cluster purity over AF on the same firing positions, at every band
size k. The formal kill (BUILD_QUEUE) does **not** fire only because its second clause never can — AJ
ignores the dead cue, so the combined result never degrades. But the mechanism contributes nothing.

Corpus: **text8** (14 MB prefix → 2.39 M words, top-N=10 000, C=400 online leader clusters). Negative
control: the **same stream word-order-shuffled** (frequency ranks identical, adjacency destroyed) —
substituting for the German slice the spec names, which `data/` does not contain (SAID SO). Fixed seed 0,
single streaming pass. Gold POS = a hand-built 283-word English closed/open-class lexicon, used **only**
to score purity (nothing is trained on it).

## The headline

The AF open-slot head `predict_open(prev)` is `P(next-word category | prev word)`. The anchor voter's
`predict_right(prev)` is `P(next category | prev is an anchor)` — the **identical count table**, masked to
the top-k frequency band. So at every position the anchor fires, anchor and AF emit the *same*
distribution. Confirmed empirically: `argmax` and top mass match to 16 digits.

### POS-cluster purity, anchor-band sweep (ENGLISH text8)

Purity is biased toward fewer-category / lower-entropy subsets, so the only fair comparison is **AF
scored on the anchor's own firing positions** (`AFsamePos`), not AF over the full eval.

| k   | anchor purity | (nCat, n)    | AF same-positions | anchor − AF |
|-----|---------------|--------------|-------------------|-------------|
| 5   | 0.590         | (2, 16 645)  | 0.590             | **+0.000**  |
| 10  | 0.536         | (2, 22 065)  | 0.536             | **+0.000**  |
| 20  | 0.540         | (2, 31 128)  | 0.540             | **+0.000**  |
| 30  | 0.527         | (2, 38 685)  | 0.527             | **+0.000**  |
| 50  | 0.490         | (2, 45 190)  | 0.490             | **+0.000**  |
| 80  | 0.468         | (3, 51 301)  | 0.472             | −0.004      |
| 120 | 0.453         | (5, 57 339)  | 0.456             | −0.004      |

(The small negatives at large k are because the *wider* band lets AF predict slightly different argmax
categories — i.e. AF is, if anything, marginally better. The anchor never wins.)

The apparent "+0.222 over AF-full" (anchor 0.540 vs AF-full 0.319) is a **purity artifact**: the anchor
only predicts at high-frequency frames, whose right-neighbours collapse to ~2 dominant categories — an
easier, lower-entropy subset. Computing AF on that *same* subset erases the gap entirely. Reporting the
unfair full-eval number would have been a fake win; it is not in the verdict.

### Next-CATEGORY perplexity where the anchor fires (k=20, ENGLISH)

| voter  | ppl   | n        |
|--------|-------|----------|
| anchor | 5.015 | 102 944  |
| frame  | 5.015 | 102 944  |
| combo  | 5.015 | 102 944  |

Identical to 3 decimals — same distribution, same predictions.

### Negative control (word-shuffled; substitutes German)

| voter            | purity | (nCat) |
|------------------|--------|--------|
| anchor           | 0.314  | (1)    |
| AF same-positions| 0.314  | (1)    |
| combo            | 0.313  | —      |

Shuffling collapses both voters to **chance** (0.31 ≈ the dominant-POS base rate) and to a single
predicted category — the diagnostic behaves exactly as the spec predicted an anchor *should* on a corpus
where the frequency band is not the categorial-adjacency class. anchor − AF stays **+0.000**: redundant
in the negative case too.

## Kill-condition (BUILD_QUEUE AX)

> *Kill:* adds <2% purity over AF on English **AND** degrades the AJ-combined result on either language
> (confirm it isn't just mis-validitied before killing).

- Clause 1 (**<2% purity over AF on English**): **TRUE** — adds +0.000 (fair, same-positions), at all k.
- Clause 2 (**combo degrades on either language**): **FALSE** — combo never degrades; AJ take-the-best
  measures the anchor cue's validity, finds it equals AF's, and never lets the redundant cue override.
  combo = AF-full on both languages.
- **Formal kill (AND) does NOT fire.** But this is the *good* failure mode: the combiner correctly
  refuses to be harmed by a useless cue. The mechanism is **redundant, not harmful**.
- FRAGILE budget honoured: swept k ∈ {5,10,20,30,50,80,120}. The +0.000 is **structural** (the two
  voters share a count table), not a mis-sized band — confirmed by reading the distributions directly,
  so this is not a mis-validitied result that further tuning would rescue.

## Why it's negative (the real lesson)

The anchor idea conflates two things that are already the same in this substrate: "the closed class" and
"frames with high token count." AF's 1-gram open-slot head **is** a per-preceding-word right-neighbour
category voter; restricting it to the top-k most frequent preceding words (= the anchors) is a no-op on
the prediction. For the anchor cue to be *new* it would need either (a) a **different key** than "the
single preceding word" (e.g. nearest-anchor-within-window, skipping intervening content words — a
non-adjacent frame AF does not build), or (b) a **generation guardrail** that sharpens the next-token
*character* distribution after emitting an anchor (the spec's second proposal, untested here because the
purity axis already came back null). Either is a real follow-up; raw frequency-rank as an *adjacency*
category signal is fully subsumed by AF.

## Rules compliance

- **Online single pass:** anchors = one `bincount` argmax; right/left tallies = one streaming pass;
  categories = jepa's one-pass signatures + one-pass leader clustering. ✓
- **No backprop / k-means / SVD / eigen / word2vec:** counts and a frequency threshold only. ✓
- **Bounded memory:** ~k anchors (k ≤ 120), each a length-C tally; C capped at 400. ✓
- **Judged on the axis it could win** (POS-cluster purity + firing perplexity), swept before judging. ✓

## Files
- `run.py` — the experiment (text8 + shuffled control, k-sweep, fair same-positions purity, firing ppl).
- `../lib/funcanchor.py` — `AnchorVoter` (top-k anchors + right/left category tallies) + `category_pos_purity`.

## Repro
```
cd /Users/sedov/Dev/kinogaki/libraries/kinogaki-cortex/experiments
exp_a_boundary/.venv/bin/python exp_ax_funcanchor/run.py
```
~7 s end-to-end.
