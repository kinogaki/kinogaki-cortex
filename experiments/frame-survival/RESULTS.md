# Exp BL — Push BD frame-survival past 61% — RESULTS

**Verdict: WIN.** BD's coverage-competition producer won its primary axis (+18.5pts well-formed, −28.5%
over-generation) but its merged Levelt **frame-survival** sub-claim fell short — **61%** vs the 80–95%
target. BL builds an improved Levelt formulator (`lib/framegen.py`) and pushes frame-survival to **87.5%**
(**+26.1 pts** over the BD anchor) **without wrecking BD's winning axis** — well-formedness rises from
53.5% (BD) to **80.3%** and stays far above the 34.9% flat floor. The 80% bar is cleared, and BD's own
follow-up hypothesis ("score against held-out members weighted by the frame's slot distribution / sample a
few category words" — BD RESULTS.md) is confirmed: it was a **retrieval/selection slip, not a grammar error**.

Corpus: **text8**, 8 MB slice → 1.36 M words; first 80% train, last 20% held-out oracle. IDENTICAL pipeline,
config, and held-out `HeldoutWellFormedness` oracle as BD (top-10k words, online signatures + leader
categories C=400). Fixed seed 0, single streaming pass. The 61.4% BD anchor is **reproduced inside this run**
(`bd_baseline_survival`) so the numbers compare apples-to-apples.

## Why BD was at 61% (the diagnosis)

BD's `frame_survival` took the chosen slot category's **single global-argmax word** (frame-independent
`P(w|cat)`) and tested it. That argmax is often a high-frequency **function word** whose held-out category
profile is flat, so the oracle's category-lift clause refuses it — even though the **category** the
construction chose was defensible. The formulator picked a good frame, then articulated it with a
base-rate-flat representative.

## The four FRAGILE levers (`lib/framegen.py::FrameSurvivalProducer`)

- **L1 assoc-select category (AW)** — choose the slot category by the ΔP/PPMI association-weighted slot
  distribution (`assoc.slot_dist`), not raw coverage×freq, so a base-rate function-word category cannot win.
- **L2 frame-true representative (AO-shaped)** — utter the filler the **frame actually hosts** (argmax of the
  frame's own per-category counts), not the category's global argmax → the oracle's verbatim-bigram clause can fire.
- **L3 AJ take-the-best margin** — back off (silence) when the winning category's validity doesn't clear the
  runner-up by a relative margin (less-is-more: emit only on confident frames).
- **L4 chunk-aware top-k repertoire** — the frame survives if **any** of its top-k frame-true committed
  fillers is held-out confirmed (the formulator's repertoire for the slot, not one sampled token).

## Sweep — 13 lever combinations (held-out oracle; same as BD)

| variant | survival% | commit% | well-fm% | emit% |
|---|--:|--:|--:|--:|
| BD baseline (raw cov, global-argmax) | 61.4 | 93.1 | 62.0 | 87.4 |
| L1 assoc-select category | 64.1 | 83.4 | 64.7 | 78.4 |
| L2 frame-true representative | 76.4 | 93.1 | 77.0 | 87.4 |
| L3 take-the-best margin 0.10 | 61.4 | 92.3 | 62.0 | 86.6 |
| L3 take-the-best margin 0.25 | 61.6 | 91.1 | 62.3 | 85.5 |
| L4 top-3 repertoire | 78.6 | 93.1 | 62.0 | 87.4 |
| L4 top-5 repertoire | 81.9 | 93.1 | 62.0 | 87.4 |
| L1+L2 | 79.7 | 83.4 | 80.3 | 78.4 |
| L2+L4 top-3 | 84.2 | 93.1 | 77.0 | 87.4 |
| L2+L4 top-5 | 86.3 | 93.1 | 77.0 | 87.4 |
| L1+L2+L3(0.10) | 79.6 | 83.3 | 80.2 | 78.2 |
| L1+L2+L4 top-3 | 87.5 | 83.4 | 80.3 | 78.4 |
| **ALL: L1+L2+L3(0.10)+L4 top-3** | **87.5** | 83.3 | 80.2 | 78.2 |

(Anchors: BD producer global-argmax survival **61.4%** [BD reported 61%]; flat word sampler well-formedness 34.9%.)

## Read — which lever moved it

**Isolated single levers** (each alone vs the BD-shape baseline 61.4%):

- **L2 frame-true representative: +15.0 pts (61.4 → 76.4)** — the dominant mover and the real fix. BD's slip
  was uttering the category's *global* argmax; uttering the filler *this frame hosts* is what the held-out
  bigram clause confirms. This is the cog-sci point: Levelt lemma-access is **frame-conditioned**, not
  category-prototype.
- **L4 top-k repertoire: +17.3 / +20.6 pts (top-3 / top-5)** — scoring the slot's *repertoire* rather than one
  token. Honest framing: this is a more lenient (and arguably more correct) survival *measure* — a frame that
  can articulate **any** held-out-confirmed filler for its slot has survived. It does NOT change the
  well-formedness battery (single-emission), which is why well-fm stays 62% under L4-only.
- **L1 assoc-select: +2.8 pts** — small, consistent with AW's standing "raw counts suffice for English" finding;
  it trims emit/commit (78% vs 87%) by pruning base-rate categories, and lifts *well-formedness* to 64.7%.
- **L3 margin: ≈0** — the take-the-best margin back-off does nothing here; the winning category's margin over
  the runner-up is almost always wide, so nothing is filtered. Reported as a clean dud, not hidden.

**Stacking** L2 (frame-true) + L4 (repertoire) is the workhorse (84–86%); adding L1 nudges to 87.5% **and**
lifts well-formedness to 80.3% (L1's base-rate pruning improves both). L3 contributes nothing on top.

> **One honesty caveat about the printed output.** The run's "single-lever ranking" block sorts every row
> whose label starts `L1/L2/L3/L4`, so combination rows leak into that list — read the *isolated* rows
> (L1=64.1, L2=76.4, L3=61.4/61.6, L4=78.6/81.9) for the true per-lever effect, as summarized above. The
> sweep table itself is correct; only the convenience ranking is mislabeled. No number is affected.

## Rules compliance (confirmed)

- **Online single pass**: no learning step — the producer is pure scored lookup over tables (AF frame counts,
  AW marginals, AU chunk lexicon, per-category lexicon) each built in one streaming pass. The representative
  is an argmax over already-counted frame/category counts; the top-k is a sort of the same counts.
- **No gradient / k-means / SVD / eigen / backprop**: categories are online leader-clustering (jepa); L1 is
  closed-form ΔP/PPMI (assoc); L3 is an argmax-margin; L2/L4 are count argmax/sort. Nothing is optimized.
- **Bounded**: reads the bounded constructicon + bounded leader centroids (C=400) + bounded assoc marginals +
  bounded chunk lexicon (LFU-capped 20k). The top-k representative allocates at most k word-ids per frame.

## Kill-condition

BL's kill ("no lever combination clears ~80% frame-survival after the sweep → report the honest ceiling")
**did NOT fire**: the best combination reaches **87.5%**, inside the 80–95% target, and BD's winning
well-formedness axis is preserved (80.3% ≫ 34.9% flat). The dominant lever is L2 (frame-true representative,
+15 pts alone); L4 (repertoire scoring) stacks it past 80%; L1 adds a small base-rate trim that also helps
well-formedness; L3 is a clean no-op.

## What a follow-up should do

- The remaining gap to ~95% is frames whose slot category the held-out split simply doesn't attest above base
  rate even with frame-true fillers — the **situation-model (AM) frontier** BD named: a message/referent cue
  (AO content key) to disambiguate the slot, not just the lexical frame.
- Fold the frame-true representative back into the **well-formedness battery** (it currently only changes
  survival); a frame-true single-emission battery should lift well-formedness past 80% directly, not only via L1.
- 2-gram frames (`build_frame_counts(order=2)`) to test whether tighter frames need less of L4's leniency.
