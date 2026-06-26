# Exp BE — Contingency-gated learning rate (G2) · the temporal-contingency dial

**2026-06-26 · acquisition / Tier-3 reactive loop · verdict: WIN (kill did not fire)**

## The claim

A reply that *answers* you teaches more than the identical words overheard cold. Goldstein & Schwade's
contingent-vs-yoked design: babies given contingent caregiver feedback advanced; babies given the *same
sounds re-timed to ignore them* did not. G2 installs that dial — in `observe()`, weight each count
increment by a soft gain `g = exp(−Δt/τ)` (Δt = steps since the agent last spoke), into two bounded
registers: `tab_hot` (warm replies, loud) + `tab_cold` (background, never silenced). Prediction pools both.

## Setup

- **Corpus.** SUBSTITUTION: G2 names CDS/dialogue (CHILDES), which is **not** in `data/`. We substitute
  **text8** (1.46M chars) as the structured "teacher reply" signal and a **frequency-matched i.i.d. scramble**
  of the same alphabet as non-contingent babble. The only difference between warm and cold input is
  structure, which isolates the dial. Held-out bpc scored on 40k held-out **real** text8.
- **World.** A fixed transcript: agent speaks → **warm** real-English reply (Δt→0) → 2 bursts of **cold**
  babble (the noisy room). 33% structured, rest babble. Every condition sees the *exact same tokens in the
  same order* — they differ only in how the dial weights them.
- **Baselines, registered before running.** PASSIVE (gain≡1, single table — the AT floor); **YOKED**
  (identical tokens + loop, gain drawn from the **scrambled** timeline → same multiset of gains, wrong
  alignment); plus a graded-contingency robustness sweep.
- **Metrics (reported separately, per spec).** held-out **bpc**; **turn-overlap** = mean prob the model
  puts on the true next char of held-out English minus mean prob on babble's next char (how well it locked
  onto the contingent channel vs the babble it also heard).

## Results

### (1) The three load-bearing conditions — same tokens, only timing→gain differs

| condition | held-bpc | turn-overlap | |
|---|---|---|---|
| PASSIVE | 2.930 | 0.1380 | no dial — reads babble and signal equally |
| YOKED   | 3.144 | 0.1045 | same tokens, **scrambled** timing → random gain |
| **ON**  | **2.692** | **0.1759** | real contingency: warm replies up-weighted |

ON beats YOKED by **+0.452 bpc** and **+0.071 turn-overlap**, and beats the PASSIVE floor by **+0.238 bpc**.
(YOKED is *worse* than passive: random gains actively mis-weight, up-counting babble and down-counting
signal — scrambled contingency hurts.)

### (2) FRAGILE sweep — τ × (cold_w, hot_pool), 12 settings

ON beats YOKED on bpc at **12/12** settings (Δ +0.444…+0.468) and on turn-overlap at **12/12** (Δ
+0.067…+0.077). τ has no effect here (in this transcript warm always lands at Δt=0, cold far out — the
warmth window never bites); the win rides the hot/cold *split*, strongest at cold_w=0.40, hot_pool=1.5.

### (2b) ROBUSTNESS — graded contingency (the control that matters most)

Make timing only a *probabilistic* cue: warm slot is signal w.p. `r`, signal leaks into cold slots w.p.
`1−r`. `r=0.5` = timing carries **no** information; the dial *should* collapse to YOKED there.

| r | bpc_ON | bpc_YK | Δbpc | Δturn-overlap |
|---|---|---|---|---|
| 1.00 | 2.680 | 3.141 | **+0.462** | +0.0755 |
| 0.90 | 2.736 | 3.074 | +0.338 | +0.0547 |
| 0.75 | 2.804 | 3.028 | +0.224 | +0.0365 |
| 0.60 | 2.891 | 2.975 | +0.083 | +0.0106 |
| 0.50 | 2.910 | 2.920 | **+0.010** | −0.0004 |

The gap shrinks monotonically to ~0 exactly as timing stops predicting content. This is the signature of a
**real** mechanism, not an artifact: the dial pays *only* when timing is informative, and correctly does
nothing (Δ≈0) when it isn't.

## Verdict — WIN

**Kill did NOT fire.** The kill-condition (ON matches YOKED on bpc AND turn-overlap at matched tokens)
fails to trigger: ON beats YOKED on **both** axes at every one of the 12 dial settings, and the
robustness sweep shows the gap is genuinely carried by timing-content alignment (it vanishes at r=0.5).
Contingency-gated counting learns the structured channel **+0.45 bpc** better than the same tokens with
scrambled timing, and **+0.24 bpc** better than reading everything equally — for free, from timing alone,
with no content label.

**Honest caveats.** (1) The win is decisive but the world is *idealized*: at r=1.0 warm≡signal is a clean
binary split, which is closer to a perfect joint-attention flag than the "soft gain" the guard asks for —
the *graded* sweep (2b) is the fair test, and the dial still wins clearly down to r≈0.6. (2) τ is inert in
this transcript (timing is all-or-nothing); a world with *graded* delays would be needed to show the
exponential window matters vs a hard gate. (3) Substituted text8+babble for CHILDES — the contingency
claim is about timing, not this text, but a real CDS corpus is the obvious follow-up.

## Rules confirmed

- **Online** — one streaming pass; `g` is a per-increment scalar inside `observe()`, no second pass.
- **No backprop / k-means / SVD / eigen** — only count increments and the calibrated geometric-mean `vote`.
- **Bounded** — two Column bands (hot/cold) under the same context-tail trim as the base agent; hot is
  small/specific; LFU/AE-protection unchanged.
- **Cognition-as-guide** — the dial is contingency/availability (Goldstein & Schwade), not optimization;
  the soft gate never silences cold input.

## Files

- `../lib/contingency.py` — `ContingencyAgent` (the dial) + `yoked_gains` (the control). Reusable.
- `run.py` — this experiment (seed 0).

## Follow-up

E3 (communicative-success reweighting) is gated to run *after* BE with G2 as baseline — cut if it adds
nothing beyond contingent timing. The natural next test for BE itself: a **graded-delay** world (replies
arrive at varied Δt, not all-or-nothing) to make τ bite and test the *soft* gain against a hard
joint-attention flag — and a real child-directed corpus in place of the babble substitution.
