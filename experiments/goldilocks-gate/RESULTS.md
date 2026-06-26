# Exp BI — the Goldilocks learning-rate gate (an inverted-U on surprisal) — 2026-06-26

*(RESULTS reconstructed from the completed run — the build agent ran the experiment clean but did not
write this file before returning. Numbers below are from `run.py`, seed 0, text8, reproducible.)*

**The bet.** A child does not learn equally from every word. The desirable-difficulty / Goldilocks-attention
literature (Kidd et al. on infant looking; the N400/cloze surprisal curve) says learning peaks in a **middle**
band of predictability — the already-known carries no news, the unparsable connects to nothing. The naive
correction "learn MORE the more surprised you are" (a **monotone** surprise-gate) is exactly wrong at the high
end, where a typo/OOV burst is maximally surprising and maximally worthless. BI makes the model's **write-weight
an inverted-U on its own surprisal**: skip the low-s news-free token AND the high-s noise, spend the bounded
write budget on the middle band that teaches. The decisive test is **at equal table size** — three gate shapes
share one streaming-pass, LFU-capped count model (**flat** = count every token once, **monotone** = the naive
gate this corrects, **goldilocks** = the inverted-U), so the only difference is *how much each token writes*.

---

## Result 1 — at equal cap, the gate never lowers bpc (the kill axis)

Held-out bpc at three table caps; lower is better. `flat` is the no-gate floor.

| cap | flat (no gate) | best monotone | best goldilocks | Δ flat→best gate |
|---:|---:|---:|---:|---:|
| 1,500 | **2.684** | 2.758 (steep) | 3.01 (wide) | **+0.073 worse** |
| 4,000 | **2.585** | 2.631 (steep) | 2.83 (wide) | **+0.046 worse** |
| 12,000 | **2.527** | 2.568 (steep) | 2.728 (wide) | **+0.041 worse** |

`kill 1500 → no gain · kill 4000 → no gain · kill 12000 → no gain.` **Flat wins at every cap.** Every gate that
throttles writes — monotone or inverted-U — *raises* bpc. Worse, it isn't even a hidden memory win: the
`gold-wide` gate writes **more** distinct contexts (≈230 k vs flat's 120 k) and **still loses**. Throttling what
a count learner writes is pure loss.

## Result 2 — the gate hurts the tail it deliberately skips

High-surprisal (top-tertile) rare-context slice — exactly the band the goldilocks gate refuses to write:

| | bpc on the high-s tail |
|---|---:|
| flat | **4.64** |
| goldilocks | 6.24 (**+1.59** worse) |

The "unparsable noise" the gate discards is, on real text, the **rare-but-real** long tail — and skipping it
costs +1.59 bpc exactly there. The literature's "high surprise = worthless" holds for typos, not for a Zipfian
vocabulary where the rare context is most of the information.

## Result 3 — the N400 read-out is faithful (a separate, real signal)

Kept distinct from the write-gate per the BUILD_QUEUE note: does `−log P_count` behave like an N400 read-out
(rise as cloze falls)? Yes — `flat` n400 r ≈ **0.855**, goldilocks ≈ 0.83 (both strongly positive). Surprisal as
a *read-out* is valid; that is orthogonal to using it as a *write-gate*, which is what failed.

---

## Verdict — **NEGATIVE (clean, well-diagnosed). Kill fired.**

At equal table size the Goldilocks (and monotone) write-gate **does not improve bpc** — `flat` write-everything-once
wins at every cap (+0.04 to +0.07), buys no memory win (the wide gate stores more and still loses), and the
gate *hurts* the rare tail it skips by +1.59 bpc. The kill-condition — *"at equal table-size the gate does not
improve bpc, OR improves it only by storing more"* — **fires cleanly.**

**Why it's the right negative.** "Goldilocks" is an *attentional* phenomenon (where an infant looks), not a
*write-rate law*. A count learner has no overwriting and no interference, so an early noisy high-order count is
**outvoted, never frozen** — there is nothing for a write-gate to protect against. This is the write-side echo of
**Exp AK** (*a count learner can't get stuck, so a difficulty curriculum is a no-op*): just as ordering the input
doesn't help, throttling the writes doesn't either. The honest keeper is the **read-out** (N400 r≈0.85), folded
out as its own diagnostic, not the gate.

**Rules.** Online single pass (one streaming write per token, gate is a per-token weight); bounded (LFU cap, the
gate's whole point); no gradient/k-means/SVD/backprop. Confirmed.

`lib/goldilocks.py`, `exp_bi_goldilocks/run.py` (text8, fixed seed 0, ~3 min CPU; FRAGILE budget: 10 gate shapes ×
3 caps × the skipped-tail check run).
