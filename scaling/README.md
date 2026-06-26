# Scaling studies

Every experiment in this lab was first measured on a few megabytes — the right
size to find out *whether* an idea works, the wrong size to find out whether it
*lasts*. A mechanism can win at 15 MB for a dull reason: the local counts have
not yet seen enough text, so anything that fills the gap looks clever. More data
closes the gap on its own.

So we re-ran everything at 30–200× the data and asked one question of each idea:
**which verdicts change when data is no longer the bottleneck?** Online only —
counting, leaky accumulators, online leader-clustering; no gradient descent, no
k-means, no SVD, no batch pass that revisits the data.

The narrated synthesis is the post **[What survives scale](https://cortex.kinogaki.com/what-survives-scale/)**.

## The two batches

- **Batch 1 — single mechanisms at scale** ([REPORT.md](REPORT.md), [RESULTS.tsv](RESULTS.tsv)).
  The 1-Billion-Word benchmark (~526 M words / ~3.03 B chars) normalized to a
  27-symbol space. The char-order scaling curve, offset-attention order
  sensitivity, topic ignition's backoff-slice gain, the heterogeneous stack, and
  the trajectory change-stream — each measured to 500 M–3 B units.
- **Batch 2 — synthesis experiments re-run at scale** ([REPORT2.md](REPORT2.md), [RESULTS2.tsv](RESULTS2.tsv)).
  LM1B for word/char work, enwik9 for the article-boundary truth (AC), text8 for
  analogy families (AD), the Darwin→Shakespeare→Bible register files for
  retention (AE). Each experiment re-measured on its **own** right-axis metric,
  not bits-per-char, with the small-scale prior recorded beside the big-scale
  result.

## The dichotomy

The results split clean, and the line is sharp enough to name.

**Held or grew with scale** — these do what counting structurally cannot:

- Bayesian-surprise boundaries (event-model AC): F1 **0.154 → 0.447** (3→36→960 MB),
  ~14× better than surprisal (0.45 vs 0.032) — the standout, *grew* with scale.
- Similarity-backoff for unseen contexts (Z): ~20× → **56×** perplexity win.
- Calibrated confidence (AB): ECE ~8–10× better, holds. Constructions (AF):
  ~5.6× unseen-filler generalization, 82% win, holds. Trajectory directionality
  (V) holds at 2.9 B chars. Char order-5 still dropping at 3 B chars
  (1.792 → 1.749, not saturated); order-4 saturated at 1.95.

**Vanished at scale** — these competed with local counts on already-seen prediction:

- Top-down topic prior (ignition T): +0.34 bits/word → **0.0** (local counts subsume the topic).
- Noise→concept-reliance shift (Y): +9 pp → **−1.9 pp** (model already ~99.9% concept-reliant at scale).
- Sleep's bpc gain (AA): −0.011 → **+0.012** (less lossy memory to consolidate; rare-context still benefits).
- Retention edge (AE): ~21× → **1.13×** — *soft*; the register corpus is too small to truly scale.

**The lesson.** At scale, "more data" absorbs any mechanism that competes with
local counts on already-seen prediction. A mechanism survives only if it does
something counting structurally cannot — generalize to the unseen, detect
discourse structure, report calibrated confidence, encode direction, or retain
under bounded memory. Every survivor is about representation and signal, the same
frontier as the open count-native combiner that sharpens without blurring.

## Honest caveats

A few measurements came back incomplete (a flat or broken big-scale result is
still data): the sim-hybrid rare-slice (Z) and ray-cortex rare-slice (W) returned
`nan` / 0 probes; analogy (AD) did not truly scale — its fixed-size profile ran at
16 MB; the retention corpus (AE) is too small to be a real scale test; and char
order-6 was skipped (a ~42 GB table, past machine memory — needs a GPU or sparse
path).
