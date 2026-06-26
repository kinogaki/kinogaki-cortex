# Exp K — does depth pay off at scale? (3/4/5 levels × data) — 2026-06-25

Multi-level FastCortex (char + word bands; 5th = topic cache) × 10/40/80 MB. The 80 MB tier ran too slow under
the per-position Python eval (later fixed by batch eval, Exp N) and was killed; 10/40 MB show the trend clearly.

| data | levels | test bpc | overfit | real-wd% | phrase% |
|---|---|---:|---:|---:|---:|
| 10 MB | 3 (char+word{1,2}) | 1.942 | +0.93 | 99.6 | 96.0 |
| 10 MB | 4 (+word trigram) | 1.943 | +0.99 | 99.1 | 94.4 |
| 10 MB | 5 (+topic cache) | 1.829 | +0.77 | 99.3 | 94.0 |
| 40 MB | 3 | 1.767 | +0.67 | 99.8 | 98.2 |
| 40 MB | 4 | 1.770 | +0.73 | 99.8 | 97.4 |
| 40 MB | 5 | 1.701 | +0.57 | 99.8 | 91.4 |

## Findings

- **More data helps a lot** — 3-level bpc 1.942 → 1.767 (10→40 MB); 5-level 1.829 → 1.701.
- **The 4th level (word trigram) is flat** (1.767 vs 1.770 at 40 MB) — local word context saturates, just as
  char-bpc did (Exp D). Stacking *more local n-gram levels* is not the win.
- **The 5th level (topic recency cache) helps bpc** (−0.07 to −0.10) and lowers the overfit gap, but the edge is
  roughly *constant* with data (not growing) and it slightly hurts phrase-coherence (pulls in topical words).
- **Lesson that set the night's direction:** the payoff is not more fixed local levels — it's a level that
  **reaches beyond local context** (topic/attention). That motivated the attention (Exp L) and boundary (Exp M)
  work, and the source-mining's top idea (top-down prior / ignition for global coherence).
