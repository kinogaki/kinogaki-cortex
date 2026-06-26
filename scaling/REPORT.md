# Overnight big-data batch — REPORT

_Generated 2026-06-25 23:47:32_

Corpus: 1-Billion-Word benchmark, normalized to a-z+space id-space (`data/lm1b_ids.i8`).
Online only — counting / leaky accumulators / online leader-clustering; no gradient descent, no k-means/SVD/eigen.

Logs: `overnight/logs/<exp>_<scale>.log`. Machine-readable: `overnight/RESULTS.tsv`.

## All measurements

| exp | scale | metric | value | verdict | note |
|---|---|---|---|---|---|
| _check | 2,000,000 | chunk_vs_dense_bpc_diff | 0.00e+00 | OK | chunk=2.082857 dense=2.082857 |
| char_order4 | 50,000,000 | bpc | 1.96311 | scales | tables=0.06GB t=2s |
| char_order4 | 150,000,000 | bpc | 1.95613 | scales | tables=0.06GB t=5s |
| char_order4 | 500,000,000 | bpc | 1.95266 | scales | tables=0.06GB t=18s |
| char_order4 | 1,000,000,000 | bpc | 1.95167 | scales | tables=0.06GB t=36s |
| char_order4 | 2,000,000,000 | bpc | 1.95119 | scales | tables=0.06GB t=74s |
| char_order4 | 3,000,000,000 | bpc | 1.95097 | scales | tables=0.06GB t=110s |
| char_order4 | 3,030,351,559 | bpc | 1.9501 | scales | tables=0.06GB t=110s |
| char_order5 | 50,000,000 | bpc | 1.7921 | scales | tables=1.61GB t=3s |
| char_order5 | 150,000,000 | bpc | 1.77071 | scales | tables=1.61GB t=8s |
| char_order5 | 500,000,000 | bpc | 1.75934 | scales | tables=1.61GB t=26s |
| char_order5 | 1,000,000,000 | bpc | 1.75556 | scales | tables=1.61GB t=51s |
| char_order5 | 2,000,000,000 | bpc | 1.75335 | scales | tables=1.61GB t=103s |
| char_order5 | 3,000,000,000 | bpc | 1.75244 | scales | tables=1.61GB t=154s |
| char_order5 | 3,030,351,559 | bpc | 1.74855 | scales | tables=1.61GB t=159s |
| char_order6 | 0 | bpc | nan | skipped-mem-safe | 27^7 float32 table ~42GB + int64 bincount temp ~84GB > 64GB RAM |
| hetero_stack | 864,000,000 | bpc_char | 1.8872 | concept-flat | static=2.41516 gate=2.06027 |
| hetero_stack | 864,000,000 | bpc_gate | 2.06027 | gate-wins | char=1.88720 static=2.41516 |
| ignition | 288,000,000 | char_dbpc_withG | 0.0 | G-marginal | no=1.6711 yes=1.6711 K=128 |
| ignition | 288,000,000 | word_backoff_dbits | 0.0 | G-marginal | no=11.6092 yes=11.6092 nbo=4984 |
| ignition | 500,000,000 | char_dbpc_withG | nan | capped | word-level too heavy |
| ignition | 864,000,000 | char_dbpc_withG | 0.0 | G-marginal | no=1.6523 yes=1.6523 K=128 |
| ignition | 864,000,000 | word_backoff_dbits | 0.0 | G-marginal | no=11.7250 yes=11.7250 nbo=2781 |
| offset_attn | 50,003,135 | top1_ordered | 0.1484 | order-sensitive | scr=0.04165 bag=0.05953 dpp=10.68 |
| offset_attn | 50,003,135 | ppl_ordered | 3601.54 | order-sensitive | scr=64438.56 bag=19534.62 |
| offset_attn | 50,003,135 | ig | {"1": 2.7404, "2": 1.5628, "3": 1.2086, "4": 1.0595, "5": 1.0016, "6": 0.9745, "7": 0.9572, "8": 0.9451} | order-sensitive |  |
| offset_attn | 150,019,797 | top1_ordered | 0.14963 | order-sensitive | scr=0.04167 bag=0.06148 dpp=10.80 |
| offset_attn | 150,019,797 | ppl_ordered | 4751.21 | order-sensitive | scr=66129.06 bag=19102.70 |
| offset_attn | 150,019,797 | ig | {"1": 2.5918, "2": 1.3258, "3": 0.9344, "4": 0.7719, "5": 0.7053, "6": 0.6724, "7": 0.6513, "8": 0.6366} | order-sensitive |  |
| offset_attn | 500,000,000 | top1_ordered | nan | capped | skipped > 150,000,000 words (memory) |
| trajectory | 2,880,000,000 | change_bpc | 1.11595 | directional | acc=0.64169 rev=19.04186 |

## Reading guide (fragile-ideas axes)

- **char_order5/6** — the anchor scaling curve: bpc must keep dropping as data grows (does-more-data-help). Compare order 5 vs 6 at each scale.
- **offset_attn** — judged on order-sensitivity (ordered minus scrambled top-1) and offset-vs-bag, NOT bpc-vs-bigram. Watch whether the gap GROWS with words.
- **ignition** — the backoff-slice bits/word gain (where the global topic G can speak); does the gain grow with train data?
- **hetero_stack** — does the concept/word level help the char level, and does the dynamic gate beat the static pool, at the largest tractable scale?
- **trajectory** — directional change-stream model; the forward-vs-reversed gap is the directionality signal.

_Verdicts are honest, including flat/negative results._
