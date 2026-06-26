# Overnight big-data batch #2 — SYNTHESIS re-run at scale — REPORT

_Generated 2026-06-26 01:25:50_

Corpus: LM1B (`data/lm1b_ids.i8`, ~3.03 B chars / ~526 M words) for word/char experiments; enwik9 for AC (the `<page>` boundary truth); text8 for AD (the analogy families are English vocab); the Darwin/Shakespeare/Bible register files for AE.
Online only — counting / leaky accumulators / online leader-clustering; no gradient descent, no k-means/SVD/eigen, no batch optimization.
Logs: `overnight/logs2/<exp>_<scale>.log`. Machine-readable: `overnight/RESULTS2.tsv`.

## Per-experiment: small-scale vs big-scale, and did the verdict change?

### Z — similarity-hybrid

- **Small-scale (prior):** text8 18MB / 3.07M words: UNSEEN-target ppl bigram→+word-rep ≈ 20× win; RARE-ctx win too.
- **Big-scale:**
    - `unseen_ppl_win_x` = 56.509  ·  verdict **unseen-win-holds**  (scale 100,010,346; big=765758462.7 wr=13551091.3 vocab=20000 probes=120000)
    - `rare_ppl_wr` = nan  ·  verdict **unseen-win-holds**  (scale 100,010,346; big=nan clusters=4000)

### Y — noise→concept-reliance

- **Small-scale (prior):** text8 12MB: concept-reliance share rises with surface noise p (the 86%→95% headline).
- **Big-scale:**
    - `concept_share_p0` = 99.87  ·  verdict **reliance-shift-flat**  (scale 460,800,000; p0.3=97.9 shift=-1.9pp)
    - `concept_share_p0.3` = 97.93  ·  verdict **reliance-shift-flat**  (scale 460,800,000; modwt 0.275->0.236)

### AA — sleep-consolidation

- **Small-scale (prior):** text8 16MB order-6: one gentle sleep Δbpc −0.0107, mem −37%, rare-ctx bpc −0.136; aggressive schedule degrades after a turning point.
- **Big-scale:**
    - `sleep_dbpc` = 0.01221  ·  verdict **sleep-flat**  (scale 150,000,000; mem+33% rareΔ-0.127 entries=8102202)
    - `agg_best_cycle` = 0  ·  verdict **turning-point-early**  (scale 150,000,000; best_bpc=1.6859 last=2.0363 degraded)

### AB — calibrated-confidence

- **Small-scale (prior):** text8 12MB orders{2..5}: (f,c)-revision ~10× lower ECE than bare-count; Q1 ppl ≈ flat.
- **Big-scale:**
    - `ece_improvement_x` = 8.38  ·  verdict **ece-win-holds**  (scale 250,000,000; bare_ece=0.2647 rev_ece=0.0316)
    - `ppl_revision` = 6.42  ·  verdict **ece-win-holds**  (scale 250,000,000; bare_ppl=9.52)

### AC — event-model boundary F1

- **Small-scale (prior):** enwik9 36MB / 5.18M words: KL boundary F1 0.099→0.154 (grew 3→36MB); KL ≥ surprisal.
- **Big-scale:**
    - `kl_f1_tol25` = 0.4467  ·  verdict **KL-beats-surprisal**  (scale 960,000,000; surp=0.032 be=0.013 words=135617929 gold=229184)
    - `kl_f1_tol50` = 0.4799  ·  verdict **KL-beats-surprisal**  (scale 960,000,000; surp=0.084)

### AD — analogy from raw counts

- **Small-scale (prior):** text8 16MB / 2.73M words: raw-count (β=0) analogy beats leader-smoothed; restricted top-1 macro ~ high.
- **Big-scale:**
    - `analogy_raw_restricted_t1` = 0.7716  ·  verdict **raw-beats-smoothed**  (scale 16,149,974; raw_t5=0.944 smooth_t1=0.646 open_t1=0.418)
    - `analogy_raw_open_t1` = 0.4184  ·  verdict **raw-beats-smoothed**  (scale 16,149,974; open_t5=0.677 N=30000 M=6000)

### AE — non-forgetting / retention

- **Small-scale (prior):** darwin→shakespeare→bible, 120k/reg, cap 3000/order: DUAL retains ~21× better than FLAT (backward fgt).
- **Big-scale:**
    - `fgt_ratio_flat_over_dual` = 1.13  ·  verdict **dual-retention-holds**  (scale 800,000; flat=+0.947 dual=+0.840 cap=8000)

### AF — constructions

- **Small-scale (prior):** text8 14MB / 2.39M words: open-slot construction beats n-gram ppl on held-out unseen-in-frame fillers.
- **Big-scale:**
    - `open_ppl_construction` = 10793.54  ·  verdict **construction-generalizes**  (scale 100,010,346; ngram=60015.9 win%=81.9 open_frames=36751)
    - `open_win_frac` = 0.8186  ·  verdict **construction-generalizes**  (scale 100,010,346; C=400 N=12000)

### W — ray-cortex proximity

- **Small-scale (prior):** text8 15MB / 2.55M words: proximity does NOT earn its keep even on the rare/unseen slice.
- **Big-scale:**
    - `rare_prox_gain_standalone` = nan  ·  verdict **proximity-fails**  (scale 80,005,922; in_combo=nan graphN=6000 rare_n=0)

## All measurements

| exp | scale | metric | value | verdict | note |
|---|---|---|---|---|---|
| AE | 800,000 | fgt_ratio_flat_over_dual | 1.13 | dual-retention-holds | flat=+0.947 dual=+0.840 cap=8000 |
| AD | 16,149,974 | analogy_raw_restricted_t1 | 0.7716 | raw-beats-smoothed | raw_t5=0.944 smooth_t1=0.646 open_t1=0.418 |
| AD | 16,149,974 | analogy_raw_open_t1 | 0.4184 | raw-beats-smoothed | open_t5=0.677 N=30000 M=6000 |
| AB | 250,000,000 | ece_improvement_x | 8.38 | ece-win-holds | bare_ece=0.2647 rev_ece=0.0316 |
| AB | 250,000,000 | ppl_revision | 6.42 | ece-win-holds | bare_ppl=9.52 |
| AA | 150,000,000 | sleep_dbpc | 0.01221 | sleep-flat | mem+33% rareΔ-0.127 entries=8102202 |
| AA | 150,000,000 | agg_best_cycle | 0 | turning-point-early | best_bpc=1.6859 last=2.0363 degraded |
| AC | 960,000,000 | kl_f1_tol25 | 0.4467 | KL-beats-surprisal | surp=0.032 be=0.013 words=135617929 gold=229184 |
| AC | 960,000,000 | kl_f1_tol50 | 0.4799 | KL-beats-surprisal | surp=0.084 |
| Z | 100,010,346 | unseen_ppl_win_x | 56.509 | unseen-win-holds | big=765758462.7 wr=13551091.3 vocab=20000 probes=120000 |
| Z | 100,010,346 | rare_ppl_wr | nan | unseen-win-holds | big=nan clusters=4000 |
| AF | 100,010,346 | open_ppl_construction | 10793.54 | construction-generalizes | ngram=60015.9 win%=81.9 open_frames=36751 |
| AF | 100,010,346 | open_win_frac | 0.8186 | construction-generalizes | C=400 N=12000 |
| W | 80,005,922 | rare_prox_gain_standalone | nan | proximity-fails | in_combo=nan graphN=6000 rare_n=0 |
| Y | 460,800,000 | concept_share_p0 | 99.87 | reliance-shift-flat | p0.3=97.9 shift=-1.9pp |
| Y | 460,800,000 | concept_share_p0.3 | 97.93 | reliance-shift-flat | modwt 0.275->0.236 |

_Verdicts are honest, including 'no change', flat, and negative results (a flat/negative big-scale result is still data — the fragile-ideas bet)._
